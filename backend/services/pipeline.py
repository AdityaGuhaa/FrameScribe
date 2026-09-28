"""Orchestrates the full video processing pipeline."""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.services.audio_extractor import AudioExtractor
from backend.services.transcriber import Transcriber
from backend.services.diarizer import Diarizer
from backend.models.schemas import TranscriptResult, JobStatus
from backend.services.visual_analyzer import VisualAnalyzer

logger = logging.getLogger(__name__)

class Pipeline:
    """Orchestrates the full video processing pipeline.
    
    Pipeline stages:
    1. Extract audio from video (FFmpeg → 16kHz mono WAV)
    2. Transcribe speech (faster-whisper → timestamped segments)
    3. Diarize speakers + Visual Analysis (in parallel)
    4. Merge all results into enriched TranscriptResult
    """

    def __init__(self, config):
        """Initialize pipeline with all processing services."""
        self.config = config
        self.audio_extractor = AudioExtractor(output_dir=config.audio_dir)
        self.transcriber = Transcriber(model_size=config.model_size, device=config.device)
        self.diarizer = Diarizer(hf_token=config.hf_token)
        # Use VisualAnalyzer instead of PySceneDetect
        self.scene_detector = VisualAnalyzer(output_dir=config.frames_dir, device=config.device)

    async def process(self, video_path: str, job_id: str, status_callback=None) -> TranscriptResult:
        """Run the full processing pipeline on a video file.
        
        Args:
            video_path: Path to the input video file
            job_id: Unique identifier for this processing job
            status_callback: Optional callback function receiving JobStatus updates
            
        Returns:
            TranscriptResult with enriched transcript data
        """
        try:
            def update_status(stage: str, progress: float):
                if status_callback:
                    status_callback(JobStatus(
                        job_id=job_id,
                        status=stage,
                        progress=progress,
                        stage_progress=progress,
                        current_stage=stage
                    ))

            # Stage 1: Extract Audio
            logger.info(f"[{job_id}] Stage 1: Extracting audio...")
            update_status("extracting_audio", 0.0)
            audio_path = await self.audio_extractor.extract(
                video_path,
                progress_callback=lambda p: update_status("extracting_audio", p)
            )
            duration = self.audio_extractor.get_duration(video_path)
            logger.info(f"[{job_id}] Audio extracted. Duration: {duration:.1f}s")

            # Stage 2: Transcribe
            logger.info(f"[{job_id}] Stage 2: Transcribing speech...")
            update_status("transcribing", 0.0)
            segments = await self.transcriber.transcribe(
                audio_path,
                progress_callback=lambda p: update_status("transcribing", p)
            )
            logger.info(f"[{job_id}] Transcription complete. {len(segments)} segments.")

            # Stage 3: Diarize + Scene Detection (in parallel)
            logger.info(f"[{job_id}] Stage 3: Diarizing speakers & detecting scenes (parallel)...")
            update_status("diarizing_and_scenes", 0.0)

            diarize_task = self.diarizer.diarize(audio_path)
            scenes_task = self.scene_detector.detect_scenes(video_path)

            speaker_turns, (scenes_summary, scenes_count) = await asyncio.gather(
                diarize_task, scenes_task
            )
            logger.info(f"[{job_id}] Diarization: {len(speaker_turns)} speaker turns. "
                        f"Scenes: {scenes_count} detected.")

            # Stage 4: Merge results
            logger.info(f"[{job_id}] Stage 4: Merging results...")
            update_status("merging", 0.0)

            segments = self.diarizer.assign_speakers(segments, speaker_turns)
            segments = self.scene_detector.assign_scenes(segments, scenes_summary)

            # Compute metadata
            languages = list(set(s.language for s in segments if s.language))
            speakers = len(set(s.speaker for s in segments if s.speaker))
            full_text = " ".join(s.text for s in segments)

            result = TranscriptResult(
                video_file=Path(video_path).name,
                duration_seconds=duration,
                languages=languages,
                speakers_detected=max(1, speakers),
                scenes_detected=scenes_count,
                created_at=datetime.now(timezone.utc).isoformat(),
                segments=segments,
                scenes_summary=scenes_summary,
                full_text=full_text
            )

            update_status("completed", 1.0)
            logger.info(f"[{job_id}] Pipeline completed successfully.")
            return result

        except Exception as e:
            logger.error(f"[{job_id}] Pipeline failed: {e}", exc_info=True)
            if status_callback:
                status_callback(JobStatus(
                    job_id=job_id,
                    status="failed",
                    error=str(e),
                    current_stage="failed"
                ))
            raise

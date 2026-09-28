import asyncio
import logging
from pyannote.audio import Pipeline
from backend.models.schemas import TranscriptSegment

logger = logging.getLogger(__name__)

class Diarizer:
    """Service for speaker diarization using pyannote-audio."""

    def __init__(self, hf_token: str | None = None):
        self.hf_token = hf_token
        self.pipeline = None
        self.available = True

    def _load_pipeline(self):
        if self.pipeline is None and self.available:
            if not self.hf_token:
                logger.warning("HF_TOKEN not set. Diarization will be disabled.")
                self.available = False
                return
            
            try:
                logger.info("Loading pyannote diarization pipeline")
                self.pipeline = Pipeline.from_pretrained(
                    'pyannote/speaker-diarization-3.1',
                    use_auth_token=self.hf_token
                )
            except Exception as e:
                logger.error(f"Failed to load pyannote pipeline: {e}")
                self.available = False

    async def diarize(self, audio_path: str, progress_callback=None) -> list[dict]:
        """Performs speaker diarization on an audio file."""
        self._load_pipeline()

        if not self.available:
            logger.warning("Diarizer unavailable, returning single segment.")
            return [{"speaker": "Speaker 1", "start": 0.0, "end": 999999.0}]

        def _run_diarization():
            logger.info(f"Starting diarization for {audio_path}")
            diarization = self.pipeline(audio_path)
            speaker_turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_turns.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end
                })
            return speaker_turns

        if progress_callback:
            progress_callback(0.1)

        results = await asyncio.to_thread(_run_diarization)
        
        if progress_callback:
            progress_callback(1.0)
            
        return results

    def assign_speakers(self, segments: list[TranscriptSegment], speaker_turns: list[dict]) -> list[TranscriptSegment]:
        """Assigns speaker labels to transcript segments based on maximum overlap."""
        if not speaker_turns:
            return segments

        for segment in segments:
            best_speaker = "Speaker 1"
            max_overlap = 0.0

            for turn in speaker_turns:
                overlap_start = max(segment.start, turn["start"])
                overlap_end = min(segment.end, turn["end"])
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = turn["speaker"]
            
            segment.speaker = best_speaker

        return segments

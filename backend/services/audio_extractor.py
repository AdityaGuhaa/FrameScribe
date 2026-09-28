import os
import asyncio
import logging
import ffmpeg
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioExtractor:
    """Service for extracting audio from video files."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def extract(self, video_path: str, progress_callback=None) -> str:
        """
        Extracts audio from video to 16kHz mono WAV.
        Returns the path to the extracted WAV file.
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_audio = self.output_dir / f"{video_file.stem}.wav"

        def _run_ffmpeg():
            try:
                (
                    ffmpeg
                    .input(str(video_path))
                    .output(str(output_audio), ac=1, ar='16k', format='wav', loglevel='error')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as e:
                logger.error(f"FFmpeg error: {e.stderr.decode('utf-8')}")
                raise RuntimeError("Failed to extract audio from video")

        logger.info(f"Extracting audio from {video_path} to {output_audio}")
        if progress_callback:
            progress_callback(0.1)

        await asyncio.to_thread(_run_ffmpeg)

        if progress_callback:
            progress_callback(1.0)

        return str(output_audio)

    def get_duration(self, file_path: str) -> float:
        """Gets media duration using ffprobe."""
        try:
            probe = ffmpeg.probe(str(file_path))
            duration = float(probe['format']['duration'])
            return duration
        except ffmpeg.Error as e:
            logger.error(f"ffprobe error: {e.stderr.decode('utf-8')}")
            raise RuntimeError("Failed to get duration")

import asyncio
import logging
from faster_whisper import WhisperModel
from backend.models.schemas import TranscriptSegment, WordSegment

logger = logging.getLogger(__name__)

class Transcriber:
    """Service for transcribing audio using faster-whisper."""

    def __init__(self, model_size: str = 'large-v3', device: str = 'auto'):
        self.model_size = model_size
        self.device = device
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading Whisper model {self.model_size} on {self.device}")
            compute_type = "float16" if self.device == "cuda" else "int8"
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type
            )

    async def transcribe(self, audio_path: str, progress_callback=None) -> list[TranscriptSegment]:
        """Transcribes audio file to word-level timestamped segments."""
        self._load_model()

        def _run_transcription():
            logger.info(f"Starting transcription for {audio_path}")
            segments_generator, info = self.model.transcribe(
                audio_path,
                word_timestamps=True
            )
            
            segments = []
            total_duration = getattr(info, "duration", 1.0)
            
            for idx, segment in enumerate(segments_generator):
                words = []
                if segment.words:
                    for w in segment.words:
                        words.append(WordSegment(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            confidence=w.probability
                        ))
                
                seg = TranscriptSegment(
                    id=idx + 1,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    language=info.language or "en",
                    confidence=segment.no_speech_prob,
                    words=words
                )
                segments.append(seg)
                
                if progress_callback and total_duration > 0:
                    progress = min(1.0, segment.end / total_duration)
                    progress_callback(progress)
            
            return segments

        return await asyncio.to_thread(_run_transcription)

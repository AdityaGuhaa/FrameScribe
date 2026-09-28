"""Application configuration for VideoAI backend."""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    """Central configuration for VideoAI backend."""

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.output_dir = self.base_dir / "output"
        self.audio_dir = self.output_dir / "audio"
        self.frames_dir = self.output_dir / "frames"
        self.exports_dir = self.output_dir / "exports"
        self.data_dir = self.base_dir / "data"
        self.uploads_dir = self.data_dir / "uploads"
        self.results_dir = self.data_dir / "results"

        # Create all directories
        for d in [self.audio_dir, self.frames_dir, self.exports_dir,
                  self.uploads_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Device detection
        self.device = self._detect_device()

        # Model settings
        self.model_size = os.getenv("WHISPER_MODEL_SIZE", "large-v3")

        # HuggingFace token for pyannote diarization (optional)
        self.hf_token = os.getenv("HF_TOKEN")

        logger.info(f"Config initialized: device={self.device}, model={self.model_size}, "
                     f"diarization={'enabled' if self.hf_token else 'disabled'}")

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available compute device."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA device detected: {device_name}")
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Apple MPS device detected")
                return "cpu"  # faster-whisper uses CTranslate2 which doesn't support MPS directly
            else:
                logger.info("No GPU detected, using CPU")
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not available, defaulting to CPU")
            return "cpu"


# Global config instance
config = Config()

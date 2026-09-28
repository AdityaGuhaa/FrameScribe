"""FastAPI application entry point for FrameScribe."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.config import config
from backend.routers import transcribe, export
from backend.services.pipeline import Pipeline
from backend.db.database import Database

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FrameScribe",
    description="Video-to-Text Intelligence Pipeline — Extract speech transcripts "
                "with visual context from videos.",
    version="0.1.0"
)

# CORS middleware for local Electron app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(transcribe.router)
app.include_router(export.router)

# Shared instances
db = Database(db_path=str(config.data_dir / "videoai.db"))


@app.on_event("startup")
async def startup_event():
    """Initialize database, pipeline, and inject into routers."""
    logger.info("Starting FrameScribe backend...")

    # Initialize database
    await db.init()
    transcribe.db = db

    # Initialize pipeline (model loading is lazy — happens on first request)
    pipeline = Pipeline(config)
    transcribe.pipeline = pipeline

    logger.info(f"FrameScribe ready — device: {config.device}, model: {config.model_size}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down FrameScribe backend...")


@app.get("/api/health")
async def health_check():
    """Health check with system info."""
    return {
        "status": "healthy",
        "device": config.device,
        "model": config.model_size,
        "diarization_available": bool(config.hf_token)
    }


@app.get("/")
async def root():
    """Root endpoint with basic app info."""
    return {
        "name": "FrameScribe",
        "version": "0.1.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

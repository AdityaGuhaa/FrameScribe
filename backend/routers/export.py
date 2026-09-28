"""Router for exporting transcripts to different formats."""

import os
import json
import logging
from io import BytesIO
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.models.schemas import ExportRequest, ExportResponse, TranscriptResult
from backend.services.formatter import Formatter

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


@router.post("/export", response_model=ExportResponse)
async def export_transcript(request: ExportRequest):
    """Export a transcript result to the requested format.
    
    Supported formats: json, srt, vtt, txt, ai_ready
    """
    try:
        return Formatter.export(request.transcript, request.format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Export failed")


@router.get("/export/download/{job_id}/{export_format}")
async def download_export(job_id: str, export_format: str):
    """Download a completed job's transcript in the specified format.
    
    Loads the result from disk and streams it as a file download.
    """
    result_path = os.path.join("data/results", f"{job_id}.json")
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result not found")

    try:
        with open(result_path, "r") as f:
            result = TranscriptResult.model_validate_json(f.read())
    except Exception as e:
        logger.error(f"Failed to load result: {e}")
        raise HTTPException(status_code=500, detail="Failed to load result")

    try:
        export_result = Formatter.export(result, export_format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    file_stream = BytesIO(export_result.content.encode("utf-8"))

    return StreamingResponse(
        file_stream,
        media_type=export_result.mime_type,
        headers={"Content-Disposition": f"attachment; filename={export_result.filename}"}
    )

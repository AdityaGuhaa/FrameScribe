"""Router for transcription and project endpoints."""

import os
import uuid
import asyncio
import logging
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect

from backend.models.schemas import JobStatus, TranscriptResult

# Shared variables, initialized by main.py
pipeline = None
db = None

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

jobs: Dict[str, JobStatus] = {}
results: Dict[str, TranscriptResult] = {}

@router.post("/transcribe", response_model=Dict[str, str])
async def transcribe_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts a video file, generates a job ID, and starts the transcription pipeline."""
    job_id = str(uuid.uuid4())
    
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{job_id}_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    initial_status = JobStatus(job_id=job_id, status="pending", progress=0.0)
    jobs[job_id] = initial_status

    if db:
        await db.save_project(job_id, file.filename, 0.0, "pending", "")

    def status_callback(status: JobStatus):
        """Update in-memory job status. Called from worker threads, so no async ops here."""
        jobs[job_id] = status

    async def process_task():
        try:
            if pipeline:
                result = await pipeline.process(file_path, job_id, status_callback)
                results[job_id] = result

                result_dir = "data/results"
                os.makedirs(result_dir, exist_ok=True)
                result_path = os.path.join(result_dir, f"{job_id}.json")
                with open(result_path, "w") as f:
                    f.write(result.model_dump_json(indent=2))

                if db:
                    await db.save_project(
                        job_id, file.filename,
                        result.duration_seconds, "completed", result_path
                    )
            else:
                logger.error("Pipeline not initialized")
                jobs[job_id] = JobStatus(
                    job_id=job_id, status="failed",
                    error="Pipeline not initialized", current_stage="failed"
                )
                if db:
                    await db.update_status(job_id, "failed")
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}", exc_info=True)
            jobs[job_id] = JobStatus(
                job_id=job_id, status="failed",
                error=str(e), current_stage="failed"
            )
            if db:
                await db.update_status(job_id, "failed")

    background_tasks.add_task(process_task)
    return {"job_id": job_id}

@router.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Returns the current status of a transcription job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@router.get("/result/{job_id}", response_model=TranscriptResult)
async def get_result(job_id: str):
    """Returns the final transcript result if the job is completed."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = jobs[job_id].status
    if status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
        
    if job_id not in results:
        # Try to load from disk if memory is cleared
        result_path = os.path.join("data/results", f"{job_id}.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, "r") as f:
                    return TranscriptResult.model_validate_json(f.read())
            except Exception as e:
                logger.error(f"Failed to load result: {e}")
                raise HTTPException(status_code=500, detail="Failed to load result")
        raise HTTPException(status_code=404, detail="Result not found")
        
    return results[job_id]

@router.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """Sends job status updates over a WebSocket connection."""
    await websocket.accept()
    if job_id not in jobs:
        await websocket.close(code=1008, reason="Job not found")
        return
        
    try:
        while True:
            current_status = jobs.get(job_id)
            if not current_status:
                break
                
            await websocket.send_text(current_status.model_dump_json())
            
            if current_status.status in ["completed", "failed"]:
                break
                
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from progress stream for job {job_id}")
    finally:
        # websocket may already be closed if disconnect exception raised
        try:
             await websocket.close()
        except Exception:
             pass

@router.get("/projects", response_model=list[dict])
async def list_projects():
    """Returns a list of all projects."""
    if db:
        return await db.get_projects()
    return []

@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Deletes a project and its associated files."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
        
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    await db.delete_project(project_id)
    
    if project.get("result_path") and os.path.exists(project["result_path"]):
        os.remove(project["result_path"])
        
    upload_path = os.path.join("data/uploads", f"{project_id}_{project.get('video_file', '')}")
    if os.path.exists(upload_path):
         os.remove(upload_path)
         
    return {"status": "success"}

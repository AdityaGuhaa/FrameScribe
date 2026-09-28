from pydantic import BaseModel, Field
from typing import List, Optional

class WordSegment(BaseModel):
    word: str
    start: float
    end: float
    confidence: float

class VisualContext(BaseModel):
    scene_id: int
    description: str
    key_frame: str

class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str = 'Speaker 1'
    language: str = 'en'
    confidence: float = 0.0
    words: List[WordSegment] = Field(default_factory=list)
    visual_context: Optional[VisualContext] = None

class SceneSummary(BaseModel):
    scene_id: int
    start: float
    end: float
    key_frame: str

class TranscriptResult(BaseModel):
    video_file: str
    duration_seconds: float
    languages: List[str] = Field(default_factory=list)
    speakers_detected: int = 1
    scenes_detected: int = 0
    created_at: str
    segments: List[TranscriptSegment]
    scenes_summary: List[SceneSummary] = Field(default_factory=list)
    full_text: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    stage_progress: float = 0.0
    current_stage: str = ''
    error: Optional[str] = None
    result: Optional[TranscriptResult] = None

class ExportRequest(BaseModel):
    transcript: TranscriptResult
    format: str

class ExportResponse(BaseModel):
    content: str
    filename: str
    mime_type: str

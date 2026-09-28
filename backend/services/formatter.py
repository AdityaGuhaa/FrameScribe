import json
from backend.models.schemas import TranscriptResult, ExportResponse

class Formatter:
    """Service for formatting transcript results into various export formats."""

    @staticmethod
    def _format_time_srt(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

    @staticmethod
    def _format_time_vtt(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{msecs:03d}"

    @staticmethod
    def to_json(result: TranscriptResult) -> str:
        return result.model_dump_json(indent=2)

    @staticmethod
    def to_srt(result: TranscriptResult) -> str:
        lines = []
        for i, segment in enumerate(result.segments, start=1):
            lines.append(str(i))
            start = Formatter._format_time_srt(segment.start)
            end = Formatter._format_time_srt(segment.end)
            lines.append(f"{start} --> {end}")
            text = segment.text
            if segment.speaker:
                text = f"[{segment.speaker}] {text}"
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_vtt(result: TranscriptResult) -> str:
        lines = ["WEBVTT", ""]
        for i, segment in enumerate(result.segments, start=1):
            lines.append(str(i))
            start = Formatter._format_time_vtt(segment.start)
            end = Formatter._format_time_vtt(segment.end)
            lines.append(f"{start} --> {end}")
            text = segment.text
            if segment.speaker:
                text = f"[{segment.speaker}] {text}"
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_txt(result: TranscriptResult) -> str:
        lines = []
        for segment in result.segments:
            speaker = segment.speaker or "Speaker 1"
            lines.append(f"[{speaker}] {segment.text}")
        return "\n".join(lines)

    @staticmethod
    def to_ai_ready(result: TranscriptResult) -> str:
        lines = [
            "=== VIDEO METADATA ===",
            f"File: {result.video_file}",
            f"Duration: {result.duration_seconds}s",
            f"Languages: {', '.join(result.languages)}",
            f"Speakers: {result.speakers_detected}",
            f"Scenes: {result.scenes_detected}",
            "======================",
            ""
        ]
        
        for segment in result.segments:
            context = ""
            if segment.visual_context:
                context = f" [Scene {segment.visual_context.scene_id}]"
            speaker = segment.speaker or "Speaker 1"
            time = f"[{segment.start:.1f} - {segment.end:.1f}]"
            
            lines.append(f"{time}{context} {speaker}: {segment.text}")
            
        return "\n".join(lines)

    @staticmethod
    def export(result: TranscriptResult, format: str) -> ExportResponse:
        """Dispatches to correct formatter and returns ExportResponse."""
        base_name = "export"
        
        if format == "json":
            return ExportResponse(content=Formatter.to_json(result), filename=f"{base_name}.json", mime_type="application/json")
        elif format == "srt":
            return ExportResponse(content=Formatter.to_srt(result), filename=f"{base_name}.srt", mime_type="text/plain")
        elif format == "vtt":
            return ExportResponse(content=Formatter.to_vtt(result), filename=f"{base_name}.vtt", mime_type="text/vtt")
        elif format == "txt":
            return ExportResponse(content=Formatter.to_txt(result), filename=f"{base_name}.txt", mime_type="text/plain")
        elif format == "ai_ready":
            return ExportResponse(content=Formatter.to_ai_ready(result), filename=f"{base_name}_ai.txt", mime_type="text/plain")
        else:
            raise ValueError(f"Unknown export format: {format}")

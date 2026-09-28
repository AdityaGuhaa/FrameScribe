import cv2
import asyncio
import logging
from pathlib import Path
from scenedetect import detect, ContentDetector
from backend.models.schemas import SceneSummary, TranscriptSegment, VisualContext

logger = logging.getLogger(__name__)

class SceneDetector:
    """Service for detecting scene changes and extracting keyframes."""

    def __init__(self, output_dir: str | Path, threshold: float = 27.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

    async def detect_scenes(self, video_path: str, progress_callback=None) -> tuple[list[SceneSummary], int]:
        """Detects scenes in a video and extracts keyframes."""
        def _run_detection():
            logger.info(f"Detecting scenes in {video_path}")
            scene_list = detect(video_path, ContentDetector(threshold=self.threshold))
            
            scenes = []
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            for i, scene in enumerate(scene_list):
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                
                mid_time = start_time + (end_time - start_time) / 2.0
                frame_num = int(mid_time * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                key_frame_path = ""
                if ret:
                    key_frame_filename = f"scene_{i+1:04d}.jpg"
                    key_frame_path = str(self.output_dir / key_frame_filename)
                    cv2.imwrite(key_frame_path, frame)
                
                scenes.append(SceneSummary(
                    scene_id=i+1,
                    start=start_time,
                    end=end_time,
                    key_frame=key_frame_path
                ))
            
            cap.release()
            return scenes, len(scenes)

        if progress_callback:
            progress_callback(0.1)

        results = await asyncio.to_thread(_run_detection)

        if progress_callback:
            progress_callback(1.0)
            
        return results

    def assign_scenes(self, segments: list[TranscriptSegment], scenes: list[SceneSummary]) -> list[TranscriptSegment]:
        """Assigns visual context to transcript segments based on time."""
        if not scenes:
            return segments

        for segment in segments:
            for scene in scenes:
                if scene.start <= segment.start <= scene.end:
                    segment.visual_context = VisualContext(
                        scene_id=scene.scene_id,
                        description=f"Scene {scene.scene_id}",
                        key_frame=scene.key_frame
                    )
                    break
        return segments

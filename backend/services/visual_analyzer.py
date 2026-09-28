import os
import logging
import asyncio
import ffmpeg
from models.schemas import SceneSummary, VisualContext, TranscriptSegment

logger = logging.getLogger(__name__)

class VisualAnalyzer:
    def __init__(self, output_dir, device="cpu"):
        self.output_dir = output_dir
        self.device = device
        self.model = None
        self.processor = None
        self.config = None
        self.last_analysis = []
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self):
        if self.model is None:
            logger.info("Loading Qwen2-VL-2B-Instruct model via Apple MLX...")
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            
            # Use MLX native loading (bypasses PyTorch entirely)
            model_path = "Qwen/Qwen2-VL-2B-Instruct"
            self.model, self.processor = load(model_path)
            self.config = load_config(model_path)
            logger.info("Qwen2-VL model loaded successfully natively on Mac GPU (MLX).")

    def _extract_frames(self, video_path: str) -> list[str]:
        """Extracts 1 frame per second from the video using FFmpeg and downscales to save memory."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        scene_dir = os.path.join(self.output_dir, f"{video_name}_frames")
        os.makedirs(scene_dir, exist_ok=True)

        logger.info(f"Extracting 1 FPS from {video_path}...")
        try:
            (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=1)
                .filter('scale', 512, -1) # Downscale!
                .output(os.path.join(scene_dir, "frame-%04d.jpg"), **{'qscale:v': 2})
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error extracting frames: {e.stderr.decode()}")
            raise

        frames = sorted([
            os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if f.endswith(".jpg")
        ])
        logger.info(f"Extracted {len(frames)} frames for analysis.")
        return frames

    def _analyze_frames(self, frames: list[str]) -> list[dict]:
        """Runs Qwen-VL via MLX on each frame."""
        self._load_model()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        
        results = []
        
        # Prepare the MLX text prompt
        prompt = "Describe what is happening in this image in one brief, literal sentence. Keep it under 10 words. Do not start with 'This is a picture of'."
        formatted_prompt = apply_chat_template(
            self.processor, self.config, prompt, num_images=1
        )
        
        for i, frame_path in enumerate(frames):
            second = i
            logger.info(f"Analyzing frame at {second}s...")
            
            # MLX handles the image preprocessing and inference natively on the Mac GPU
            output = generate(
                self.model,
                self.processor,
                image=[frame_path],
                prompt=formatted_prompt,
                max_tokens=20,
                verbose=False
            )
            
            output_text = output.strip()
            
            results.append({
                "second": second,
                "description": output_text,
                "frame_path": frame_path
            })
            
        return results

    async def detect_scenes(self, video_path: str, progress_callback=None) -> tuple[list[SceneSummary], int]:
        """Extracts frames, analyzes them, and groups identical sequential descriptions into 'scenes'."""
        
        def _run_analysis():
            frames = self._extract_frames(video_path)
            if not frames:
                return [], 0
                
            raw_analysis = self._analyze_frames(frames)
            
            # Grouping logic: merge consecutive identical (or very similar) descriptions
            scenes = []
            current_scene = None
            
            for item in raw_analysis:
                sec = item["second"]
                desc = item["description"]
                
                normalized_desc = desc.lower().strip().rstrip('.')
                
                if current_scene is None:
                    current_scene = {
                        "scene_id": 1,
                        "start": float(sec),
                        "end": float(sec + 1),
                        "description": desc,
                        "norm_desc": normalized_desc,
                        "key_frame": item["frame_path"]
                    }
                elif current_scene["norm_desc"] == normalized_desc:
                    current_scene["end"] = float(sec + 1)
                else:
                    scenes.append(SceneSummary(
                        scene_id=current_scene["scene_id"],
                        start=current_scene["start"],
                        end=current_scene["end"],
                        key_frame=current_scene["key_frame"]
                    ))
                    
                    current_scene = {
                        "scene_id": current_scene["scene_id"] + 1,
                        "start": float(sec),
                        "end": float(sec + 1),
                        "description": desc,
                        "norm_desc": normalized_desc,
                        "key_frame": item["frame_path"]
                    }
            
            if current_scene:
                scenes.append(SceneSummary(
                    scene_id=current_scene["scene_id"],
                    start=current_scene["start"],
                    end=current_scene["end"],
                    key_frame=current_scene["key_frame"]
                ))
                
            return scenes, len(scenes), raw_analysis
            
        if progress_callback:
            progress_callback(0.1)

        scenes, count, raw_analysis = await asyncio.to_thread(_run_analysis)
        self.last_analysis = raw_analysis

        if progress_callback:
            progress_callback(1.0)
            
        return scenes, count

    def assign_scenes(self, segments: list[TranscriptSegment], scenes: list[SceneSummary]) -> list[TranscriptSegment]:
        """Assigns visual context (with Qwen description) to transcript segments based on time."""
        if not scenes:
            return segments

        scene_desc_map = {}
        if hasattr(self, 'last_analysis'):
            current_id = 1
            last_norm = ""
            for item in self.last_analysis:
                norm = item["description"].lower().strip().rstrip('.')
                if norm != last_norm and item["second"] > 0:
                    current_id += 1
                scene_desc_map[current_id] = item["description"]
                last_norm = norm

        for segment in segments:
            best_scene = None
            max_overlap = 0
            
            for scene in scenes:
                overlap_start = max(segment.start, scene.start)
                overlap_end = min(segment.end, scene.end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap or (best_scene is None and scene.start <= segment.start <= scene.end):
                    max_overlap = overlap
                    best_scene = scene
            
            if best_scene:
                desc = scene_desc_map.get(best_scene.scene_id, f"Scene {best_scene.scene_id}")
                segment.visual_context = VisualContext(
                    scene_id=best_scene.scene_id,
                    description=desc,
                    key_frame=best_scene.key_frame
                )
                
        return segments

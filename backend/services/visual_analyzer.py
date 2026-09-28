import os
import asyncio
import logging
import ffmpeg
import torch
from pathlib import Path
from PIL import Image

from backend.models.schemas import SceneSummary, TranscriptSegment, VisualContext

logger = logging.getLogger(__name__)

class VisualAnalyzer:
    """Service for analyzing video frames every second using Qwen-VL."""

    def __init__(self, output_dir: str | Path, device: str = "cpu"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.model = None
        self.processor = None

    def _load_model(self):
        if self.model is None:
            logger.info("Loading Qwen2-VL-2B-Instruct model (this may take a moment on first run)...")
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            # Determine precision based on device (bfloat16 for CUDA, float16 for Mac Apple Silicon)
            dtype = torch.float32
            if self.device == "cuda":
                dtype = torch.bfloat16
            elif self.device == "cpu" and torch.backends.mps.is_available():
                # We fallback to CPU for some ops but can keep model in float16 for memory
                dtype = torch.float16

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct", 
                torch_dtype=dtype
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            logger.info("Qwen2-VL model loaded successfully.")

    def _extract_frames(self, video_path: str) -> list[str]:
        """Extracts 1 frame per second from the video using FFmpeg."""
        video_file = Path(video_path)
        logger.info(f"Extracting 1 FPS from {video_path}...")
        
        # We save frames as frame_0001.jpg, frame_0002.jpg, etc.
        output_pattern = str(self.output_dir / f"{video_file.stem}_frame_%04d.jpg")
        
        try:
            (
                ffmpeg
                .input(str(video_path))
                .filter('fps', fps=1)
                .output(output_pattern, loglevel='error')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode('utf-8')}")
            raise RuntimeError("Failed to extract frames for visual analysis")

        frames = sorted([str(p) for p in self.output_dir.glob(f"{video_file.stem}_frame_*.jpg")])
        logger.info(f"Extracted {len(frames)} frames for analysis.")
        return frames

    def _analyze_frames(self, frames: list[str]) -> list[dict]:
        """Runs Qwen-VL on each frame to generate a description."""
        self._load_model()
        from qwen_vl_utils import process_vision_info
        
        results = []
        for i, frame_path in enumerate(frames):
            second = i # Since it's 1 FPS, index i = second i
            logger.info(f"Analyzing frame at {second}s...")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_path},
                        {"type": "text", "text": "Describe what is happening in this image in one brief, literal sentence. Keep it under 10 words. Do not start with 'This is a picture of'."}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=20)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
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
                
                # Normalize description for comparison (lowercase, remove punctuation)
                normalized_desc = desc.lower().strip().rstrip('.')
                
                if current_scene is None:
                    current_scene = {
                        "scene_id": 1,
                        "start": float(sec),
                        "end": float(sec + 1), # spans 1 second
                        "description": desc,
                        "norm_desc": normalized_desc,
                        "key_frame": item["frame_path"]
                    }
                elif current_scene["norm_desc"] == normalized_desc:
                    # Same description, extend the scene
                    current_scene["end"] = float(sec + 1)
                else:
                    # Description changed, save old scene and start a new one
                    scenes.append(SceneSummary(
                        scene_id=current_scene["scene_id"],
                        start=current_scene["start"],
                        end=current_scene["end"],
                        key_frame=current_scene["key_frame"]
                        # Note: we temporarily hijack key_frame or rely on mapping later
                    ))
                    # Attach description to the key_frame field as a hack, or we need to map it properly.
                    # Wait, SceneSummary schema doesn't have a 'description' field natively, but VisualContext does.
                    # We will store the description in a dictionary first, then create the SceneSummary.
                    
                    current_scene = {
                        "scene_id": current_scene["scene_id"] + 1,
                        "start": float(sec),
                        "end": float(sec + 1),
                        "description": desc,
                        "norm_desc": normalized_desc,
                        "key_frame": item["frame_path"]
                    }
            
            # Append the last scene
            if current_scene:
                scenes.append(SceneSummary(
                    scene_id=current_scene["scene_id"],
                    start=current_scene["start"],
                    end=current_scene["end"],
                    key_frame=current_scene["key_frame"]
                ))
                
            # We need a way to pass the descriptions to the assign_scenes method.
            # We'll attach a custom attribute dynamically, or just use the class state.
            # Python allows setting attributes dynamically on instances.
            for s, c_scene in zip(scenes, [s for s in scenes]): # Wait, this isn't right.
                pass 
                
            # Let's map it safely by returning a tuple of (scenes, dict_mapping) 
            # Actually, to keep Pipeline signature happy, let's inject description directly into SceneSummary dynamically
            # or update the assign_scenes method.
            
            return scenes, len(scenes), raw_analysis
            
        if progress_callback:
            progress_callback(0.1)

        scenes, count, raw_analysis = await asyncio.to_thread(_run_analysis)
        self.last_analysis = raw_analysis # Store for assign_scenes

        if progress_callback:
            progress_callback(1.0)
            
        return scenes, count

    def assign_scenes(self, segments: list[TranscriptSegment], scenes: list[SceneSummary]) -> list[TranscriptSegment]:
        """Assigns visual context (with Qwen description) to transcript segments based on time."""
        if not scenes:
            return segments

        # Build a lookup from scene_id to description
        scene_desc_map = {}
        if hasattr(self, 'last_analysis'):
            # Group by scene_id logic
            current_id = 1
            last_norm = ""
            for item in self.last_analysis:
                norm = item["description"].lower().strip().rstrip('.')
                if norm != last_norm and item["second"] > 0:
                    current_id += 1
                scene_desc_map[current_id] = item["description"]
                last_norm = norm

        for segment in segments:
            # Find which scene overlaps most with the segment
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

import os
import json
import time
os.makedirs("data/processed", exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
import gc
import cv2

class VisionEngine:
    FRAME_INTERVAL_SECONDS = 10
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[VisionEngine] Initializing SAM 2 on {self.device}...")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            self.model = build_sam2("configs/sam2/sam2_hiera_t.yaml", "sam2_hiera_tiny.pt", device=self.device)
            
            if isinstance(self.model, dict):
                raise ValueError("[VisionEngine] Config missing from paths.")
            
            if self.device == "cuda":
                self.model.half()
            
            self.predictor = SAM2ImagePredictor(self.model)
            print("[VisionEngine] SAM 2 loaded.")
        except Exception as e:
            print(f"[VisionEngine] Failed to initialize: {e}")
            self.predictor = None

    def resize_image(self, image_np: np.ndarray, max_size: int = 640) -> np.ndarray:
        """Resizes the image array so its longest side is max_size, maintaining aspect ratio."""
        height, width = image_np.shape[:2]
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            return cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        return image_np

    def process_frame(self, frame_np: np.ndarray, output_path: str = None) -> str:
        """
        Processes a single frame. Uses try-except to catch processing and OOM errors,
        and ensures memory is freed up in finally block.
        """
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            img_np = self.resize_image(frame_rgb, max_size=640)
            original = img_np.copy()
            
            with torch.no_grad():
                h, w = img_np.shape[:2]
                masks, scores_array = None, None
                
                if self.predictor is not None:
                    input_tensor = torch.from_numpy(img_np).to(device=self.device, dtype=torch.float16)
                    with torch.amp.autocast("cuda"):
                        self.predictor.set_image(img_np)
                        
                    points = [
                        [w * 0.25, h * 0.25], [w * 0.75, h * 0.25],
                        [w * 0.25, h * 0.75], [w * 0.75, h * 0.75]
                    ]
                    
                    all_masks = []
                    all_scores = []
                    
                    with torch.amp.autocast("cuda"):
                        for pt in points:
                            point_coords = np.array([[pt[0], pt[1]]])
                            point_labels = np.array([1])
                            
                            m_out, s_out, _ = self.predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                                multimask_output=True
                            )
                            
                            if m_out is not None and len(m_out) > 0:
                                best_idx = np.argmax(s_out)
                                best_mask = m_out[best_idx]
                                best_score = s_out[best_idx]
                                
                                if isinstance(best_mask, torch.Tensor):
                                    mask_bool = (best_mask > 0.0).cpu().numpy().astype(bool)
                                else:
                                    mask_bool = (best_mask > 0.0).astype(bool)
                                
                                all_masks.append(mask_bool)
                                all_scores.append(best_score)
                                        
                    if len(all_masks) > 0:
                        masks = np.array(all_masks)
                        scores_array = np.array(all_scores)
                
                num_masks = 0
                if masks is not None:
                    num_masks = masks.shape[0]
                    
                scores = []
                if scores_array is not None:
                    scores = list(scores_array)
                    
                mask_colors: list[list[int] | None] = []
                if masks is not None:
                    for i in range(num_masks):
                        if i < len(scores) and scores[i] <= 0.6:
                            mask_colors.append(None)
                            continue
                            
                        mask = masks[i]
                        bright_color = [int(np.random.randint(100, 255)) for _ in range(3)]
                        mask_colors.append(bright_color)
                        
                        img_np[mask] = (img_np[mask] * 0.5 + np.array(bright_color) * 0.5).astype(np.uint8)
                    
                    for i in range(num_masks):
                        if mask_colors[i] is None:
                            continue
                            
                        mask = masks[i]
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img_np, contours, -1, (255, 255, 255), 1)

            if output_path is not None:
                try:
                    cv2.imwrite(output_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"[VisionEngine] ERROR: Could not save image: {e}")
                
            proc_time = time.time() - start_time
            file_name = os.path.basename(output_path) if output_path else "unknown"
            return json.dumps({
                "file_processed": file_name,
                "processing_time": f"{proc_time:.2f}s",
                "status": "SAM2_Success"
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[VisionEngine] Runtime Error/OOM: {e}")
            return "Error: Runtime/OOM in VisionEngine."
        except Exception as e:
            print(f"[VisionEngine] General exception: {e}")
            return "Error in vision processing."
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def process_image(self, image_path: str, output_path: str) -> str:
        """Processes a single image file, generates SAM masks, and saves the segmented output."""
        start_time = time.time()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return f"Error: Could not read image at {image_path}"
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # Dynamic Scaling
            img_np = self.resize_image(img_rgb, max_size=640)
            original = img_np.copy()
            
            with torch.no_grad():
                h, w = img_np.shape[:2]
                masks, scores_array = None, None
                
                if self.predictor is not None:
                    input_tensor = torch.from_numpy(img_np).to(device=self.device, dtype=torch.float16)
                    with torch.amp.autocast("cuda"):
                        self.predictor.set_image(img_np)
                    
                    point_coords = np.array([[w//2, h//2]])
                    point_labels = np.array([1])
                    
                    with torch.amp.autocast("cuda"):
                        masks, scores_array, logits = self.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True
                        )
                        
                    if masks is not None:
                        if isinstance(masks, torch.Tensor):
                            masks = (masks > 0.0).cpu().numpy().astype(bool)
                        else:
                            masks = (masks > 0.0).astype(bool)
                
                num_masks = 0
                if masks is not None:
                    num_masks = masks.shape[0]
                    
                scores = []
                if scores_array is not None:
                    scores = list(scores_array)
                    
                mask_colors: list[list[int] | None] = []
                if masks is not None:
                    for i in range(num_masks):
                        if i < len(scores) and scores[i] <= 0.6:
                            mask_colors.append(None)
                            continue
                            
                        mask = masks[i]
                        bright_color = [int(np.random.randint(100, 255)) for _ in range(3)]
                        mask_colors.append(bright_color)
                        
                        img_np[mask] = (img_np[mask] * 0.5 + np.array(bright_color) * 0.5).astype(np.uint8)
                    
                    for i in range(num_masks):
                        if mask_colors[i] is None:
                            continue
                            
                        mask = masks[i]
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img_np, contours, -1, (255, 255, 255), 1)

            try:
                cv2.imwrite(output_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"[VisionEngine] ERROR: Could not save image: {e}")
                
            proc_time = time.time() - start_time
            file_name = os.path.basename(image_path)
            return json.dumps({
                "file_processed": file_name,
                "processing_time": f"{proc_time:.2f}s",
                "status": "SAM2_Success"
            })
        except Exception as e:
            print(f"[VisionEngine] Image processing failed: {e}")
            return "Error: Image processing failed."
        finally:
            torch.cuda.empty_cache()
            gc.collect()

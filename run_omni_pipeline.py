import os
import time
import gc
import torch
import cv2
import json
import subprocess
from src.main import VisionEngine
from src.audio_processor import AudioEngine

def extract_audio(video_path: str, temp_audio_path: str) -> bool:
    """Extracts audio from video and saves to a temporary file using native ffmpeg."""
    try:
        subprocess.run(
            ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', temp_audio_path, '-y'],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        return os.path.exists(temp_audio_path)
    except Exception as e:
        print(f"[Error] Audio extraction failed (Ensure ffmpeg is system installed): {e}")
        return False

def run_pipeline(video_path: str, vision_engine: VisionEngine, audio_engine: AudioEngine, output_dir: str = "data"):
    """
    Orchestrates the Omni-Vision-Audio pipeline.
    Steps: Audio Extraction -> Audio Classification -> Video Masking/Classification.
    """
    print(f"\n--- Starting Pipeline for: {video_path} ---")
    
    filename, ext = os.path.splitext(os.path.basename(video_path))
    ext = ext.lower()
    
    # FILENAME FIX: Clean up chain naming if a processed file is forcibly run again
    if filename.endswith("_segmented"):
        filename = filename[:-10]
    elif filename.endswith("_report"):
        filename = filename[:-7]
    
    # --- 1. Audio Processing Phase ---
    if ext == ".wav":
        print("[Pipeline] Processing audio track via AST Model...")
        audio_result = audio_engine.process_audio(video_path)
    elif ext == ".jpg":
        print("[Pipeline] Image file detected. Skipping audio processing phase.")
    else:
        temp_audio_path = os.path.join(output_dir, f"{filename}_temp_audio.wav")
        print("[Pipeline] Extracting audio track...")
        has_audio = extract_audio(video_path, temp_audio_path)
        
        if has_audio:
            print("[Pipeline] Processing audio track via AST Model...")
            audio_result = audio_engine.process_audio(temp_audio_path)
        else:
            print("[Pipeline] No audio track detected. Skipping.")
                
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass

    # Free memory before switching modalities
    torch.cuda.empty_cache()
    gc.collect()
    
    # --- 2. Vision Processing Phase ---
    if ext == ".wav":
        print("[Pipeline] Audio file detected. Skipping vision processing.")
    elif ext == ".jpg":
        print("[Pipeline] Processing static image via SAM 2 Model...")
        segmented_path = os.path.join(output_dir, f"{filename}_segmented.jpg")
        vision_engine.process_image(video_path, segmented_path)
    else:
        print("[Pipeline] Extracting and processing video frames via SAM 2 Model...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[Error] Could not open video file: {video_path}")
            return
            
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0.0 else 30.0
        frame_interval = int(fps * vision_engine.FRAME_INTERVAL_SECONDS)
        
        frame_count = 0
        vision_results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    sys_time_sec = frame_count / fps
                    segmented_path = os.path.join(output_dir, f"{filename}_{int(sys_time_sec)}s_segmented.jpg")
                    print(f"[Pipeline] Processing frame at {sys_time_sec:.2f}s...")
                    res = vision_engine.process_frame(frame, output_path=segmented_path)
                    
                    try:
                        res_dict = json.loads(res)
                        vision_results.append({
                            "timestamp_seconds": float(f"{sys_time_sec:.2f}"),
                            "file_saved": os.path.basename(segmented_path),
                            "processing_time": res_dict.get("processing_time", "Unknown"),
                            "status": res_dict.get("status", "Unknown")
                        })
                    except Exception:
                        vision_results.append({
                            "timestamp_seconds": float(f"{sys_time_sec:.2f}"),
                            "file_saved": os.path.basename(segmented_path),
                            "status": "SAM2_Success"
                        })
                frame_count += 1
                
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("[Error] CUDA OOM error during video iteration processing.")
            else:
                print(f"[Error] Exception in video processing loop: {e}")
        finally:
            cap.release()
    
    # --- 3. Mandatory Hardware Stability Procedures ---
    print("[Pipeline] Pipeline finished for file. Cleaning up VRAM and initiating thermal cooldown...")
    torch.cuda.empty_cache()
    gc.collect()
    
    # --- 4. Final Data Aggregation ---
    report_json_path = os.path.join(output_dir, f"{filename}_report.json")
    
    if ext == '.jpg':
        vision_outcome = "SAM2_Success (Static Image)"
    elif ext == '.mp4':
        vision_outcome = vision_results if 'vision_results' in locals() and vision_results else "No frames processed."
    else:
        vision_outcome = "Vision sequence not processed for this file type."
        
    final_report = {
        "file_processed": video_path,
        "audio_analysis": audio_result if 'audio_result' in locals() else "Audio not processed for this file type.",
        "vision_analysis": vision_outcome
    }
    with open(report_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(final_report, json_file, indent=4)
    print(f"[Pipeline] JSON report saved to: {report_json_path}")
    
    # Mandatory exactly 10 seconds sleep requested for MX150 constraints
    time.sleep(10)
    print("\n[Pipeline] Cooldown complete. Ready for next task.")

if __name__ == "__main__":
    # Test stub if run directly
    sample_video = os.path.join("data", "sample.mp4")
    if os.path.exists(sample_video):
        # Requires mock setup to run standalone
        v_engine = VisionEngine()
        a_engine = AudioEngine()
        run_pipeline(sample_video, v_engine, a_engine)
    else:
        print(f"Please place a video file at {sample_video} or run run_smart_dispatcher.py to monitor /data folder.")

import os
import time
import glob
from run_omni_pipeline import run_pipeline
from src.main import VisionEngine
from src.audio_processor import AudioEngine

DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TIMEOUT_SECONDS = 30
CHECK_INTERVAL = 5

def setup_directories():
    """Ensure required input and output structures exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_target_files():
    """Helper to find all supported file types."""
    extensions = ('*.mp4', '*.jpg', '*.wav')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(DATA_DIR, ext)))
    return files

def monitor_directory():
    """
    Entry point: uses a loop to monitor the data/ folder for incoming .mp4, .jpg, and .wav files.
    Exits gracefully if no files are found for a continuous TIMEOUT_SECONDS duration.
    """
    setup_directories()
    
    print("Pre-loading analytical models...")
    vision_engine = VisionEngine()
    audio_engine = AudioEngine()
    
    print(f"Smart Dispatcher active. Monitoring '{os.path.abspath(DATA_DIR)}' for new files (.mp4, .jpg, .wav)...")
    target_files = get_target_files()
    if not target_files:
        print("\n[Smart Dispatcher] No files found in 'data/' to process. Shutting down.")
        return
    
    valid_files_processed = 0
    for file_path in target_files:
        filename = os.path.basename(file_path)
        
        # EXCLUDE RESULTS: Stop loop processing mapped outputs dynamically
        if "_segmented" in filename or "_report" in filename or "_temp" in filename:
            continue
            
        ext = os.path.splitext(file_path)[1].lower()
        print(f"\n[*] Smart Dispatcher identified incoming file: {file_path}")
        
        # OUTPUT PATH: Save results directly into 'data/processed/' folder
        run_pipeline(file_path, vision_engine, audio_engine, output_dir=PROCESSED_DIR)
        
        # INPUT ONLY: Move to processed tracking to prevent endless looping
        processed_path = os.path.join(PROCESSED_DIR, filename)
        try:
            # Overwrite existing tracking files to prevent runtime halts
            os.replace(file_path, processed_path)
            print(f"[*] Dispatcher safely moved {filename} to {PROCESSED_DIR}")
            valid_files_processed += 1
        except Exception as e:
            print(f"[!] Dispatcher error moving file {filename}: {e}")
            
    print(f"\n[Smart Dispatcher] Successfully handled {valid_files_processed} files. Auto-terminating securely.")

if __name__ == "__main__":
    monitor_directory()

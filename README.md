# OmniSeg-Audio-Pipeline: Multimodal Intelligence 🤖🎙️👁️

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Model: SAM2](https://img.shields.io/badge/Model-Meta_SAM2-green)](https://github.com/facebookresearch/segment-anything-2)
[![Model: AST](https://img.shields.io/badge/Model-MIT_AST-red)](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)

A high-performance, modular processing engine that synchronizes **Computer Vision** and **Acoustic Intelligence**. This pipeline leverages Meta's State-of-the-Art **Segment Anything Model 2 (SAM 2)** for visual isolation and MIT's **Audio Spectrogram Transformer (AST)** for environmental sound classification.

---

## 🚀 Key Architectural Advantages

* **Hybrid Multimodality**: Simultaneously processes `.mp4` video, `.jpg` images, and `.wav` audio through unified dispatching logic.
* **$O(1)$ Resource Management**: Optimized for edge hardware (e.g., NVIDIA MX150). Implements a strict "Process & Purge" cycle, clearing VRAM between tasks to prevent memory leakage.
* **Temporal Video Slicing**: Efficiently extracts keyframes at defined intervals (e.g., 10s) to track visual changes without the overhead of full-frame processing.
* **Production-Ready Output**: Generates standardized JSON reports and segmented visual overlays for seamless database integration.

---

## 📊 Technical Showcase

### 1. Static Image Segmentation (SAM 2)
The engine isolates objects with surgical precision. By mapping central mass coordinates, it generates high-fidelity masks for complex subjects like wildlife or landscapes.

| Source: Tiger | Output: Segmented Mask |
| :---: | :---: |
| ![Original Tiger](./assets/picture.jpg) | ![Segmented Tiger](./assets/picture_segmented.jpg) |



### 2. Acoustic Event Detection (AST)
The system extracts audio streams and classifies environmental contexts in real-time. Using the Audio Spectrogram Transformer, it identifies specific sound signatures (e.g., musical instruments, speech, or nature) with high confidence scores.

**Standardized JSON Result:**
![Audio Report JSON](./assets/audioJSON.png)

### 3. Unified Video & Metadata Intelligence
For video payloads, the pipeline merges temporal visual slicing with synchronized audio analysis. This creates a multi-layered report containing both object tracking data and acoustic timelines.

**Video Pipeline Metadata:**
![Video JSON Output](./assets/videoJSON.png)

---

## 🛠️ System Requirements

### 1. External Dependencies
- **FFmpeg**: Required for native audio stream extraction and video demuxing.
  - *Windows*: `winget install "FFmpeg (Shared)"`
  - *Linux*: `sudo apt install ffmpeg`

### 2. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/LTolo/OmniSeg-Audio-Pipeline.git](https://github.com/LTolo/OmniSeg-Audio-Pipeline.git)
cd OmniSeg-Audio-Pipeline

# Create a clean virtual environment
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

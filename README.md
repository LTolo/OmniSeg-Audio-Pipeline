# OmniSeg-Audio-Pipeline: Multimodal Intelligence 🤖🎙️👁️

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Model: SAM2](https://img.shields.io/badge/Model-Meta_SAM2-green)](https://github.com/facebookresearch/segment-anything-2)
[![Model: AST](https://img.shields.io/badge/Model-MIT_AST-red)](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)

A high-performance, modular processing engine that synchronizes **Computer Vision** and **Acoustic Intelligence**. This pipeline leverages Meta's State-of-the-Art **Segment Anything Model 2 (SAM 2)** for visual isolation and MIT's **Audio Spectrogram Transformer (AST)** for environmental sound classification.

---

## 🚀 Key Architectural Advantages

* **Hybrid Multimodality**: Simultaneously processes `.mp4` video, `.jpg` images, and `.wav` audio through a unified dispatching logic.
* **$O(1)$ Resource Management**: Optimized for local hardware (e.g., NVIDIA MX150). The pipeline implements a strict "Process & Purge" cycle, ensuring VRAM is cleared between tasks to prevent memory leakage.
* **Temporal Video Slicing**: Automatically extracts keyframes at defined intervals (e.g., 0s, 10s) to track visual changes without the computational overhead of full-frame processing.
* **Production-Ready Output**: Generates standardized JSON reports and segmented visual overlays, ready for integration into larger databases.



---

## 📊 Visual Showcase

### Computer Vision (SAM 2)
In der Bild- und Videoverarbeitung isoliert das System Objekte mit chirurgischer Präzision. Hier ein Beispiel der Tier- und Landschaftssegmentierung:

| Target: Tiger |
| :---: |:---: |
| ![Tiger Segmentation](./assets/picture.jpg) | ![Beach Segmentation](./assets/picture_segmented.jpg) |

### Acoustic Intelligence (AST)
Das System extrahiert die Audiospur und klassifiziert die Umgebungskontexte in Echtzeit. Die Ergebnisse werden in einer strukturierten Topologie ausgegeben:

**Sample JSON Output:**
![Audio Report JSON](./assets/audio_report_json.png)

---

## 🛠️ System Requirements

### 1. External Dependencies
- **FFmpeg**: Required for native audio stream extraction from video files.
  - *Windows*: `winget install "FFmpeg (Shared)"`
  - *Linux*: `sudo apt install ffmpeg`

### 2. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/LTolo/OmniSeg-Audio-Pipeline.git](https://github.com/LTolo/OmniSeg-Audio-Pipeline.git)
cd OmniSeg-Audio-Pipeline

# Create a clean virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

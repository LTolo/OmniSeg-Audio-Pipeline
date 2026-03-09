# Multi-Modal Vision & Audio Analysis (SAM 2 & AST)

## Description
This project provides a unified, production-ready multimedia pipeline combining **Meta's Segment Anything Model 2 (SAM 2)** for highly optimized image and video segmentation, alongside **MIT's Audio Spectrogram Transformer (AST)** for robust audio classification.

Built explicitly for minimal-footprint hardware environments, this tool operates fully autonomously across dynamic media tracking natively via zero-overhead sequences. It maps direct visual structural overlays while consolidating comprehensive runtime properties internally.

## Features
- **$O(1)$ Streamlined Pipeline**: Intelligent dispatcher (`run_smart_dispatcher.py`) sequentially processes massive video, audio, and image arrays applying automated VRAM teardown protocols between transitions, guaranteeing extreme stability.
- **Hybrid Vision Logic**:
  - Dynamically isolates central mass targeting tracks for static image `.jpg` prediction mapping.
  - Slices complex `.mp4` video sequences processing broad native 2x2 grids across 10-second intervals asynchronously without scaling failure boundaries.
- **Strict Output Topology**: Operations restrict rigorously exactly to a modular `1 Source, 1 Segmented-JPG, 1 JSON Report` paradigm avoiding sprawling dataset footprints mapping safe sequential directory management.

## Visuals
<!-- IMPORTANT: Upload and commit your segmented output demo into a new `./assets` root folder -->
![Segmentation Result](./assets/demo_result.jpg)

## Installation

Ensure you have [FFmpeg](https://ffmpeg.org/) installed securely inside your system environment PATH to manage native audio track separation capabilities smoothly.

Clone the repository and securely build the strictly enforced footprint environment:

```bash
git clone <your-repo-url>
cd omni-vision-audio-prototyper

# Set up the VENV boundary
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix:
# source .venv/bin/activate 

pip install -r requirements.txt
```

*(Note: SAM 2 downloads natively from Meta's source registry requiring `pip` compilation).*

## Usage

1. Create a `data/` directory natively inside the application root if not strictly tracked via `.gitkeep`.
2. Move any combinations of native `.mp4`, `.jpg`, or `.wav` payload files directly inside the base `data/` folder.
3. Automatically process everything silently parsing all files strictly down the pipeline mappings linearly:

```bash
python run_smart_dispatcher.py
```

The Smart Dispatcher uniquely monitors file processing execution tracking, shifting source formats recursively entirely mapping exclusively nested into `data/processed/`, automatically terminating gracefully out of sequence bounds upon encountering zero active backlogs!

# Soccer Action Attribution

This project implements a real-time player action attribution system for soccer videos. It uses computer vision and deep learning to detect players, track their movements, estimate their poses, and recognize jersey numbers to attribute actions to specific players.

## Features

- Player and ball detection using YOLOv8
- Player tracking using Norfair
- Jersey number recognition (stub/OCR)
- Real-time processing capabilities
- Modular codebase

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for speed)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>.git
cd <repo-folder>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download model weights:
- YOLOv8 weights will be downloaded automatically on first run (default: `yolov8n.pt`)
- (Optional) Download OCR/pose weights if you want to enable those modules

## Usage

1. Prepare your input:
   - Video file (e.g., `match_clip_01.mp4`)
   - Evaluation CSV file (`eval.csv`) with columns:
     - `timestamp`: Event timestamp (any of `ss.s`, `mm:ss.s`, or `hh:mm:ss.s`)
     - `event`: Event type (e.g., "pass", "goal", "free kick")

2. Run the pipeline:
```bash
python main.py eval.csv
```

3. The output will be saved in `submission.csv` with columns:
   - `timestamp`: Event timestamp in `hh:mm:ss.s` format
   - `event`: Event type
   - `player`: First player's jersey number (or "NONE")
   - `receiver`: Second player's jersey number (for pass events)
   - `latency`: Processing time in seconds

## Speed & Troubleshooting

- **Slow processing?**
  - Use a smaller YOLO model (e.g., `yolov8n.pt`)
  - Lower the video resolution
  - Reduce the window size in `predictor.py` (e.g., `window=0.2`)
  - Run on GPU if available
  - For fastest results, process only a single frame per event
- **No players detected?**
  - Check if YOLO weights are present and model is loading
  - Try lowering detection thresholds
- **No module named norfair?**
  - Run `pip install norfair`
- **No module named ultralytics?**
  - Run `pip install ultralytics`
- **Jersey numbers always NONE?**
  - OCR model/weights may be missing or jersey numbers not visible

## Project Structure

```
soccer-action-attribution/
├── main.py                # Entry point
├── event_processor.py     # Event loop and CSV I/O
├── predictor.py           # Player/receiver detection and attribution
├── detection.py           # YOLOv8 player and ball detection
├── tracking.py            # Norfair player tracking
├── ocr.py                 # Jersey number recognition (stub/OCR)
├── video_utils.py         # Video frame extraction and timestamp parsing
├── requirements.txt       # Project dependencies
├── README.md              # This file
```

## Submission
- The script will generate `submission.csv` as required for your assignment.
- Timestamps in output are always in `hh:mm:ss.s` format.
- Player and receiver columns will be filled if detected, otherwise "NONE".

## GitHub Push Instructions

1. Initialize git (if not already):
```bash
git init
git add .
git commit -m "Initial commit: working soccer action attribution pipeline"
```
2. Add your remote and push:
```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

---

For any issues, check the troubleshooting section above or open an issue on GitHub. 
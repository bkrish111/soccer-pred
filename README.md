# Soccer Action Attribution

## Overview

This project is designed to automatically identify which soccer player performed a specific action (like a pass, goal, or free kick) in a match video. It uses computer vision and deep learning to detect players, track their movements, and (optionally) recognize jersey numbers. The system is modular and can be extended for more advanced player attribution.

## How It Works

1. **Input:**  
   - A soccer match video (e.g., `match_clip_01.mp4`)
   - A CSV file (`eval.csv`) listing events with timestamps and event types (e.g., pass, goal)

2. **Processing Steps:**  
   - For each event, the system extracts the relevant video frame(s) at the event timestamp.
   - It uses a deep learning model (YOLOv8) to detect all players and the ball in the frame.
   - It uses a tracking algorithm (Norfair) to follow players across multiple frames.
   - For pass events, it tries to identify both the passer and receiver by associating the ball with the nearest players.
   - (Optional) It can use OCR to read jersey numbers from player regions.
   - The system outputs a CSV file (`submission.csv`) listing, for each event, the predicted player(s) and the time taken to process.

3. **Output:**  
   - A CSV file with columns: `timestamp`, `event`, `player`, `receiver`, `latency`

## Models and Libraries Used

- **YOLOv8 (Ultralytics):**  
  For fast and accurate detection of players and the ball in each frame.
- **Norfair:**  
  For tracking players across frames, which helps in associating actions with the correct player.
- **(Optional) OCR (e.g., EasyOCR, PARSeq):**  
  For reading jersey numbers from player images (currently a stub, can be extended).
- **OpenCV:**  
  For video processing and frame extraction.
- **Pandas:**  
  For CSV file handling.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bkrish111/soccer-pred.git
   cd soccer-pred
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights:**
   - YOLOv8 weights (`yolov8n.pt`) will be downloaded automatically on first run.
   - (Optional) Download OCR weights if you want to enable jersey number recognition.

## Usage

1. **Prepare your input:**
   - Place your video file (e.g., `match_clip_01.mp4`) in the project directory.
   - Prepare your `eval.csv` file with columns:
     - `timestamp` (e.g., `00:12.3`, `03:45.1`, or `00:12:03.5`)
     - `event` (e.g., `pass`, `goal`, `free kick`)

2. **Run the pipeline:**
   ```bash
   python main.py eval.csv
   ```

3. **Check the output:**
   - The results will be saved in `submission.csv` with columns:
     - `timestamp` (in `hh:mm:ss.s` format)
     - `event`
     - `player` (jersey number or `NONE`)
     - `receiver` (for pass events, else `NONE`)
     - `latency` (processing time in seconds)

## Troubleshooting & Tips

- **Slow processing?**
  - Use a smaller YOLO model (e.g., `yolov8n.pt`)
  - Lower the video resolution
  - Reduce the number of frames processed per event in `predictor.py`
  - Run on a machine with a GPU if possible

- **No players detected?**
  - Make sure YOLO weights are present and the model loads correctly
  - Try lowering detection thresholds

- **No module named norfair/ultralytics?**
  - Run `pip install norfair ultralytics`

- **Jersey numbers always NONE?**
  - OCR model/weights may be missing, or jersey numbers are not visible in the video

## Project Structure

```
soccer-pred/
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

## Contribution

Feel free to fork the repo, open issues, or submit pull requests to improve the system!

---

If you have any questions or run into issues, check the troubleshooting section above or open an issue on [GitHub](https://github.com/bkrish111/soccer-pred).

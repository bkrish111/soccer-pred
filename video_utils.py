import cv2

def parse_timestamp(ts):
    # Handles formats like "mm:ss.s" or "hh:mm:ss.s"
    parts = str(ts).split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        # fallback: try float
        return float(ts)

def get_frame_at_timestamp(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Warning: Could not read frame at {timestamp}s (frame {frame_idx})")
        return None
    return frame 
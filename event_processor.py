import pandas as pd
from video_utils import get_frame_at_timestamp, parse_timestamp
from predictor import predict_player_for_event
import time
import cv2

def format_timestamp_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:04.1f}" if secs % 1 else f"{hours:02}:{minutes:02}:{int(secs):02}"

def process_events(event_csv, video_path, output_csv):
    events = pd.read_csv(event_csv)
    results = []
    # Get FPS from video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    for idx, row in events.iterrows():
        timestamp_raw = row['timestamp']
        timestamp = parse_timestamp(timestamp_raw)
        event = row['event']
        frame = get_frame_at_timestamp(video_path, timestamp)
        start_time = time.time()
        # Pass video_path, timestamp, and fps for tracking-based attribution
        player, receiver = predict_player_for_event(frame, event, video_path=video_path, timestamp=timestamp, fps=fps)
        latency = round(time.time() - start_time, 3)
        out_timestamp = format_timestamp_hms(timestamp)
        if event.lower() == 'pass':
            results.append([out_timestamp, event, player, receiver, latency])
        else:
            results.append([out_timestamp, event, player, 'NONE', latency])
    pd.DataFrame(results, columns=['timestamp', 'event', 'player', 'receiver', 'latency']).to_csv(output_csv, index=False)
    print(f"Predictions written to {output_csv}") 
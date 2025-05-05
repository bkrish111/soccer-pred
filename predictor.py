from detection import PlayerDetector
from tracking import PlayerTracker
try:
    from ocr import JerseyNumberRecognizer
except ImportError:
    JerseyNumberRecognizer = None
import numpy as np
import cv2

# Initialize detector and OCR (if available)
player_detector = PlayerDetector()
player_tracker = PlayerTracker()
if JerseyNumberRecognizer:
    jersey_ocr = JerseyNumberRecognizer()
else:
    jersey_ocr = None

def get_frames_around_event(video_path, timestamp, window=1.0, fps=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames = []
    start = max(0, int((timestamp - window) * fps))
    end = int((timestamp + window) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for idx in range(start, end+1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def predict_player_for_event(frame, event_type, video_path=None, timestamp=None, fps=20):
    # If video_path and timestamp are provided, use tracking over a window
    if video_path is not None and timestamp is not None:
        frames = get_frames_around_event(video_path, timestamp, window=1.0, fps=fps)
        all_player_boxes = []
        all_ball_boxes = []
        for f in frames:
            player_boxes = player_detector.get_player_boxes(f)
            ball_box = player_detector.get_ball_box(f)
            all_player_boxes.append(player_boxes)
            all_ball_boxes.append(ball_box)
        # Flatten player boxes and track
        tracked_players = player_tracker.get_tracked_players(frames[-1], [box for sublist in all_player_boxes for box in sublist])
        # Use last frame's ball box
        ball_box = all_ball_boxes[-1]
        # For pass: associate two closest players to ball
        if event_type.lower() == 'pass' and ball_box is not None and tracked_players:
            # Sort players by distance to ball
            def center(box):
                x1, y1, x2, y2 = box
                return ((x1+x2)/2, (y1+y2)/2)
            bx, by = center(ball_box)
            dists = [((center(tp['bbox'])[0]-bx)**2 + (center(tp['bbox'])[1]-by)**2, tp) for tp in tracked_players]
            dists.sort(key=lambda x: x[0])
            if len(dists) >= 2:
                p1, p2 = dists[0][1], dists[1][1]
                jersey1 = jersey_ocr.recognize_number(frames[-1], p1['bbox']) if jersey_ocr else 'NONE'
                jersey2 = jersey_ocr.recognize_number(frames[-1], p2['bbox']) if jersey_ocr else 'NONE'
                return jersey1, jersey2
            elif len(dists) == 1:
                jersey1 = jersey_ocr.recognize_number(frames[-1], dists[0][1]['bbox']) if jersey_ocr else 'NONE'
                return jersey1, 'NONE'
            else:
                return 'NONE', 'NONE'
        # For other events: pick closest player to ball
        elif ball_box is not None and tracked_players:
            def center(box):
                x1, y1, x2, y2 = box
                return ((x1+x2)/2, (y1+y2)/2)
            bx, by = center(ball_box)
            dists = [((center(tp['bbox'])[0]-bx)**2 + (center(tp['bbox'])[1]-by)**2, tp) for tp in tracked_players]
            dists.sort(key=lambda x: x[0])
            if dists:
                jersey = jersey_ocr.recognize_number(frames[-1], dists[0][1]['bbox']) if jersey_ocr else 'NONE'
                return jersey, None
            else:
                return 'NONE', None
        else:
            return 'NONE', 'NONE' if event_type.lower() == 'pass' else ( 'NONE', None )
    # Fallback: single frame logic
    if frame is None:
        if event_type.lower() == 'pass':
            return 'NONE', 'NONE'
        else:
            return 'NONE', None
    player_boxes = player_detector.get_player_boxes(frame)
    if not player_boxes:
        if event_type.lower() == 'pass':
            return 'NONE', 'NONE'
        else:
            return 'NONE', None
    bbox = player_boxes[0]
    if jersey_ocr:
        jersey_number = jersey_ocr.recognize_number(frame, bbox)
    else:
        jersey_number = 'NONE'
    if event_type.lower() == 'pass':
        if len(player_boxes) >= 2:
            bbox2 = player_boxes[1]
            jersey_number2 = jersey_ocr.recognize_number(frame, bbox2) if jersey_ocr else 'NONE'
            return jersey_number, jersey_number2
        else:
            return jersey_number, 'NONE'
    else:
        return jersey_number, None 
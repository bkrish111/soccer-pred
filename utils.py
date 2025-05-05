import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1 (list): First bounding box [x1, y1, x2, y2]
        box2 (list): Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU score
    """
    # Get coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def associate_players_with_ball(
    player_tracks: List[Dict],
    ball_box: Optional[List[float]],
    iou_threshold: float = 0.1
) -> Tuple[Optional[int], Optional[int]]:
    """Associate players with the ball for pass events.
    
    Args:
        player_tracks (list): List of tracked players
        ball_box (list): Ball bounding box [x1, y1, x2, y2]
        iou_threshold (float): IoU threshold for association
        
    Returns:
        tuple: (passer_id, receiver_id) or (None, None) if association fails
    """
    if not ball_box or not player_tracks:
        return None, None
        
    # Calculate IoU between ball and each player
    ious = [calculate_iou(track['bbox'], ball_box) for track in player_tracks]
    
    # Get players with IoU above threshold
    associated_players = [
        (i, track['id']) for i, (iou, track) in enumerate(zip(ious, player_tracks))
        if iou > iou_threshold
    ]
    
    if len(associated_players) >= 2:
        # Sort by IoU score
        associated_players.sort(key=lambda x: ious[x[0]], reverse=True)
        return associated_players[0][1], associated_players[1][1]
    elif len(associated_players) == 1:
        return associated_players[0][1], None
    else:
        return None, None

def get_frame_at_timestamp(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """Get video frame at specific timestamp.
    
    Args:
        video_path (str): Path to video file
        timestamp (float): Timestamp in seconds
        
    Returns:
        np.ndarray: Frame at timestamp or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(timestamp * fps)
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    # Read frame
    success, frame = cap.read()
    cap.release()
    
    return frame if success else None

def format_output_row(
    timestamp: float,
    event_type: str,
    player1: str,
    player2: Optional[str] = None,
    latency: float = 0.0
) -> List:
    """Format output row for submission CSV.
    
    Args:
        timestamp (float): Event timestamp
        event_type (str): Type of event
        player1 (str): First player (or only player)
        player2 (str): Second player (for pass events)
        latency (float): Processing latency in seconds
        
    Returns:
        list: Formatted row for CSV
    """
    if event_type.lower() == 'pass' and player2 is not None:
        return [timestamp, event_type, player1, player2, latency]
    else:
        return [timestamp, event_type, player1, latency] 
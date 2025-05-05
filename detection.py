from ultralytics import YOLO
import torch
import cv2
import numpy as np

class PlayerDetector:
    def __init__(self, model_path='yolov8n.pt', device=None):
        """Initialize the YOLOv8 model for player and ball detection.
        
        Args:
            model_path (str): Path to YOLOv8 weights
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        
    def detect(self, frame):
        """Detect players and ball in a frame.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            list: List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class: int (0 for person, 32 for sports ball)
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                # Only keep person (0) and sports ball (32) detections
                if int(cls) in [0, 32]:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': int(cls)
                    })
        
        return detections
    
    def get_player_boxes(self, frame):
        """Get only player bounding boxes from the frame.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            list: List of player bounding boxes [x1, y1, x2, y2]
        """
        detections = self.detect(frame)
        return [det['bbox'] for det in detections if det['class'] == 0]
    
    def get_ball_box(self, frame):
        """Get the ball bounding box from the frame.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            list: Ball bounding box [x1, y1, x2, y2] or None if not found
        """
        detections = self.detect(frame)
        ball_dets = [det for det in detections if det['class'] == 32]
        return ball_dets[0]['bbox'] if ball_dets else None 
from norfair import Detection, Tracker
import numpy as np

class PlayerTracker:
    def __init__(self, distance_threshold=30):
        """Initialize the Norfair tracker for player tracking.
        
        Args:
            distance_threshold (float): Maximum distance between detections to be considered the same player
        """
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=distance_threshold,
            hit_counter_max=30,  # Keep tracks alive for longer
            initialization_delay=1  # Start tracking immediately
        )
        
    def update(self, detections):
        """Update tracks with new detections.
        
        Args:
            detections (list): List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class: int
                
        Returns:
            list: List of tracked objects, each containing:
                - bbox: [x1, y1, x2, y2]
                - id: int (track ID)
                - confidence: float
        """
        # Convert detections to Norfair format
        norfair_detections = []
        for det in detections:
            if det['class'] == 0:  # Only track players
                x1, y1, x2, y2 = det['bbox']
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                norfair_detections.append(
                    Detection(
                        points=np.array([[xc, yc]]),
                        scores=np.array([det['confidence']])
                    )
                )
        
        # Update tracks
        tracked_objects = self.tracker.update(detections=norfair_detections)
        
        # Convert tracks back to our format
        tracks = []
        for obj in tracked_objects:
            xc, yc = obj.estimate[0]
            # Create a small bounding box around the center point
            size = 20  # Size of the bounding box
            bbox = [xc - size/2, yc - size/2, xc + size/2, yc + size/2]
            tracks.append({
                'bbox': bbox,
                'id': obj.id,
                'confidence': obj.last_detection.scores[0] if obj.last_detection else 0.0
            })
            
        return tracks
    
    def get_tracked_players(self, frame, player_boxes):
        """Get tracked players from player bounding boxes.
        
        Args:
            frame (np.ndarray): Input frame (not used, but kept for API consistency)
            player_boxes (list): List of player bounding boxes [x1, y1, x2, y2]
            
        Returns:
            list: List of tracked players with IDs
        """
        detections = [
            {
                'bbox': box,
                'confidence': 1.0,  # Assume high confidence for detected boxes
                'class': 0  # Class 0 for person
            }
            for box in player_boxes
        ]
        return self.update(detections) 
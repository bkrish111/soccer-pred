import torch
import cv2
import numpy as np
from vitpose_pytorch import ViTPose

class PoseEstimator:
    def __init__(self, model_name='vitpose-b', device=None):
        """Initialize the ViTPose model for pose estimation.
        
        Args:
            model_name (str): ViTPose model variant ('vitpose-b', 'vitpose-l', etc.)
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.model = ViTPose.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        
    def estimate_pose(self, frame, bbox):
        """Estimate pose for a player in the given bounding box.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            bbox (list): Bounding box [x1, y1, x2, y2]
            
        Returns:
            dict: Pose keypoints and confidence scores
        """
        x1, y1, x2, y2 = map(int, bbox)
        player_img = frame[y1:y2, x1:x2]
        
        if player_img.size == 0:
            return None
            
        # Resize to model input size
        player_img = cv2.resize(player_img, (256, 256))
        
        # Convert to RGB and normalize
        player_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)
        player_img = player_img.astype(np.float32) / 255.0
        
        # Convert to tensor
        player_img = torch.from_numpy(player_img).permute(2, 0, 1).unsqueeze(0)
        player_img = player_img.to(self.device)
        
        # Get pose keypoints
        with torch.no_grad():
            keypoints = self.model(player_img)
            
        # Convert keypoints to original image coordinates
        keypoints = keypoints.cpu().numpy()[0]
        keypoints[:, 0] = keypoints[:, 0] * (x2 - x1) / 256 + x1
        keypoints[:, 1] = keypoints[:, 1] * (y2 - y1) / 256 + y1
        
        return {
            'keypoints': keypoints[:, :2],  # x, y coordinates
            'scores': keypoints[:, 2]  # confidence scores
        }
    
    def get_torso_region(self, pose_data):
        """Get the torso region from pose keypoints.
        
        Args:
            pose_data (dict): Pose keypoints and scores
            
        Returns:
            tuple: (x1, y1, x2, y2) coordinates of torso region
        """
        if pose_data is None:
            return None
            
        # Get relevant keypoints for torso
        left_shoulder = pose_data['keypoints'][5]  # Left shoulder
        right_shoulder = pose_data['keypoints'][6]  # Right shoulder
        left_hip = pose_data['keypoints'][11]  # Left hip
        right_hip = pose_data['keypoints'][12]  # Right hip
        
        # Calculate torso region
        x1 = min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])
        x2 = max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])
        y1 = min(left_shoulder[1], right_shoulder[1])
        y2 = max(left_hip[1], right_hip[1])
        
        # Add some padding
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = x2 + padding
        y2 = y2 + padding
        
        return (int(x1), int(y1), int(x2), int(y2)) 
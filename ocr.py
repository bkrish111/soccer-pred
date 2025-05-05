import torch
import cv2
import numpy as np
from parseq import PARSeq
from PIL import Image

class JerseyNumberRecognizer:
    def __init__(self, device=None):
        """Initialize the PARSeq model for jersey number recognition.
        
        Args:
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.model = PARSeq.from_pretrained('parseq')
        self.model.to(device)
        self.device = device
        
    def preprocess_image(self, img):
        """Preprocess image for PARSeq model.
        
        Args:
            img (np.ndarray): Input image in BGR format
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        # Resize to model input size
        img = img.resize((128, 32))
        
        # Convert to tensor and normalize
        img = torch.from_numpy(np.array(img)).float()
        img = img.permute(2, 0, 1) / 255.0
        img = img.unsqueeze(0)
        
        return img.to(self.device)
        
    def recognize_number(self, frame, bbox):
        """Recognize jersey number in the given region.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            
        Returns:
            str: Recognized number or 'NONE' if recognition fails
        """
        x1, y1, x2, y2 = map(int, bbox)
        region = frame[y1:y2, x1:x2]
        
        if region.size == 0:
            return 'NONE'
            
        # Preprocess image
        img = self.preprocess_image(region)
        
        # Get prediction
        with torch.no_grad():
            pred = self.model(img)
            
        # Convert prediction to text
        text = pred[0]
        
        # Clean up prediction (keep only digits)
        text = ''.join(c for c in text if c.isdigit())
        
        return text if text else 'NONE'
    
    def recognize_from_torso(self, frame, torso_bbox):
        """Recognize jersey number from torso region.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            torso_bbox (tuple): Torso bounding box (x1, y1, x2, y2)
            
        Returns:
            str: Recognized number or 'NONE' if recognition fails
        """
        if torso_bbox is None:
            return 'NONE'
            
        return self.recognize_number(frame, torso_bbox) 
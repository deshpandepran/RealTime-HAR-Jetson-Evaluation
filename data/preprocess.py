import torch
import torchvision.transforms as T
import numpy as np
import cv2

class VideoPreprocessor:
    """
    Handles frame resizing, normalization, and tensor conversion for video action recognition.
    Enforces spatial dimensions H x W = 256 x 344 (multiples of 8 for feature scaling).
    """
    def __init__(self, target_h: int = 256, target_w: int = 344, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
        self.target_h = target_h
        self.target_w = target_w
        self.normalize = T.Normalize(mean=mean, std=std)
        
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resizes a single frame (H x W x C, RGB) to target dimensions (256 x 344).
        """
        return cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

    def process_clip(self, clip_frames: list) -> torch.Tensor:
        """
        Processes a list of frames representing a video clip of length K.
        Returns tensor of shape (K, 3, H, W) normalized and ready for feature extraction.
        """
        processed_frames = []
        for frame in clip_frames:
            if frame.dtype != np.float32:
                frame = frame.astype(np.float32) / 255.0
            # Convert H x W x C to C x H x W
            try:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
            except RuntimeError:
                frame_tensor = torch.frombuffer(bytearray(frame.tobytes()), dtype=torch.float32).reshape(frame.shape).permute(2, 0, 1)
            frame_tensor = self.normalize(frame_tensor)
            processed_frames.append(frame_tensor)
            
        return torch.stack(processed_frames, dim=0)
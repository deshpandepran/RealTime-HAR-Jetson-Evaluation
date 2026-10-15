import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from .preprocess import VideoPreprocessor

class ActivityNetSubsetDataset(Dataset):
    """
    ActivityNet dataset subset loader for model training and distillation (IMFE).
    Total: 4,359 videos (4,059 training, 300 validation).
    """
    def __init__(self, data_root: str, split: str = 'train', clip_length: int = 6, preprocessor: VideoPreprocessor = None):
        self.data_root = data_root
        self.split = split
        self.clip_length = clip_length
        self.preprocessor = preprocessor or VideoPreprocessor()
        
        self.samples = self._load_metadata()
        
    def _load_metadata(self) -> list:
        meta_file = os.path.join(self.data_root, f"activitynet_{self.split}.json")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                return json.load(f)
        else:
            # Fallback directory scan if JSON metadata is not pre-generated
            split_dir = os.path.join(self.data_root, self.split)
            if os.path.exists(split_dir):
                videos = [os.path.join(split_dir, v) for v in os.listdir(split_dir) if v.endswith(('.mp4', '.avi', '.mkv'))]
                return [{"video_path": p} for p in videos]
            return []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        video_path = sample['video_path']
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.preprocessor.resize_frame(frame)
            frames.append(frame)
        cap.release()
        
        if len(frames) < self.clip_length:
            while len(frames) < self.clip_length:
                frames.append(frames[-1] if frames else np.zeros((256, 344, 3), dtype=np.uint8))
                
        # Sample a random clip of length K (default K=6)
        start_idx = np.random.randint(0, len(frames) - self.clip_length + 1) if len(frames) >= self.clip_length else 0
        clip_frames = frames[start_idx : start_idx + self.clip_length]
        
        clip_tensor = self.preprocessor.process_clip(clip_frames) # Shape: (K, 3, H, W)
        
        return {
            'clip': clip_tensor,
            'video_path': video_path
        }
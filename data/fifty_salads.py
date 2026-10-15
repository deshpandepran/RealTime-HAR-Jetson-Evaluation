import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from .preprocess import VideoPreprocessor

class FiftySaladsDataset(Dataset):
    """
    50 Salads benchmark dataset loader with 5-fold cross-validation support.
    50 top-view camera recordings at 30 FPS.
    """
    def __init__(self, data_root: str, fold: int = 1, total_folds: int = 5, split: str = 'train', target_fps: int = 30, clip_length: int = 6, preprocessor: VideoPreprocessor = None):
        self.data_root = data_root
        self.fold = fold
        self.total_folds = total_folds
        self.split = split
        self.target_fps = target_fps
        self.clip_length = clip_length
        self.preprocessor = preprocessor or VideoPreprocessor()
        
        self.video_list, self.annotations = self._setup_folds_and_metadata()

    def _setup_folds_and_metadata(self) -> tuple:
        # 50 Salads contains 50 videos (rgb-seq-01 to rgb-seq-50)
        all_videos = [f"rgb-seq-{i:02d}" for i in range(1, 51)]
        
        # Deterministic 5-fold split (10 videos per fold)
        fold_size = len(all_videos) // self.total_folds
        folds = [all_videos[i * fold_size : (i + 1) * fold_size] for i in range(self.total_folds)]
        
        val_videos = folds[self.fold - 1]
        train_videos = [v for f_idx, f in enumerate(folds) if f_idx != (self.fold - 1) for v in f]
        
        selected_videos = train_videos if self.split == 'train' else val_videos
        
        # Load action segment annotations if available in groundTruth/ directory
        annotations = {}
        anno_dir = os.path.join(self.data_root, 'groundTruth')
        for v in selected_videos:
            anno_path = os.path.join(anno_dir, f"{v}.txt")
            if os.path.exists(anno_path):
                with open(anno_path, 'r') as f:
                    lines = [line.strip().split() for line in f if line.strip()]
                    annotations[v] = lines
            else:
                annotations[v] = []
                
        return selected_videos, annotations

    def __len__(self) -> int:
        return len(self.video_list)

    def __getitem__(self, idx: int) -> dict:
        video_name = self.video_list[idx]
        video_path = os.path.join(self.data_root, 'videos', f"{video_name}.avi")
        
        if not os.path.exists(video_path):
            video_path = os.path.join(self.data_root, 'videos', f"{video_name}.mp4")
            
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(native_fps / self.target_fps)))
        
        frames = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.preprocessor.resize_frame(frame)
                frames.append(frame)
            frame_idx += 1
        cap.release()
        
        if len(frames) < self.clip_length:
            while len(frames) < self.clip_length:
                frames.append(frames[-1] if frames else np.zeros((256, 344, 3), dtype=np.uint8))
                
        start_idx = np.random.randint(0, len(frames) - self.clip_length + 1) if len(frames) >= self.clip_length else 0
        clip_frames = frames[start_idx : start_idx + self.clip_length]
        clip_tensor = self.preprocessor.process_clip(clip_frames)
        
        return {
            'clip': clip_tensor,
            'video_name': video_name,
            'labels': self.annotations.get(video_name, [])
        }
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import VideoPreprocessor

DOWNLOAD_HINT = (
    "UCSD Ped2 not found at '{root}'. Download the UCSD Anomaly Detection Dataset "
    "(http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm), extract it, and point "
    "data_root at the 'UCSDped2' folder, which must contain 'Train/' and 'Test/'."
)

CLASS_NAMES = ["normal", "anomaly"]


def _list_frames(folder: str, exts: Tuple[str, ...]) -> List[str]:
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)
    )


class UCSDPed2Dataset(Dataset):
    """
    UCSD Ped2 anomaly-detection loader (pedestrian walkway, ~10 FPS, grayscale 240x360).

    Layout expected under data_root (the 'UCSDped2' folder):
        Train/Train001/001.tif ...        (normal frames only)
        Test/Test001/001.tif ...          (normal + anomalous frames)
        Test/Test001_gt/001.bmp ...       (pixel masks; a frame is anomalous if its mask has any pixel > 0)

    Only the Test videos carry anomalies, so supervised training needs them. They are split
    into k folds by video; 'train' uses the other folds, 'val' uses the held-out fold.
    With include_normal_train=True the Train videos (all label 0) are added to the train split.

    Each item is one clip of `clip_length` consecutive frames (default 6, as in RT-HARE).
    The clip label is the label of its LAST frame (the "current timestamp" in online recognition).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        fold: int = 1,
        total_folds: int = 4,
        clip_length: int = 6,
        stride: Optional[int] = None,
        include_normal_train: bool = False,
        preprocessor: Optional[VideoPreprocessor] = None,
    ):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        assert 1 <= fold <= total_folds, "fold must be in [1, total_folds]"
        self.data_root = data_root
        self.split = split
        self.fold = fold
        self.total_folds = total_folds
        self.clip_length = clip_length
        self.stride = stride or clip_length  # non-overlapping = one prediction per K frames
        self.include_normal_train = include_normal_train
        self.preprocessor = preprocessor or VideoPreprocessor()

        test_dir = os.path.join(data_root, "Test")
        train_dir = os.path.join(data_root, "Train")
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(DOWNLOAD_HINT.format(root=data_root))

        test_videos = sorted(
            d for d in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, d)) and not d.endswith("_gt")
        )
        if not test_videos:
            raise FileNotFoundError(DOWNLOAD_HINT.format(root=data_root))

        fold_size = max(1, len(test_videos) // total_folds)
        folds = [test_videos[i * fold_size:(i + 1) * fold_size] for i in range(total_folds)]
        folds[-1].extend(test_videos[total_folds * fold_size:])  # leftover videos
        val_videos = folds[fold - 1]
        train_videos = [v for i, f in enumerate(folds) if i != fold - 1 for v in f]

        selected = val_videos if split == "val" else train_videos
        # video name -> (list of frame paths, np.ndarray of per-frame labels)
        self.videos = {}
        for name in selected:
            frames = _list_frames(os.path.join(test_dir, name), (".tif", ".tiff", ".png", ".jpg"))
            labels = self._load_labels(os.path.join(test_dir, name + "_gt"), len(frames))
            self.videos[name] = (frames, labels)

        if split == "train" and include_normal_train and os.path.isdir(train_dir):
            for name in sorted(os.listdir(train_dir)):
                path = os.path.join(train_dir, name)
                if os.path.isdir(path):
                    frames = _list_frames(path, (".tif", ".tiff", ".png", ".jpg"))
                    self.videos[name] = (frames, np.zeros(len(frames), dtype=np.int64))

        self.index = []  # (video_name, start_frame)
        for name, (frames, _) in self.videos.items():
            for start in range(0, len(frames) - clip_length + 1, self.stride):
                self.index.append((name, start))

    @staticmethod
    def _load_labels(gt_dir: str, n_frames: int) -> np.ndarray:
        labels = np.zeros(n_frames, dtype=np.int64)
        if not os.path.isdir(gt_dir):
            return labels
        masks = _list_frames(gt_dir, (".bmp", ".png", ".tif"))
        for i, m in enumerate(masks[:n_frames]):
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.max() > 0:
                labels[i] = 1
        return labels

    def get_video_labels(self, video_name: str) -> np.ndarray:
        """Per-frame 0/1 labels of a video (for evaluation)."""
        return self.videos[video_name][1]

    def __len__(self) -> int:
        return len(self.index)

    def _read_frame(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not read frame: {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # grayscale -> 3 channels
        return self.preprocessor.resize_frame(rgb)

    def __getitem__(self, idx: int) -> dict:
        name, start = self.index[idx]
        frames, labels = self.videos[name]
        K = self.clip_length
        clip = self.preprocessor.process_clip(
            [self._read_frame(frames[start + k]) for k in range(K)]
        )
        try:
            frame_labels = torch.from_numpy(labels[start:start + K])
        except RuntimeError:
            frame_labels = torch.tensor(labels[start:start + K].tolist(), dtype=torch.long)
        return {
            "clip": clip,                              # (K, 3, 256, 344)
            "label": int(labels[start + K - 1]),       # label of the last frame
            "frame_labels": frame_labels,              # (K,)
            "video_name": name,
            "start": start,
        }

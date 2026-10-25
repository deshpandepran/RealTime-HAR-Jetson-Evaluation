"""Tests for data/ucsd_ped2.py using a tiny synthetic dataset (no real download needed).

Run from the repository root:  python tests/test_ucsd_ped2.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.ucsd_ped2 import UCSDPed2Dataset  # noqa: E402

N_FRAMES = 12


def make_fake_ped2(root):
    rng = np.random.RandomState(0)
    for v in range(1, 3):  # 2 normal train videos
        d = os.path.join(root, "Train", f"Train{v:03d}")
        os.makedirs(d)
        for i in range(1, N_FRAMES + 1):
            cv2.imwrite(os.path.join(d, f"{i:03d}.tif"), rng.randint(0, 255, (240, 360), np.uint8))
    for v in range(1, 5):  # 4 test videos, anomaly in frames 7-9 (1-indexed) of videos 1 and 3
        d = os.path.join(root, "Test", f"Test{v:03d}")
        g = os.path.join(root, "Test", f"Test{v:03d}_gt")
        os.makedirs(d)
        os.makedirs(g)
        for i in range(1, N_FRAMES + 1):
            cv2.imwrite(os.path.join(d, f"{i:03d}.tif"), rng.randint(0, 255, (240, 360), np.uint8))
            mask = np.zeros((240, 360), np.uint8)
            if v in (1, 3) and 7 <= i <= 9:
                mask[50:100, 50:100] = 255
            cv2.imwrite(os.path.join(g, f"{i:03d}.bmp"), mask)


def test_all():
    passed = 0
    with tempfile.TemporaryDirectory() as root:
        make_fake_ped2(root)

        val = UCSDPed2Dataset(root, split="val", fold=1, total_folds=2)
        train = UCSDPed2Dataset(root, split="train", fold=1, total_folds=2)
        assert set(val.videos).isdisjoint(train.videos), "train/val videos overlap"
        passed += 1

        item = val[0]
        assert item["clip"].shape == (6, 3, 256, 344), item["clip"].shape
        assert item["frame_labels"].shape == (6,)
        passed += 1

        # Test001 is in fold 1 (val): anomaly frames 7-9 -> 0-indexed 6-8
        labels = val.get_video_labels("Test001")
        assert labels.tolist() == [0] * 6 + [1] * 3 + [0] * 3, labels.tolist()
        passed += 1

        # non-overlapping clips of 6 frames from a 12-frame video -> 2 clips per video
        assert len(val) == 2 * len(val.videos), (len(val), len(val.videos))
        passed += 1

        # second clip of Test001 starts at frame 6, last frame is 11 (normal)
        starts = [(n, s) for n, s in val.index if n == "Test001"]
        assert starts == [("Test001", 0), ("Test001", 6)], starts
        passed += 1

        with_normal = UCSDPed2Dataset(root, split="train", fold=1, total_folds=2,
                                      include_normal_train=True)
        assert len(with_normal.videos) == len(train.videos) + 2
        assert with_normal.get_video_labels("Train001").sum() == 0
        passed += 1

        try:
            UCSDPed2Dataset(os.path.join(root, "missing"))
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            passed += 1

    print(f"{passed} passed, 0 failed")


if __name__ == "__main__":
    test_all()

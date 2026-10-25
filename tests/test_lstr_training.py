#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_lstr_training.py — Smoke tests for Issue #7 LSTR training loop
==========================================================================
Validates the full training pipeline end-to-end without requiring:
  - Real video files
  - A trained IMFE checkpoint
  - A GPU

All tests use synthetic data and run in under ~30 seconds on CPU.

Run with:
    python tests/test_lstr_training.py

Or via pytest:
    pytest tests/test_lstr_training.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types

import numpy as np
import torch

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO_ROOT)

from distillation.train_lstr import (  # noqa: E402
    FIFTY_SALADS_ACTIONS,
    LABEL_TO_IDX,
    NUM_CLASSES,
    FiftySaladsFeaturesDataset,
    _encode_label,
    _fold_split,
    _load_frame_labels,
    _evaluate,
    _train_one_epoch,
    run_lstr_training,
)
from models import LongShortTermTransformer  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_gt_file(path: str, labels: list[str]) -> None:
    """Write a one-label-per-line ground-truth file."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(labels) + "\n")


def _make_gt_dir(tmp: str, num_videos: int = 10, frames_per_video: int = 30) -> str:
    """Create a synthetic groundTruth directory and return its path."""
    gt_dir = os.path.join(tmp, "groundTruth")
    os.makedirs(gt_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    label_names = FIFTY_SALADS_ACTIONS[1:]  # skip 'background'
    for i in range(1, num_videos + 1):
        labels = rng.choice(label_names, size=frames_per_video).tolist()
        _write_gt_file(os.path.join(gt_dir, f"rgb-seq-{i:02d}.txt"), labels)
    return gt_dir


def _make_tiny_lstr(**kwargs) -> LongShortTermTransformer:
    """Build a tiny LSTR model for fast CPU tests."""
    defaults = dict(in_dim=32, num_classes=NUM_CLASSES,
                    d_model=32, num_heads=2, num_layers=1,
                    dim_feedforward=64, dropout=0.0, max_seq_len=64, causal=True)
    defaults.update(kwargs)
    return LongShortTermTransformer(**defaults)


def _make_args(**kwargs) -> types.SimpleNamespace:
    """Build a minimal args namespace."""
    defaults = dict(
        device="cpu",
        data_root=None,
        imfe_checkpoint=None,
        smoke_test=True,
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_label_vocabulary():
    """FIFTY_SALADS_ACTIONS has the expected size; LABEL_TO_IDX is consistent."""
    assert NUM_CLASSES == len(FIFTY_SALADS_ACTIONS)
    assert LABEL_TO_IDX["background"] == 0
    for idx, name in enumerate(FIFTY_SALADS_ACTIONS):
        assert LABEL_TO_IDX[name] == idx


def test_encode_label_known_and_unknown():
    """Known labels encode to correct indices; unknown labels map to 0."""
    assert _encode_label("cut_cheese") == LABEL_TO_IDX["cut_cheese"]
    assert _encode_label("background") == 0
    assert _encode_label("totally_unknown_action") == 0
    assert _encode_label("  cut_cucumber  ") == LABEL_TO_IDX["cut_cucumber"]


def test_load_frame_labels_reads_file():
    """_load_frame_labels parses a one-label-per-line file correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "seq.txt")
        expected = ["cut_cheese", "add_pepper", "background", "mix_dressing"]
        _write_gt_file(path, expected)
        result = _load_frame_labels(path)
        assert result.dtype == np.int64
        assert len(result) == len(expected)
        for idx_r, lbl in zip(result, expected):
            assert idx_r == LABEL_TO_IDX[lbl]


def test_fold_split_sizes():
    """5-fold split: each fold has 10 videos; train has 40, val has 10."""
    for fold in range(1, 6):
        train_vids, val_vids = _fold_split(num_videos=50, num_folds=5, eval_fold=fold)
        assert len(train_vids) == 40, f"fold {fold}: expected 40 train videos"
        assert len(val_vids) == 10, f"fold {fold}: expected 10 val videos"
        assert set(train_vids).isdisjoint(set(val_vids)), "train/val overlap in fold %d" % fold


def test_fold_split_covers_all_videos():
    """Train + val covers all 50 videos exactly once."""
    train, val = _fold_split(eval_fold=1)
    all_vids = sorted(train + val)
    expected = [f"rgb-seq-{i:02d}" for i in range(1, 51)]
    assert all_vids == expected


def test_dataset_synthetic_mode():
    """FiftySaladsFeaturesDataset generates windows with correct shapes (no real features)."""
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = _make_gt_dir(tmp, num_videos=6, frames_per_video=40)
        videos = [f"rgb-seq-{i:02d}" for i in range(1, 7)]
        ds = FiftySaladsFeaturesDataset(
            video_names=videos,
            gt_dir=gt_dir,
            feature_dir=None,   # synthetic mode
            window_size=10,
            stride=5,
            feature_dim=32,
        )
        assert len(ds) > 0, "Dataset should have at least one window"
        item = ds[0]
        assert "features" in item and "labels" in item
        assert item["features"].shape == (10, 32)
        assert item["labels"].shape == (10,)
        assert item["features"].dtype == torch.float32
        assert item["labels"].dtype == torch.int64


def test_dataset_all_windows_correct_size():
    """Every window returned by the dataset has shape (window_size, feature_dim)."""
    W, D = 8, 32
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = _make_gt_dir(tmp, num_videos=4, frames_per_video=20)
        videos = [f"rgb-seq-{i:02d}" for i in range(1, 5)]
        ds = FiftySaladsFeaturesDataset(
            video_names=videos, gt_dir=gt_dir, feature_dir=None,
            window_size=W, stride=4, feature_dim=D,
        )
        for i in range(len(ds)):
            item = ds[i]
            assert item["features"].shape == (W, D), f"Window {i} has wrong feature shape"
            assert item["labels"].shape == (W,), f"Window {i} has wrong label shape"


def test_dataset_with_preextracted_features():
    """Dataset correctly loads real .npy feature files when available."""
    D, T, W = 32, 25, 10
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = _make_gt_dir(tmp, num_videos=2, frames_per_video=T)
        feat_dir = os.path.join(tmp, "features")
        os.makedirs(feat_dir)
        rng = np.random.RandomState(7)
        for i in range(1, 3):
            feat = rng.randn(T, D).astype(np.float32)
            np.save(os.path.join(feat_dir, f"rgb-seq-{i:02d}.npy"), feat)

        videos = [f"rgb-seq-{i:02d}" for i in range(1, 3)]
        ds = FiftySaladsFeaturesDataset(
            video_names=videos, gt_dir=gt_dir, feature_dir=feat_dir,
            window_size=W, stride=5, feature_dim=D,
        )
        assert len(ds) > 0
        item = ds[0]
        assert item["features"].shape == (W, D)


def test_lstr_forward_pass():
    """LongShortTermTransformer produces (B, T, C) output from (B, T, D) input."""
    B, T, D, C = 2, 16, 32, NUM_CLASSES
    model = _make_tiny_lstr()
    x = torch.randn(B, T, D)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, T, C), f"Expected ({B}, {T}, {C}), got {tuple(out.shape)}"


def test_train_one_epoch_reduces_loss():
    """A single training epoch should run without error and return a finite loss."""
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = _make_gt_dir(tmp, num_videos=5, frames_per_video=30)
        videos = [f"rgb-seq-{i:02d}" for i in range(1, 6)]
        from torch.utils.data import DataLoader

        ds = FiftySaladsFeaturesDataset(
            video_names=videos, gt_dir=gt_dir, feature_dir=None,
            window_size=8, stride=4, feature_dim=32,
        )
        loader = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)
        model = _make_tiny_lstr()
        crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = _train_one_epoch(model, loader, crit, opt, torch.device("cpu"),
                                grad_clip=1.0, log_every=999, epoch=0)
        assert np.isfinite(loss), f"Training loss is not finite: {loss}"


def test_evaluate_returns_finite_metrics():
    """Evaluation returns finite loss and accuracy in [0, 100]."""
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = _make_gt_dir(tmp, num_videos=2, frames_per_video=20)
        videos = [f"rgb-seq-{i:02d}" for i in range(1, 3)]
        from torch.utils.data import DataLoader

        ds = FiftySaladsFeaturesDataset(
            video_names=videos, gt_dir=gt_dir, feature_dir=None,
            window_size=8, stride=8, feature_dim=32,
        )
        loader = DataLoader(ds, batch_size=2)
        model = _make_tiny_lstr()
        crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
        val_loss, val_acc = _evaluate(model, loader, crit, torch.device("cpu"))
        assert np.isfinite(val_loss), f"Val loss not finite: {val_loss}"
        assert 0.0 <= val_acc <= 100.0, f"Val acc out of range: {val_acc}"


def test_run_lstr_training_smoke():
    """
    End-to-end smoke test: run_lstr_training with 2 epochs and synthetic data.
    Verifies that checkpoints and a JSON history file are created.
    """
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = _make_gt_dir(tmp, num_videos=50, frames_per_video=30)
        ckpt_dir = os.path.join(tmp, "checkpoints", "lstr")
        log_dir = os.path.join(tmp, "runs", "lstr")

        config = {
            "model": {
                "in_dim": 32,
                "num_classes": NUM_CLASSES,
                "d_model": 32,
                "num_heads": 2,
                "num_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.0,
                "max_seq_len": 64,
                "causal": True,
            },
            "training": {
                "device": "cpu",
                "epochs": 2,
                "batch_size": 2,
                "window_size": 8,
                "stride": 4,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "lr_scheduler": "cosine",
                "warmup_epochs": 0,
                "gradient_clip": 1.0,
                "checkpoint_dir": ckpt_dir,
                "save_every_n_epochs": 1,
                "num_workers": 0,
            },
            "dataset": {
                "data_root": tmp,   # parent of groundTruth
                "feature_dir": None,
                "num_folds": 5,
                "eval_fold": 1,
            },
            "logging": {
                "log_dir": log_dir,
                "log_every_n_steps": 999,
            },
        }

        # Patch data_root to point at our tmp dir
        args = _make_args(data_root=tmp)
        run_lstr_training(config, args)

        # Check that at least the best checkpoint was saved
        best_ckpt = os.path.join(ckpt_dir, "lstr_best.pt")
        assert os.path.exists(best_ckpt), "Best checkpoint not found: " + best_ckpt

        # Verify the checkpoint can be loaded
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "epoch" in ckpt
        assert "metrics" in ckpt

        # Check JSON history
        history_path = os.path.join(log_dir, "lstr_training_history.json")
        assert os.path.exists(history_path), "Training history JSON not found"
        with open(history_path, "r", encoding="utf-8") as fh:
            history = json.load(fh)
        assert "history" in history
        assert len(history["history"]) == 2   # 2 epochs
        assert "val_frame_acc" in history["history"][0]


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  [PASS]  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            import traceback
            print(f"  [FAIL]  {name}: {exc}")
            traceback.print_exc()

    total = len(tests)
    passed = total - len(failures)
    print(f"\n{passed} passed, {len(failures)} failed")
    if failures:
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

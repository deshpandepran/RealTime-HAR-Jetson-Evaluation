#!/usr/bin/env python3
"""
distillation/train_lstr.py — LSTR Training Loop  (Issue #7)
============================================================
Fine-tunes the Long Short-Term Transformer (LSTR) on the 50 Salads dataset
using the ground-truth frame-level labels that are already checked-in under
``data/fifty_salads/groundTruth/``.

Two input modes are supported, selected automatically at runtime:

  **Mode A – Pre-extracted features** (preferred / fast):
    IMFE features are stored as ``<feature_dir>/<video_name>.npy`` files,
    each of shape ``(T, feature_dim)``.  If ``feature_dir`` is set in the
    config and at least one ``.npy`` file exists there, this mode is used.

  **Mode B – On-the-fly with synthetic features** (smoke-test / no GPU):
    When pre-extracted features are absent, random Gaussian vectors of the
    configured ``in_dim`` are used so that the entire training loop
    (data loading → forward → loss → backward → checkpoint) can be verified
    without any video files or a trained IMFE checkpoint.

Usage (from repo root)
----------------------
    # Full run (Mode A, requires pre-extracted .npy features)
    python train.py --stage lstr --config configs/lstr_training.yaml

    # Smoke-test with synthetic features (no videos needed)
    python train.py --stage lstr --config configs/lstr_training.yaml --smoke-test

    # Override device
    python train.py --stage lstr --config configs/lstr_training.yaml --device cpu

Public API
----------
    run_lstr_training(config, args)  – called by train.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Label vocabulary — 50 Salads fine-grained 18-class set + background
# ─────────────────────────────────────────────────────────────────────────────

FIFTY_SALADS_ACTIONS: List[str] = [
    "background",
    "cut_cheese",
    "cut_cucumber",
    "cut_tomato",
    "peel_cucumber",
    "place_cheese_into_bowl",
    "place_cucumber_into_bowl",
    "place_tomato_into_bowl",
    "add_dressing",
    "add_oil",
    "add_pepper",
    "add_salt",
    "add_vinegar",
    "mix_dressing",
    "mix_ingredients",
    "serve_salad_onto_plate",
    "add_croutons",
    "peel_tomato",
    "add_lettuce",
]

# Build a frozen vocabulary (label string → integer index)
LABEL_TO_IDX: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(FIFTY_SALADS_ACTIONS)}
NUM_CLASSES: int = len(FIFTY_SALADS_ACTIONS)   # 19


def _encode_label(label_str: str) -> int:
    """Map a label string to its integer index, defaulting to 'background'."""
    return LABEL_TO_IDX.get(label_str.strip(), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_frame_labels(gt_path: str) -> np.ndarray:
    """Read a one-label-per-line ground truth file → integer label array."""
    with open(gt_path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    return np.array([_encode_label(ln) for ln in lines], dtype=np.int64)


def _fold_split(
    num_videos: int = 50,
    num_folds: int = 5,
    eval_fold: int = 1,
) -> Tuple[List[str], List[str]]:
    """Return (train_videos, val_videos) for the given fold (1-indexed)."""
    all_videos = [f"rgb-seq-{i:02d}" for i in range(1, num_videos + 1)]
    fold_size = num_videos // num_folds
    folds = [all_videos[i * fold_size: (i + 1) * fold_size] for i in range(num_folds)]
    val_videos = folds[eval_fold - 1]
    train_videos = [v for fi, f in enumerate(folds) if fi != (eval_fold - 1) for v in f]
    return train_videos, val_videos


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset – sliding window over per-video feature sequences
# ─────────────────────────────────────────────────────────────────────────────

class FiftySaladsFeaturesDataset(Dataset):
    """Dataset of (feature_window, label_window) pairs from 50 Salads.

    Each item is a fixed-length temporal window (``window_size`` time-steps)
    sampled with a given ``stride`` from one of the videos assigned to the
    split.

    Features are loaded from pre-extracted NumPy files when available; otherwise
    synthetic Gaussian vectors are generated on the fly for smoke-testing.

    Parameters
    ----------
    video_names : list of str
        Video identifiers (e.g. ``["rgb-seq-01", …]``).
    gt_dir : str
        Directory containing ``<video_name>.txt`` ground-truth files.
    feature_dir : str or None
        Directory containing ``<video_name>.npy`` feature files, shape
        ``(T, feature_dim)``.  If ``None`` or the files are absent, synthetic
        features are generated.
    window_size : int
        Number of time-steps per training sample.
    stride : int
        Sliding-window stride (overlap = window_size - stride).
    feature_dim : int
        Dimensionality of the feature vectors (used for synthetic mode).
    """

    def __init__(
        self,
        video_names: List[str],
        gt_dir: str,
        feature_dir: Optional[str],
        window_size: int = 64,
        stride: int = 32,
        feature_dim: int = 2048,
    ):
        self.window_size = window_size
        self.stride = stride
        self.feature_dim = feature_dim

        # Build list of (feature_array, label_array) per video
        self._windows: List[Tuple[np.ndarray, np.ndarray]] = []

        for video_name in video_names:
            gt_path = os.path.join(gt_dir, f"{video_name}.txt")
            if not os.path.exists(gt_path):
                logger.warning("Ground truth not found, skipping: %s", gt_path)
                continue

            labels = _load_frame_labels(gt_path)   # (T,)
            T = len(labels)

            # Try to load pre-extracted features
            features = None
            if feature_dir is not None:
                feat_path = os.path.join(feature_dir, f"{video_name}.npy")
                if os.path.exists(feat_path):
                    features = np.load(feat_path).astype(np.float32)   # (T, D)
                    if features.shape[0] != T:
                        # Resample via linear interpolation if lengths differ
                        idx = np.linspace(0, features.shape[0] - 1, T).astype(int)
                        features = features[idx]

            if features is None:
                # Smoke-test: generate deterministic synthetic features
                rng = np.random.RandomState(seed=hash(video_name) & 0xFFFFFFFF)
                features = rng.randn(T, feature_dim).astype(np.float32)

            # Sliding-window extraction
            for start in range(0, max(1, T - window_size + 1), stride):
                end = start + window_size
                feat_win = features[start:end]
                lbl_win = labels[start:end]

                # Pad if the window is shorter than window_size (last chunk)
                if len(feat_win) < window_size:
                    pad = window_size - len(feat_win)
                    feat_win = np.concatenate(
                        [feat_win, np.zeros((pad, feature_dim), dtype=np.float32)], axis=0
                    )
                    lbl_win = np.concatenate([lbl_win, np.zeros(pad, dtype=np.int64)], axis=0)

                self._windows.append((feat_win, lbl_win))

        if not self._windows:
            raise RuntimeError(
                "FiftySaladsFeaturesDataset: no valid windows found. "
                "Check that ground-truth files exist in the gt_dir and that "
                "video names are correct."
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        features, labels = self._windows[idx]
        return {
            "features": torch.from_numpy(features),   # (T, D) float32
            "labels": torch.from_numpy(labels),        # (T,) int64
        }


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    num_epochs: int,
    warmup_epochs: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build a learning-rate scheduler with optional linear warm-up."""
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=1e-6)
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return None   # "none" or unknown


def _warmup_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    warmup_epochs: int,
    current_epoch: int,
) -> None:
    """Linearly scale LR from 0 → base_lr during warm-up epochs."""
    if current_epoch < warmup_epochs:
        scale = (current_epoch + 1) / max(warmup_epochs, 1)
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * scale


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    log_every: int = 10,
    epoch: int = 0,
) -> float:
    """Run one full training epoch, returning mean cross-entropy loss."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_frames = 0

    for step, batch in enumerate(dataloader):
        features = batch["features"].to(device)   # (B, T, D)
        labels = batch["labels"].to(device)        # (B, T)

        optimizer.zero_grad()

        logits = model(features)                   # (B, T, C)
        B, T, C = logits.shape

        # Flatten for cross-entropy: (B*T, C) vs (B*T,)
        loss = criterion(logits.view(B * T, C), labels.view(B * T))
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)              # (B, T)
        total_correct += (preds == labels).sum().item()
        total_frames += B * T

        if (step + 1) % log_every == 0:
            running_acc = 100.0 * total_correct / max(total_frames, 1)
            logger.info(
                "  Epoch %d | Step %d/%d | loss=%.4f | frame_acc=%.2f%%",
                epoch + 1, step + 1, len(dataloader),
                total_loss / (step + 1), running_acc,
            )

    mean_loss = total_loss / max(len(dataloader), 1)
    return mean_loss


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate on the validation set. Returns (mean_loss, frame_accuracy %)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_frames = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        logits = model(features)                   # (B, T, C)
        B, T, C = logits.shape
        loss = criterion(logits.view(B * T, C), labels.view(B * T))
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_frames += B * T

    mean_loss = total_loss / max(len(dataloader), 1)
    frame_acc = 100.0 * total_correct / max(total_frames, 1)
    return mean_loss, frame_acc


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    checkpoint_dir: str,
    filename: str,
) -> str:
    """Save model + optimiser state to disk. Returns the saved path."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    logger.info("Checkpoint saved → %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point (called by train.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_lstr_training(config: dict, args) -> None:
    """
    Full LSTR training loop on 50 Salads (Issue #7).

    Parameters
    ----------
    config : dict
        Parsed YAML configuration (``configs/lstr_training.yaml``).
    args : argparse.Namespace
        CLI arguments from ``train.py``.
    """
    from models import LongShortTermTransformer

    # ── Device ───────────────────────────────────────────────────────────────
    device_str = getattr(args, "device", None) or config.get("training", {}).get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("LSTR training device: %s", device)

    # ── Config extraction ────────────────────────────────────────────────────
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("dataset", {})
    log_cfg = config.get("logging", {})
    smoke_test: bool = getattr(args, "smoke_test", False)

    in_dim: int = model_cfg.get("in_dim", 2048)
    num_classes: int = model_cfg.get("num_classes", NUM_CLASSES)
    epochs: int = 2 if smoke_test else train_cfg.get("epochs", 30)
    batch_size: int = 2 if smoke_test else train_cfg.get("batch_size", 4)
    window_size: int = train_cfg.get("window_size", 64)
    stride: int = train_cfg.get("stride", 32)
    lr: float = train_cfg.get("learning_rate", 1e-4)
    weight_decay: float = train_cfg.get("weight_decay", 1e-5)
    warmup_epochs: int = train_cfg.get("warmup_epochs", 3)
    grad_clip: float = train_cfg.get("gradient_clip", 1.0)
    save_every: int = train_cfg.get("save_every_n_epochs", 5)
    num_workers: int = 0 if smoke_test else train_cfg.get("num_workers", 2)
    scheduler_name: str = train_cfg.get("lr_scheduler", "cosine")
    checkpoint_dir: str = train_cfg.get("checkpoint_dir", "checkpoints/lstr/")

    data_root: str = getattr(args, "data_root", None) or data_cfg.get(
        "data_root", "data/fifty_salads/"
    )
    feature_dir: Optional[str] = data_cfg.get("feature_dir", None)
    num_folds: int = data_cfg.get("num_folds", 5)
    eval_fold: int = data_cfg.get("eval_fold", 1)
    log_every: int = log_cfg.get("log_every_n_steps", 10)

    # ── Validate data root ────────────────────────────────────────────────────
    gt_dir = os.path.join(data_root, "groundTruth")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(
            f"50 Salads groundTruth directory not found: {gt_dir}\n"
            "Ensure 'data/fifty_salads/groundTruth/' exists (it was added in a recent PR)."
        )

    # Detect whether pre-extracted features are available
    has_features = False
    if feature_dir and os.path.isdir(feature_dir):
        npy_files = list(Path(feature_dir).glob("*.npy"))
        has_features = len(npy_files) > 0

    mode = "pre-extracted features" if has_features else "synthetic features (smoke-test mode)"
    logger.info("Feature mode: %s", mode)
    if not has_features:
        logger.warning(
            "No pre-extracted feature files found in '%s'. "
            "Using synthetic Gaussian features so the training loop can be verified "
            "without video files or a trained IMFE checkpoint. "
            "Run IMFE feature extraction first for real training.",
            feature_dir or "(feature_dir not set)",
        )

    # ── Fold split ────────────────────────────────────────────────────────────
    train_videos, val_videos = _fold_split(
        num_videos=50, num_folds=num_folds, eval_fold=eval_fold
    )
    logger.info(
        "Fold %d/%d — train: %d videos, val: %d videos",
        eval_fold, num_folds, len(train_videos), len(val_videos),
    )

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    feature_dir_real = feature_dir if has_features else None

    train_dataset = FiftySaladsFeaturesDataset(
        video_names=train_videos,
        gt_dir=gt_dir,
        feature_dir=feature_dir_real,
        window_size=window_size,
        stride=stride,
        feature_dim=in_dim,
    )
    val_dataset = FiftySaladsFeaturesDataset(
        video_names=val_videos,
        gt_dir=gt_dir,
        feature_dir=feature_dir_real,
        window_size=window_size,
        stride=window_size,   # No overlap for eval
        feature_dim=in_dim,
    )

    logger.info(
        "Dataset: %d train windows, %d val windows (window=%d, stride=%d)",
        len(train_dataset), len(val_dataset), window_size, stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    lstr = LongShortTermTransformer(
        in_dim=in_dim,
        num_classes=num_classes,
        d_model=model_cfg.get("d_model", 512),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 2),
        dim_feedforward=model_cfg.get("dim_feedforward", 2048),
        dropout=model_cfg.get("dropout", 0.1),
        max_seq_len=model_cfg.get("max_seq_len", 512),
        causal=model_cfg.get("causal", True),
    ).to(device)

    total_params = sum(p.numel() for p in lstr.parameters())
    trainable_params = sum(p.numel() for p in lstr.parameters() if p.requires_grad)
    logger.info(
        "LSTR initialised — total params: %s | trainable: %s",
        f"{total_params:,}", f"{trainable_params:,}",
    )

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    # Class-frequency weighting can help with imbalanced 50 Salads labels;
    # we default to uniform weighting for simplicity.
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = AdamW(lstr.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _build_lr_scheduler(optimizer, scheduler_name, epochs, warmup_epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_ckpt_path: Optional[str] = None
    history: List[Dict] = []

    logger.info("Starting LSTR training for %d epoch(s)…", epochs)
    t0 = time.time()

    for epoch in range(epochs):
        epoch_t0 = time.time()

        # Warm-up LR override
        _warmup_lr(optimizer, lr, warmup_epochs, epoch)

        # Train
        train_loss = _train_one_epoch(
            lstr, train_loader, criterion, optimizer,
            device, grad_clip, log_every=log_every, epoch=epoch,
        )

        # Step LR scheduler (after warm-up)
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Validate
        val_loss, val_acc = _evaluate(lstr, val_loader, criterion, device)

        epoch_time = time.time() - epoch_t0
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%% | lr=%.2e | %.1fs",
            epoch + 1, epochs, train_loss, val_loss, val_acc, current_lr, epoch_time,
        )

        entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_frame_acc": round(val_acc, 4),
            "lr": current_lr,
        }
        history.append(entry)

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            _save_checkpoint(
                lstr, optimizer, epoch + 1, entry, checkpoint_dir,
                f"lstr_epoch_{epoch + 1:03d}.pt",
            )

        # Best model checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = _save_checkpoint(
                lstr, optimizer, epoch + 1, entry, checkpoint_dir, "lstr_best.pt"
            )
            logger.info("  ★ New best val_acc=%.2f%% → %s", best_val_acc, best_ckpt_path)

    total_time = time.time() - t0
    logger.info(
        "Training complete in %.1fs | Best val_frame_acc=%.2f%% | checkpoint → %s",
        total_time, best_val_acc, best_ckpt_path,
    )

    # ── Save training history ────────────────────────────────────────────────
    log_dir = log_cfg.get("log_dir", "runs/lstr/")
    os.makedirs(log_dir, exist_ok=True)
    history_path = os.path.join(log_dir, "lstr_training_history.json")
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": {
                    "model": model_cfg,
                    "training": train_cfg,
                    "dataset": data_cfg,
                },
                "feature_mode": mode,
                "best_val_frame_acc": round(best_val_acc, 4),
                "best_checkpoint": best_ckpt_path,
                "history": history,
            },
            fh,
            indent=2,
        )
    logger.info("Training history saved → %s", history_path)

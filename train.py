#!/usr/bin/env python3
"""
train.py — RT-HARE / IMFE Main Training Pipeline Entry Point
=============================================================
Orchestrates the full training workflow:

  Stage 1 (Issue #6):  Knowledge Distillation — train IMFE student from
                        frozen TV-L1 + ResNetFlowExtractor teacher via MSE loss.

  Stage 2 (Issue #7):  LSTR training on 50 Salads with ground-truth labels.
                        Uses pre-extracted IMFE feature .npy files when
                        available, or synthetic features for smoke-testing when
                        no video files are present.

Usage
-----
    # Stage 1: Knowledge Distillation
    python train.py --stage distill --config configs/imfe_distillation.yaml

    # Stage 2: LSTR training (pre-extracted features recommended)
    python train.py --stage lstr --config configs/lstr_training.yaml

    # Stage 2: LSTR smoke-test (no videos or GPU needed)
    python train.py --stage lstr --config configs/lstr_training.yaml --smoke-test

    # Verify environment only
    python train.py --verify-only

See configs/ for all hyperparameters.
"""

import argparse
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RT-HARE / IMFE Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["distill", "lstr"],
        default="distill",
        help="Training stage: 'distill' (IMFE KD) or 'lstr' (LSTR fine-tune).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/imfe_distillation.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--imfe-checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained IMFE checkpoint (.pt) for LSTR stage.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda / cpu) from config.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run environment + architecture verification and exit.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "LSTR stage only: run 2 epochs with synthetic features and a tiny "
            "batch so the full training loop can be verified without video files "
            "or a trained IMFE checkpoint."
        ),
    )
    return parser


# ─────────────────────────────────────────────────────────────
# Environment verification (quick sanity check)
# ─────────────────────────────────────────────────────────────

def verify_environment():
    """Run basic import checks and architecture smoke test."""
    logger.info("Running environment verification...")

    import torch
    logger.info(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

    # Architecture imports
    from models import IMFE, LongShortTermTransformer, RGBOnlyBaseline

    # Quick forward pass
    imfe = IMFE(clip_length=6, feature_dim=2048)
    lstr = LongShortTermTransformer(in_dim=2048, num_classes=50, causal=True)
    x_clip = torch.randn(1, 6, 3, 256, 344)
    with torch.no_grad():
        feat = imfe(x_clip)
        logits = lstr(feat.unsqueeze(0))
    logger.info(f"IMFE output: {tuple(feat.shape)}")
    logger.info(f"LSTR output: {tuple(logits.shape)}")
    logger.info("✅ Environment verification passed.")


# ─────────────────────────────────────────────────────────────
# Stage 1: Knowledge Distillation (IMFE training)
# ─────────────────────────────────────────────────────────────

def run_distillation(config: dict, args: argparse.Namespace):
    """
    Trains the IMFE student network via knowledge distillation from a
    frozen TV-L1 + ResNetFlowExtractor teacher. Requires Issue #6 to be
    implemented in distillation/train_kd.py.

    Args:
        config: Parsed YAML configuration dictionary.
        args: CLI arguments namespace.
    """
    try:
        from distillation import run_kd_training
    except ImportError:
        try:
            from distillation.train_kd import run_kd_training
        except ImportError:
            logger.error("distillation training module not found.")
            sys.exit(1)

    logger.info("Starting Knowledge Distillation training...")
    run_kd_training(config, args)


# ─────────────────────────────────────────────────────────────
# Stage 2: LSTR Training
# ─────────────────────────────────────────────────────────────

def run_lstr_training(config: dict, args: argparse.Namespace):
    """
    Full LSTR training loop on 50 Salads (Issue #7).

    Delegates to ``distillation/train_lstr.py::run_lstr_training`` which
    implements the complete pipeline:
      - 5-fold deterministic split matching the 50 Salads evaluation protocol
      - FiftySaladsFeaturesDataset with sliding-window sampling
      - Pre-extracted IMFE feature loading *or* synthetic feature fallback
      - AdamW + cosine-annealing LR schedule with warm-up
      - Per-epoch train / val metrics logging
      - Periodic + best-model checkpointing
      - JSON training-history export

    Args:
        config: Parsed YAML configuration dictionary.
        args: CLI arguments namespace.
    """
    try:
        from distillation.train_lstr import run_lstr_training as _run
    except ImportError as exc:
        logger.error("Failed to import LSTR training loop: %s", exc)
        sys.exit(1)

    # Validate data root early so the user gets a clear error message
    data_root = args.data_root or config.get("dataset", {}).get(
        "data_root", "data/fifty_salads/"
    )
    if not os.path.exists(data_root):
        logger.error(
            "50 Salads data root not found at '%s'. "
            "Ensure 'data/fifty_salads/groundTruth/' exists "
            "(it was added via a recent PR). "
            "Override with --data-root if needed.",
            data_root,
        )
        sys.exit(1)

    logger.info("Starting LSTR training (Issue #7)…")
    _run(config, args)


# ─────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    import yaml
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verify_only:
        verify_environment()
        return

    config = load_config(args.config)

    if args.stage == "distill":
        run_distillation(config, args)
    elif args.stage == "lstr":
        run_lstr_training(config, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

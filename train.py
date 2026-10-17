#!/usr/bin/env python3
"""
train.py — RT-HARE / IMFE Main Training Pipeline Entry Point
=============================================================
Orchestrates the full training workflow:

  Stage 1 (Issue #6):  Knowledge Distillation — train IMFE student from
                        frozen TV-L1 + ResNetFlowExtractor teacher via MSE loss.

  Stage 2 (Issue #7):  LSTR fine-tuning on 50 Salads with the pre-trained IMFE
                        backbone frozen (or jointly trained).

Usage
-----
    # Stage 1: Knowledge Distillation
    python train.py --stage distill --config configs/imfe_distillation.yaml

    # Stage 2: LSTR training (requires pre-trained IMFE checkpoint)
    python train.py --stage lstr --config configs/lstr_inference.yaml \
                    --imfe-checkpoint checkpoints/imfe_best.pt

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
        from distillation.train_kd import run_kd_training
    except ImportError:
        logger.error(
            "distillation/train_kd.py not yet implemented (Issue #6). "
            "Run this stage once Issue #6 is merged into main."
        )
        sys.exit(1)

    logger.info("Starting Knowledge Distillation training (Issue #6)...")
    run_kd_training(config, args)


# ─────────────────────────────────────────────────────────────
# Stage 2: LSTR Training
# ─────────────────────────────────────────────────────────────

def run_lstr_training(config: dict, args: argparse.Namespace):
    """
    Fine-tunes the LSTR module on 50 Salads using pre-extracted IMFE features.
    Loads a pre-trained IMFE checkpoint from --imfe-checkpoint.

    Args:
        config: Parsed YAML configuration dictionary.
        args: CLI arguments namespace.
    """
    import torch
    from models import IMFE, LongShortTermTransformer
    from data.fifty_salads import FiftySaladsDataset

    device_str = args.device or config.get("inference", {}).get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"LSTR training on device: {device}")

    # Load IMFE
    imfe = IMFE(clip_length=6, feature_dim=2048).to(device)
    if args.imfe_checkpoint:
        if not os.path.exists(args.imfe_checkpoint):
            logger.error(f"IMFE checkpoint not found: {args.imfe_checkpoint}")
            sys.exit(1)
        imfe.load_state_dict(torch.load(args.imfe_checkpoint, map_location=device))
        logger.info(f"Loaded IMFE checkpoint: {args.imfe_checkpoint}")
    else:
        logger.warning(
            "No --imfe-checkpoint provided. IMFE will use random weights. "
            "Run distillation stage first (Issue #6) to get a trained IMFE."
        )

    # Freeze IMFE for LSTR training
    for param in imfe.parameters():
        param.requires_grad = False
    imfe.eval()
    logger.info("IMFE backbone frozen.")

    # Initialize LSTR
    num_classes = config.get("model", {}).get("num_classes", 50)
    lstr = LongShortTermTransformer(
        in_dim=2048,
        num_classes=num_classes,
        d_model=config.get("model", {}).get("d_model", 512),
        num_heads=config.get("model", {}).get("num_heads", 8),
        num_layers=config.get("model", {}).get("num_layers", 2),
        causal=config.get("model", {}).get("causal", True),
    ).to(device)
    logger.info(f"LSTR initialized: {sum(p.numel() for p in lstr.parameters()):,} params")

    # Dataset (requires actual data files on disk)
    data_root = args.data_root or config.get("dataset", {}).get("data_root", "data/fifty_salads/")
    logger.info(f"Dataset root: {data_root}")

    if not os.path.exists(data_root):
        logger.error(
            f"50 Salads data root not found at '{data_root}'. "
            "Download the dataset and update --data-root or configs/lstr_inference.yaml."
        )
        sys.exit(1)

    # Placeholder: full training loop belongs to Issue #7
    logger.warning(
        "Full LSTR training loop belongs to Issue #7. "
        "Implement distillation/train_lstr.py for the complete loop."
    )


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

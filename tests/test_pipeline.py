# -*- coding: utf-8 -*-
"""
RT-HARE / IMFE - Full Architecture Pipeline Smoke Test
======================================================
Validates all model forward passes, shape contracts, and parameter counts
using synthetic random tensors. No real datasets or GPU required.

Run from the repo root:
    python tests/test_pipeline.py
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

PASS  = "[PASS]"
FAIL  = "[FAIL]"
SKIP  = "[SKIP]"

results = []

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, condition))
    return condition

def section(title: str):
    print(f"\n" + "-"*60)
    print(f"  {title}")
    print("-"*60)

def run_test(name, fn):
    try:
        fn()
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         Exception: {e}")
        traceback.print_exc()
        results.append((name, False))

# ─────────────────────────────────────────────────────────────
# Constants matching the paper / README
# ─────────────────────────────────────────────────────────────
B = 2          # batch size
K = 6          # clip length (frames per clip)
H, W = 256, 344  # spatial dims (multiples of 8)
FLOW_CHANNELS = 10  # 5 flow pairs × 2 (u, v)
MOTION_DIM = 2048   # IMFE output feature dim
RGB_DIM = 2048      # RGB baseline output feature dim
FUSED_DIM = 2560    # LSTR default: IMFE (2048) + RGB pooled (512) — see lstr.py
T = 16              # temporal sequence length fed to LSTR
NUM_CLASSES = 50    # 50-class setting (50 Salads)

device = torch.device("cpu")  # CPU-only for portability on Windows

# ─────────────────────────────────────────────────────────────
# 1. VideoPreprocessor
# ─────────────────────────────────────────────────────────────
section("1. VideoPreprocessor")

def test_preprocessor():
    from data.preprocess import VideoPreprocessor
    proc = VideoPreprocessor(target_h=H, target_w=W)

    # Simulate K=6 raw uint8 frames
    raw_frames = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(K)]
    clip_tensor = proc.process_clip(raw_frames)

    check("Output shape == (K, 3, H, W)",
          clip_tensor.shape == (K, 3, H, W),
          f"got {tuple(clip_tensor.shape)}")
    check("Dtype == float32", clip_tensor.dtype == torch.float32)
    check("Values roughly normalised (mean near 0)",
          abs(float(clip_tensor.mean())) < 2.0,
          f"mean={float(clip_tensor.mean()):.3f}")

run_test("VideoPreprocessor", test_preprocessor)

# ─────────────────────────────────────────────────────────────
# 2. IMFE — Single-Shot Motion Feature Extractor
# ─────────────────────────────────────────────────────────────
section("2. IMFE (Integrated Motion Feature Extractor)")

def test_imfe():
    from models.imfe import IMFE
    model = IMFE(clip_length=K, feature_dim=MOTION_DIM).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"         Parameters: {n_params:,}")

    # Input: (B, K, 3, H, W)
    x = torch.randn(B, K, 3, H, W)
    with torch.no_grad():
        out = model(x)

    check("Output shape == (B, MOTION_DIM)",
          out.shape == (B, MOTION_DIM),
          f"got {tuple(out.shape)}")
    check("No NaN in output", not torch.isnan(out).any())
    check("Parameter count > 1M", n_params > 1_000_000,
          f"{n_params:,} params")

run_test("IMFE forward pass", test_imfe)

# ─────────────────────────────────────────────────────────────
# 3. RGBOnlyBaseline
# ─────────────────────────────────────────────────────────────
section("3. RGBOnlyBaseline (ResNet-50 spatial features)")

def test_rgb_baseline():
    from models.baselines import RGBOnlyBaseline
    model = RGBOnlyBaseline(pretrained=False).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"         Parameters: {n_params:,}")

    # 5-D input: (B, K, 3, H, W)
    x_5d = torch.randn(B, K, 3, H, W)
    with torch.no_grad():
        out_5d = model(x_5d)
    check("5-D input → (B, K, 2048)",
          out_5d.shape == (B, K, RGB_DIM),
          f"got {tuple(out_5d.shape)}")

    # 4-D input: (B, 3, H, W)
    x_4d = torch.randn(B, 3, H, W)
    with torch.no_grad():
        out_4d = model(x_4d)
    check("4-D input → (B, 2048)",
          out_4d.shape == (B, RGB_DIM),
          f"got {tuple(out_4d.shape)}")

    check("No NaN in output", not torch.isnan(out_5d).any())

run_test("RGBOnlyBaseline forward pass", test_rgb_baseline)

# ─────────────────────────────────────────────────────────────
# 4. ResNetFlowExtractor
# ─────────────────────────────────────────────────────────────
section("4. ResNetFlowExtractor (Multi-channel flow features)")

def test_resnet_flow():
    from models.resnet_flow import ResNetFlowExtractor
    model = ResNetFlowExtractor(in_channels=FLOW_CHANNELS, pretrained=False).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"         Parameters: {n_params:,}")

    # Input: (B, 10, H, W) — 5 flow pairs stacked
    x = torch.randn(B, FLOW_CHANNELS, H, W)
    with torch.no_grad():
        out = model(x)

    check("Output shape == (B, 2048)",
          out.shape == (B, 2048),
          f"got {tuple(out.shape)}")
    check("No NaN in output", not torch.isnan(out).any())

run_test("ResNetFlowExtractor forward pass", test_resnet_flow)

# ─────────────────────────────────────────────────────────────
# 5. LongShortTermTransformer — fused features
# ─────────────────────────────────────────────────────────────
section("5. LSTR — Fused Features (in_dim=2560, causal=True)")

def test_lstr_fused():
    from models.lstr import LongShortTermTransformer
    model = LongShortTermTransformer(
        in_dim=FUSED_DIM, num_classes=NUM_CLASSES,
        d_model=512, num_heads=8, num_layers=2,
        causal=True
    ).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"         Parameters: {n_params:,}")

    x = torch.randn(B, T, FUSED_DIM)
    with torch.no_grad():
        logits = model(x)
        preds  = model.predict(x)

    check("Logits shape == (B, T, num_classes)",
          logits.shape == (B, T, NUM_CLASSES),
          f"got {tuple(logits.shape)}")
    check("Predict shape == (B, T)",
          preds.shape == (B, T),
          f"got {tuple(preds.shape)}")
    check("No NaN in logits", not torch.isnan(logits).any())
    check("Predictions are valid class indices",
          int(preds.min()) >= 0 and int(preds.max()) < NUM_CLASSES,
          f"range [{int(preds.min())}, {int(preds.max())}]")

run_test("LSTR fused forward pass", test_lstr_fused)

# ─────────────────────────────────────────────────────────────
# 6. LongShortTermTransformer — RGB-only features
# ─────────────────────────────────────────────────────────────
section("6. LSTR — RGB-Only Features (in_dim=2048)")

def test_lstr_rgb():
    from models.lstr import LongShortTermTransformer
    model = LongShortTermTransformer(
        in_dim=RGB_DIM, num_classes=NUM_CLASSES,
        d_model=512, num_heads=8, num_layers=2,
        causal=True
    ).eval()

    x = torch.randn(B, T, RGB_DIM)
    with torch.no_grad():
        logits = model(x)

    check("RGB-only logits shape == (B, T, num_classes)",
          logits.shape == (B, T, NUM_CLASSES),
          f"got {tuple(logits.shape)}")
    check("No NaN in RGB-only output", not torch.isnan(logits).any())

run_test("LSTR RGB-only forward pass", test_lstr_rgb)

# ─────────────────────────────────────────────────────────────
# 7. Non-causal (bidirectional) LSTR
# ─────────────────────────────────────────────────────────────
section("7. LSTR — Non-Causal (offline / bidirectional)")

def test_lstr_noncausal():
    from models.lstr import LongShortTermTransformer
    model = LongShortTermTransformer(
        in_dim=FUSED_DIM, num_classes=NUM_CLASSES, causal=False
    ).eval()

    x = torch.randn(B, T, FUSED_DIM)
    with torch.no_grad():
        logits = model(x)

    check("Non-causal LSTR output shape correct",
          logits.shape == (B, T, NUM_CLASSES),
          f"got {tuple(logits.shape)}")

run_test("LSTR non-causal forward pass", test_lstr_noncausal)

# ─────────────────────────────────────────────────────────────
# 8. LSTR with padding mask
# ─────────────────────────────────────────────────────────────
section("8. LSTR — Padding Mask (variable-length batches)")

def test_lstr_padding_mask():
    from models.lstr import LongShortTermTransformer
    model = LongShortTermTransformer(
        in_dim=FUSED_DIM, num_classes=NUM_CLASSES, causal=True
    ).eval()

    x = torch.randn(B, T, FUSED_DIM)
    # Mask last 4 positions of second sequence as PAD
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    pad_mask[1, -4:] = True

    with torch.no_grad():
        logits = model(x, src_key_padding_mask=pad_mask)

    check("Masked LSTR output shape correct",
          logits.shape == (B, T, NUM_CLASSES),
          f"got {tuple(logits.shape)}")
    check("No NaN in masked output", not torch.isnan(logits).any())

run_test("LSTR padding mask", test_lstr_padding_mask)

# ─────────────────────────────────────────────────────────────
# 9. End-to-End Pipeline: Preprocessor → IMFE → LSTR
# ─────────────────────────────────────────────────────────────
section("9. End-to-End: VideoPreprocessor → IMFE → LSTR")

def test_e2e_pipeline():
    from data.preprocess import VideoPreprocessor
    from models.imfe import IMFE
    from models.lstr import LongShortTermTransformer

    preprocessor = VideoPreprocessor(target_h=H, target_w=W)
    imfe  = IMFE(clip_length=K, feature_dim=MOTION_DIM).eval()
    # Use IMFE-only features into LSTR (in_dim=2048)
    lstr  = LongShortTermTransformer(in_dim=MOTION_DIM, num_classes=NUM_CLASSES,
                                      causal=True).eval()

    # Simulate a batch of T=8 sequential clips (streaming scenario)
    T_stream = 8
    feature_sequence = []

    with torch.no_grad():
        for t in range(T_stream):
            # One clip = 6 random frames
            raw_clip = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
                        for _ in range(K)]
            clip_tensor = preprocessor.process_clip(raw_clip)   # (K, 3, H, W)
            clip_tensor = clip_tensor.unsqueeze(0)               # (1, K, 3, H, W)
            feat = imfe(clip_tensor)                             # (1, 2048)
            feature_sequence.append(feat)

    # Stack into sequence for LSTR: (1, T_stream, 2048)
    feature_seq_tensor = torch.cat(feature_sequence, dim=0).unsqueeze(0)
    check("IMFE feature sequence shape == (1, T_stream, MOTION_DIM)",
          feature_seq_tensor.shape == (1, T_stream, MOTION_DIM),
          f"got {tuple(feature_seq_tensor.shape)}")

    with torch.no_grad():
        logits = lstr(feature_seq_tensor)
    check("E2E LSTR output shape == (1, T_stream, num_classes)",
          logits.shape == (1, T_stream, NUM_CLASSES),
          f"got {tuple(logits.shape)}")
    check("E2E no NaN", not torch.isnan(logits).any())

    preds = logits.argmax(dim=-1)
    print(f"         Per-frame predictions: {preds[0].tolist()}")

run_test("End-to-End pipeline", test_e2e_pipeline)

# ─────────────────────────────────────────────────────────────
# 10. models/__init__.py exports
# ─────────────────────────────────────────────────────────────
section("10. models/__init__.py — all exports accessible")

def test_package_exports():
    import models
    for name in ["RGBOnlyBaseline", "RAFTFlowExtractor",
                 "ResNetFlowExtractor", "IMFE", "LongShortTermTransformer"]:
        check(f"models.{name} importable", hasattr(models, name))

run_test("Package exports", test_package_exports)

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
total   = len(results)
passed  = sum(1 for _, ok in results if ok)
failed  = total - passed
print(f"  Total checks : {total}")
print(f"  Passed       : {passed}")
print(f"  Failed       : {failed}")
print("="*60)

if failed > 0:
    print("\n  FAILED CHECKS:")
    for name, ok in results:
        if not ok:
            print(f"    [FAIL]  {name}")
    sys.exit(1)
else:
    print("\n  All checks passed. Architecture is correct and consistent.")
    sys.exit(0)

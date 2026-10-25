#!/usr/bin/env python3
"""Frame-rate degradation benchmark for RT-HARE (issue #12).

For each frame rate (default 30, 15, 6, 3 FPS) this script:
  1. Subsamples the 50 Salads ground truth (recorded at 30 FPS) by stride = 30 // fps.
  2. Loads predictions from a model retrained at that frame rate.
  3. Computes Frame Accuracy (MoF), Edit score, and F1@10 / F1@25 / F1@50.
  4. Optionally attaches latency statistics and deadline miss ratios.
  5. Writes a JSON report and, optionally, the trade-off plot.

Expected layout:
  --gt-dir    data/fifty_salads/groundTruth/
                one .txt per video, one integer label id per line (30 FPS)
  --pred-dir  predictions/
                fps30/, fps15/, fps6/, fps3/  (one subfolder per retrained model)
                each video as <video>.npy (1-D labels, or (T, C) scores -> argmax)
                or <video>.txt (one integer label per line), at that frame rate
  --latency-json   optional {"30": [ms, ...], "15": [...], ...} per-frame latencies
  --deadlines-json optional {"30": ms, ...}. Default: one frame interval, 1000 / fps ms.

Notes:
  - Metrics are computed in the downsampled frame space, so each rate is scored
    against ground truth sampled at that rate.
  - Results are the mean over videos. The paper averages over 5 folds; pass
    per-fold prediction folders if you need that exact protocol.
  - Background is treated as an ordinary label. Check this against the paper's protocol.

Usage:
  python eval/benchmark_fps.py --gt-dir data/fifty_salads/groundTruth \
      --pred-dir predictions --output eval/results/fps_benchmark.json \
      --plot assets/fps_tradeoff_plots.png
  python eval/benchmark_fps.py --self-test
"""

import argparse
import json
from pathlib import Path

import numpy as np

BASE_FPS = 30
DEFAULT_FPS = [30, 15, 6, 3]
IOU_THRESHOLDS = (0.10, 0.25, 0.50)  # F1@10, F1@25, F1@50
METRIC_KEYS = ["mof", "edit", "f1@10", "f1@25", "f1@50"]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def stride_for_fps(fps: int, base_fps: int = BASE_FPS) -> int:
    """Return how many 30 FPS frames make one frame at `fps`."""
    if fps <= 0 or base_fps % fps != 0:
        raise ValueError(f"{fps} FPS must divide the base rate {base_fps} FPS evenly")
    return base_fps // fps


# ---------------------------------------------------------------------------
# Metrics (NumPy only, so this file runs without the rest of the repo)
# ---------------------------------------------------------------------------

def _segments(labels):
    """Collapse per-frame labels into (label, start, end) runs; end is exclusive."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [labels.size]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def frame_accuracy(gt, pred) -> float:
    """Frame-wise accuracy (MoF), in percent."""
    return 100.0 * float(np.mean(np.asarray(gt) == np.asarray(pred)))


def _levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def edit_score(gt, pred) -> float:
    """100 * (1 - Levenshtein / max(len)) over the ordered segment-label sequences."""
    g = [label for label, _, _ in _segments(gt)]
    p = [label for label, _, _ in _segments(pred)]
    denom = max(len(g), len(p))
    if denom == 0:
        return 100.0
    return 100.0 * (1.0 - _levenshtein(p, g) / denom)


def f1_at(gt, pred, tau: float) -> float:
    """Segment F1 at temporal IoU threshold `tau`, in percent.

    Each predicted segment takes the unmatched ground-truth segment of the same
    label with the highest IoU; it counts as a true positive if IoU >= tau.
    """
    gt_segs = _segments(gt)
    pred_segs = _segments(pred)
    if not gt_segs or not pred_segs:
        return 0.0
    matched = set()
    tp = 0
    for pl, ps, pe in pred_segs:
        best, best_iou = None, 0.0
        for k, (gl, gs, ge) in enumerate(gt_segs):
            if k in matched or gl != pl:
                continue
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = (pe - ps) + (ge - gs) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best, best_iou = k, iou
        if best is not None and best_iou >= tau:
            matched.add(best)
            tp += 1
    if tp == 0:
        return 0.0
    precision = tp / len(pred_segs)
    recall = tp / len(gt_segs)
    return 100.0 * 2 * precision * recall / (precision + recall)


def score_video(gt, pred) -> dict:
    scores = {"mof": frame_accuracy(gt, pred), "edit": edit_score(gt, pred)}
    for tau in IOU_THRESHOLDS:
        scores[f"f1@{round(tau * 100)}"] = f1_at(gt, pred, tau)
    return scores


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def latency_stats(ms, deadline_ms: float) -> dict:
    ms = np.asarray(ms, dtype=float)
    return {
        "n": int(ms.size),
        "mean_ms": float(ms.mean()),
        "std_ms": float(ms.std()),
        "p95_ms": float(np.percentile(ms, 95)),
        "deadline_ms": float(deadline_ms),
        "deadline_miss_ratio": float(np.mean(ms > deadline_ms)),
    }


# ---------------------------------------------------------------------------
# Loading and evaluation
# ---------------------------------------------------------------------------

def find_prediction(pred_dir: Path, stem: str) -> Path:
    for ext in (".npy", ".txt"):
        candidate = pred_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no prediction for '{stem}' in {pred_dir}")


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 2:  # (T, C) scores
            arr = arr.argmax(axis=1)
        return arr.astype(int).ravel()
    return np.loadtxt(path, dtype=int).ravel()


def evaluate_fps(gt_dir: Path, pred_dir: Path, fps: int) -> dict:
    stride = stride_for_fps(fps)
    gt_files = sorted(gt_dir.glob("*.txt"))
    if not gt_files:
        raise FileNotFoundError(f"no ground-truth .txt files in {gt_dir}")

    rows = []
    for gt_path in gt_files:
        gt = np.loadtxt(gt_path, dtype=int).ravel()[::stride]
        pred = load_prediction(find_prediction(pred_dir, gt_path.stem))
        if len(pred) != len(gt):
            raise ValueError(
                f"{gt_path.stem}: {len(pred)} predicted frames but {len(gt)} "
                f"ground-truth frames at {fps} FPS (stride {stride})"
            )
        rows.append({"video": gt_path.stem, **score_video(gt, pred)})

    summary = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
    return {"num_videos": len(rows), "stride": stride, "summary": summary, "per_video": rows}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(results: dict, latency_samples: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps_list = sorted(int(k) for k in results)
    keys = [str(f) for f in fps_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for metric, label in [("mof", "MoF"), ("edit", "Edit"), ("f1@10", "F1@10")]:
        ax1.plot(fps_list, [results[k]["summary"][metric] for k in keys], marker="o", label=label)
    ax1.set_xlabel("Frame rate (FPS)")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Accuracy vs frame rate")
    ax1.set_xticks(fps_list)
    ax1.grid(alpha=0.3)
    ax1.legend()

    xs = [f for f in fps_list if str(f) in latency_samples]
    if xs:
        positions = list(range(1, len(xs) + 1))
        ax2.boxplot([latency_samples[str(f)] for f in xs], positions=positions)
        ax2.set_xticks(positions)
        ax2.set_xticklabels([f"{f} FPS" for f in xs])
        deadlines = [results[str(f)]["latency"]["deadline_ms"] for f in xs]
        ax2.plot(positions, deadlines, "r--", marker="_", label="deadline")
        ax2.set_ylabel("Latency (ms)")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No latency data supplied", ha="center", va="center",
                 transform=ax2.transAxes)
    ax2.set_title("Latency distribution vs deadline")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Frame-rate degradation benchmark (issue #12).")
    p.add_argument("--gt-dir", type=Path, default=Path("data/fifty_salads/groundTruth"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions"))
    p.add_argument("--fps", type=int, nargs="+", default=DEFAULT_FPS)
    p.add_argument("--latency-json", type=Path, default=None)
    p.add_argument("--deadlines-json", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("eval/results/fps_benchmark.json"))
    p.add_argument("--plot", type=Path, default=None, help="e.g. assets/fps_tradeoff_plots.png")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        _self_test()
        return

    latency = json.loads(args.latency_json.read_text()) if args.latency_json else {}
    deadlines = json.loads(args.deadlines_json.read_text()) if args.deadlines_json else {}

    results = {}
    for fps in args.fps:
        key = str(fps)
        metrics = evaluate_fps(args.gt_dir, args.pred_dir / f"fps{fps}", fps)
        if key in latency:
            deadline = float(deadlines.get(key, 1000.0 / fps))
            metrics["latency"] = latency_stats(latency[key], deadline)
        results[key] = metrics
        s = metrics["summary"]
        print(f"{fps:>3} FPS | MoF {s['mof']:6.2f} | Edit {s['edit']:6.2f} | "
              f"F1@10 {s['f1@10']:6.2f} | videos {metrics['num_videos']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.output}")

    if args.plot:
        latency_samples = {k: [float(x) for x in v] for k, v in latency.items()}
        plot(results, latency_samples, args.plot)
        print(f"Wrote {args.plot}")


# ---------------------------------------------------------------------------
# Self-test (no dataset or trained model needed)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    gt = np.array([0] * 10 + [1] * 10 + [2] * 5)

    assert stride_for_fps(30) == 1
    assert stride_for_fps(15) == 2
    assert stride_for_fps(6) == 5
    assert stride_for_fps(3) == 10

    # Perfect predictions score 100 on every metric.
    s = score_video(gt, gt)
    for k in METRIC_KEYS:
        assert abs(s[k] - 100.0) < 1e-9, (k, s[k])

    # A wrong block lowers frame accuracy.
    wrong = gt.copy()
    wrong[:3] = 1
    assert 0 < frame_accuracy(gt, wrong) < 100

    # Flicker (over-segmentation) lowers the Edit score.
    flicker = gt.copy()
    flicker[5], flicker[6] = 1, 0
    assert edit_score(gt, flicker) < 100

    # Latency: 1 of 3 samples exceeds 33.3 ms.
    stats = latency_stats([10.0, 20.0, 80.0], deadline_ms=33.3)
    assert abs(stats["deadline_miss_ratio"] - 1 / 3) < 1e-9

    print("benchmark_fps self-test passed.")


if __name__ == "__main__":
    main()
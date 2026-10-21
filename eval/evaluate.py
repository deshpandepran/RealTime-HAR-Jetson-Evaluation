#!/usr/bin/env python3
"""Benchmark entry point for 50 Salads (Issue #11).

Evaluates saved per-frame predictions across the 5 folds, computing Frame
Accuracy (MoF), Edit Score, and F1@{10,25,50}, plus optional latency
statistics.  Ground truth is read from the 50 Salads ``groundTruth`` folder.

Prediction folder layout (one file per video, stem = video name)::

    predictions/fold1/rgb-seq-01.npy
    predictions/fold1/rgb-seq-02.npy
    ...

Each ``.npy`` file may contain:

* ``(T,)`` integer frame labels,
* ``(T, C)`` logits or probabilities (argmax is applied),

``.txt`` / ``.csv`` files with one label per line or one score row per frame,
and ``.npz`` files with keys ``labels``, ``predictions``, ``probs`` or
``logits`` are also supported.

Usage
-----
    # Evaluate all folds from one prediction folder
    python eval/evaluate.py --pred-dir predictions/ --gt-dir data/fifty_salads/groundTruth

    # One fold, with hybrid post-processing on (T, C) probabilities
    python eval/evaluate.py --pred-dir predictions/ --fold 1 --post-process --fps 30

    # Add latency statistics from a JSON list of per-frame milliseconds
    python eval/evaluate.py --pred-dir predictions/ --latency-json latency.json
"""

import argparse
import json
import os
import re
import sys

import numpy as np

# Allow both `python eval/evaluate.py` and `python -m eval.evaluate`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval.metrics import (  # noqa: E402
    aggregate_folds,
    aggregate_metrics,
    evaluate_sequence,
    latency_stats,
)


VIDEOS_PER_FOLD = 10
TOTAL_VIDEOS = 50


def fold_videos(fold):
    """Deterministic 5-fold split matching ``data/fifty_salads.py``."""
    all_videos = ["rgb-seq-{:02d}".format(i) for i in range(1, TOTAL_VIDEOS + 1)]
    start = (int(fold) - 1) * VIDEOS_PER_FOLD
    return all_videos[start:start + VIDEOS_PER_FOLD]


def _read_numeric_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = [token for token in re.split(r"[,\s]+", line) if token]
            try:
                rows.append([int(round(float(token))) for token in tokens])
            except ValueError:
                continue
    return rows


def load_ground_truth(path, fmt="auto"):
    """Load a 50 Salads ground-truth file into a per-frame label array.

    Supported layouts

    * ``auto``      - 3+ columns -> ``start end label`` segments; 1 column ->
      one label per frame; 2 columns -> ``frame label`` pairs.
    * ``segments``  - explicit ``start end label`` rows.
    * ``frames``    - one label per frame (or ``frame label`` pairs).
    """
    rows = _read_numeric_rows(path)
    if not rows:
        return np.empty((0,), dtype=np.int64)

    widths = {len(row) for row in rows}
    if fmt == "auto":
        fmt = "segments" if min(widths) >= 3 else "frames"

    if fmt == "segments":
        segments = []
        for row in rows:
            if len(row) < 3:
                continue
            start, end, label = row[0], row[1], row[2]
            if end <= start:
                continue
            segments.append((start, end, label))
        if not segments:
            return np.empty((0,), dtype=np.int64)
        total = max(end for _, end, _ in segments)
        labels = np.zeros((total,), dtype=np.int64)
        for start, end, label in segments:
            labels[start:end] = label
        return labels

    # frames
    if max(widths) == 1:
        return np.asarray([row[0] for row in rows], dtype=np.int64)

    pairs = [(row[0], row[-1]) for row in rows if len(row) >= 2]
    if not pairs:
        return np.empty((0,), dtype=np.int64)
    total = max(frame for frame, _ in pairs) + 1
    labels = np.zeros((total,), dtype=np.int64)
    for frame, label in pairs:
        if 0 <= frame < total:
            labels[frame] = label
    # Forward-fill unannotated frames.
    for i in range(1, labels.size):
        if labels[i] == 0 and labels[i - 1] != 0:
            labels[i] = labels[i - 1]
    return labels


def _postprocess(scores, fps):
    from models.post_processing import HybridPostProcessor

    processor = HybridPostProcessor(fps=fps)
    return processor(scores)


def load_prediction(path, post_process=False, fps=30):
    """Load a prediction file and return a 1-D array of frame labels."""
    extension = os.path.splitext(path)[1].lower()

    if extension == ".npz":
        with np.load(path) as data:
            array = None
            for key in ("labels", "predictions", "probs", "logits"):
                if key in data:
                    array = data[key]
                    break
            if array is None:
                raise KeyError("No labels/probs/logits array in {}".format(path))
    elif extension == ".npy":
        array = np.load(path)
    else:
        rows = _read_numeric_rows(path)
        if not rows:
            return np.empty((0,), dtype=np.int64)
        widths = {len(row) for row in rows}
        array = np.asarray(rows, dtype=np.int64) if len(widths) == 1 else None
        if array is None:
            raise ValueError("Inconsistent row widths in {}".format(path))

    array = np.asarray(array)
    if array.ndim == 1:
        return array.astype(np.int64)

    if array.ndim != 2:
        raise ValueError("Prediction array must be 1-D or 2-D, got {}".format(array.shape))

    if post_process:
        return _postprocess(array, fps)
    return np.argmax(array, axis=1).astype(np.int64)


def find_prediction(pred_dir, video):
    for extension in (".npy", ".npz", ".txt", ".csv"):
        path = os.path.join(pred_dir, video + extension)
        if os.path.exists(path):
            return path
    return None


def load_latencies(path):
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("latencies_ms", payload.get("latencies", []))
    return [float(value) for value in payload]


def evaluate_fold(pred_dir, gt_dir, fold, fps=30, gt_format="auto", post_process=False):
    per_video = {}
    missing_predictions = []

    for video in fold_videos(fold):
        gt_path = os.path.join(gt_dir, video + ".txt")
        if not os.path.exists(gt_path):
            continue

        pred_path = find_prediction(pred_dir, video)
        if pred_path is None:
            missing_predictions.append(video)
            continue

        ground_truth = load_ground_truth(gt_path, fmt=gt_format)
        prediction = load_prediction(pred_path, post_process=post_process, fps=fps)

        if ground_truth.size == 0 or prediction.size == 0:
            continue

        length = min(ground_truth.size, prediction.size)
        per_video[video] = evaluate_sequence(
            ground_truth[:length], prediction[:length]
        )

    summary = aggregate_metrics(list(per_video.values()))
    summary["fold"] = int(fold)
    summary["num_videos"] = len(per_video)
    summary["missing_predictions"] = missing_predictions
    summary["per_video"] = per_video
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="RT-HARE evaluation on 50 Salads (Issue #11)"
    )
    parser.add_argument(
        "--pred-dir", required=True,
        help="Directory containing per-video prediction files.",
    )
    parser.add_argument(
        "--gt-dir", default=os.path.join(_REPO_ROOT, "data", "fifty_salads", "groundTruth"),
        help="Directory containing 50 Salads ground-truth .txt files.",
    )
    parser.add_argument(
        "--fold", default="all",
        help="Fold to evaluate: 'all' (default) or 1-5.",
    )
    parser.add_argument(
        "--gt-format", default="auto", choices=["auto", "segments", "frames"],
        help="Ground-truth file layout (auto detects 3-column segment files).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate.")
    parser.add_argument(
        "--post-process", action="store_true",
        help="Apply HybridPostProcessor to (T, C) probability/logit files.",
    )
    parser.add_argument(
        "--deadline-ms", type=float, default=68.83,
        help="Per-frame latency deadline used for the miss ratio.",
    )
    parser.add_argument(
        "--latency-json", default=None,
        help="JSON list (or {'latencies_ms': [...]}) of measured per-frame latencies.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write the full JSON results.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.fold == "all":
        folds = list(range(1, 6))
    else:
        folds = [int(args.fold)]
        if folds[0] < 1 or folds[0] > 5:
            raise SystemExit("--fold must be 'all' or an integer in 1..5")

    fold_results = []
    for fold in folds:
        result = evaluate_fold(
            args.pred_dir, args.gt_dir, fold,
            fps=args.fps, gt_format=args.gt_format,
            post_process=args.post_process,
        )
        fold_results.append(result)
        print(
            "Fold {} | videos {:2d} | MoF {:6.2f}% | Edit {:6.2f}% | "
            "F1@10 {:6.2f}% | F1@25 {:6.2f}% | F1@50 {:6.2f}%".format(
                fold,
                result["num_videos"],
                result["frame_accuracy"]["mean"],
                result["edit_score"]["mean"],
                result["f1@10"]["mean"],
                result["f1@25"]["mean"],
                result["f1@50"]["mean"],
            )
        )
        if result["missing_predictions"]:
            print("           missing predictions: {}".format(
                ", ".join(result["missing_predictions"])
            ))

    # Aggregate fold means (standard 5-fold protocol: mean of fold scores).
    fold_summary = aggregate_folds(fold_results)

    latencies = load_latencies(args.latency_json)
    latency = latency_stats(latencies, deadline_ms=args.deadline_ms) if latencies else None

    print("-" * 72)
    print("5-fold summary (mean +/- std across folds)")
    for key in ("frame_accuracy", "edit_score", "f1@10", "f1@25", "f1@50"):
        print("  {:14s} {:6.2f} +/- {:.2f}".format(
            key, fold_summary[key]["mean"], fold_summary[key]["std"]
        ))
    if latency:
        print("  latency        {:.2f} ms (p95 {:.2f} ms, deadline miss {:.2f}%)".format(
            latency["mean_ms"], latency["p95_ms"], latency["deadline_miss_ratio"]
        ))

    payload = {
        "config": {
            "pred_dir": os.path.abspath(args.pred_dir),
            "gt_dir": os.path.abspath(args.gt_dir),
            "fps": args.fps,
            "post_process": args.post_process,
        },
        "folds": {str(result["fold"]): result for result in fold_results},
        "summary": fold_summary,
        "latency": latency,
    }

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("Wrote {}".format(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())

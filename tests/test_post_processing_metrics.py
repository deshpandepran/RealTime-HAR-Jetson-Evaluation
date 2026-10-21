# -*- coding: utf-8 -*-
"""Unit tests for Issue #9 (post-processing) and Issue #11 (metrics).

Runs with pytest, or standalone with no test runner installed:

    python tests/test_post_processing_metrics.py

Only NumPy is required; no model checkpoint, GPU, or dataset is needed.
"""

import importlib.util
import os
import sys
import tempfile

import numpy as np

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO_ROOT)


def _load_module(name, relative_path):
    """Load a module by file path (avoids package __init__ importing torch)."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO_ROOT, relative_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_post_processing = _load_module("post_processing", "models/post_processing.py")

from eval.metrics import (  # noqa: E402
    ActionSegment,
    edit_score,
    evaluate_sequence,
    f1_at_threshold,
    frame_accuracy,
    latency_stats,
    segments_from_frame_labels,
    temporal_iou,
)
from eval.evaluate import load_ground_truth, main as evaluate_main  # noqa: E402

HybridPostProcessor = _post_processing.HybridPostProcessor
boundary_scores = _post_processing.boundary_scores
calibrate_min_segment_frames = _post_processing.calibrate_min_segment_frames
threshold_filtering = _post_processing.threshold_filtering


def _close(actual, expected, tolerance=1e-6):
    return abs(float(actual) - float(expected)) <= tolerance


# ---------------------------------------------------------------------------
# Segment helpers / metrics
# ---------------------------------------------------------------------------

def test_segments_from_frame_labels():
    labels = [1, 1, 1, 2, 2, 3, 3, 3, 3]
    segments = segments_from_frame_labels(labels)
    assert segments == [
        ActionSegment(0, 3, 1),
        ActionSegment(3, 5, 2),
        ActionSegment(5, 9, 3),
    ]
    assert segments[1].duration == 2
    assert temporal_iou(segments[0], ActionSegment(0, 3, 1)) == 1.0
    assert _close(temporal_iou(ActionSegment(0, 10, 1),
                               ActionSegment(5, 15, 1)), 5.0 / 15.0)


def test_frame_accuracy():
    assert frame_accuracy([1, 1, 2, 2], [1, 1, 2, 2]) == 100.0
    assert frame_accuracy([1, 1, 2, 2], [1, 2, 2, 3]) == 50.0
    assert frame_accuracy([1, 2, 3], [1, 9, 3], ignore_label=2) == 100.0


def test_edit_score():
    # GT transcript [1, 2, 3] vs predicted [1, 3]: one deletion -> 2/3.
    gt = [1, 1, 2, 2, 3, 3]
    pred = [1, 1, 1, 1, 3, 3]
    assert _close(edit_score(gt, pred), 100.0 * (1.0 - 1.0 / 3.0))


def test_f1_at_threshold():
    gt = [1] * 10 + [2] * 10
    perfect = [1] * 10 + [2] * 10
    assert f1_at_threshold(gt, perfect, 0.1) == 100.0

    # One extra segment: F1 = 2*2 / (2*2 + 1 + 0) = 80%.
    fragmented = [1] * 10 + [2] * 5 + [3] * 5
    assert _close(f1_at_threshold(gt, fragmented, 0.1), 80.0)
    assert _close(f1_at_threshold(gt, fragmented, 0.5), 80.0)
    # At IoU 0.75 only the first segment matches -> 2*1 / (2 + 2 + 1) = 40%.
    assert _close(f1_at_threshold(gt, fragmented, 0.75), 40.0)


def test_evaluate_sequence_keys():
    result = evaluate_sequence([1] * 5, [1] * 5)
    for key in ("frame_accuracy", "edit_score", "f1@10", "f1@25", "f1@50"):
        assert key in result
    assert result["frame_accuracy"] == 100.0


def test_latency_stats():
    stats = latency_stats([10, 20, 30, 100], deadline_ms=50)
    assert stats["count"] == 4
    assert _close(stats["mean_ms"], 40.0)
    assert _close(stats["deadline_miss_ratio"], 25.0)
    assert _close(stats["p50_ms"], 25.0)


# ---------------------------------------------------------------------------
# Hybrid post-processing
# ---------------------------------------------------------------------------

def _make_probabilities(true_labels, num_classes=6, confidence=0.9, seed=0):
    probs = np.full((len(true_labels), num_classes), (1.0 - confidence) / (num_classes - 1))
    for t, label in enumerate(true_labels):
        probs[t, label] = confidence
    return probs


def test_boundary_scores_on_clean_sequence():
    probs = _make_probabilities([1] * 10 + [2] * 10, confidence=1.0)
    scores = boundary_scores(probs, smooth_kernel=1)
    assert int(np.argmax(scores)) == 10
    assert scores[10] > 0.9
    assert scores[5] < 0.1


def test_post_processing_removes_flicker_and_short_segment():
    true_labels = np.array([1] * 30 + [2] * 30 + [3] * 30, dtype=np.int64)
    probs = _make_probabilities(true_labels, confidence=0.9, seed=7)

    # Spurious 2-frame action in the middle of class 2.
    probs[44:46, :] = 0.01
    probs[44:46, 5] = 0.95

    processor = HybridPostProcessor(fps=30)
    labels = processor(probs)
    assert labels.shape == true_labels.shape
    assert frame_accuracy(true_labels, labels) >= 95.0

    for start, end, _label in segments_from_frame_labels(labels):
        assert (end - start) >= 1
    # Every surviving non-boundary segment must satisfy the calibrated minimum.
    kept = segments_from_frame_labels(labels)
    assert all(segment.duration >= processor.min_segment_frames
               for segment in kept[1:-1]) if len(kept) > 2 else True


def test_threshold_filtering_merges_short_segment():
    labels = np.array([1] * 20 + [2] * 3 + [1] * 20, dtype=np.int64)
    filtered = threshold_filtering(labels, min_duration=10)
    assert np.array_equal(filtered, np.array([1] * 43))


def test_fps_calibration():
    assert calibrate_min_segment_frames(30, 0.33) == 10
    assert calibrate_min_segment_frames(15, 0.33) == 5
    assert calibrate_min_segment_frames(6, 0.33) == 2
    assert calibrate_min_segment_frames(3, 0.33) == 1

    processor = HybridPostProcessor(fps=3)
    assert processor.min_segment_frames == 1
    processor.calibrate(30)
    assert processor.min_segment_frames == 10


def test_processor_accepts_per_frame_labels():
    processor = HybridPostProcessor(fps=30)
    labels = [1] * 20 + [2] * 2 + [1] * 20
    result = processor(labels=labels)
    assert np.array_equal(result, np.array([1] * 42))


# ---------------------------------------------------------------------------
# GT parsing + end-to-end evaluate.py
# ---------------------------------------------------------------------------

def test_load_ground_truth_formats():
    with tempfile.TemporaryDirectory() as tmp:
        segment_path = os.path.join(tmp, "seg.txt")
        with open(segment_path, "w", encoding="utf-8") as handle:
            handle.write("0 10 1\n10 20 2\n")
        labels = load_ground_truth(segment_path)
        assert labels.size == 20
        assert np.all(labels[:10] == 1)
        assert np.all(labels[10:] == 2)

        frame_path = os.path.join(tmp, "frames.txt")
        with open(frame_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(["1", "1", "2", "2"]))
        labels = load_ground_truth(frame_path)
        assert np.array_equal(labels, np.array([1, 1, 2, 2]))


def test_evaluate_cli_perfect_predictions():
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = os.path.join(tmp, "groundTruth")
        pred_dir = os.path.join(tmp, "predictions")
        os.makedirs(gt_dir)
        os.makedirs(pred_dir)

        for index in range(1, 11):
            video = "rgb-seq-{:02d}".format(index)
            with open(os.path.join(gt_dir, video + ".txt"), "w", encoding="utf-8") as handle:
                handle.write("0 10 1\n10 20 2\n20 30 3\n")
            np.save(os.path.join(pred_dir, video + ".npy"),
                    np.array([1] * 10 + [2] * 10 + [3] * 10, dtype=np.int64))

        output = os.path.join(tmp, "results.json")
        exit_code = evaluate_main([
            "--pred-dir", pred_dir,
            "--gt-dir", gt_dir,
            "--fold", "1",
            "--output", output,
        ])
        assert exit_code == 0
        assert os.path.exists(output)


def main():
    tests = [
        (name, value) for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = []
    for name, test in tests:
        try:
            test()
            print("  [PASS]  {}".format(name))
        except Exception as error:  # noqa: BLE001
            failures.append((name, error))
            print("  [FAIL]  {}: {}".format(name, error))

    print("\n{} passed, {} failed".format(len(tests) - len(failures), len(failures)))
    if failures:
        for name, error in failures:
            print("  - {}: {}".format(name, error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

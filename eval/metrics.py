"""Evaluation metrics for RT-HARE on 50 Salads.

Implements Issue #11:
    * Frame Accuracy / Mean over Frames (MoF)
    * Edit Score (normalised Levenshtein distance on segment sequences)
    * F1@{10, 25, 50} (temporal IoU matching of action segments)
    * Latency statistics (mean, std, percentiles, deadline miss ratio)

All accuracy metrics are reported in percent (0-100), matching the paper.
The module depends only on NumPy, so it can be imported and unit-tested
without a trained model, GPU, or the 50 Salads videos.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Segment utilities
# ---------------------------------------------------------------------------

class ActionSegment(tuple):
    """A half-open temporal segment ``[start, end)`` with an action label."""

    __slots__ = ()

    def __new__(cls, start, end, label):
        return tuple.__new__(cls, (int(start), int(end), int(label)))

    @property
    def start(self):
        return self[0]

    @property
    def end(self):
        return self[1]

    @property
    def label(self):
        return self[2]

    @property
    def duration(self):
        return self.end - self.start

    def __repr__(self):
        return "ActionSegment(start={}, end={}, label={})".format(
            self.start, self.end, self.label
        )


def _as_label_array(labels):
    """Convert a list / tensor / ndarray of frame labels to a 1-D int array."""
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(
            "Expected a 1-D sequence of frame labels, got shape {}".format(arr.shape)
        )
    return arr.astype(np.int64)


def segments_from_frame_labels(labels, background_label=0, ignore_background=True):
    """Group consecutive identical frame labels into :class:`ActionSegment` objects.

    Parameters
    ----------
    labels : array-like, shape (T,)
        Per-frame class indices.
    background_label : int
        Label treated as background when ``ignore_background`` is True.
    ignore_background : bool
        Drop background segments from the returned list (used by Edit / F1).

    Returns
    -------
    list[ActionSegment]
    """
    arr = _as_label_array(labels)
    if arr.size == 0:
        return []

    segments = []
    start = 0
    for t in range(1, arr.size):
        if arr[t] != arr[t - 1]:
            segments.append(ActionSegment(start, t, arr[t - 1]))
            start = t
    segments.append(ActionSegment(start, arr.size, arr[-1]))

    if ignore_background:
        segments = [s for s in segments if s.label != background_label]
    return segments


def temporal_iou(seg_a, seg_b):
    """Intersection-over-union of two temporal segments (0.0 - 1.0)."""
    inter = max(0, min(seg_a.end, seg_b.end) - max(seg_a.start, seg_b.start))
    if inter == 0:
        return 0.0
    union = seg_a.duration + seg_b.duration - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _levenshtein_distance(seq_a, seq_b):
    """Classic dynamic-programming Levenshtein distance between two sequences."""
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    previous = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        current = [i]
        for j, token_b in enumerate(seq_b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if token_a == token_b else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def frame_accuracy(y_true, y_pred, ignore_label=None):
    """Frame-wise accuracy (MoF) in percent.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (T,)
        Ground-truth and predicted per-frame labels.
    ignore_label : int or None
        Optional label excluded from the comparison (e.g. -1 for padding).
    """
    y_true = _as_label_array(y_true)
    y_pred = _as_label_array(y_pred)

    n = min(y_true.size, y_pred.size)
    if n == 0:
        return float("nan")

    y_true, y_pred = y_true[:n], y_pred[:n]
    if ignore_label is not None:
        valid = y_true != ignore_label
        if not np.any(valid):
            return float("nan")
        y_true, y_pred = y_true[valid], y_pred[valid]

    return 100.0 * float(np.mean(y_true == y_pred))


def edit_score(y_true, y_pred, background_label=0, ignore_background=True):
    """Edit score in percent: ``100 * (1 - ED / max(len(gt), len(pred)))``.

    The distance is computed on the ordered sequences of *segment labels*,
    not on individual frames, following the 50 Salads / action-segmentation
    protocol (Levenshtein distance over predicted and ground-truth transcripts).
    """
    gt_segments = segments_from_frame_labels(
        y_true, background_label=background_label,
        ignore_background=ignore_background,
    )
    pred_segments = segments_from_frame_labels(
        y_pred, background_label=background_label,
        ignore_background=ignore_background,
    )

    gt_seq = [s.label for s in gt_segments]
    pred_seq = [s.label for s in pred_segments]

    max_len = max(len(gt_seq), len(pred_seq))
    if max_len == 0:
        return 100.0

    distance = _levenshtein_distance(gt_seq, pred_seq)
    return 100.0 * (1.0 - float(distance) / float(max_len))


def f1_at_threshold(
    y_true,
    y_pred,
    iou_threshold,
    background_label=0,
    ignore_background=True,
):
    """Segment F1 score at a temporal-IoU threshold, in percent.

    A predicted segment is a true positive when it has the same label and an
    IoU ``>= iou_threshold`` with an unmatched ground-truth segment.  Matches
    are chosen greedily by descending IoU, which mirrors the standard
    action-segmentation evaluation (F1@10 uses ``iou_threshold=0.1``).
    """
    gt_segments = segments_from_frame_labels(
        y_true, background_label=background_label,
        ignore_background=ignore_background,
    )
    pred_segments = segments_from_frame_labels(
        y_pred, background_label=background_label,
        ignore_background=ignore_background,
    )

    if not gt_segments and not pred_segments:
        return 100.0

    candidates = []
    for gt_idx, gt_seg in enumerate(gt_segments):
        for pred_idx, pred_seg in enumerate(pred_segments):
            if gt_seg.label != pred_seg.label:
                continue
            iou = temporal_iou(gt_seg, pred_seg)
            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    true_positives = 0
    for _iou, gt_idx, pred_idx in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        true_positives += 1

    false_positives = len(pred_segments) - true_positives
    false_negatives = len(gt_segments) - true_positives

    denominator = 2 * true_positives + false_positives + false_negatives
    if denominator == 0:
        return 100.0
    return 100.0 * (2.0 * true_positives / float(denominator))


def f1_at_10(y_true, y_pred, **kwargs):
    """Convenience wrapper for F1@10 (IoU threshold 0.1)."""
    return f1_at_threshold(y_true, y_pred, 0.1, **kwargs)


def f1_at_25(y_true, y_pred, **kwargs):
    """Convenience wrapper for F1@25 (IoU threshold 0.25)."""
    return f1_at_threshold(y_true, y_pred, 0.25, **kwargs)


def f1_at_50(y_true, y_pred, **kwargs):
    """Convenience wrapper for F1@50 (IoU threshold 0.5)."""
    return f1_at_threshold(y_true, y_pred, 0.5, **kwargs)


def evaluate_sequence(
    y_true,
    y_pred,
    iou_thresholds=(0.1, 0.25, 0.5),
    background_label=0,
    ignore_background=True,
    ignore_label=None,
):
    """Compute all per-video metrics for one prediction sequence."""
    result = {
        "num_frames": int(min(len(y_true), len(y_pred))),
        "frame_accuracy": frame_accuracy(y_true, y_pred, ignore_label=ignore_label),
        "edit_score": edit_score(
            y_true, y_pred, background_label=background_label,
            ignore_background=ignore_background,
        ),
        "num_gt_segments": len(segments_from_frame_labels(
            y_true, background_label=background_label,
            ignore_background=ignore_background,
        )),
        "num_pred_segments": len(segments_from_frame_labels(
            y_pred, background_label=background_label,
            ignore_background=ignore_background,
        )),
    }
    for threshold in iou_thresholds:
        key = "f1@{:g}".format(100 * threshold)
        result[key] = f1_at_threshold(
            y_true, y_pred, threshold,
            background_label=background_label,
            ignore_background=ignore_background,
        )
    return result


# ---------------------------------------------------------------------------
# Aggregation across videos / folds
# ---------------------------------------------------------------------------

METRIC_KEYS = ("frame_accuracy", "edit_score", "f1@10", "f1@25", "f1@50")


def aggregate_metrics(per_item, keys=METRIC_KEYS):
    """Mean (and sample std) of metrics over a list of result dictionaries.

    ``per_item`` may be a list of per-video dicts or per-fold dicts.  NaN
    entries are ignored.
    """
    summary = {}
    for key in keys:
        values = []
        for item in per_item:
            value = item.get(key)
            if value is None:
                continue
            value = float(value)
            if not math.isnan(value):
                values.append(value)
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        else:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
    return summary


def evaluate_dataset(samples, **kwargs):
    """Evaluate a collection of ``(name, y_true, y_pred)`` triplets.

    Returns a dictionary with the aggregate summary and per-video results.
    """
    per_video = {}
    for name, y_true, y_pred in samples:
        per_video[name] = evaluate_sequence(y_true, y_pred, **kwargs)

    summary = aggregate_metrics(list(per_video.values()))
    summary["num_videos"] = len(per_video)
    summary["per_video"] = per_video
    return summary


def aggregate_folds(fold_summaries, keys=METRIC_KEYS):
    """Aggregate per-fold metrics into mean +/- std across folds.

    ``fold_summaries`` is an iterable of dictionaries returned by
    :func:`evaluate_dataset` (one per fold).  Metrics may be stored either as
    plain floats or as ``{"mean": ..., "std": ...}`` dictionaries.
    """
    fold_summaries = list(fold_summaries)
    summary = {}
    for key in keys:
        values = []
        for fold in fold_summaries:
            entry = fold.get(key)
            if isinstance(entry, dict):
                entry = entry.get("mean")
            if entry is None:
                continue
            entry = float(entry)
            if not math.isnan(entry):
                values.append(entry)
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        else:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
    summary["num_folds"] = len(fold_summaries)
    return summary


# ---------------------------------------------------------------------------
# Latency metrics
# ---------------------------------------------------------------------------

def latency_stats(latencies_ms, deadline_ms=68.83):
    """Latency summary in milliseconds, including deadline miss ratio (%)."""
    values = np.asarray(latencies_ms, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "count": 0,
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
            "deadline_ms": float(deadline_ms),
            "deadline_miss_ratio": float("nan"),
            "throughput_fps": float("nan"),
        }

    mean_ms = float(np.mean(values))
    return {
        "count": int(values.size),
        "mean_ms": mean_ms,
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "deadline_ms": float(deadline_ms),
        "deadline_miss_ratio": 100.0 * float(np.mean(values > deadline_ms)),
        "throughput_fps": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
    }

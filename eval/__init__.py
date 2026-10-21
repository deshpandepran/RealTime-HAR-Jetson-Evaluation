"""Evaluation utilities for RT-HARE (training-free metrics and benchmarking)."""

from .metrics import (
    ActionSegment,
    aggregate_folds,
    aggregate_metrics,
    edit_score,
    evaluate_dataset,
    evaluate_sequence,
    f1_at_10,
    f1_at_25,
    f1_at_50,
    f1_at_threshold,
    frame_accuracy,
    latency_stats,
    segments_from_frame_labels,
    temporal_iou,
)

__all__ = [
    "ActionSegment",
    "aggregate_folds",
    "aggregate_metrics",
    "edit_score",
    "evaluate_dataset",
    "evaluate_sequence",
    "f1_at_10",
    "f1_at_25",
    "f1_at_50",
    "f1_at_threshold",
    "frame_accuracy",
    "latency_stats",
    "segments_from_frame_labels",
    "temporal_iou",
]

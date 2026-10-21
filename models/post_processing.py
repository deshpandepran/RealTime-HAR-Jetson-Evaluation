"""Hybrid temporal post-processing for online action recognition.

Implements Issue #9 by combining the two filtering strategies described in
RT-HARE:

1. **Boundary regression / boundary barriers** - per-frame class probabilities
   are differentiated over time to obtain a boundary score.  Peaks above a
   calibrated threshold are refined to their local maximum and used as hard
   barriers; each barrier-delimited region is then collapsed to a single
   action label (probability-weighted majority).  This removes the
   frame-level flicker that causes over-segmentation.
2. **Threshold filtering** - after boundary barriers, any segment shorter than
   a calibrated minimum duration is merged into its strongest neighbour.  The
   minimum duration is expressed in seconds and converted to frames for the
   target frame rate (30 / 15 / 6 / 3 FPS), which keeps the behaviour stable
   when the pipeline is down-sampled on the Jetson.

The module only depends on NumPy and works on raw logits or probabilities,
so it can run directly on TensorRT / ONNX outputs.

Example
-------
    from models.post_processing import HybridPostProcessor

    processor = HybridPostProcessor(fps=30)
    labels = processor(logits)          # logits: (T, num_classes)

    # or with already-computed per-frame labels:
    labels = processor(labels=raw_labels)
"""

import numpy as np


# Minimum segment length (in seconds) used when the caller does not override
# it.  0.33 s is roughly 10 frames at 30 FPS and keeps single-frame noise from
# surviving while still preserving short real actions.
DEFAULT_MIN_SEGMENT_SECONDS = 0.33

# Frame-rate calibration table.  Lower frame rates produce fewer samples per
# action, so the minimum length in frames shrinks and the boundary threshold
# is raised slightly to avoid splitting on a single noisy transition.
FPS_CALIBRATION = {
    30: {"min_segment_frames": 10, "boundary_threshold": 0.30},
    15: {"min_segment_frames": 5, "boundary_threshold": 0.30},
    6: {"min_segment_frames": 2, "boundary_threshold": 0.35},
    3: {"min_segment_frames": 1, "boundary_threshold": 0.40},
}


def calibrate_min_segment_frames(fps, min_segment_seconds=DEFAULT_MIN_SEGMENT_SECONDS):
    """Convert a duration in seconds to a frame count for ``fps``."""
    return max(1, int(round(float(fps) * float(min_segment_seconds))))


def _as_2d_float(array):
    """Return a float64 ``(T, C)`` array from logits/probabilities."""
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(
            "Expected per-frame scores with shape (T, C), got {}".format(arr.shape)
        )
    return arr


def _softmax(scores):
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _is_probability_matrix(scores):
    """Heuristic check: rows are non-negative and (approximately) sum to 1."""
    if np.any(scores < 0):
        return False
    row_sums = np.sum(scores, axis=1)
    return bool(np.allclose(row_sums, 1.0, atol=1e-3))


def to_probabilities(scores):
    """Convert logits to probabilities, leaving already-normalised input intact."""
    arr = _as_2d_float(scores)
    if _is_probability_matrix(arr):
        return arr
    return _softmax(arr)


def _moving_average(array, kernel_size):
    """Centered (edge-padded) moving average over the first axis."""
    if kernel_size is None or int(kernel_size) <= 1:
        return array.copy()
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2

    if array.ndim == 1:
        padded = np.pad(array, radius, mode="edge")
    else:
        pad_width = [(radius, radius)] + [(0, 0)] * (array.ndim - 1)
        padded = np.pad(array, pad_width, mode="edge")

    if array.ndim == 1:
        kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
        return np.convolve(padded, kernel, mode="valid")

    kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
    return np.apply_along_axis(
        lambda column: np.convolve(column, kernel, mode="valid"), 0, padded
    )


def boundary_scores(probabilities, smooth_kernel=1):
    """Per-frame boundary score in ``[0, 1]``.

    The score is half the L1 distance between consecutive class-probability
    vectors, optionally smoothed over time.  A score near 1.0 indicates a
    confident action change; a score near 0.0 indicates a stable segment.
    """
    probs = to_probabilities(probabilities)
    if probs.shape[0] <= 1:
        return np.zeros(probs.shape[0], dtype=np.float64)

    scores = 0.5 * np.sum(np.abs(np.diff(probs, axis=0)), axis=1)
    scores = np.concatenate(([0.0], scores))
    if smooth_kernel and int(smooth_kernel) > 1:
        scores = _moving_average(scores, smooth_kernel)
    return np.clip(scores, 0.0, 1.0)


def _refine_boundaries(candidates, scores, refine_radius, min_gap):
    """Snap candidate boundaries to local maxima and remove duplicates."""
    if not candidates:
        return []

    refined = set()
    for candidate in candidates:
        lo = max(1, candidate - refine_radius)
        hi = min(scores.size - 1, candidate + refine_radius)
        local = scores[lo:hi + 1]
        peak = lo + int(np.argmax(local))
        refined.add(int(peak))

    ordered = sorted(refined)
    kept = []
    for boundary in ordered:
        if not kept or boundary - kept[-1] >= max(1, min_gap):
            kept.append(boundary)
    return kept


def boundary_regression(
    scores,
    boundary_threshold=0.30,
    smooth_kernel=1,
    refine_radius=2,
    min_boundary_gap=2,
):
    """Apply boundary barriers and return one label per frame.

    Parameters
    ----------
    scores : array-like, shape (T, C)
        Per-frame logits or class probabilities.
    boundary_threshold : float
        Boundary scores at or above this value become segment barriers.
    smooth_kernel : int
        Temporal smoothing window used before differentiation.
    refine_radius : int
        Each candidate boundary is moved to the highest-scoring frame within
        +/- ``refine_radius`` ("boundary regression" step).
    min_boundary_gap : int
        Minimum spacing between two consecutive barriers, in frames.

    Returns
    -------
    np.ndarray, shape (T,), dtype int64
        Smoothed per-frame labels.
    """
    probs = to_probabilities(scores)
    num_frames = probs.shape[0]
    if num_frames == 0:
        return np.empty((0,), dtype=np.int64)

    smoothed = _moving_average(probs, smooth_kernel)
    scores_1d = boundary_scores(smoothed, smooth_kernel=1)

    candidates = [
        int(i) for i in range(1, num_frames)
        if scores_1d[i] >= float(boundary_threshold)
    ]
    boundaries = _refine_boundaries(
        candidates, scores_1d, int(refine_radius), int(min_boundary_gap)
    )

    starts = [0] + [b for b in boundaries if 0 < b < num_frames]
    ends = starts[1:] + [num_frames]

    labels = np.empty((num_frames,), dtype=np.int64)
    for start, end in zip(starts, ends):
        if end <= start:
            continue
        mean_probs = np.mean(smoothed[start:end], axis=0)
        labels[start:end] = int(np.argmax(mean_probs))

    return labels


def threshold_filtering(labels, min_duration):
    """Merge segments shorter than ``min_duration`` frames into a neighbour.

    The shortest offending segment is merged repeatedly (longest neighbour
    wins, previous frame on ties) until every segment satisfies the minimum
    duration or only one segment remains.
    """
    arr = np.asarray(labels).reshape(-1).astype(np.int64).copy()
    min_duration = max(1, int(min_duration))
    if arr.size == 0:
        return arr

    while True:
        segments = _runs(arr)
        if len(segments) <= 1:
            break

        short = [(end - start, start, end) for start, end, _ in segments
                 if (end - start) < min_duration]
        if not short:
            break

        short.sort()
        _, start, end = short[0]
        index = [i for i, seg in enumerate(segments) if seg[0] == start][0]

        if index == 0:
            target = segments[1][2]
        elif index == len(segments) - 1:
            target = segments[index - 1][2]
        else:
            previous_len = segments[index - 1][1] - segments[index - 1][0]
            next_len = segments[index + 1][1] - segments[index + 1][0]
            target = (segments[index - 1][2] if previous_len >= next_len
                      else segments[index + 1][2])

        arr[start:end] = target

    return arr


def _runs(arr):
    """Return ``(start, end, label)`` runs for a 1-D label array."""
    if arr.size == 0:
        return []
    runs = []
    start = 0
    for i in range(1, arr.size):
        if arr[i] != arr[i - 1]:
            runs.append((start, i, int(arr[i - 1])))
            start = i
    runs.append((start, arr.size, int(arr[-1])))
    return runs


class HybridPostProcessor(object):
    """Boundary-barrier + short-segment filtering for online HAR.

    Parameters
    ----------
    fps : int
        Target inference frame rate.  If the rate is present in
        :data:`FPS_CALIBRATION`, the calibrated threshold and minimum segment
        length are loaded automatically.
    boundary_threshold : float or None
        Override the calibrated boundary threshold.
    smooth_kernel : int
        Temporal smoothing window used before computing boundary scores.
    min_segment_frames : int or None
        Override the calibrated minimum segment length in frames.
    min_segment_seconds : float
        Used to derive the minimum length when no calibration entry exists.
    use_fps_calibration : bool
        Set to False to ignore :data:`FPS_CALIBRATION` and derive the minimum
        length from ``min_segment_seconds`` instead.
    """

    def __init__(
        self,
        fps=30,
        boundary_threshold=None,
        smooth_kernel=1,
        min_segment_frames=None,
        min_segment_seconds=DEFAULT_MIN_SEGMENT_SECONDS,
        use_fps_calibration=True,
        refine_radius=2,
        min_boundary_gap=2,
    ):
        self.fps = int(fps)
        self.smooth_kernel = int(smooth_kernel)
        self.refine_radius = int(refine_radius)
        self.min_boundary_gap = int(min_boundary_gap)
        self.min_segment_seconds = float(min_segment_seconds)
        self.use_fps_calibration = bool(use_fps_calibration)

        calibrated = (FPS_CALIBRATION.get(self.fps, {})
                      if self.use_fps_calibration else {})

        self.boundary_threshold = float(
            boundary_threshold
            if boundary_threshold is not None
            else calibrated.get("boundary_threshold", 0.30)
        )
        self.min_segment_frames = int(
            min_segment_frames
            if min_segment_frames is not None
            else calibrated.get(
                "min_segment_frames",
                calibrate_min_segment_frames(self.fps, self.min_segment_seconds),
            )
        )

    def calibrate(self, fps):
        """Re-calibrate the processor for a new target frame rate."""
        self.fps = int(fps)
        calibrated = FPS_CALIBRATION.get(self.fps, {})
        if self.use_fps_calibration and calibrated:
            self.boundary_threshold = float(calibrated["boundary_threshold"])
            self.min_segment_frames = int(calibrated["min_segment_frames"])
        else:
            self.min_segment_frames = calibrate_min_segment_frames(
                self.fps, self.min_segment_seconds
            )
        return self

    def smooth(self, scores):
        """Return smoothed class probabilities (useful for debugging/plots)."""
        return _moving_average(to_probabilities(scores), self.smooth_kernel)

    def boundary_scores(self, scores):
        return boundary_scores(scores, smooth_kernel=self.smooth_kernel)

    def boundaries(self, scores):
        """Return the refined boundary frame indices for a sequence."""
        scores_1d = self.boundary_scores(scores)
        candidates = [
            int(i) for i in range(1, scores_1d.size)
            if scores_1d[i] >= self.boundary_threshold
        ]
        return _refine_boundaries(
            candidates, scores_1d, self.refine_radius, self.min_boundary_gap
        )

    def boundary_regression(self, scores):
        return boundary_regression(
            scores,
            boundary_threshold=self.boundary_threshold,
            smooth_kernel=self.smooth_kernel,
            refine_radius=self.refine_radius,
            min_boundary_gap=self.min_boundary_gap,
        )

    def threshold_filtering(self, labels):
        return threshold_filtering(labels, self.min_segment_frames)

    def __call__(self, scores=None, labels=None):
        """Run boundary regression (if scores given) then threshold filtering."""
        if scores is None and labels is None:
            raise ValueError("Provide either per-frame scores or labels.")

        if scores is not None:
            labels = self.boundary_regression(scores)
        return self.threshold_filtering(labels)

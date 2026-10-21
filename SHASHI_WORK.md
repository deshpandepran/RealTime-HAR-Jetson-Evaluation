# Shashi - Completed Work (Issues #9 and #11)

This document summarises the work completed for:

- **Issue #9 - Hybrid Post-Processing Module** (`models/post_processing.py`)
- **Issue #11 - Comprehensive Benchmarking Suite** (`eval/metrics.py`, `eval/evaluate.py`)

Everything below is implemented, unit-tested locally, and does not require a
trained checkpoint, a GPU, or the real 50 Salads videos to run.

---

## 1. Issue #9 - Hybrid Post-Processing

**File:** `models/post_processing.py`

The module combines the two filtering strategies requested in the issue.

### 1.1 Boundary regression / boundary barriers

- Per-frame logits or probabilities are converted to probabilities.
- A temporal boundary score is computed as half the L1 distance between
  consecutive class-probability vectors.
- Scores above a threshold become candidate boundaries. Each candidate is
  refined ("regressed") to the local maximum within a +/-2 frame window.
- Duplicate / too-close boundaries are removed, and each barrier-delimited
  region is collapsed to its probability-weighted majority label.

### 1.2 Threshold filtering

- Segments shorter than a calibrated minimum duration are merged into the
  longest neighbouring segment (previous neighbour on ties).
- Merging repeats until no short segment remains or only one segment is left.

### 1.3 Frame-rate calibration

| FPS | Min segment (frames) | Boundary threshold |
|----:|---------------------:|-------------------:|
| 30  | 10                   | 0.30               |
| 15  | 5                    | 0.30               |
| 6   | 2                    | 0.35               |
| 3   | 1                    | 0.40               |

Unknown frame rates fall back to `min_segment_seconds = 0.33 s`.

### 1.4 API

```python
from models.post_processing import HybridPostProcessor

processor = HybridPostProcessor(fps=30)
labels = processor(logits)                 # logits/probs: (T, num_classes)
labels = processor(labels=raw_labels)      # or post-filter existing labels

processor.calibrate(15)                    # re-calibrate for 15 FPS
```

Module-level helpers are also exported: `boundary_scores`,
`boundary_regression`, `threshold_filtering`, `calibrate_min_segment_frames`.

---

## 2. Issue #11 - Metrics and Benchmarking

### 2.1 `eval/metrics.py`

Pure NumPy implementation of:

- **Frame Accuracy / MoF** - percentage of correctly classified frames.
- **Edit Score** - `100 * (1 - Levenshtein / max(len(gt), len(pred)))` over
  the ordered segment-label transcripts, matching the 50 Salads protocol.
- **F1@10 / F1@25 / F1@50** - temporal-IoU matching with greedy
  highest-overlap assignment; reports percentages.
- **Latency statistics** - mean, std, min, max, p50/p95/p99, throughput, and
  deadline miss ratio (default deadline: 68.83 ms).
- Helpers: segment extraction, temporal IoU, per-sequence evaluation,
  per-video and per-fold aggregation.

### 2.2 `eval/evaluate.py`

Command-line 5-fold benchmark:

- Uses the same deterministic 10-videos-per-fold split as
  `data/fifty_salads.py`.
- Reads ground truth from `data/fifty_salads/groundTruth/*.txt`
  (supports 3-column `start end label` files, one-label-per-frame files, and
  `frame label` pairs; selectable with `--gt-format`).
- Loads predictions from `.npy`, `.npz`, `.txt`, or `.csv`
  (1-D labels or `(T, C)` scores -> argmax).
- Optional `--post-process` applies `HybridPostProcessor` to `(T, C)` scores.
- Optional `--latency-json` adds latency/deadline statistics.
- Writes a full JSON report with per-video, per-fold, and summary results.

### 2.3 Usage

```bash
# All 5 folds
python eval/evaluate.py --pred-dir predictions \
    --gt-dir data/fifty_salads/groundTruth \
    --fold all --output eval/results.json

# Single fold with hybrid post-processing
python eval/evaluate.py --pred-dir predictions --fold 1 \
    --post-process --fps 30

# Add measured latency values
python eval/evaluate.py --pred-dir predictions --latency-json latency.json
```

---

## 3. Tests

**File:** `tests/test_post_processing_metrics.py`

13 tests covering:

- Segment extraction and temporal IoU
- Frame accuracy (including `ignore_label`)
- Edit score
- F1 at multiple IoU thresholds
- Latency statistics
- Boundary-score detection
- Flicker / short-segment removal
- FPS calibration
- Label-only post-processing path
- Ground-truth parsing for both file formats
- End-to-end `eval/evaluate.py` run on synthetic 5-fold data

Run with:

```bash
python tests/test_post_processing_metrics.py
```

Result:

```
13 passed, 0 failed
```

The end-to-end test writes a synthetic fold and verifies that perfect
predictions produce MoF, Edit, F1@10, F1@25, and F1@50 all equal to 100%.

---

## 4. Files Added

| File | Purpose |
|------|---------|
| `models/post_processing.py` | Issue #9 hybrid post-processing |
| `eval/metrics.py` | Issue #11 metric implementations |
| `eval/evaluate.py` | Issue #11 5-fold benchmark CLI |
| `eval/__init__.py` | Metric exports |
| `tests/test_post_processing_metrics.py` | Unit + end-to-end tests |
| `SHASHI_WORK.md` | This summary |

---

## 5. Notes / Limitations

- Real benchmark numbers require prediction files from the trained LSTR
  model (Issue #7). The evaluation suite is ready and can be run as soon as
  those predictions exist.
- The default 50 Salads column mapping (`background = 0`) follows the common
  action-segmentation convention; use `background_label` if the dataset
  encoding differs.
- Post-processing is deliberately model-agnostic and works on TensorRT /
  ONNX outputs, so it can be applied at inference time on the Jetson.

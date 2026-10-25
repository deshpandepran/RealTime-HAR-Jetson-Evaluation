# Real-Time Human Action Recognition (RT-HARE & IMFE)

> **Project Scope:** Implementation, evaluation, and novel extensions of the paper **"Real-Time Human Action Recognition on Embedded Platforms"** (Wang et al., 2024).

---

## Overview

Video-based Human Action Recognition (HAR) on live streams suffers from excessive computational latency on embedded platforms such as the NVIDIA Jetson Xavier NX. Traditional two-stream HAR pipelines rely on TV-L1 Optical Flow (OF) extraction, which constitutes approximately 93% of total inference latency.

This project evaluates RT-HARE and its core component, the Integrated Motion Feature Extractor (IMFE):

* **IMFE Architecture:** Replaces the traditional two-stage Optical Flow and Flow Feature Extractor pipeline with a single-shot neural network that directly generates 2048-dimensional motion representations.
* **Knowledge Distillation:** Trains the IMFE module utilizing a frozen TV-L1 and ResNet-50 teacher configuration optimized via Mean Squared Error (MSE) loss.
* **Online Recognition:** Integrates the Long Short-Term Transformer (LSTR) for online action recognition and temporal segmentation.

---

## Repository Structure

```text
├── assets/                  # Diagrams, plots, and visual artifacts
├── configs/                 # Hyperparameters and model configuration files
├── data/                    # Dataset loaders and preprocessing scripts
│   ├── __init__.py
│   ├── preprocess.py        # Spatial resizing (256x344), normalization, and tensor batching
│   ├── activitynet.py       # ActivityNet dataset subset loader (4,359 videos)
│   └── fifty_salads.py      # 50 Salads benchmark loader with 5-fold cross-validation
├── models/                  # Architecture implementations
│   ├── __init__.py
│   ├── imfe.py              # Single-shot IMFE network
│   ├── lstr.py              # Long Short-Term Transformer
│   ├── baselines.py         # RGB-only & RAFT baselines
│   ├── resnet_flow.py       # ResNet-50 flow feature extractor (multi-channel input)
│   └── tvl1_baseline.py     # OpenCV GPU-accelerated TV-L1 optical flow extractor
├── distillation/            # Knowledge Distillation training loop (MSE Loss)
├── eval/                    # Metrics calculation (Accuracy, Edit, F1@10, Latency)
│   └── profile_latency.py   # Latency profiling and component-wise bottleneck breakdown
├── extensions/              # Team-developed novel extensions
├── scripts/                 # Setup and run scripts
│   ├── setup_jetson.sh      # Jetson Xavier NX configuration (Mode 6: 20W, 2-Core @ 1.9GHz)
│   ├── export_tensorrt.py   # ONNX to TensorRT export with FP16/DLA acceleration
│   └── verify_env.py        # Environment & hardware verification suite
├── requirements.txt         # Pinned Python package dependencies
├── environment.yml          # Conda environment configuration
├── train.py                 # Main training pipeline
└── README.md                # Project documentation
```

---

## Environment Configuration

### Workstation Setup (Dual RTX 3090, Ubuntu 20.04)

```bash
conda env create -f environment.yml
conda activate rt-hare
python3 scripts/verify_env.py
```

### Embedded Testbed Setup (NVIDIA Jetson Xavier NX, JetPack 5.1.1)

In accordance with Section III-B of the reference literature, peak inference performance requires Power Mode 6 (20W, 2-Core) with Dynamic Voltage and Frequency Scaling (DVFS) disabled and CPU clocks locked to 1.9 GHz:

```bash
sudo bash scripts/setup_jetson.sh --clocks-only
sudo bash scripts/setup_jetson.sh --status
```

### TensorRT Model Optimization

```bash
python3 scripts/export_tensorrt.py --onnx <model.onnx> --fp16
```

---

## Usage Instructions

### Data Pipeline and Preprocessing

The preprocessing module enforces spatial dimensions of $H \times W = 256 \times 344$ (multiples of 8 to support network feature scaling) alongside multi-frame batching for video clip inputs. Dataset loaders manage the ActivityNet distillation subset and the 50 Salads benchmark configuration with 5-fold cross-validation splits.

### Baseline and Feature Extraction Modules

Baseline models provide standalone spatial features via ResNet-50 backbones and high-accuracy optical flow via TorchVision RAFT-Large wrappers. The ResNet-50 flow extractor adapts multi-channel input structures to process sequential optical flow representations into 2048-dimensional embeddings.

### Latency Profiling

To replicate component-wise execution latency benchmarks and quantify the TV-L1 computational bottleneck:

```bash
python eval/profile_latency.py
```

### IMFE Model Execution

The IMFE architecture processes temporal video clips through batched feature encoding, correlation modules, dimension compression layers, and a final residual encoder to yield motion feature representations in a single forward pass.


## Windows Install (CPU-only laptop)

TensorRT and PyCUDA are Jetson/Linux-only. On Windows:

    conda create -n rt-hare python=3.10 -y
    conda activate rt-hare
    pip install -r requirements.txt

On a Jetson or a CUDA workstation, additionally run `pip install -r requirements-jetson.txt`.

## Live Demo

    python extensions/live_demo.py                          # webcam
    python extensions/live_demo.py --video clip.mp4 --no-display --max-clips 5

Without trained IMFE/LSTR checkpoints the weights are random, so predictions
are meaningless and only latency is measured. Pass checkpoints with
`--imfe-ckpt` and `--lstr-ckpt`.

## New Use Case: Anomaly Detection (UCSD Ped2)

In progress. Results, latency and limitations will be added here.

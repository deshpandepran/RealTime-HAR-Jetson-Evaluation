# Real-Time Human Action Recognition (RT-HARE & IMFE)

> **Project Scope:** Implementation, evaluation, and novel extensions of the paper **"Real-Time Human Action Recognition on Embedded Platforms"** (Wang et al., 2024).

---

## 📌 Project Overview

Video-based Human Action Recognition (HAR) on live streams typically suffers from excessive latency on embedded platforms (e.g., NVIDIA Jetson Xavier NX). Traditional two-stream HAR pipelines rely on **TV-L1 Optical Flow (OF)** extraction, which accounts for **~93% of total inference latency**.

This project evaluates **RT-HARE** and its core component, **IMFE (Integrated Motion Feature Extractor)**[cite: 2]:
* **IMFE** replaces the two-stage Optical Flow + Flow Feature Extractor pipeline with a single-shot neural network that directly generates motion features (2048-dim) without generating high-resolution optical flow vectors.
* **Knowledge Distillation** is used to train IMFE using a frozen TV-L1 + ResNet-50 teacher model with Mean Squared Error (MSE) loss.
* **LSTR (Long Short-Term Transformer)** is used for online action recognition and segmentation.

---

## 🎯 Primary Deliverables

1. **GitHub Repository:** Clean, well-documented codebase with training, distillation, evaluation, and extension modules.
2. **Technical Extensions:** Novel architectural or optimization improvements over the original paper.
3. **Benchmarking Suite:** FPS, Latency (ms), Accuracy, Edit Score, and F1@10 metrics across frame rates (30, 15, 6, 3 FPS)[cite: 8, 10, 11].
4. **Video Presentation:** A concise video detailing problem statement, architecture, baseline comparisons, and extension results.

---

## 📂 Expected Repository Structure

```text
├── assets/                  # Diagrams, plots, and visual artifacts
├── configs/                 # Hyperparameters and model configuration files
├── data/                    # Dataset loaders and preprocessing scripts
├── models/                  # Architecture implementations
│   ├── imfe.py              # Single-shot IMFE network
│   ├── lstr.py              # Long Short-Term Transformer
│   └── baselines.py         # RGB-only & RAFT baselines
├── distillation/            # Knowledge Distillation training loop (MSE Loss)
├── eval/                    # Metrics calculation (Accuracy, Edit, F1@10, Latency)
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

## ⚙️ Environment Setup (Issue #1)

### 1. Workstation Setup (Dual RTX 3090, Ubuntu 20.04)
```bash
# Using conda
conda env create -f environment.yml
conda activate rt-hare

# Verify environment
python3 scripts/verify_env.py
```

### 2. Embedded Testbed Setup (NVIDIA Jetson Xavier NX, JetPack 5.1.1)
As specified in **Section III-B** of the paper, maximum inference performance is obtained in **Power Mode 6 (20W, 2-Core)** with DVFS disabled and CPU clocks locked to 1.9 GHz:
```bash
# Apply power mode 6 and maximize clocks
sudo bash scripts/setup_jetson.sh --clocks-only

# Check hardware status
sudo bash scripts/setup_jetson.sh --status
```

### 3. TensorRT Model Optimization & ONNX Export Pipeline
```bash
# Convert ONNX to TensorRT FP16 engine
python3 scripts/export_tensorrt.py --onnx <model.onnx> --fp16
```

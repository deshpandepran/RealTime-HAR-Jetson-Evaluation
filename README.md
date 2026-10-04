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
├── train.py                 # Main training pipeline
└── README.md                # Project documentation

#!/usr/bin/env python3
"""
Environment & Dependency Verification Script
===========================================
Validates the software stack and hardware configuration for:
- DL Training Workstation (Dual RTX 3090, Ubuntu 20.04)
- NVIDIA Jetson Xavier NX Embedded Testbed (JetPack 5.1.1, Power Mode 6, 1.9 GHz)
"""

import sys
import os
import platform
import subprocess


def main():
    print("=" * 60)
    print("   RT-HARE ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print(f"OS:             {platform.system()} {platform.release()}")
    print(f"Architecture:   {platform.machine()}")
    print(f"Python Version: {platform.python_version()} ({sys.executable})")

    # PyTorch check
    try:
        import torch
        print(f"[PASS] PyTorch:     {torch.__version__}")
        print(f"       CUDA Avail:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"       Device:      {torch.cuda.get_device_name(0)}")
            print(f"       GPU Count:   {torch.cuda.device_count()}")
    except ImportError:
        print("[FAIL] PyTorch not installed.")

    # TensorRT check
    try:
        import tensorrt as trt
        print(f"[PASS] TensorRT:    {trt.__version__}")
    except ImportError:
        print("[NOTE] TensorRT python package not installed.")

    # OpenCV check
    try:
        import cv2
        print(f"[PASS] OpenCV:      {cv2.__version__}")
    except ImportError:
        print("[FAIL] OpenCV not installed.")

    # Metrics check
    try:
        import editdistance
        print(f"[PASS] editdistance installed.")
    except ImportError:
        print("[NOTE] editdistance not installed.")

    # Jetson check
    if os.path.isfile("/etc/nv_tegra_release"):
        print("[INFO] NVIDIA Jetson detected.")
        try:
            res = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True)
            print(res.stdout.strip())
        except Exception:
            pass

    print("=" * 60)


if __name__ == "__main__":
    main()

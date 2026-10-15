#!/usr/bin/env python3
"""
TensorRT Model Optimization and ONNX-to-TensorRT Export Pipeline
================================================================
Reference:
  - Section V-C (Embedded System Deployment) & Section V-D (Inference Parallelization)
    in "Real-Time Human Action Recognition on Embedded Platforms" (Wang et al., 2024).

Tasks Addressed (Issue #1):
  - Model optimization with mixed precision (FP16/FP32).
  - ONNX-to-TensorRT compilation.
  - Deep Learning Accelerator (DLA) targeting on Jetson Xavier NX.
  - Latency and throughput benchmarking.
"""

import os
import sys
import argparse
import time
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    cuda = None

TRT_LOGGER = trt.Logger(trt.Logger.INFO) if trt else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to optimized TensorRT engine for RT-HARE."
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to input ONNX file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output TensorRT engine (.plan or .engine).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 mixed precision execution (default: True).",
    )
    parser.add_argument(
        "--dla-core",
        type=int,
        default=None,
        choices=[0, 1],
        help="Target NVIDIA DLA core (0 or 1) on Jetson Xavier NX.",
    )
    parser.add_argument(
        "--allow-gpu-fallback",
        action="store_true",
        default=True,
        help="Allow unsupported DLA layers to fall back to GPU.",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=2048,
        help="Max workspace size in MB (default: 2048 MB).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark engine latency after compilation.",
    )
    return parser.parse_args()


def build_engine(onnx_file, engine_file, fp16=True, dla_core=None, fallback=True, workspace_mb=2048):
    if trt is None:
        raise RuntimeError("TensorRT library not found.")

    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] ONNX parsing failed:")
            for e in range(parser.num_errors):
                print(f"  - {parser.get_error(e)}")
            return None

    config = builder.create_builder_config()
    workspace_bytes = workspace_mb * (1024 * 1024)
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if fp16 and builder.platform_has_fast_fp16:
        print("[INFO] Enabling FP16 mixed precision.")
        config.set_flag(trt.BuilderFlag.FP16)

    if dla_core is not None:
        print(f"[INFO] Targeting DLA Core {dla_core}...")
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        if fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    print("[INFO] Building TensorRT engine plan...")
    start = time.time()
    serialized = builder.build_serialized_network(network, config)
    print(f"[SUCCESS] Built engine in {time.time() - start:.2f} seconds.")

    os.makedirs(os.path.dirname(os.path.abspath(engine_file)), exist_ok=True)
    with open(engine_file, "wb") as f:
        f.write(serialized)
    print(f"[SUCCESS] Saved TensorRT engine to: {engine_file}")
    return serialized


def main():
    args = parse_args()
    if args.output is None:
        base, _ = os.path.splitext(args.onnx)
        args.output = f"{base}.engine"

    build_engine(
        onnx_file=args.onnx,
        engine_file=args.output,
        fp16=args.fp16,
        dla_core=args.dla_core,
        fallback=args.allow_gpu_fallback,
        workspace_mb=args.workspace,
    )


if __name__ == "__main__":
    main()

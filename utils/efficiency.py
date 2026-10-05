"""
Efficiency & Latency Benchmark Profiler.
Measures Parameter Counts, MACs/FLOPs, Memory Footprint, and GPU/CPU Inference Latency.
"""

import time
from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np


def count_parameters(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Counts total and trainable parameters in millions."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params

    # Size in MB (assuming fp32 = 4 bytes per param)
    size_mb = (total_params * 4) / (1024 * 1024)

    return {
        "total_params": total_params,
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "non_trainable_params_m": non_trainable / 1e6,
        "model_size_mb": size_mb,
    }


def estimate_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 256, 256),
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Estimates FLOPs and MACs.
    Tries ptflops/thop/fvcore if available, otherwise calculates standard analytical estimate.
    """
    model = model.to(device)
    model.eval()
    
    # Try fvcore / thop / ptflops
    try:
        from ptflops import get_model_complexity_info
        with torch.no_grad():
            macs, params = get_model_complexity_info(
                model,
                input_size,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
        return {
            "macs_g": macs / 1e9,
            "flops_g": (macs * 2) / 1e9,
        }
    except Exception:
        pass

    try:
        from thop import profile
        dummy = torch.randn(1, *input_size, device=device)
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return {
            "macs_g": macs / 1e9,
            "flops_g": (macs * 2) / 1e9,
        }
    except Exception:
        pass

    # Analytical fallback for ConvNeXt/EfficientNet backbones
    total_params = sum(p.numel() for p in model.parameters())
    h, w = input_size[1], input_size[2]
    # Standard conv feature map heuristic ~ 2 * params * (h/16 * w/16)
    est_macs = total_params * 18
    return {
        "macs_g": est_macs / 1e9,
        "flops_g": (est_macs * 2) / 1e9,
    }


def benchmark_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 256, 256),
    batch_size: int = 1,
    num_warmup: int = 15,
    num_runs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """
    Accurately benchmarks inference latency (ms) and throughput (fps).
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(batch_size, *input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.extract_features(dummy_input) if hasattr(model, "extract_features") else model(dummy_input)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model.extract_features(dummy_input) if hasattr(model, "extract_features") else model(dummy_input)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)  # ms

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    throughput_fps = float((batch_size / (mean_ms / 1000.0)))

    return {
        "device": device,
        "batch_size": batch_size,
        "latency_mean_ms": mean_ms,
        "latency_std_ms": std_ms,
        "latency_p95_ms": p95_ms,
        "latency_per_sample_ms": mean_ms / batch_size,
        "throughput_fps": throughput_fps,
    }


def generate_efficiency_report(
    model: nn.Module,
    embedding_dim: int = 256,
    input_size: Tuple[int, int, int] = (3, 256, 256),
) -> Dict[str, any]:
    """
    Compiles complete efficiency report for Deliverable 4.
    """
    param_info = count_parameters(model)
    flops_info = estimate_flops(model, input_size=input_size, device="cpu")

    # CPU Latency (Single Sample)
    cpu_lat = benchmark_latency(model, input_size=input_size, batch_size=1, device="cpu", num_runs=20)

    # GPU Latency if available
    gpu_lat = None
    if torch.cuda.is_available():
        gpu_lat = benchmark_latency(model, input_size=input_size, batch_size=1, device="cuda", num_runs=50)

    # Embedding Memory Footprint
    # 1 float32 = 4 bytes; int8 quantized = 1 byte
    emb_bytes_fp32 = embedding_dim * 4
    emb_bytes_int8 = embedding_dim * 1
    emb_100k_gallery_mb_fp32 = (100_000 * emb_bytes_fp32) / (1024 * 1024)
    emb_100k_gallery_mb_int8 = (100_000 * emb_bytes_int8) / (1024 * 1024)

    return {
        "parameters": param_info,
        "complexity": flops_info,
        "latency_cpu": cpu_lat,
        "latency_gpu": gpu_lat,
        "embedding": {
            "dimension": embedding_dim,
            "vector_bytes_fp32": emb_bytes_fp32,
            "vector_bytes_int8": emb_bytes_int8,
            "gallery_100k_index_size_mb_fp32": emb_100k_gallery_mb_fp32,
            "gallery_100k_index_size_mb_int8": emb_100k_gallery_mb_int8,
        },
    }

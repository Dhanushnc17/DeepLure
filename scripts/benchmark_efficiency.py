"""
Benchmark Efficiency Script for Deliverable 4 (Efficiency Report).
Measures Parameter Counts, FLOPs, Embedding Size, and Inference Latency.
Usage:
    python scripts/benchmark_efficiency.py --config configs/config.yaml
"""

import os
import sys
import argparse
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.saree_model import ColorInvariantSareeNet
from utils.efficiency import generate_efficiency_report


def main():
    parser = argparse.ArgumentParser(description="Benchmark Model Efficiency for DeepLure Saree Recognition")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone to benchmark")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Override embedding dimension")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    backbone_name = args.backbone or cfg["model"]["backbone"]
    embedding_dim = args.embedding_dim or cfg["model"]["embedding_dim"]
    img_size = tuple(cfg["data"]["image_size"])

    print(f"[*] Benchmarking Model Efficiency: Backbone='{backbone_name}', Dim={embedding_dim}, Input={img_size}...")

    # Instantiate model
    model = ColorInvariantSareeNet(
        backbone_name=backbone_name,
        num_classes=100,
        embedding_dim=embedding_dim,
        pretrained=False,
        pooling=cfg["model"]["pooling"],
        gem_p=cfg["model"]["gem_p"],
        gem_trainable=False,
        use_color_disentanglement=False,
    )

    report = generate_efficiency_report(
        model=model,
        embedding_dim=embedding_dim,
        input_size=(3, img_size[0], img_size[1]),
    )

    params = report["parameters"]
    complexity = report["complexity"]
    cpu_lat = report["latency_cpu"]
    gpu_lat = report["latency_gpu"]
    emb = report["embedding"]

    print("\n" + "=" * 68)
    print("           DEEPLURE MODEL EFFICIENCY & PERFORMANCE REPORT")
    print("=" * 68)
    print(f"  Architecture Backbone        : {backbone_name}")
    print(f"  Input Resolution             : {img_size[0]} x {img_size[1]}")
    print(f"  Total Parameters             : {params['total_params_m']:.2f} M ({params['total_params']:,} params)")
    print(f"  Model Size (fp32 weights)    : {params['model_size_mb']:.2f} MB")
    print(f"  Computation (MACs / FLOPs)   : {complexity['macs_g']:.2f} GMACs / {complexity['flops_g']:.2f} GFLOPs")
    print("-" * 68)
    print(f"  Embedding Dimension          : {emb['dimension']}-D")
    print(f"  Vector Size (fp32 / int8)    : {emb['vector_bytes_fp32']} bytes / {emb['vector_bytes_int8']} bytes")
    print(f"  100k Gallery RAM (fp32)      : {emb['gallery_100k_index_size_mb_fp32']:.2f} MB")
    print(f"  100k Gallery RAM (int8 Quant): {emb['gallery_100k_index_size_mb_int8']:.2f} MB")
    print("-" * 68)
    print(f"  CPU Latency (Batch=1, mean)  : {cpu_lat['latency_mean_ms']:.2f} ms (p95: {cpu_lat['latency_p95_ms']:.2f} ms)")
    print(f"  CPU Throughput               : {cpu_lat['throughput_fps']:.1f} FPS")
    if gpu_lat:
        print(f"  GPU Latency (Batch=1, mean)  : {gpu_lat['latency_mean_ms']:.2f} ms (p95: {gpu_lat['latency_p95_ms']:.2f} ms)")
        print(f"  GPU Throughput               : {gpu_lat['throughput_fps']:.1f} FPS")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()

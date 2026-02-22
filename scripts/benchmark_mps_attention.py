"""
Benchmark MPS Attention Optimizations.

This script compares different attention implementations on Apple Silicon:
1. Standard attention
2. MPS-optimized chunked attention
3. PyTorch's scaled_dot_product_attention
"""

# Add parent directory to path for imports
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.models.mps_optimizations import (
    MPSEfficientAttention,
    MPSOptimizedAttention,
    is_mps_available,
)


def benchmark_attention(
    batch_size: int = 8,
    num_heads: int = 12,
    seq_lengths: list = [196, 384, 768, 1024],
    head_dim: int = 64,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> dict[str, Any]:
    """
    Benchmark different attention implementations.
    """
    if not is_mps_available():
        print("MPS not available. Please run on a Mac with Apple Silicon.")
        return {}

    device = torch.device("mps")
    results = {}

    for seq_len in seq_lengths:
        print(f"\nBenchmarking sequence length: {seq_len}")
        print("-" * 50)

        # Create random inputs
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        seq_results = {}

        # 1. Standard Attention
        print("Testing standard attention...")
        attn = MPSEfficientAttention(num_heads, head_dim)
        attn = attn.to(device)

        # Warmup
        for _ in range(warmup_iterations):
            _ = attn._standard_attention(q, k, v)
        torch.mps.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            _ = attn._standard_attention(q, k, v)
        torch.mps.synchronize()
        standard_time = (time.time() - start_time) / num_iterations

        seq_results["standard"] = {
            "time_ms": standard_time * 1000,
            "throughput": batch_size * seq_len / standard_time,
        }

        # 2. Chunked Attention (for larger sequences)
        if seq_len > 256:
            print("Testing chunked attention...")
            # Warmup
            for _ in range(warmup_iterations):
                _ = attn._chunked_attention(q, k, v)
            torch.mps.synchronize()

            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = attn._chunked_attention(q, k, v)
            torch.mps.synchronize()
            chunked_time = (time.time() - start_time) / num_iterations

            seq_results["chunked"] = {
                "time_ms": chunked_time * 1000,
                "throughput": batch_size * seq_len / chunked_time,
                "speedup_vs_standard": standard_time / chunked_time,
            }

        # 3. PyTorch SDPA
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("Testing PyTorch SDPA...")
            # Warmup
            for _ in range(warmup_iterations):
                _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)
            torch.mps.synchronize()

            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)
            torch.mps.synchronize()
            sdpa_time = (time.time() - start_time) / num_iterations

            seq_results["pytorch_sdpa"] = {
                "time_ms": sdpa_time * 1000,
                "throughput": batch_size * seq_len / sdpa_time,
                "speedup_vs_standard": standard_time / sdpa_time,
            }

        # 4. MPS Optimized (adaptive)
        print("Testing MPS optimized attention...")
        opt_attn = MPSOptimizedAttention(num_heads, head_dim, use_pytorch_sdpa=True).to(device)

        # Warmup
        for _ in range(warmup_iterations):
            _ = opt_attn(q, k, v)
        torch.mps.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            _ = opt_attn(q, k, v)
        torch.mps.synchronize()
        opt_time = (time.time() - start_time) / num_iterations

        seq_results["mps_optimized"] = {
            "time_ms": opt_time * 1000,
            "throughput": batch_size * seq_len / opt_time,
            "speedup_vs_standard": standard_time / opt_time,
        }

        results[f"seq_len_{seq_len}"] = seq_results

        # Print results for this sequence length
        print(f"\nResults for sequence length {seq_len}:")
        for name, metrics in seq_results.items():
            print(f"  {name}:")
            print(f"    Time: {metrics['time_ms']:.2f} ms")
            print(f"    Throughput: {metrics['throughput']:.0f} tokens/sec")
            if "speedup_vs_standard" in metrics:
                print(f"    Speedup: {metrics['speedup_vs_standard']:.2f}x")

    return results


def main():
    """Main benchmark function."""
    print("=" * 60)
    print("MPS Attention Optimization Benchmark")
    print("=" * 60)

    # Check device
    if not torch.backends.mps.is_available():
        print("Error: MPS backend is not available.")
        print("This script requires an Apple Silicon Mac.")
        return

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    # Run benchmarks with different configurations
    print("\n" + "=" * 60)
    print("Configuration 1: Small model (ViT-Base like)")
    print("=" * 60)
    benchmark_attention(
        batch_size=8,
        num_heads=12,
        seq_lengths=[196, 384],  # 14x14 and ~20x20 patches
        head_dim=64,
        num_iterations=100,
    )

    print("\n" + "=" * 60)
    print("Configuration 2: Large model (ViT-Large like)")
    print("=" * 60)
    benchmark_attention(
        batch_size=4,
        num_heads=16,
        seq_lengths=[196, 384, 768],  # Various patch counts
        head_dim=64,
        num_iterations=50,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nKey Findings:")
    print("1. Standard attention: Good for small sequences (<= 256)")
    print("2. Chunked attention: Better for large sequences on MPS")
    print("3. PyTorch SDPA: Usually best if available and optimized")
    print("4. MPS Optimized: Automatically selects best backend")

    print("\nRecommendations for MPS:")
    print("- Use MPS-optimized attention wrapper for automatic backend selection")
    print("- For sequences > 512, chunked attention provides memory benefits")
    print("- Monitor memory usage for very large models")
    print("- Consider torch.compile for additional optimizations")


if __name__ == "__main__":
    main()

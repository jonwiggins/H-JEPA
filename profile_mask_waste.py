#!/usr/bin/env python3
"""
Profile script to quantify computational waste in HierarchicalMaskGenerator.

This script measures:
1. Time spent generating masks for each level
2. Memory allocated for each level
3. Total waste percentage
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.masks import HierarchicalMaskGenerator


def profile_mask_generation(batch_size=32, num_iterations=100):
    """Profile mask generation performance."""

    print("=" * 80)
    print("Hierarchical Mask Generator Waste Analysis")
    print("=" * 80)
    print()

    # Setup
    mask_gen = HierarchicalMaskGenerator(
        input_size=224,
        patch_size=16,
        num_hierarchies=3,
        num_target_masks=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Device: {device}")
    print(f"  Number of hierarchies: {mask_gen.num_hierarchies}")
    print(f"  Number of patches: {mask_gen.num_patches}")
    print()

    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = mask_gen(batch_size=batch_size, device=device)

    if device == "cuda":
        torch.cuda.synchronize()

    # Profile full generation (all 3 levels)
    print("\nProfiling FULL generation (all 3 levels)...")
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        masks = mask_gen(batch_size=batch_size, device=device)
        if device == "cuda":
            torch.cuda.synchronize()

    full_time = time.perf_counter() - start_time

    # Calculate memory usage
    memory_per_mask_set = 0
    for level_key, level_masks in masks.items():
        context_mem = level_masks["context"].element_size() * level_masks["context"].nelement()
        targets_mem = level_masks["targets"].element_size() * level_masks["targets"].nelement()
        memory_per_mask_set += context_mem + targets_mem

    print(f"  Total time: {full_time:.4f}s")
    print(f"  Time per iteration: {full_time/num_iterations*1000:.2f}ms")
    print(f"  Memory per batch: {memory_per_mask_set / 1024 / 1024:.2f} MB")

    # Profile level 0 only (what's actually used)
    print("\nProfiling LEVEL 0 ONLY generation...")

    # Temporarily modify to generate only level 0
    original_num_hierarchies = mask_gen.num_hierarchies
    original_level_configs = mask_gen.level_configs
    mask_gen.num_hierarchies = 1
    mask_gen.level_configs = [mask_gen.level_configs[0]]

    start_time = time.perf_counter()

    for _ in range(num_iterations):
        masks = mask_gen(batch_size=batch_size, device=device)
        if device == "cuda":
            torch.cuda.synchronize()

    level0_time = time.perf_counter() - start_time

    # Calculate memory for level 0 only
    memory_level0 = 0
    for level_key, level_masks in masks.items():
        context_mem = level_masks["context"].element_size() * level_masks["context"].nelement()
        targets_mem = level_masks["targets"].element_size() * level_masks["targets"].nelement()
        memory_level0 += context_mem + targets_mem

    print(f"  Total time: {level0_time:.4f}s")
    print(f"  Time per iteration: {level0_time/num_iterations*1000:.2f}ms")
    print(f"  Memory per batch: {memory_level0 / 1024 / 1024:.2f} MB")

    # Restore original configuration
    mask_gen.num_hierarchies = original_num_hierarchies
    mask_gen.level_configs = original_level_configs

    # Calculate waste
    print("\n" + "=" * 80)
    print("WASTE ANALYSIS")
    print("=" * 80)

    wasted_time = full_time - level0_time
    wasted_time_pct = (wasted_time / full_time) * 100

    wasted_memory = memory_per_mask_set - memory_level0
    wasted_memory_pct = (wasted_memory / memory_per_mask_set) * 100

    print(f"\nTime Waste:")
    print(f"  Full generation time: {full_time:.4f}s")
    print(f"  Level 0 only time: {level0_time:.4f}s")
    print(f"  Wasted time: {wasted_time:.4f}s ({wasted_time_pct:.1f}%)")
    print(f"  Speedup potential: {full_time/level0_time:.2f}x")

    print(f"\nMemory Waste:")
    print(f"  Full generation memory: {memory_per_mask_set / 1024 / 1024:.2f} MB")
    print(f"  Level 0 only memory: {memory_level0 / 1024 / 1024:.2f} MB")
    print(f"  Wasted memory: {wasted_memory / 1024 / 1024:.2f} MB ({wasted_memory_pct:.1f}%)")

    # Extrapolate to training
    print("\n" + "=" * 80)
    print("TRAINING IMPACT ESTIMATION")
    print("=" * 80)

    # Assume 1000 steps per epoch, 300 epochs
    steps_per_epoch = 1000
    total_epochs = 300
    total_steps = steps_per_epoch * total_epochs

    total_wasted_time = (wasted_time / num_iterations) * total_steps
    total_wasted_hours = total_wasted_time / 3600

    print(f"\nFor a typical training run:")
    print(f"  Total steps: {total_steps:,} ({total_epochs} epochs × {steps_per_epoch} steps/epoch)")
    print(f"  Wasted time per step: {wasted_time/num_iterations*1000:.2f}ms")
    print(f"  Total wasted time: {total_wasted_hours:.2f} hours")
    print(f"  Wasted GPU memory per batch: {wasted_memory / 1024 / 1024:.2f} MB")

    # Categorize severity
    print("\n" + "=" * 80)
    print("SEVERITY ASSESSMENT")
    print("=" * 80)

    print("\nClassification: ", end="")
    if wasted_time_pct > 50:
        print("🔴 CRITICAL - Major bottleneck")
        severity = "CRITICAL"
    elif wasted_time_pct > 25:
        print("🟡 MODERATE - Significant inefficiency")
        severity = "MODERATE"
    else:
        print("🟢 MINOR - Low priority optimization")
        severity = "MINOR"

    print(f"\nJustification:")
    print(f"  - {wasted_time_pct:.1f}% of mask generation time is wasted")
    print(f"  - {wasted_memory_pct:.1f}% of mask memory is unused")
    print(f"  - {total_wasted_hours:.2f} hours wasted over full training")

    # Is mask generation a bottleneck?
    print(f"\nBottleneck Analysis:")
    ms_per_step = (full_time / num_iterations) * 1000
    print(f"  - Mask generation takes ~{ms_per_step:.2f}ms per step")

    # Typical forward+backward pass takes 100-500ms depending on model size
    # For ViT-Base with batch size 32: ~200ms
    typical_fwd_bwd = 200  # ms
    mask_overhead = (ms_per_step / typical_fwd_bwd) * 100

    print(f"  - Typical forward+backward pass: ~{typical_fwd_bwd}ms")
    print(f"  - Mask generation overhead: {mask_overhead:.1f}% of training step")

    if mask_overhead > 10:
        print(f"  - ⚠️  Mask generation is {mask_overhead:.1f}% of training time - optimization recommended")
    else:
        print(f"  - ✓ Mask generation is only {mask_overhead:.1f}% of training time - not a critical bottleneck")

    print("\n" + "=" * 80)

    return {
        "severity": severity,
        "time_waste_pct": wasted_time_pct,
        "memory_waste_pct": wasted_memory_pct,
        "speedup_potential": full_time / level0_time,
        "wasted_hours_per_training": total_wasted_hours,
    }


if __name__ == "__main__":
    results = profile_mask_generation(batch_size=32, num_iterations=100)

    print("\nRecommendation:")
    if results["severity"] == "CRITICAL" or results["time_waste_pct"] > 30:
        print("  → IMMEDIATE ACTION: Simplify to generate only level_0 masks")
        print("  → Expected improvement: {:.1f}x faster mask generation".format(results["speedup_potential"]))
    else:
        print("  → LOW PRIORITY: Waste is measurable but not critical")
        print("  → Consider fixing during code cleanup phase")

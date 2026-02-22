#!/usr/bin/env python3
"""
Test script to verify mask semantics in H-JEPA.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.masks import HierarchicalMaskGenerator

# Create mask generator with config values
mask_gen = HierarchicalMaskGenerator(
    input_size=224,
    patch_size=16,
    num_hierarchies=3,
    num_target_masks=4,
    base_scale=(0.15, 0.2),  # From default config
    aspect_ratio_range=(0.75, 1.5),
)

# Generate masks
batch_size = 8
masks_dict = mask_gen(batch_size=batch_size, device="cpu")

# Extract level 0 masks
context_masks = masks_dict["level_0"]["context"]  # [B, N]
target_masks = masks_dict["level_0"]["targets"]  # [B, num_targets, N]

print("=" * 80)
print("MASK SEMANTICS VERIFICATION")
print("=" * 80)

print("\nMask shapes:")
print(f"  context_masks: {context_masks.shape}")
print(f"  target_masks: {target_masks.shape}")

# Compute coverage statistics
num_patches = context_masks.shape[1]
print(f"\nTotal patches per image: {num_patches}")

# For each sample in batch
context_coverages = []
target_coverages = []
overlap_counts = []

for i in range(batch_size):
    context = context_masks[i]  # [N]
    targets = target_masks[i]  # [num_targets, N]

    # Count True values
    context_count = context.sum().item()
    target_union = targets.any(dim=0)  # Combine all targets
    target_count = target_union.sum().item()

    # Check overlap
    overlap = (context & target_union).sum().item()

    context_pct = (context_count / num_patches) * 100
    target_pct = (target_count / num_patches) * 100

    context_coverages.append(context_pct)
    target_coverages.append(target_pct)
    overlap_counts.append(overlap)

print("\nContext Coverage (VISIBLE patches):")
print(f"  Mean: {np.mean(context_coverages):.1f}%")
print(f"  Std:  {np.std(context_coverages):.1f}%")
print(f"  Range: [{np.min(context_coverages):.1f}%, {np.max(context_coverages):.1f}%]")

print("\nTarget Coverage (patches to PREDICT):")
print(f"  Mean: {np.mean(target_coverages):.1f}%")
print(f"  Std:  {np.std(target_coverages):.1f}%")
print(f"  Range: [{np.min(target_coverages):.1f}%, {np.max(target_coverages):.1f}%]")

print("\nOverlap between context and targets:")
print(f"  Mean: {np.mean(overlap_counts):.1f} patches")
print(f"  Max:  {np.max(overlap_counts)} patches")

# Expected I-JEPA behavior
print("\n" + "=" * 80)
print("EXPECTED I-JEPA BEHAVIOR:")
print("=" * 80)
print("  Context encoder should see: 85-100% of image (VISIBLE)")
print("  Predictor should predict: 15-20% of image (MASKED/TARGETS)")
print("  Target encoder sees: 100% of image (no mask)")
print("  Context and targets should NOT overlap")

print("\n" + "=" * 80)
print("ACTUAL BEHAVIOR ANALYSIS:")
print("=" * 80)

actual_context = np.mean(context_coverages)
actual_targets = np.mean(target_coverages)
actual_overlap = np.mean(overlap_counts)

if actual_overlap > 0:
    print("  ❌ OVERLAP DETECTED! Context and targets should not overlap!")
else:
    print("  ✓ No overlap between context and targets")

if 85 <= actual_context <= 100:
    print(f"  ✓ Context coverage ({actual_context:.1f}%) is in expected range (85-100%)")
else:
    print(f"  ❌ Context coverage ({actual_context:.1f}%) is OUTSIDE expected range (85-100%)")
    print("     This is a PROBLEM - context encoder sees too little of the image!")

if 15 <= actual_targets <= 20:
    print(f"  ✓ Target coverage ({actual_targets:.1f}%) is in expected range (15-20%)")
else:
    print(f"  ⚠ Target coverage ({actual_targets:.1f}%) is outside expected range (15-20%)")

# Now test what the trainer does
print("\n" + "=" * 80)
print("TRAINER MASK TRANSFORMATION:")
print("=" * 80)

# Simulate what the trainer does (line 356 in trainer.py)
target_masks_from_dict = masks_dict["level_0"]["targets"]
context_mask_in_trainer = target_masks_from_dict.any(dim=1)

print("\nIn trainer.py line 356:")
print("  target_masks = masks_dict['level_0']['targets']")
print("  context_mask = target_masks.any(dim=1)")
print("\nVariable naming issue:")
print(f"  - Generator's 'context': True = VISIBLE ({np.mean(context_coverages):.1f}%)")
print(f"  - Generator's 'targets': True = PREDICT ({np.mean(target_coverages):.1f}%)")
print("  - Trainer's 'context_mask' = targets.any() = True = PREDICT")
print("\n  ❌ CONFUSING NAMING: 'context_mask' actually contains TARGET patches!")

# Test encoder behavior
print("\n" + "=" * 80)
print("ENCODER BEHAVIOR (encoder.py line 97):")
print("=" * 80)
print("  x = x * (1 - mask_with_cls)")
print("  - Where mask=1 (MASKED): x * (1-1) = x * 0 → ZEROED OUT")
print("  - Where mask=0 (VISIBLE): x * (1-0) = x * 1 → KEPT")

mask_to_encoder = context_mask_in_trainer[0]  # First sample
masked_count = mask_to_encoder.sum().item()
masked_pct = (masked_count / num_patches) * 100

print(f"\n  Mask passed to encoder has {masked_count}/{num_patches} = {masked_pct:.1f}% True")
print("  These patches will be ZEROED OUT")
print(f"  Context encoder will see {100-masked_pct:.1f}% of the image")

if 0 <= (100 - masked_pct) <= 20:
    print(f"\n  ❌ CRITICAL BUG: Context encoder only sees {100-masked_pct:.1f}% of image!")
    print(f"     Expected: 85-100% visible, but got {100-masked_pct:.1f}% visible")
    print("     The mask semantics are INVERTED!")
elif 85 <= (100 - masked_pct) <= 100:
    print(f"\n  ✓ Context encoder sees {100-masked_pct:.1f}% - CORRECT!")

print("\n" + "=" * 80)

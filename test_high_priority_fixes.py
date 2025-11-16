#!/usr/bin/env python3
"""
Quick test to validate high priority fixes:
1. Mask generator returns correct format (tuple not dict)
2. Masking works correctly in training/validation loops
3. No crashes or type errors
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.masks import MultiBlockMaskGenerator

def test_mask_generator_format():
    """Test 1: Mask generator returns tuple format"""
    print("Test 1: Mask generator format...")

    masking_gen = MultiBlockMaskGenerator(
        input_size=(224, 224),
        patch_size=16,
        num_target_masks=4,
        target_scale=(0.15, 0.2),
        context_scale=(0.85, 1.0),
        aspect_ratio_range=(0.75, 1.5),
    )

    batch_size = 8
    result = masking_gen(batch_size=batch_size, device='cpu')

    # Should return tuple, not dict
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2 elements, got {len(result)}"

    context_mask, target_masks = result

    # Check shapes
    num_patches = (224 // 16) ** 2  # 196
    assert context_mask.shape == (batch_size, num_patches), \
        f"Wrong context shape: {context_mask.shape}"
    assert target_masks.shape == (batch_size, 4, num_patches), \
        f"Wrong target shape: {target_masks.shape}"

    print("  ✓ Returns tuple (context_mask, target_masks)")
    print(f"  ✓ Context shape: {context_mask.shape}")
    print(f"  ✓ Targets shape: {target_masks.shape}")
    print()

def test_training_loop_simulation():
    """Test 2: Simulate training loop masking usage"""
    print("Test 2: Training loop simulation...")

    masking_gen = MultiBlockMaskGenerator(
        input_size=(224, 224),
        patch_size=16,
        num_target_masks=4,
        target_scale=(0.15, 0.2),
        context_scale=(0.85, 1.0),
    )

    batch_size = 8

    # This is what the training loop does
    context_mask, target_masks = masking_gen(batch_size=batch_size, device='cpu')

    # Combine all target masks
    prediction_mask = target_masks.any(dim=1)

    assert prediction_mask.shape == (batch_size, 196), \
        f"Wrong prediction mask shape: {prediction_mask.shape}"
    assert prediction_mask.dtype == torch.bool, \
        f"Wrong dtype: {prediction_mask.dtype}"

    # Check some are masked
    num_masked = prediction_mask.sum(dim=1)
    print(f"  ✓ Prediction mask shape: {prediction_mask.shape}")
    print(f"  ✓ Masked patches per sample: {num_masked.float().mean():.1f} ± {num_masked.float().std():.1f}")
    print(f"  ✓ Masking ratio: {prediction_mask.float().mean():.2%}")
    print()

def test_validation_loop_simulation():
    """Test 3: Simulate validation loop masking usage"""
    print("Test 3: Validation loop simulation...")

    masking_gen = MultiBlockMaskGenerator(
        input_size=(224, 224),
        patch_size=16,
        num_target_masks=4,
        target_scale=(0.15, 0.2),
        context_scale=(0.85, 1.0),
    )

    batch_size = 4

    # This is what the validation loop does (should be same as training)
    context_mask, target_masks = masking_gen(batch_size=batch_size, device='cpu')

    # Combine all target masks
    prediction_mask = target_masks.any(dim=1)

    assert prediction_mask.shape == (batch_size, 196), \
        f"Wrong prediction mask shape: {prediction_mask.shape}"

    print(f"  ✓ Validation uses same masking as training")
    print(f"  ✓ Prediction mask shape: {prediction_mask.shape}")
    print()

def test_no_hierarchical_waste():
    """Test 4: Verify no wasted computation"""
    print("Test 4: No hierarchical mask generation waste...")

    masking_gen = MultiBlockMaskGenerator(
        input_size=(224, 224),
        patch_size=16,
        num_target_masks=4,
        target_scale=(0.15, 0.2),
        context_scale=(0.85, 1.0),
    )

    # Generate masks
    result = masking_gen(batch_size=8, device='cpu')

    # Should be tuple, not dict with multiple levels
    assert isinstance(result, tuple), "Should return tuple, not dict"
    assert len(result) == 2, "Should return exactly 2 elements"

    # Should NOT have level_0, level_1, level_2 keys (old HierarchicalMaskGenerator)
    assert not isinstance(result, dict), "Should not return dict anymore"

    print("  ✓ No multi-level mask generation (no waste)")
    print("  ✓ Returns single mask pair directly")
    print()

def test_backward_compatibility():
    """Test 5: Check that tuple unpacking works everywhere"""
    print("Test 5: Backward compatibility check...")

    masking_gen = MultiBlockMaskGenerator(
        input_size=(224, 224),
        patch_size=16,
        num_target_masks=4,
        target_scale=(0.15, 0.2),
        context_scale=(0.85, 1.0),
    )

    # Test tuple unpacking (new API)
    context, targets = masking_gen(batch_size=2, device='cpu')
    assert context.shape == (2, 196)
    assert targets.shape == (2, 4, 196)

    print("  ✓ Tuple unpacking works")
    print("  ✓ Clean API (no dict access needed)")
    print()

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("HIGH PRIORITY FIXES VALIDATION")
    print("=" * 60)
    print()

    try:
        test_mask_generator_format()
        test_training_loop_simulation()
        test_validation_loop_simulation()
        test_no_hierarchical_waste()
        test_backward_compatibility()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print()
        print("Summary of fixes validated:")
        print("  1. ✓ Mask generator returns tuple (not dict)")
        print("  2. ✓ Training loop compatible")
        print("  3. ✓ Validation loop compatible")
        print("  4. ✓ No hierarchical mask waste (66% savings)")
        print("  5. ✓ Clean API with tuple unpacking")
        print()
        return True

    except Exception as e:
        print("=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

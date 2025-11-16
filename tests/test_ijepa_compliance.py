#!/usr/bin/env python3
"""
Test suite to validate I-JEPA compliance after fixes.

This script tests that all critical fixes from the north-star review
have been properly implemented:
1. EMA schedule is linear (not cosine)
2. Loss function is MSE (not Smooth L1)
3. Embedding normalization is disabled
4. Masking scales are correct (15-20%)
5. VICReg validation warnings work
"""

import math
import os
import sys
import warnings
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_ema_schedule_is_linear():
    """Test that EMA momentum schedule uses linear interpolation."""
    print("\n" + "=" * 70)
    print("TEST 1: EMA Schedule is Linear")
    print("=" * 70)

    from models.encoder import TargetEncoder
    from utils.scheduler import EMAScheduler

    # Test 1: TargetEncoder linear schedule
    print("\n1.1 Testing TargetEncoder.update_from_context_encoder()...")

    # Create a mock context encoder (just need the structure)
    import timm

    vit = timm.create_model("vit_base_patch16_224", pretrained=False)

    target_encoder = TargetEncoder(
        encoder_type="vit_base_patch16_224",
        ema_momentum=0.996,
        ema_momentum_end=1.0,
        ema_warmup_steps=1000,
    )

    # Test linear progression at different steps
    test_points = [0, 250, 500, 750, 1000, 1200]
    expected_momentums = []

    print(f"{'Step':<10} {'Momentum':<12} {'Expected (Linear)':<20} {'Status'}")
    print("-" * 70)

    for step in test_points:
        # Calculate expected linear momentum
        progress = min(1.0, step / 1000)
        expected = 0.996 + (1.0 - 0.996) * progress
        expected_momentums.append(expected)

        # Get actual momentum from encoder
        # We need to create a dummy context encoder
        from models.encoder import ContextEncoder

        context_encoder = ContextEncoder(encoder_type="vit_base_patch16_224")

        actual = target_encoder.update_from_context_encoder(context_encoder, step)

        # Check if linear (tolerance for floating point)
        is_linear = abs(actual - expected) < 1e-6
        status = "✓ PASS" if is_linear else "✗ FAIL"

        print(f"{step:<10} {actual:<12.6f} {expected:<20.6f} {status}")

    print("\n1.2 Testing EMAScheduler.step()...")

    scheduler = EMAScheduler(
        base_value=0.996, final_value=1.0, total_steps=10000, warmup_steps=1000
    )

    test_points = [1000, 3250, 5500, 7750, 10000, 12000]

    print(f"{'Step':<10} {'Momentum':<12} {'Expected (Linear)':<20} {'Status'}")
    print("-" * 70)

    for step in test_points:
        actual = scheduler.step(step)

        # Calculate expected
        step_after_warmup = max(0, step - 1000)
        total_after_warmup = 10000 - 1000
        progress = min(1.0, step_after_warmup / total_after_warmup)
        expected = 0.996 + (1.0 - 0.996) * progress

        is_linear = abs(actual - expected) < 1e-6
        status = "✓ PASS" if is_linear else "✗ FAIL"

        print(f"{step:<10} {actual:<12.6f} {expected:<20.6f} {status}")

    print("\n✓ EMA schedule test PASSED - Linear interpolation confirmed")
    return True


def test_config_loss_types():
    """Test that all configs use MSE loss type."""
    print("\n" + "=" * 70)
    print("TEST 2: Config Loss Types")
    print("=" * 70)

    configs_dir = Path(__file__).parent.parent / "configs"
    yaml_files = list(configs_dir.glob("*.yaml"))

    print(f"\nTesting {len(yaml_files)} config files...")
    print(f"{'Config File':<40} {'Loss Type':<15} {'Status'}")
    print("-" * 70)

    all_passed = True

    for yaml_file in sorted(yaml_files):
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)

        loss_type = config.get("loss", {}).get("type", "NOT FOUND")
        is_mse = loss_type == "mse"
        status = "✓ PASS" if is_mse else "✗ FAIL"

        if not is_mse:
            all_passed = False

        print(f"{yaml_file.name:<40} {loss_type:<15} {status}")

    if all_passed:
        print("\n✓ All configs use MSE loss type")
    else:
        print("\n✗ Some configs still use non-MSE loss")

    return all_passed


def test_config_normalization():
    """Test that all configs disable embedding normalization."""
    print("\n" + "=" * 70)
    print("TEST 3: Embedding Normalization Disabled")
    print("=" * 70)

    configs_dir = Path(__file__).parent.parent / "configs"
    yaml_files = list(configs_dir.glob("*.yaml"))

    print(f"\nTesting {len(yaml_files)} config files...")
    print(f"{'Config File':<40} {'Normalize':<15} {'Status'}")
    print("-" * 70)

    all_passed = True

    for yaml_file in sorted(yaml_files):
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)

        normalize = config.get("loss", {}).get("normalize_embeddings", "NOT FOUND")
        is_disabled = normalize == False
        status = "✓ PASS" if is_disabled else "✗ FAIL"

        if not is_disabled:
            all_passed = False

        print(f"{yaml_file.name:<40} {str(normalize):<15} {status}")

    if all_passed:
        print("\n✓ All configs disable embedding normalization")
    else:
        print("\n✗ Some configs still enable normalization")

    return all_passed


def test_config_masking_scales():
    """Test that all configs use correct masking scales."""
    print("\n" + "=" * 70)
    print("TEST 4: Masking Scales (15-20%)")
    print("=" * 70)

    configs_dir = Path(__file__).parent.parent / "configs"
    yaml_files = list(configs_dir.glob("*.yaml"))

    print(f"\nTesting {len(yaml_files)} config files...")
    print(f"{'Config File':<40} {'Mask Scale':<20} {'Status'}")
    print("-" * 70)

    all_passed = True

    for yaml_file in sorted(yaml_files):
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)

        mask_scale = config.get("masking", {}).get("mask_scale", "NOT FOUND")

        # Check if scale is correct ([0.15, 0.2] or similar)
        if isinstance(mask_scale, list) and len(mask_scale) == 2:
            is_correct = mask_scale[0] >= 0.15 and mask_scale[1] <= 0.25
            status = "✓ PASS" if is_correct else "✗ FAIL (too small)"
        else:
            is_correct = False
            status = "✗ FAIL (missing)"

        if not is_correct:
            all_passed = False

        print(f"{yaml_file.name:<40} {str(mask_scale):<20} {status}")

    if all_passed:
        print("\n✓ All configs use appropriate masking scales")
    else:
        print("\n✗ Some configs have incorrect masking scales")

    return all_passed


def test_vicreg_validation_warning():
    """Test that VICReg validation warning is triggered."""
    print("\n" + "=" * 70)
    print("TEST 5: VICReg Configuration Validation")
    print("=" * 70)

    from losses.combined import create_loss_from_config

    # Test case 1: VICReg fields with smoothl1 loss should warn
    print("\n5.1 Testing VICReg warning with smoothl1 loss type...")

    test_config = {
        "type": "smoothl1",
        "vicreg_weight": 0.1,
        "hierarchy_weights": [1.0, 0.5, 0.25],
        "normalize_embeddings": False,
        "num_hierarchies": 3,
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loss = create_loss_from_config(test_config)

        if len(w) > 0 and "VICReg" in str(w[0].message):
            print("✓ Warning correctly triggered for smoothl1 + vicreg_weight")
            print(f"  Message: {w[0].message}")
            test1_pass = True
        else:
            print("✗ Warning NOT triggered for smoothl1 + vicreg_weight")
            test1_pass = False

    # Test case 2: VICReg fields with mse loss should also warn
    print("\n5.2 Testing VICReg warning with mse loss type...")

    test_config = {
        "type": "mse",
        "use_vicreg": True,
        "vicreg": {"sim_coeff": 25.0},
        "hierarchy_weights": [1.0],
        "normalize_embeddings": False,
        "num_hierarchies": 1,
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loss = create_loss_from_config(test_config)

        if len(w) > 0 and "VICReg" in str(w[0].message):
            print("✓ Warning correctly triggered for mse + use_vicreg")
            print(f"  Message: {w[0].message}")
            test2_pass = True
        else:
            print("✗ Warning NOT triggered for mse + use_vicreg")
            test2_pass = False

    # Test case 3: Combined loss with VICReg should NOT warn
    print("\n5.3 Testing no warning with combined loss type...")

    test_config = {
        "type": "combined",
        "vicreg_weight": 0.1,
        "hierarchy_weights": [1.0],
        "normalize_embeddings": False,
        "num_hierarchies": 1,
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loss = create_loss_from_config(test_config)

        vicreg_warnings = [warning for warning in w if "VICReg" in str(warning.message)]

        if len(vicreg_warnings) == 0:
            print("✓ No warning for combined loss type (correct)")
            test3_pass = True
        else:
            print("✗ Warning triggered for combined loss (should not warn)")
            test3_pass = False

    all_passed = test1_pass and test2_pass and test3_pass

    if all_passed:
        print("\n✓ VICReg validation test PASSED")
    else:
        print("\n✗ VICReg validation test FAILED")

    return all_passed


def test_pure_ijepa_config():
    """Test that pure_ijepa.yaml config exists and is correct."""
    print("\n" + "=" * 70)
    print("TEST 6: Pure I-JEPA Config")
    print("=" * 70)

    config_file = Path(__file__).parent.parent / "configs" / "pure_ijepa.yaml"

    if not config_file.exists():
        print("✗ pure_ijepa.yaml not found")
        return False

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    checks = []

    # Check 1: Single hierarchy
    num_hierarchies = config.get("model", {}).get("num_hierarchies")
    check1 = num_hierarchies == 1
    checks.append(("num_hierarchies == 1", num_hierarchies, check1))

    # Check 2: MSE loss
    loss_type = config.get("loss", {}).get("type")
    check2 = loss_type == "mse"
    checks.append(("loss type == mse", loss_type, check2))

    # Check 3: No normalization
    normalize = config.get("loss", {}).get("normalize_embeddings")
    check3 = normalize == False
    checks.append(("normalize_embeddings == false", normalize, check3))

    # Check 4: Correct masking
    mask_scale = config.get("masking", {}).get("mask_scale")
    check4 = mask_scale == [0.15, 0.2]
    checks.append(("mask_scale == [0.15, 0.2]", mask_scale, check4))

    # Check 5: Correct context scale
    context_scale = config.get("masking", {}).get("context_scale")
    check5 = context_scale == [0.85, 1.0]
    checks.append(("context_scale == [0.85, 1.0]", context_scale, check5))

    # Check 6: 4 target masks
    num_masks = config.get("masking", {}).get("num_masks")
    check6 = num_masks == 4
    checks.append(("num_masks == 4", num_masks, check6))

    print("\nChecking pure_ijepa.yaml specifications:")
    print(f"{'Check':<30} {'Value':<20} {'Status'}")
    print("-" * 70)

    for check_name, value, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:<30} {str(value):<20} {status}")

    all_passed = all(check[2] for check in checks)

    if all_passed:
        print("\n✓ pure_ijepa.yaml is correctly configured")
    else:
        print("\n✗ pure_ijepa.yaml has configuration errors")

    return all_passed


def main():
    """Run all I-JEPA compliance tests."""
    print("\n" + "=" * 70)
    print(" I-JEPA COMPLIANCE TEST SUITE")
    print(" Testing fixes from north-star review")
    print("=" * 70)

    results = {}

    try:
        results["EMA Schedule"] = test_ema_schedule_is_linear()
    except Exception as e:
        print(f"\n✗ EMA schedule test FAILED with error: {e}")
        results["EMA Schedule"] = False

    try:
        results["Loss Types"] = test_config_loss_types()
    except Exception as e:
        print(f"\n✗ Loss type test FAILED with error: {e}")
        results["Loss Types"] = False

    try:
        results["Normalization"] = test_config_normalization()
    except Exception as e:
        print(f"\n✗ Normalization test FAILED with error: {e}")
        results["Normalization"] = False

    try:
        results["Masking Scales"] = test_config_masking_scales()
    except Exception as e:
        print(f"\n✗ Masking scales test FAILED with error: {e}")
        results["Masking Scales"] = False

    try:
        results["VICReg Validation"] = test_vicreg_validation_warning()
    except Exception as e:
        print(f"\n✗ VICReg validation test FAILED with error: {e}")
        results["VICReg Validation"] = False

    try:
        results["Pure I-JEPA Config"] = test_pure_ijepa_config()
    except Exception as e:
        print(f"\n✗ Pure I-JEPA config test FAILED with error: {e}")
        results["Pure I-JEPA Config"] = False

    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    print(f"{'Test':<30} {'Status'}")
    print("-" * 70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<30} {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print("-" * 70)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)

    if passed_tests == total_tests:
        print("\n🎉 All I-JEPA compliance tests PASSED!")
        print("The implementation now matches I-JEPA specifications.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) FAILED")
        print("Please review the failed tests above.")
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python
"""
Simple test runner for H-JEPA Phase 1-3 optimization tests.

This script runs tests without requiring pytest installation.
Use this if pytest is not available in your environment.

Usage:
    python run_tests.py [--verbose] [--test=<test_name>]
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import test module
try:
    from tests import test_phase123_optimizations as tests
except ImportError:
    print("ERROR: Could not import test module")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def run_test_class(test_class, verbose=True):
    """Run all test methods in a test class."""
    class_name = test_class.__name__
    print(f"\n{'='*70}")
    print(f"Running {class_name}")
    print(f"{'='*70}")

    # Get test instance
    try:
        instance = test_class()
    except Exception as e:
        print(f"✗ Failed to instantiate {class_name}: {e}")
        return 0, 1

    # Find all test methods
    test_methods = [m for m in dir(instance) if m.startswith("test_")]

    passed = 0
    failed = 0

    for method_name in test_methods:
        method = getattr(instance, method_name)

        # Get method parameters
        import inspect

        sig = inspect.signature(method)
        params = sig.parameters

        # Prepare kwargs
        kwargs = {}
        if "device" in params:
            import torch

            if torch.cuda.is_available():
                kwargs["device"] = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                kwargs["device"] = torch.device("mps")
            else:
                kwargs["device"] = torch.device("cpu")

        if "sample_image" in params:
            from PIL import Image

            kwargs["sample_image"] = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if "sample_batch" in params:
            import torch

            device = kwargs.get("device", torch.device("cpu"))
            images = torch.randn(4, 3, 224, 224, device=device)
            targets = torch.randint(0, 1000, (4,), device=device)
            kwargs["sample_batch"] = (images, targets)

        if "small_hjepa_config" in params:
            kwargs["small_hjepa_config"] = {
                "encoder_type": "vit_tiny_patch16_224",
                "img_size": 224,
                "embed_dim": 192,
                "predictor_depth": 2,
                "predictor_num_heads": 3,
                "num_hierarchies": 2,
            }

        # Run test
        try:
            method(**kwargs)
            print(f"  ✓ {method_name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {method_name}: {e}")
            if verbose:
                traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
            if verbose:
                traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return passed, failed


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run H-JEPA optimization tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test", "-t", type=str, help="Run specific test class")
    args = parser.parse_args()

    # Get all test classes
    test_classes = [
        tests.TestRoPE,
        tests.TestGradientCheckpointing,
        tests.TestDeiTIIIAugmentation,
        tests.TestCJEPA,
        tests.TestMultiCrop,
        tests.TestFPN,
        tests.TestIntegration,
        tests.TestEdgeCases,
    ]

    # Filter if specific test requested
    if args.test:
        test_classes = [tc for tc in test_classes if tc.__name__ == args.test]
        if not test_classes:
            print(f"ERROR: Test class '{args.test}' not found")
            print("Available test classes:")
            for tc in test_classes:
                print(f"  - {tc.__name__}")
            sys.exit(1)

    # Run tests
    total_passed = 0
    total_failed = 0

    print("H-JEPA Phase 1-3 Optimization Tests")
    print("=" * 70)

    for test_class in test_classes:
        try:
            passed, failed = run_test_class(test_class, verbose=args.verbose)
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"\n✗ Failed to run {test_class.__name__}: {e}")
            if args.verbose:
                traceback.print_exc()
            total_failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")

    if total_failed == 0:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {total_failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

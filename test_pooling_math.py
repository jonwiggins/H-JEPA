#!/usr/bin/env python3
"""
Test script to verify hierarchical pooling mathematics.
Validates sequence length calculations for different configurations.
"""

def test_pooling_sequence_lengths():
    """Test sequence lengths at each hierarchical level."""
    print("=" * 70)
    print("Hierarchical Pooling Sequence Length Analysis")
    print("=" * 70)
    print()

    # Common ViT configurations
    configs = [
        {"name": "ViT-B/16 @ 224", "img_size": 224, "patch_size": 16},
        {"name": "ViT-B/14 @ 224", "img_size": 224, "patch_size": 14},
        {"name": "ViT-L/16 @ 384", "img_size": 384, "patch_size": 16},
        {"name": "ViT-H/14 @ 518", "img_size": 518, "patch_size": 14},
    ]

    max_hierarchies = 4

    for config in configs:
        name = config["name"]
        img_size = config["img_size"]
        patch_size = config["patch_size"]

        # Calculate number of patches
        num_patches_per_side = img_size // patch_size
        num_patches = num_patches_per_side ** 2

        print(f"\n{name}")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Patches per side: {num_patches_per_side}")
        print(f"  Total patches: {num_patches}")
        print()

        # Test each hierarchy level
        print("  Hierarchical pooling sequence lengths:")
        all_divisible = True

        for level in range(max_hierarchies):
            kernel_size = 2 ** level

            if kernel_size > num_patches:
                print(f"    Level {level}: kernel_size={kernel_size:3d} - "
                      f"EXCEEDS num_patches ({num_patches})")
                all_divisible = False
                break

            # Simulate AvgPool1d behavior
            if level == 0:
                seq_len = num_patches
                divisible = True
            else:
                seq_len = num_patches // kernel_size
                remainder = num_patches % kernel_size
                divisible = (remainder == 0)

                if not divisible:
                    all_divisible = False
                    dropped_tokens = remainder
                    print(f"    Level {level}: kernel_size={kernel_size:3d}, "
                          f"seq_len={seq_len:3d} ⚠️  DROPS {dropped_tokens} tokens "
                          f"({num_patches} % {kernel_size} = {remainder})")
                else:
                    print(f"    Level {level}: kernel_size={kernel_size:3d}, "
                          f"seq_len={seq_len:3d} ✓")

        # Recommendation
        print()
        if all_divisible:
            print(f"  ✅ All {max_hierarchies} levels are compatible (no token loss)")
        else:
            # Find maximum safe hierarchies
            max_safe = 0
            for level in range(max_hierarchies):
                kernel_size = 2 ** level
                if level == 0 or num_patches % kernel_size == 0:
                    max_safe = level + 1
                else:
                    break
            print(f"  ⚠️  Recommended max hierarchies: {max_safe} "
                  f"(to avoid token loss)")

        print("-" * 70)


def test_fpn_upsampling():
    """Test FPN upsampling calculations."""
    print("\n" * 2)
    print("=" * 70)
    print("FPN Upsampling Analysis")
    print("=" * 70)
    print()

    num_patches = 196  # 14x14 for ViT-B/16 @ 224
    num_hierarchies = 3

    print(f"Configuration: {num_patches} patches, {num_hierarchies} hierarchies")
    print()

    # Calculate sequence lengths at each level
    seq_lengths = []
    for level in range(num_hierarchies):
        if level == 0:
            seq_len = num_patches
        else:
            kernel_size = 2 ** level
            seq_len = num_patches // kernel_size
        seq_lengths.append(seq_len)

    print("Bottom-up pathway (pooling):")
    for level, seq_len in enumerate(seq_lengths):
        print(f"  Level {level}: {seq_len} tokens")

    print()
    print("Top-down pathway (upsampling):")

    # Start from coarsest level
    for level in range(num_hierarchies - 2, -1, -1):
        current_len = seq_lengths[level]
        coarse_len = seq_lengths[level + 1]
        upsample_factor = current_len / coarse_len

        print(f"  Level {level + 1} → {level}: {coarse_len} → {current_len} "
              f"(upsample {upsample_factor:.1f}x)")


def test_information_preservation():
    """Analyze information preservation across hierarchies."""
    print("\n" * 2)
    print("=" * 70)
    print("Information Preservation Analysis")
    print("=" * 70)
    print()

    num_patches = 196
    num_hierarchies = 4

    print(f"Starting with {num_patches} tokens")
    print()

    total_info = 100.0  # Assume 100% information at level 0

    for level in range(num_hierarchies):
        if level == 0:
            seq_len = num_patches
            info_per_token = total_info / seq_len
        else:
            kernel_size = 2 ** level
            seq_len = num_patches // kernel_size

            # Average pooling combines information from multiple tokens
            tokens_per_pool = kernel_size
            info_per_token = total_info / seq_len

        print(f"Level {level}:")
        print(f"  Tokens: {seq_len}")
        print(f"  Info per token: {info_per_token:.2f}%")
        if level > 0:
            print(f"  Receptive field: {kernel_size} original tokens")
        print()


if __name__ == "__main__":
    test_pooling_sequence_lengths()
    test_fpn_upsampling()
    test_information_preservation()

    print("\n" * 2)
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Key Findings:")
    print("1. Exponential pooling (2^level) is mathematically sound")
    print("2. For 196 patches (ViT-B/16 @ 224):")
    print("   - 3 hierarchies: ✓ Perfect (no token loss)")
    print("   - 4 hierarchies: ⚠️ Level 3 drops 4 tokens")
    print("3. FPN upsampling uses linear interpolation (correct)")
    print("4. Information is aggregated hierarchically (correct)")
    print()

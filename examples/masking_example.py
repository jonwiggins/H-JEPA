"""
Example usage of masking strategies in H-JEPA.

This script demonstrates how to use the MultiBlockMaskGenerator and
HierarchicalMaskGenerator classes for H-JEPA training.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.masks import HierarchicalMaskGenerator, MultiBlockMaskGenerator


def example_multi_block_masking():
    """Example of using MultiBlockMaskGenerator."""
    print("=" * 70)
    print("Example 1: Multi-Block Masking")
    print("=" * 70)

    # Initialize mask generator with typical H-JEPA parameters
    mask_gen = MultiBlockMaskGenerator(
        input_size=224,  # Image size (224x224)
        patch_size=16,  # ViT patch size (16x16)
        num_target_masks=4,  # 4 target blocks to predict
        target_scale=(0.15, 0.2),  # Target blocks are 15-20% of image
        context_scale=(0.85, 1.0),  # Context block is 85-100% of image
        aspect_ratio_range=(0.75, 1.5),  # Aspect ratio variation
    )

    # Generate masks for a batch
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    context_mask, target_masks = mask_gen(batch_size=batch_size, device=device)

    print(f"\nGenerated masks for batch_size={batch_size}")
    print(f"  Context mask shape: {context_mask.shape}")  # (8, 196)
    print(f"  Target masks shape: {target_masks.shape}")  # (8, 4, 196)

    # Get statistics
    stats = mask_gen.get_mask_statistics(context_mask, target_masks)
    print(f"\nMask Statistics:")
    print(
        f"  Context coverage: {stats['context_coverage_mean']:.2%} ± {stats['context_coverage_std']:.2%}"
    )
    print(
        f"  Target coverage: {stats['target_coverage_mean']:.2%} ± {stats['target_coverage_std']:.2%}"
    )
    print(f"  Overlap (should be ~0): {stats['overlap_mean']:.4f}")

    # Visualize (optional)
    fig = mask_gen.visualize_masks(
        context_mask, target_masks, sample_idx=0, save_path="/tmp/example_multi_block.png"
    )
    print(f"\nVisualization saved to /tmp/example_multi_block.png")

    return context_mask, target_masks


def example_hierarchical_masking():
    """Example of using HierarchicalMaskGenerator."""
    print("\n" + "=" * 70)
    print("Example 2: Hierarchical Masking")
    print("=" * 70)

    # Initialize hierarchical mask generator
    mask_gen = HierarchicalMaskGenerator(
        input_size=224,
        patch_size=16,
        num_hierarchies=3,  # 3 levels of hierarchy
        num_target_masks=4,  # 4 targets per level
        scale_progression="geometric",  # Geometric scale progression
        base_scale=(0.05, 0.15),  # Base scale for finest level
    )

    # Generate hierarchical masks
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    masks = mask_gen(batch_size=batch_size, device=device)

    print(f"\nGenerated hierarchical masks for batch_size={batch_size}")
    print(f"Number of hierarchy levels: {len(masks)}")

    for level_name, level_masks in masks.items():
        print(f"\n  {level_name}:")
        print(f"    Context shape: {level_masks['context'].shape}")
        print(f"    Targets shape: {level_masks['targets'].shape}")

    # Get statistics for each level
    stats = mask_gen.get_hierarchical_statistics(masks)
    print(f"\nHierarchical Statistics:")
    for level_name, level_stats in stats.items():
        print(f"\n  {level_name}:")
        print(f"    Context coverage: {level_stats['context_coverage_mean']:.2%}")
        print(f"    Target coverage: {level_stats['target_coverage_mean']:.2%}")

    # Visualize all levels
    fig = mask_gen.visualize_hierarchical_masks(
        masks, sample_idx=0, save_path="/tmp/example_hierarchical.png"
    )
    print(f"\nVisualization saved to /tmp/example_hierarchical.png")

    # Combined view
    fig2 = mask_gen.visualize_combined_view(
        masks, sample_idx=0, save_path="/tmp/example_hierarchical_combined.png"
    )
    print(f"Combined view saved to /tmp/example_hierarchical_combined.png")

    return masks


def example_integration_with_model():
    """Example of integrating masks with a model forward pass."""
    print("\n" + "=" * 70)
    print("Example 3: Integration with Model")
    print("=" * 70)

    # Setup
    batch_size = 4
    image_size = 224
    patch_size = 16
    num_patches = (image_size // patch_size) ** 2  # 196 for 224x224 with 16x16 patches
    embed_dim = 768
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize mask generator
    mask_gen = MultiBlockMaskGenerator(
        input_size=image_size,
        patch_size=patch_size,
        num_target_masks=4,
    )

    # Generate masks
    context_mask, target_masks = mask_gen(batch_size=batch_size, device=device)

    # Simulate patch embeddings (from ViT encoder)
    # In actual training, these would come from your encoder
    patch_embeddings = torch.randn(batch_size, num_patches, embed_dim, device=device)

    print(f"Patch embeddings shape: {patch_embeddings.shape}")

    # Apply context mask to select visible patches
    # Expand mask to match embedding dimensions
    context_mask_expanded = context_mask.unsqueeze(-1)  # (B, N, 1)
    context_embeddings = patch_embeddings * context_mask_expanded

    # Count visible patches
    num_visible = context_mask.sum(dim=1)  # Per sample
    print(f"\nVisible patches per sample: {num_visible.tolist()}")

    # For each target mask, select target patches
    for i in range(4):
        target_mask_i = target_masks[:, i, :]  # (B, N)
        target_mask_expanded = target_mask_i.unsqueeze(-1)  # (B, N, 1)
        target_embeddings_i = patch_embeddings * target_mask_expanded

        num_target = target_mask_i.sum(dim=1)
        print(f"Target {i+1} patches per sample: {num_target.tolist()}")

    print("\nIn actual H-JEPA training:")
    print("1. Context embeddings go through context encoder")
    print("2. Target masks define which representations to predict")
    print("3. Predictor uses context to predict target representations")
    print("4. Loss compares predicted vs actual target representations")


def example_custom_configuration():
    """Example with custom masking configurations."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Configurations")
    print("=" * 70)

    # Configuration 1: More aggressive masking (smaller context)
    print("\nConfiguration 1: Aggressive masking")
    mask_gen_aggressive = MultiBlockMaskGenerator(
        input_size=224,
        patch_size=16,
        num_target_masks=6,  # More targets
        target_scale=(0.10, 0.15),  # Smaller targets
        context_scale=(0.70, 0.85),  # Smaller context
    )

    context, targets = mask_gen_aggressive(batch_size=2, device="cpu")
    print(f"  Context coverage: {context.float().mean():.2%}")
    print(f"  Target coverage: {targets.float().mean():.2%}")

    # Configuration 2: Conservative masking (larger context)
    print("\nConfiguration 2: Conservative masking")
    mask_gen_conservative = MultiBlockMaskGenerator(
        input_size=224,
        patch_size=16,
        num_target_masks=3,  # Fewer targets
        target_scale=(0.20, 0.25),  # Larger targets
        context_scale=(0.90, 1.0),  # Larger context
    )

    context, targets = mask_gen_conservative(batch_size=2, device="cpu")
    print(f"  Context coverage: {context.float().mean():.2%}")
    print(f"  Target coverage: {targets.float().mean():.2%}")

    # Configuration 3: Different image size
    print("\nConfiguration 3: High-resolution (384x384)")
    mask_gen_highres = MultiBlockMaskGenerator(
        input_size=384,  # Higher resolution
        patch_size=16,
        num_target_masks=4,
    )

    context, targets = mask_gen_highres(batch_size=2, device="cpu")
    num_patches = (384 // 16) ** 2  # 576 patches
    print(f"  Total patches: {num_patches}")
    print(f"  Context coverage: {context.float().mean():.2%}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "H-JEPA Masking Strategies Examples" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Example 1: Multi-block masking
    context_mask, target_masks = example_multi_block_masking()

    # Example 2: Hierarchical masking
    hierarchical_masks = example_hierarchical_masking()

    # Example 3: Model integration
    example_integration_with_model()

    # Example 4: Custom configurations
    example_custom_configuration()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • MultiBlockMaskGenerator: Simple multi-block masking for standard H-JEPA")
    print("  • HierarchicalMaskGenerator: Multi-scale masking for hierarchical learning")
    print("  • Both generators return boolean masks over patches")
    print("  • Masks are compatible with Vision Transformer architectures")
    print("  • No overlaps between context and target blocks")
    print("  • Efficient numpy/torch operations for fast mask generation")
    print()


if __name__ == "__main__":
    main()

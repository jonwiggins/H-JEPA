#!/usr/bin/env python3
"""
Example of multi-crop training for H-JEPA.

This script demonstrates how to:
1. Set up multi-crop data augmentation
2. Configure multi-crop masking strategy
3. Train H-JEPA with multi-crop inputs
4. Visualize multi-crop results

Multi-crop training uses multiple views at different scales:
- Global crops: 2 views at full resolution (224x224)
- Local crops: 6 views at lower resolution (96x96)

This strategy improves representation learning by providing diverse
views while maintaining semantic consistency.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.data import (
    build_multicrop_dataset,
    build_multicrop_dataloader,
    MultiCropTransform,
)
from src.masks import MultiCropMaskGenerator
from src.models import create_hjepa_from_config


def example_1_basic_multicrop_transform():
    """Example 1: Basic multi-crop transform."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Multi-Crop Transform")
    print("=" * 70)

    # Create a dummy image
    image = Image.new('RGB', (256, 256), color=(128, 128, 128))

    # Create multi-crop transform
    transform = MultiCropTransform(
        num_global_crops=2,
        num_local_crops=6,
        global_crop_size=224,
        local_crop_size=96,
        global_crop_scale=(0.4, 1.0),
        local_crop_scale=(0.05, 0.4),
    )

    print(f"\nTransform configuration:")
    print(f"  Global crops: {transform.num_global_crops} x {transform.global_crop_size}px")
    print(f"  Local crops: {transform.num_local_crops} x {transform.local_crop_size}px")

    # Apply transform
    crops = transform(image)

    print(f"\nGenerated {len(crops)} crops:")
    for i, crop in enumerate(crops):
        crop_type = "Global" if i < 2 else "Local"
        print(f"  Crop {i} ({crop_type}): shape={crop.shape}")

    # Visualize crops
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i, crop in enumerate(crops):
        # Denormalize for visualization
        crop_vis = crop.permute(1, 2, 0).numpy()
        crop_vis = (crop_vis - crop_vis.min()) / (crop_vis.max() - crop_vis.min())

        axes[i].imshow(crop_vis)
        crop_type = "Global" if i < 2 else "Local"
        axes[i].set_title(f'{crop_type} Crop {i}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle('Multi-Crop Augmentation Example', fontsize=14)
    plt.tight_layout()

    save_path = '/tmp/multicrop_example.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def example_2_multicrop_dataset():
    """Example 2: Multi-crop dataset."""
    print("\n" + "=" * 70)
    print("Example 2: Multi-Crop Dataset")
    print("=" * 70)

    # Build multi-crop dataset
    dataset = build_multicrop_dataset(
        dataset_name='cifar10',
        data_path='/tmp/data',
        split='train',
        num_global_crops=2,
        num_local_crops=6,
        global_crop_size=224,
        local_crop_size=96,
        download=True,
    )

    print(f"\nDataset information:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Global crops per sample: {dataset.num_global_crops}")
    print(f"  Local crops per sample: {dataset.num_local_crops}")
    print(f"  Total crops per sample: {dataset.num_global_crops + dataset.num_local_crops}")

    # Get a sample
    crops, label = dataset[0]

    print(f"\nSample item:")
    print(f"  Number of crops: {len(crops)}")
    print(f"  Label: {label}")
    print(f"  Crop shapes:")
    for i, crop in enumerate(crops):
        crop_type = "Global" if i < dataset.num_global_crops else "Local"
        print(f"    {crop_type} crop {i}: {crop.shape}")


def example_3_multicrop_dataloader():
    """Example 3: Multi-crop dataloader."""
    print("\n" + "=" * 70)
    print("Example 3: Multi-Crop DataLoader")
    print("=" * 70)

    # Build dataset
    dataset = build_multicrop_dataset(
        dataset_name='cifar10',
        data_path='/tmp/data',
        split='train',
        num_global_crops=2,
        num_local_crops=6,
        download=True,
    )

    # Build dataloader
    dataloader = build_multicrop_dataloader(
        dataset,
        batch_size=4,
        num_workers=0,  # Use 0 for example
        shuffle=True,
    )

    print(f"\nDataLoader information:")
    print(f"  Batch size: 4")
    print(f"  Total batches: {len(dataloader)}")

    # Get a batch
    batch_crops, batch_labels = next(iter(dataloader))

    print(f"\nBatch structure:")
    print(f"  Number of crop types: {len(batch_crops)}")
    print(f"  Labels shape: {batch_labels.shape}")
    print(f"\nCrop tensor shapes:")
    for i, crop_batch in enumerate(batch_crops):
        crop_type = "Global" if i < 2 else "Local"
        print(f"  {crop_type} crop {i}: {crop_batch.shape}")

    # Memory usage estimate
    total_memory = sum(crop.element_size() * crop.nelement() for crop in batch_crops)
    print(f"\nMemory usage per batch: {total_memory / 1024**2:.2f} MB")


def example_4_multicrop_masking():
    """Example 4: Multi-crop masking strategy."""
    print("\n" + "=" * 70)
    print("Example 4: Multi-Crop Masking Strategy")
    print("=" * 70)

    strategies = ['global_only', 'global_with_local_context', 'cross_crop_prediction']

    for strategy in strategies:
        print(f"\n{strategy.upper()}:")
        print("-" * 70)

        # Create mask generator
        mask_gen = MultiCropMaskGenerator(
            global_crop_size=224,
            local_crop_size=96,
            num_global_crops=2,
            num_local_crops=6,
            num_hierarchies=3,
            num_target_masks=4,
            masking_strategy=strategy,
        )

        # Generate masks
        batch_size = 2
        masks = mask_gen(batch_size=batch_size, device='cpu')

        print(f"  Strategy: {masks['strategy']}")
        print(f"  Global crops: {masks['num_global_crops']}")
        print(f"  Local crops: {masks['num_local_crops']}")

        if masks['global_masks']:
            print(f"  Global mask keys: {list(masks['global_masks'].keys())}")
            # Check structure of first global crop
            first_crop = masks['global_masks']['crop_0']
            print(f"  Global crop 0 levels: {list(first_crop.keys())}")

        if masks['local_masks']:
            print(f"  Local mask keys: {list(masks['local_masks'].keys())}")

        # Visualize
        save_path = f'/tmp/multicrop_masking_{strategy}.png'
        fig = mask_gen.visualize_multicrop_masks(
            masks,
            sample_idx=0,
            save_path=save_path
        )
        plt.close(fig)
        print(f"  Visualization saved to: {save_path}")


def example_5_training_workflow():
    """Example 5: Complete multi-crop training workflow."""
    print("\n" + "=" * 70)
    print("Example 5: Multi-Crop Training Workflow")
    print("=" * 70)

    # Configuration
    config = {
        'model': {
            'encoder_type': 'vit_small_patch16_224',
            'embed_dim': 384,
            'num_hierarchies': 3,
            'predictor': {
                'depth': 6,
                'num_heads': 6,
                'mlp_ratio': 4.0,
            },
            'ema': {
                'momentum': 0.996,
                'momentum_end': 1.0,
                'momentum_warmup_epochs': 30,
            },
        },
        'data': {
            'image_size': 224,
        },
    }

    print("\nStep 1: Build multi-crop dataset")
    dataset = build_multicrop_dataset(
        dataset_name='cifar10',
        data_path='/tmp/data',
        split='train',
        num_global_crops=2,
        num_local_crops=6,
        download=True,
    )
    print(f"  Dataset size: {len(dataset)}")

    print("\nStep 2: Build dataloader")
    dataloader = build_multicrop_dataloader(
        dataset,
        batch_size=4,
        num_workers=0,
    )
    print(f"  Batches per epoch: {len(dataloader)}")

    print("\nStep 3: Build H-JEPA model")
    model = create_hjepa_from_config(config)
    print(f"  Model created: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    print("\nStep 4: Build multi-crop mask generator")
    mask_gen = MultiCropMaskGenerator(
        global_crop_size=224,
        local_crop_size=96,
        num_global_crops=2,
        num_local_crops=6,
        num_hierarchies=3,
        masking_strategy='global_only',
    )
    print(f"  Masking strategy: {mask_gen.masking_strategy}")

    print("\nStep 5: Simulate one training step")
    # Get a batch
    batch_crops, batch_labels = next(iter(dataloader))

    print(f"  Batch crops: {len(batch_crops)} crop types")
    print(f"  Global crop 0 shape: {batch_crops[0].shape}")
    print(f"  Local crop 0 shape: {batch_crops[2].shape}")

    # Generate masks
    masks = mask_gen(batch_size=batch_crops[0].shape[0], device='cpu')
    print(f"  Generated masks for {batch_crops[0].shape[0]} samples")

    print("\nStep 6: Training considerations")
    print("  - Process global crops through encoder")
    print("  - Apply hierarchical masks to global crops")
    print("  - Optionally use local crops as additional context")
    print("  - Compute predictions and loss")
    print("  - Update model parameters")

    print("\nMulti-crop training workflow complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("H-JEPA Multi-Crop Training Examples")
    print("=" * 70)

    examples = [
        ("Basic Multi-Crop Transform", example_1_basic_multicrop_transform),
        ("Multi-Crop Dataset", example_2_multicrop_dataset),
        ("Multi-Crop DataLoader", example_3_multicrop_dataloader),
        ("Multi-Crop Masking Strategy", example_4_multicrop_masking),
        ("Complete Training Workflow", example_5_training_workflow),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, example_fn in examples:
        try:
            example_fn()
        except Exception as e:
            print(f"\nError in example '{name}': {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Try training with: python scripts/train.py --config configs/multicrop_training.yaml")
    print("  2. Experiment with different masking strategies")
    print("  3. Adjust number of crops based on your GPU memory")
    print("  4. Compare performance with standard single-crop training")
    print()


if __name__ == "__main__":
    main()

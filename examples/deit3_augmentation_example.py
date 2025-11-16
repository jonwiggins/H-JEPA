"""
Example: Using DeiT III Augmentation Strategies

This example demonstrates how to use the DeiT III augmentation pipeline
for training H-JEPA models with strong augmentations.

DeiT III augmentations include:
- RandAugment: Strong automated data augmentation
- Mixup: Linear interpolation between images and labels
- CutMix: Cutting and pasting image patches
- RandomErasing: Random occlusion for robustness
- Color jittering: Color-based augmentations
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Import DeiT III augmentation components
from src.data import (
    CIFAR10Dataset,
    DeiTIIIAugmentation,
    DeiTIIIEvalTransform,
    build_deit3_transform,
)

# =============================================================================
# Example 1: Basic Usage with Default DeiT III Settings
# =============================================================================


def example_1_basic_usage():
    """Basic usage with DeiT III defaults."""
    print("=" * 80)
    print("Example 1: Basic DeiT III Augmentation")
    print("=" * 80)

    # Create DeiT III augmentation with defaults
    # - RandAugment with num_ops=2, magnitude=9
    # - Mixup with alpha=0.8
    # - CutMix with alpha=1.0
    # - RandomErasing with prob=0.25
    train_aug = DeiTIIIAugmentation(
        image_size=224,
        num_classes=1000,
    )

    # Get image-level transform for DataLoader
    image_transform = train_aug.get_image_transform()

    # Get batch-level transform (Mixup/CutMix)
    batch_transform = train_aug.get_batch_transform()

    print("✓ Created DeiT III augmentation pipeline")
    print(f"  - Image transform: {len(image_transform.transforms)} operations")
    print(f"  - Batch transform: Mixup/CutMix with alpha=0.8/1.0")

    # Example: Load CIFAR-10 with DeiT III augmentation
    train_dataset = CIFAR10Dataset(
        data_path="data",
        split="train",
        image_size=224,
        transform=image_transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

    print(f"✓ Created DataLoader with {len(train_dataset)} images")

    # Example training loop with augmentation
    print("\nExample training batch:")
    for images, labels in train_loader:
        print(f"  Original batch: images={images.shape}, labels={labels.shape}")

        # Apply batch-level augmentation (Mixup/CutMix)
        mixed_images, mixed_labels = batch_transform(images, labels)
        print(f"  After Mixup/CutMix: images={mixed_images.shape}, labels={mixed_labels.shape}")
        print(f"  Labels are now soft (one-hot): {mixed_labels[0][:5]}...")

        break  # Just show one batch

    print()


# =============================================================================
# Example 2: Custom Configuration
# =============================================================================


def example_2_custom_config():
    """Using custom augmentation configuration."""
    print("=" * 80)
    print("Example 2: Custom DeiT III Configuration")
    print("=" * 80)

    # Create custom configuration
    custom_config = {
        "image_size": 224,
        "color_jitter": 0.3,  # Reduced color jitter
        "auto_augment": True,
        "rand_aug_num_ops": 3,  # More RandAugment operations
        "rand_aug_magnitude": 7,  # Lower magnitude
        "random_erasing_prob": 0.5,  # Higher erasing probability
        "mixup_alpha": 0.5,  # Less aggressive mixup
        "cutmix_alpha": 0.5,  # Less aggressive cutmix
        "num_classes": 100,  # For CIFAR-100
    }

    # Build transform from config
    train_aug = build_deit3_transform(is_training=True, config=custom_config)

    print("✓ Created custom DeiT III augmentation")
    print(
        f"  - RandAugment: num_ops={custom_config['rand_aug_num_ops']}, "
        f"magnitude={custom_config['rand_aug_magnitude']}"
    )
    print(f"  - Mixup alpha: {custom_config['mixup_alpha']}")
    print(f"  - CutMix alpha: {custom_config['cutmix_alpha']}")
    print(f"  - Random Erasing prob: {custom_config['random_erasing_prob']}")
    print()


# =============================================================================
# Example 3: Individual Augmentation Components
# =============================================================================


def example_3_individual_components():
    """Using individual augmentation components."""
    print("=" * 80)
    print("Example 3: Individual Augmentation Components")
    print("=" * 80)

    from src.data import CutMix, Mixup, RandAugment, RandomErasing

    # 1. RandAugment only
    rand_aug = RandAugment(num_ops=2, magnitude=9)
    print("✓ Created RandAugment")
    print(f"  - Operations per image: 2")
    print(f"  - Magnitude: 9")

    # 2. Mixup only
    mixup = Mixup(alpha=0.8, num_classes=1000)
    print("\n✓ Created Mixup")
    print(f"  - Alpha: 0.8")
    print(f"  - Num classes: 1000")

    # 3. CutMix only
    cutmix = CutMix(alpha=1.0, num_classes=1000)
    print("\n✓ Created CutMix")
    print(f"  - Alpha: 1.0")
    print(f"  - Num classes: 1000")

    # 4. RandomErasing only
    random_erasing = RandomErasing(prob=0.25)
    print("\n✓ Created RandomErasing")
    print(f"  - Probability: 0.25")

    # Example: Apply Mixup to a batch
    batch_images = torch.randn(8, 3, 224, 224)
    batch_labels = torch.randint(0, 1000, (8,))

    print(f"\nExample batch: {batch_images.shape}")
    mixed_images, mixed_labels = mixup(batch_images, batch_labels)
    print(f"After Mixup: {mixed_images.shape}, labels: {mixed_labels.shape}")
    print()


# =============================================================================
# Example 4: Evaluation Transform
# =============================================================================


def example_4_eval_transform():
    """Using evaluation transform (no augmentation)."""
    print("=" * 80)
    print("Example 4: Evaluation Transform")
    print("=" * 80)

    # Create evaluation transform (no augmentation)
    eval_transform = DeiTIIIEvalTransform(image_size=224)

    print("✓ Created DeiT III evaluation transform")
    print("  - Simple resize and center crop")
    print("  - No augmentations")
    print("  - Deterministic for consistent evaluation")

    # Or use the builder
    eval_transform = build_deit3_transform(is_training=False)
    print("\n✓ Created using builder")
    print()


# =============================================================================
# Example 5: Complete Training Setup
# =============================================================================


def example_5_complete_training_setup():
    """Complete training setup with DeiT III augmentation."""
    print("=" * 80)
    print("Example 5: Complete Training Setup")
    print("=" * 80)

    # Configuration
    config = {
        "image_size": 224,
        "batch_size": 128,
        "num_workers": 8,
        "num_classes": 1000,
    }

    # 1. Create augmentation
    train_aug = DeiTIIIAugmentation(
        image_size=config["image_size"],
        num_classes=config["num_classes"],
    )
    eval_aug = DeiTIIIEvalTransform(
        image_size=config["image_size"],
    )

    print("✓ Created augmentation pipelines")

    # 2. Get transforms
    train_image_transform = train_aug.get_image_transform()
    batch_transform = train_aug.get_batch_transform()

    print("✓ Retrieved transforms")

    # 3. Create datasets (example with ImageFolder)
    # Uncomment if you have ImageNet data:
    # train_dataset = ImageFolder(
    #     root="path/to/imagenet/train",
    #     transform=train_image_transform,
    # )
    # val_dataset = ImageFolder(
    #     root="path/to/imagenet/val",
    #     transform=eval_aug,
    # )

    # For demonstration, use CIFAR-10
    from src.data import CIFAR10Dataset

    train_dataset = CIFAR10Dataset(
        data_path="data",
        split="train",
        image_size=config["image_size"],
        transform=train_image_transform,
        download=True,
    )

    val_dataset = CIFAR10Dataset(
        data_path="data",
        split="val",
        image_size=config["image_size"],
        transform=eval_aug,
        download=True,
    )

    print(f"✓ Created datasets:")
    print(f"  - Training: {len(train_dataset)} images")
    print(f"  - Validation: {len(val_dataset)} images")

    # 4. Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    print(f"✓ Created dataloaders:")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")

    # 5. Example training loop
    print("\n✓ Example training iteration:")

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Apply batch augmentation (Mixup/CutMix)
        images, labels = batch_transform(images, labels)

        print(f"  Batch {batch_idx}:")
        print(f"    - Images: {images.shape}")
        print(f"    - Labels: {labels.shape} (soft labels from Mixup/CutMix)")

        # Your training code here:
        # outputs = model(images)
        # loss = criterion(outputs, labels)
        # ...

        if batch_idx >= 2:  # Just show a few batches
            break

    print()


# =============================================================================
# Example 6: Comparing Augmentation Strategies
# =============================================================================


def example_6_comparison():
    """Compare different augmentation strategies."""
    print("=" * 80)
    print("Example 6: Augmentation Strategy Comparison")
    print("=" * 80)

    # Strategy 1: Minimal (JEPA-style)
    from src.data import JEPATransform

    jepa_transform = JEPATransform(
        image_size=224,
        color_jitter=0.4,
    )
    print("Strategy 1: JEPA (Minimal Augmentation)")
    print("  - Resize + Random crop")
    print("  - Horizontal flip")
    print("  - Mild color jitter")
    print("  - No RandAugment, Mixup, CutMix, or RandomErasing")

    # Strategy 2: DeiT III (Strong Augmentation)
    deit3_transform = DeiTIIIAugmentation(
        image_size=224,
        num_classes=1000,
    )
    print("\nStrategy 2: DeiT III (Strong Augmentation)")
    print("  - All JEPA augmentations")
    print("  - RandAugment (2 ops, magnitude 9)")
    print("  - Mixup (alpha=0.8)")
    print("  - CutMix (alpha=1.0)")
    print("  - RandomErasing (prob=0.25)")

    # Strategy 3: Custom Moderate
    moderate_transform = DeiTIIIAugmentation(
        image_size=224,
        auto_augment=True,
        rand_aug_num_ops=1,
        rand_aug_magnitude=5,
        random_erasing_prob=0.1,
        mixup_alpha=0.4,
        cutmix_alpha=0.4,
        num_classes=1000,
    )
    print("\nStrategy 3: Moderate Augmentation")
    print("  - Reduced RandAugment (1 op, magnitude 5)")
    print("  - Lighter Mixup/CutMix (alpha=0.4)")
    print("  - Lower RandomErasing (prob=0.1)")

    print("\n" + "=" * 80)
    print("Use Case Recommendations:")
    print("=" * 80)
    print("• JEPA style: Best for self-supervised pretraining (H-JEPA)")
    print("• DeiT III style: Best for supervised training with Vision Transformers")
    print("• Moderate style: Good balance for smaller datasets or fine-tuning")
    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DeiT III Augmentation Examples for H-JEPA")
    print("=" * 80 + "\n")

    # Run examples
    example_1_basic_usage()
    example_2_custom_config()
    example_3_individual_components()
    example_4_eval_transform()
    example_5_complete_training_setup()
    example_6_comparison()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)

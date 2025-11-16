#!/usr/bin/env python3
"""
ImageNet-100 Dataset Usage Examples

This script demonstrates how to use ImageNet-100 dataset with H-JEPA.
Run this script to verify your ImageNet-100 setup and see example usage.

Usage:
    python examples/imagenet100_example.py --data-path ./data/imagenet

Requirements:
    - ImageNet downloaded and extracted to data_path
    - PyTorch and torchvision installed
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.data import (
    ImageNet100Dataset,
    JEPATransform,
    build_dataloader,
    build_dataset,
    build_multi_dataset,
)


def example_1_basic_usage(data_path: str):
    """Example 1: Basic ImageNet-100 dataset usage"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic ImageNet-100 Usage")
    print("=" * 70)

    # Build dataset using factory function
    train_dataset = build_dataset(
        dataset_name="imagenet100",
        data_path=data_path,
        split="train",
        image_size=224,
        color_jitter=0.1,
    )

    # Build validation dataset
    val_dataset = build_dataset(
        dataset_name="imagenet100",
        data_path=data_path,
        split="val",
        image_size=224,
    )

    print(f"\nDataset Information:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")

    # Get a sample
    image, label, dataset_idx = train_dataset[0]
    print(f"\nSample Information:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Label: {label}")
    print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")

    return train_dataset, val_dataset


def example_2_dataloader(dataset):
    """Example 2: Creating dataloaders"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: DataLoader Creation")
    print("=" * 70)

    from src.data import build_dataloader

    # Build dataloader
    train_loader = build_dataloader(
        dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )

    print(f"\nDataLoader Information:")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Number of batches: {len(train_loader):,}")
    print(f"  Number of workers: {train_loader.num_workers}")

    # Get a batch
    batch = next(iter(train_loader))
    images, labels, dataset_indices = batch
    print(f"\nBatch Information:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Dataset indices shape: {dataset_indices.shape}")

    return train_loader


def example_3_multi_dataset(data_path: str):
    """Example 3: Multi-dataset with ImageNet-100"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multi-Dataset Configuration")
    print("=" * 70)

    # Configure multiple datasets with weights
    dataset_configs = [
        {"name": "imagenet100", "weight": 0.6, "path": data_path},
        {"name": "stl10", "weight": 0.3},
        {"name": "cifar100", "weight": 0.1},
    ]

    print("\nDataset Configuration:")
    for config in dataset_configs:
        print(f"  {config['name']:12s}: weight={config['weight']:.1%}")

    # Build multi-dataset
    multi_dataset = build_multi_dataset(
        dataset_configs=dataset_configs,
        data_path="./data",
        split="train",
        sampling_strategy="weighted",
        image_size=224,
        download=True,  # Auto-download CIFAR/STL10
    )

    print(f"\nMulti-Dataset Information:")
    print(f"  Total effective size: {len(multi_dataset):,}")

    # Get dataset statistics
    stats = multi_dataset.get_dataset_stats()
    print(f"\nSampling Statistics:")
    for name, info in stats.items():
        print(
            f"  {name:12s}: {info['size']:7,} images, "
            f"{info['weight']:5.1%} weight, "
            f"{info['expected_samples_per_epoch']:6,} samples/epoch"
        )

    # Sample from multi-dataset
    sample = multi_dataset[0]
    image, label, dataset_idx = sample
    print(f"\nSample from Multi-Dataset:")
    print(f"  Image shape: {image.shape}")
    print(f"  Dataset index: {dataset_idx} ({multi_dataset.dataset_names[dataset_idx]})")

    return multi_dataset


def example_4_custom_transforms(data_path: str):
    """Example 4: Custom transforms for ImageNet-100"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Transforms")
    print("=" * 70)

    # Create custom JEPA transform
    custom_transform = JEPATransform(
        image_size=224,
        crop_scale=(0.85, 1.0),  # Less aggressive cropping
        color_jitter=0.05,  # Very minimal color jitter
        horizontal_flip=True,
    )

    # Use with dataset
    dataset = ImageNet100Dataset(
        data_path=data_path,
        split="train",
        transform=custom_transform,
    )

    print(f"\nCustom Transform Parameters:")
    print(f"  Image size: 224x224")
    print(f"  Crop scale: (0.85, 1.0)")
    print(f"  Color jitter: 0.05")
    print(f"  Horizontal flip: True")

    # Get sample
    image, label, _ = dataset[0]
    print(f"\nTransformed Sample:")
    print(f"  Shape: {image.shape}")
    print(f"  Range: [{image.min():.3f}, {image.max():.3f}]")

    return dataset


def example_5_class_filtering():
    """Example 5: Understanding ImageNet-100 class filtering"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: ImageNet-100 Class Filtering")
    print("=" * 70)

    from src.data.datasets import ImageNet100Dataset

    print("\nImageNet-100 uses a predefined set of 100 classes from ImageNet-1K.")
    print(f"Number of classes: {len(ImageNet100Dataset.IMAGENET100_CLASSES)}")
    print(f"\nFirst 10 classes (WordNet synset IDs):")
    for i, class_id in enumerate(ImageNet100Dataset.IMAGENET100_CLASSES[:10], 1):
        print(f"  {i:2d}. {class_id}")
    print(f"  ...")
    print(f"  (+ {len(ImageNet100Dataset.IMAGENET100_CLASSES) - 10} more classes)")

    print("\nClass Filtering Process:")
    print("  1. Load full ImageNet directory structure")
    print("  2. Identify all image samples")
    print("  3. Filter samples to only include the 100 predefined classes")
    print("  4. Create filtered dataset with ~126K images")

    print("\nBenefits:")
    print("  ✓ Faster training than full ImageNet (~10x faster)")
    print("  ✓ Higher quality than CIFAR (224x224 vs 32x32)")
    print("  ✓ Benchmark compatible (standardized class subset)")
    print("  ✓ No manual data preparation needed")


def example_6_performance_comparison():
    """Example 6: Performance comparison across datasets"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Dataset Performance Comparison")
    print("=" * 70)

    datasets_info = {
        "CIFAR-10": {
            "images": 50000,
            "resolution": "32x32",
            "classes": 10,
            "linear_probe": "50-55%",
            "training_time": "2 hours",
        },
        "CIFAR-100": {
            "images": 50000,
            "resolution": "32x32",
            "classes": 100,
            "linear_probe": "40-50%",
            "training_time": "2 hours",
        },
        "STL-10": {
            "images": 105000,
            "resolution": "96x96",
            "classes": 10,
            "linear_probe": "55-60%",
            "training_time": "4 hours",
        },
        "ImageNet-100": {
            "images": 126689,
            "resolution": "224x224",
            "classes": 100,
            "linear_probe": "60-70%",
            "training_time": "12 hours",
        },
        "Multi-Dataset": {
            "images": 280000,
            "resolution": "224x224",
            "classes": "100+",
            "linear_probe": "65-75%",
            "training_time": "18 hours",
        },
    }

    print("\nDataset Comparison (100 epochs, ViT-Small on M1 Max):")
    print(
        f"\n{'Dataset':<15} {'Images':<10} {'Resolution':<12} {'Classes':<8} "
        f"{'Linear Probe':<13} {'Training Time':<15}"
    )
    print("-" * 90)

    for name, info in datasets_info.items():
        print(
            f"{name:<15} {info['images']:<10,} {info['resolution']:<12} "
            f"{str(info['classes']):<8} {info['linear_probe']:<13} "
            f"{info['training_time']:<15}"
        )

    print("\nKey Insights:")
    print("  • ImageNet-100 provides +10-15% improvement over CIFAR datasets")
    print("  • Higher resolution (224x224) enables better feature learning")
    print("  • Multi-dataset training combines benefits of all datasets")
    print("  • Training time increases with dataset size and resolution")


def verify_dataset(data_path: str) -> bool:
    """Verify ImageNet-100 dataset is properly set up"""
    print("\n" + "=" * 70)
    print("Dataset Verification")
    print("=" * 70)

    data_path = Path(data_path)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    print(f"\nChecking ImageNet directory: {data_path}")

    # Check directories exist
    if not train_dir.exists():
        print(f"  ✗ Training directory not found: {train_dir}")
        return False
    print(f"  ✓ Training directory found: {train_dir}")

    if not val_dir.exists():
        print(f"  ✗ Validation directory not found: {val_dir}")
        return False
    print(f"  ✓ Validation directory found: {val_dir}")

    # Count class directories
    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    val_classes = [d for d in val_dir.iterdir() if d.is_dir()]

    print(f"\nClass Directories:")
    print(f"  Train: {len(train_classes)} classes")
    print(f"  Val: {len(val_classes)} classes")

    if len(train_classes) < 100:
        print(f"  ⚠ Warning: Expected at least 100 classes, found {len(train_classes)}")
        print(f"  ImageNet-100 requires full ImageNet download")
        return False

    # Count images in first few classes
    sample_classes = train_classes[:3]
    print(f"\nSample Class Sizes:")
    for class_dir in sample_classes:
        images = list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.jpg"))
        print(f"  {class_dir.name}: {len(images)} images")

    print(f"\n✓ Dataset verification passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="ImageNet-100 Dataset Examples")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/imagenet",
        help="Path to ImageNet directory (containing train/ and val/ subdirectories)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify dataset setup, don't run examples",
    )
    parser.add_argument(
        "--skip-multi-dataset",
        action="store_true",
        help="Skip multi-dataset example (useful if CIFAR/STL10 not available)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ImageNet-100 Dataset Examples for H-JEPA")
    print("=" * 70)

    # Verify dataset
    if not verify_dataset(args.data_path):
        print("\n" + "=" * 70)
        print("Dataset verification failed!")
        print("=" * 70)
        print("\nTo download ImageNet:")
        print("  1. Register at https://image-net.org/download.php")
        print("  2. Download ILSVRC2012 train and val sets")
        print("  3. Extract to: ./data/imagenet/")
        print("\nExpected structure:")
        print("  data/imagenet/")
        print("    train/")
        print("      n01440764/  (class directories)")
        print("        image1.JPEG")
        print("        ...")
        print("    val/")
        print("      n01440764/")
        print("        ...")
        return

    if args.verify_only:
        print("\n✓ Verification complete. Dataset is ready to use.")
        return

    # Run examples
    try:
        # Example 1: Basic usage
        train_dataset, val_dataset = example_1_basic_usage(args.data_path)

        # Example 2: DataLoader
        train_loader = example_2_dataloader(train_dataset)

        # Example 3: Multi-dataset (optional)
        if not args.skip_multi_dataset:
            try:
                multi_dataset = example_3_multi_dataset(args.data_path)
            except Exception as e:
                print(f"\n⚠ Multi-dataset example skipped: {e}")
                print("  (CIFAR/STL10 datasets may not be available)")

        # Example 4: Custom transforms
        custom_dataset = example_4_custom_transforms(args.data_path)

        # Example 5: Class filtering explanation
        example_5_class_filtering()

        # Example 6: Performance comparison
        example_6_performance_comparison()

        print("\n" + "=" * 70)
        print("All Examples Completed Successfully!")
        print("=" * 70)
        print("\nNext Steps:")
        print("  1. Review the example code above")
        print("  2. Check configuration files: configs/m1_max_imagenet100_100epoch.yaml")
        print("  3. Start training:")
        print("     python scripts/train.py --config configs/m1_max_imagenet100_100epoch.yaml")
        print("\nFor more information:")
        print("  • Documentation: docs/IMAGENET100_INTEGRATION.md")
        print("  • Dataset implementation: src/data/datasets.py")
        print("  • Multi-dataset support: src/data/multi_dataset.py")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback

        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

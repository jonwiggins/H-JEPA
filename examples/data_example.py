"""
Example script demonstrating dataset usage for H-JEPA.

This script shows how to:
1. Download and load datasets
2. Create dataloaders
3. Iterate through batches
4. Use different datasets and configurations
"""

import argparse
from pathlib import Path

import torch

from src.data import (
    DATASET_INFO,
    build_dataloader,
    build_dataset,
    print_dataset_summary,
    verify_dataset,
)


def example_basic_usage():
    """Basic dataset usage example."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Dataset Usage")
    print("=" * 70 + "\n")

    # Build CIFAR-10 dataset
    print("Building CIFAR-10 dataset...")
    train_dataset = build_dataset(
        dataset_name="cifar10",
        data_path="./data/cifar10",
        split="train",
        image_size=224,
        download=True,
    )

    val_dataset = build_dataset(
        dataset_name="cifar10",
        data_path="./data/cifar10",
        split="val",
        image_size=224,
        download=True,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Get a sample
    img, label = train_dataset[0]
    print(f"Sample image shape: {img.shape}")
    print(f"Sample label: {label}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")


def example_dataloader():
    """Dataloader usage example."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: DataLoader Usage")
    print("=" * 70 + "\n")

    # Build dataset
    dataset = build_dataset(
        dataset_name="cifar10",
        data_path="./data/cifar10",
        split="train",
        download=True,
    )

    # Build dataloader
    print("Building dataloader...")
    dataloader = build_dataloader(
        dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Number of batches: {len(dataloader)}")

    # Iterate through a few batches
    print("\nIterating through first 3 batches...")
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= 3:
            break

        print(f"Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels[:8].tolist()}")  # Show first 8 labels


def example_multiple_datasets():
    """Example using multiple datasets."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multiple Datasets")
    print("=" * 70 + "\n")

    datasets_to_try = [
        ("cifar10", "./data/cifar10"),
        ("cifar100", "./data/cifar100"),
    ]

    for dataset_name, data_path in datasets_to_try:
        print(f"\n{dataset_name.upper()}:")
        print("-" * 40)

        try:
            dataset = build_dataset(
                dataset_name=dataset_name,
                data_path=data_path,
                split="train",
                download=True,
            )

            print(f"✓ Loaded {dataset_name}: {len(dataset)} images")

            # Get sample
            img, label = dataset[0]
            print(f"  Image shape: {img.shape}")
            print(f"  Number of classes: {DATASET_INFO[dataset_name]['num_classes']}")

        except Exception as e:
            print(f"✗ Failed to load {dataset_name}: {e}")


def example_custom_config():
    """Example with custom configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Configuration")
    print("=" * 70 + "\n")

    # Build dataset with custom settings
    print("Building dataset with custom settings...")
    dataset = build_dataset(
        dataset_name="cifar10",
        data_path="./data/cifar10",
        split="train",
        image_size=192,  # Custom image size
        color_jitter=0.2,  # Reduced color jitter
        download=True,
    )

    # Build dataloader with custom settings
    dataloader = build_dataloader(
        dataset,
        batch_size=64,  # Larger batch size
        num_workers=8,  # More workers
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: 64")

    # Get one batch
    images, labels = next(iter(dataloader))
    print(f"Batch shape: {images.shape}")


def example_verification():
    """Example of dataset verification."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Dataset Verification")
    print("=" * 70 + "\n")

    datasets_to_verify = [
        ("cifar10", "./data/cifar10"),
        ("cifar100", "./data/cifar100"),
    ]

    for dataset_name, data_path in datasets_to_verify:
        print(f"Verifying {dataset_name}...")
        is_valid = verify_dataset(dataset_name, Path(data_path))

        if is_valid:
            print(f"✓ {dataset_name} is valid")
        else:
            print(f"✗ {dataset_name} verification failed")


def example_imagenet_loading():
    """Example of ImageNet loading (if available)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: ImageNet Loading (if available)")
    print("=" * 70 + "\n")

    imagenet_path = "./data/imagenet"

    print(f"Checking for ImageNet at {imagenet_path}...")

    try:
        dataset = build_dataset(
            dataset_name="imagenet",
            data_path=imagenet_path,
            split="train",
            download=False,  # ImageNet cannot be auto-downloaded
        )

        print(f"✓ ImageNet loaded: {len(dataset)} images")
        print(f"  Number of classes: {len(dataset.classes)}")

        # Get sample
        img, label = dataset[0]
        print(f"  Sample image shape: {img.shape}")

    except FileNotFoundError:
        print("✗ ImageNet not found")
        print("  ImageNet requires manual download.")
        print("  Run: ./scripts/download_data.sh imagenet")
        print("  for download instructions.")


def example_performance_comparison():
    """Example comparing data loading performance."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Performance Comparison")
    print("=" * 70 + "\n")

    import time

    # Build dataset
    dataset = build_dataset(
        dataset_name="cifar10",
        data_path="./data/cifar10",
        split="train",
        download=True,
    )

    # Test different configurations
    configs = [
        {"num_workers": 0, "pin_memory": False},
        {"num_workers": 4, "pin_memory": False},
        {"num_workers": 4, "pin_memory": True},
    ]

    for config in configs:
        print(
            f"\nTesting: num_workers={config['num_workers']}, " f"pin_memory={config['pin_memory']}"
        )

        dataloader = build_dataloader(
            dataset,
            batch_size=128,
            shuffle=False,
            **config,
        )

        # Time 100 batches
        start_time = time.time()
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= 100:
                break

        elapsed = time.time() - start_time
        throughput = 100 * 128 / elapsed

        print(f"  Time for 100 batches: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.1f} images/sec")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Dataset usage examples for H-JEPA")
    parser.add_argument(
        "--example",
        type=str,
        choices=[
            "basic",
            "dataloader",
            "multiple",
            "custom",
            "verify",
            "imagenet",
            "performance",
            "all",
        ],
        default="all",
        help="Which example to run",
    )
    parser.add_argument("--summary", action="store_true", help="Show dataset summary")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("H-JEPA Dataset Usage Examples")
    print("=" * 70)

    if args.summary:
        print_dataset_summary()
        return

    examples = {
        "basic": example_basic_usage,
        "dataloader": example_dataloader,
        "multiple": example_multiple_datasets,
        "custom": example_custom_config,
        "verify": example_verification,
        "imagenet": example_imagenet_loading,
        "performance": example_performance_comparison,
    }

    if args.example == "all":
        # Run all examples
        for name, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"\n✗ Example '{name}' failed: {e}")
    else:
        # Run specific example
        examples[args.example]()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

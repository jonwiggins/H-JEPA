"""
Dataset downloading and verification utilities for H-JEPA.

This module provides utilities to download, verify, and prepare datasets
for H-JEPA training. It handles automatic downloads where possible and
provides clear instructions for manual downloads when required.
"""

import hashlib
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast
from urllib.error import URLError

import torch
from torchvision import datasets
from tqdm import tqdm


class DatasetInfo(TypedDict):
    """Type definition for dataset information."""

    name: str
    size_gb: float
    num_images: int
    num_classes: int
    resolution: str
    auto_download: bool
    description: str
    url: str


# Dataset information and requirements
DATASET_INFO: Dict[str, DatasetInfo] = {
    "cifar10": {
        "name": "CIFAR-10",
        "size_gb": 0.17,
        "num_images": 60000,
        "num_classes": 10,
        "resolution": "32x32",
        "auto_download": True,
        "description": "10 classes of natural images",
        "url": "https://www.cs.toronto.edu/~kriz/cifar.html",
    },
    "cifar100": {
        "name": "CIFAR-100",
        "size_gb": 0.17,
        "num_images": 60000,
        "num_classes": 100,
        "resolution": "32x32",
        "auto_download": True,
        "description": "100 classes of natural images",
        "url": "https://www.cs.toronto.edu/~kriz/cifar.html",
    },
    "stl10": {
        "name": "STL-10",
        "size_gb": 2.5,
        "num_images": 113000,  # 5k train + 8k test + 100k unlabeled
        "num_classes": 10,
        "resolution": "96x96",
        "auto_download": True,
        "description": "10 classes with unlabeled data",
        "url": "https://cs.stanford.edu/~acoates/stl10/",
    },
    "imagenet": {
        "name": "ImageNet (ILSVRC2012)",
        "size_gb": 144.0,  # ~144GB for training, ~6.3GB for validation
        "num_images": 1331167,  # 1.28M train + 50k val
        "num_classes": 1000,
        "resolution": "varies (resized to 224x224)",
        "auto_download": False,
        "description": "1000 classes of natural images (requires manual download)",
        "url": "https://image-net.org/download.php",
    },
    "imagenet100": {
        "name": "ImageNet-100",
        "size_gb": 14.4,  # ~10% of ImageNet
        "num_images": 130000,  # Approximate
        "num_classes": 100,
        "resolution": "varies (resized to 224x224)",
        "auto_download": False,
        "description": "100-class subset of ImageNet (requires ImageNet download first)",
        "url": "https://image-net.org/download.php",
    },
}


def get_disk_usage(path: Path) -> Tuple[float, float, float]:
    """
    Get disk usage statistics for a path.

    Args:
        path: Directory path

    Returns:
        Tuple of (total_gb, used_gb, free_gb)
    """
    try:
        stat = shutil.disk_usage(path)
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        free_gb = stat.free / (1024**3)
        return total_gb, used_gb, free_gb
    except Exception as e:
        warnings.warn(f"Could not get disk usage: {e}")
        return 0.0, 0.0, 0.0


def check_disk_space(data_path: Path, required_gb: float, buffer_gb: float = 5.0) -> bool:
    """
    Check if there's enough disk space for dataset download.

    Args:
        data_path: Directory where data will be stored
        required_gb: Required space in GB
        buffer_gb: Additional buffer space in GB

    Returns:
        True if enough space, False otherwise
    """
    total_gb, used_gb, free_gb = get_disk_usage(data_path)

    if free_gb < required_gb + buffer_gb:
        print(f"\n⚠️  WARNING: Insufficient disk space!")
        print(f"  Required: {required_gb:.1f} GB + {buffer_gb:.1f} GB buffer")
        print(f"  Available: {free_gb:.1f} GB")
        print(f"  Total: {total_gb:.1f} GB, Used: {used_gb:.1f} GB")
        return False

    print(f"✓ Disk space check passed: {free_gb:.1f} GB available")
    return True


def verify_dataset(dataset_name: str, data_path: Path) -> bool:
    """
    Verify that a dataset is properly downloaded and can be loaded.

    Args:
        dataset_name: Name of the dataset
        data_path: Path to dataset directory

    Returns:
        True if dataset is valid, False otherwise
    """
    dataset_name = dataset_name.lower()

    print(f"\nVerifying {dataset_name}...")

    try:
        if dataset_name == "cifar10":
            # Try loading CIFAR-10
            train_data = datasets.CIFAR10(root=data_path, train=True, download=False)
            test_data = datasets.CIFAR10(root=data_path, train=False, download=False)
            assert len(train_data) == 50000, f"Expected 50000 train images, got {len(train_data)}"
            assert len(test_data) == 10000, f"Expected 10000 test images, got {len(test_data)}"
            print(f"✓ CIFAR-10 verified: {len(train_data)} train + {len(test_data)} test images")
            return True

        elif dataset_name == "cifar100":
            # Try loading CIFAR-100
            train_data = datasets.CIFAR100(root=data_path, train=True, download=False)
            test_data = datasets.CIFAR100(root=data_path, train=False, download=False)
            assert len(train_data) == 50000, f"Expected 50000 train images, got {len(train_data)}"
            assert len(test_data) == 10000, f"Expected 10000 test images, got {len(test_data)}"
            print(f"✓ CIFAR-100 verified: {len(train_data)} train + {len(test_data)} test images")
            return True

        elif dataset_name == "stl10":
            # Try loading STL-10
            train_data = datasets.STL10(root=data_path, split="train", download=False)
            test_data = datasets.STL10(root=data_path, split="test", download=False)
            unlabeled_data = datasets.STL10(root=data_path, split="unlabeled", download=False)
            print(
                f"✓ STL-10 verified: {len(train_data)} train + {len(test_data)} test + "
                f"{len(unlabeled_data)} unlabeled images"
            )
            return True

        elif dataset_name in ["imagenet", "imagenet100"]:
            # Check for ImageNet directory structure
            train_dir = data_path / "train"
            val_dir = data_path / "val"

            if not train_dir.exists() or not val_dir.exists():
                print(f"✗ ImageNet not found at {data_path}")
                print(f"  Expected directories: {train_dir} and {val_dir}")
                return False

            # Count number of class directories
            train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
            val_classes = [d for d in val_dir.iterdir() if d.is_dir()]

            if len(train_classes) == 0 or len(val_classes) == 0:
                print(f"✗ ImageNet directories exist but appear empty")
                return False

            print(
                f"✓ ImageNet found: {len(train_classes)} train classes, "
                f"{len(val_classes)} val classes"
            )

            if dataset_name == "imagenet100":
                if len(train_classes) < 100:
                    print(
                        f"  Note: Found {len(train_classes)} classes, expected >= 100 for ImageNet-100"
                    )
                    print(f"  Will filter to 100 classes at runtime")

            return True

        else:
            print(f"✗ Unknown dataset: {dataset_name}")
            return False

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def download_dataset(
    dataset_name: str,
    data_path: Path,
    force: bool = False,
    verify: bool = True,
) -> bool:
    """
    Download a dataset if it supports automatic downloading.

    Args:
        dataset_name: Name of the dataset
        data_path: Path to store dataset
        force: Force re-download even if already exists
        verify: Verify dataset after download

    Returns:
        True if successful, False otherwise
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_INFO:
        print(f"✗ Unknown dataset: {dataset_name}")
        return False

    info = DATASET_INFO[dataset_name]

    print(f"\n{'='*70}")
    print(f"Dataset: {info['name']}")
    print(f"Size: {info['size_gb']:.2f} GB")
    print(f"Images: {info['num_images']:,}")
    print(f"Classes: {info['num_classes']}")
    print(f"Resolution: {info['resolution']}")
    print(f"{'='*70}\n")

    # Check if auto-download is supported
    if not info["auto_download"]:
        print(f"⚠️  {info['name']} requires manual download")
        print_manual_download_instructions(dataset_name)
        return False

    # Check disk space
    if not check_disk_space(data_path, info["size_gb"]):
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != "y":
            print("Download cancelled")
            return False

    # Create data directory
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading {info['name']}...")

        if dataset_name == "cifar10":
            # Download CIFAR-10
            print("Downloading training set...")
            datasets.CIFAR10(root=data_path, train=True, download=True)
            print("Downloading test set...")
            datasets.CIFAR10(root=data_path, train=False, download=True)

        elif dataset_name == "cifar100":
            # Download CIFAR-100
            print("Downloading training set...")
            datasets.CIFAR100(root=data_path, train=True, download=True)
            print("Downloading test set...")
            datasets.CIFAR100(root=data_path, train=False, download=True)

        elif dataset_name == "stl10":
            # Download STL-10
            print("Downloading training set...")
            datasets.STL10(root=data_path, split="train", download=True)
            print("Downloading test set...")
            datasets.STL10(root=data_path, split="test", download=True)
            print("Downloading unlabeled set...")
            datasets.STL10(root=data_path, split="unlabeled", download=True)

        print(f"\n✓ {info['name']} downloaded successfully!")

        # Verify if requested
        if verify:
            if verify_dataset(dataset_name, data_path):
                print(f"✓ {info['name']} verified successfully!")
            else:
                print(f"⚠️  {info['name']} verification failed")
                return False

        return True

    except URLError as e:
        print(f"\n✗ Download failed due to network error: {e}")
        print("Please check your internet connection and try again.")
        return False

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def print_manual_download_instructions(dataset_name: str) -> None:
    """
    Print instructions for manually downloading a dataset.

    Args:
        dataset_name: Name of the dataset
    """
    dataset_name = dataset_name.lower()

    print("\n" + "=" * 70)
    print(f"MANUAL DOWNLOAD INSTRUCTIONS: {DATASET_INFO[dataset_name]['name']}")
    print("=" * 70 + "\n")

    if dataset_name == "imagenet":
        print(
            """
1. Register and Download:
   - Go to: https://image-net.org/download.php
   - Register for an account (required for download)
   - Download ILSVRC2012 training images (tar file)
   - Download ILSVRC2012 validation images (tar file)
   - Download ILSVRC2012 development kit (for labels)

2. Organize the data:
   - Create directory structure:
     {data_path}/
       train/
         n01440764/
           n01440764_18.JPEG
           ...
         n01443537/
           ...
       val/
         n01440764/
           ...

3. Extract and organize:
   # Extract training data
   mkdir -p train && cd train
   tar -xvf ILSVRC2012_img_train.tar
   # Each class tar file needs to be extracted
   for f in *.tar; do
     d=$(basename "$f" .tar)
     mkdir -p "$d"
     tar -xf "$f" -C "$d"
     rm "$f"
   done

   # Extract validation data
   mkdir -p ../val && cd ../val
   tar -xvf ILSVRC2012_img_val.tar
   # Organize val images into class folders using devkit labels
   # (Use provided script or manual organization)

4. Verify:
   - Training: Should have 1000 class folders with ~1300 images each
   - Validation: Should have 1000 class folders with 50 images each

Note: ImageNet is ~144GB for training + ~6.3GB for validation.
Make sure you have sufficient disk space!
        """
        )

    elif dataset_name == "imagenet100":
        print(
            """
ImageNet-100 is a 100-class subset of ImageNet.

Option 1: Download full ImageNet first
   - Follow ImageNet download instructions above
   - The dataset class will automatically filter to 100 classes

Option 2: Download ImageNet-100 directly (if available)
   - Some research groups provide pre-filtered ImageNet-100
   - Check: https://github.com/HobbitLong/CMC
   - Or search for "ImageNet-100 dataset download"

The 100 classes are a standard subset used in self-supervised learning research.
They will be automatically selected from the full ImageNet if you download it.
        """
        )

    print("=" * 70 + "\n")


def print_dataset_summary() -> None:
    """Print a summary of all supported datasets."""
    print("\n" + "=" * 70)
    print("SUPPORTED DATASETS FOR H-JEPA")
    print("=" * 70 + "\n")

    total_size = 0.0

    for name, info in DATASET_INFO.items():
        auto = "✓ Auto" if info["auto_download"] else "✗ Manual"
        print(
            f"{info['name']:25} {auto:12} {info['size_gb']:6.1f} GB  "
            f"{info['num_images']:>9,} images  {info['num_classes']:>4} classes"
        )
        total_size += info["size_gb"]

    print("\n" + "-" * 70)
    print(f"{'Total (all datasets)':25} {total_size:19.1f} GB")
    print("=" * 70 + "\n")

    print("Recommended for quick start: CIFAR-10 or CIFAR-100 (auto-download)")
    print("Recommended for research: ImageNet or ImageNet-100 (manual download)")
    print("\nUse download_data.sh script or this module to download datasets.\n")


def main() -> None:
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Download and verify datasets for H-JEPA training")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset(s) to download (cifar10, cifar100, stl10, imagenet, imagenet100). "
        "Leave empty to show summary.",
    )
    parser.add_argument(
        "--data-path", type=str, default="./data", help="Path to store datasets (default: ./data)"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify datasets, don't download"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if dataset exists"
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip verification after download")

    args = parser.parse_args()

    # Convert to Path
    data_path = Path(args.data_path).resolve()

    # Show summary if no datasets specified
    if not args.datasets:
        print_dataset_summary()
        print(f"Current data path: {data_path}")
        total_gb, used_gb, free_gb = get_disk_usage(data_path)
        print(f"Disk space: {free_gb:.1f} GB available ({total_gb:.1f} GB total)\n")
        return

    # Process each dataset
    for dataset_name in args.datasets:
        dataset_name = dataset_name.lower()

        if dataset_name not in DATASET_INFO:
            print(f"\n✗ Unknown dataset: {dataset_name}")
            print(f"Supported datasets: {', '.join(DATASET_INFO.keys())}")
            continue

        dataset_path = data_path / dataset_name

        if args.verify_only:
            # Just verify
            verify_dataset(dataset_name, dataset_path)
        else:
            # Download
            success = download_dataset(
                dataset_name,
                dataset_path,
                force=args.force,
                verify=not args.no_verify,
            )

            if not success and DATASET_INFO[dataset_name]["auto_download"]:
                print(f"\n✗ Failed to download {dataset_name}")

    print("\nDone!")


if __name__ == "__main__":
    main()

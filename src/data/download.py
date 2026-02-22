"""
Dataset downloading and verification utilities for H-JEPA.

This module provides utilities to download, verify, and prepare datasets
for H-JEPA training. It handles automatic downloads where possible and
provides clear instructions for manual downloads when required.
"""

import logging
import shutil
import warnings
from pathlib import Path
from typing import TypedDict
from urllib.error import URLError

from torchvision import datasets

logger = logging.getLogger(__name__)


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
DATASET_INFO: dict[str, DatasetInfo] = {
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


def get_disk_usage(path: Path) -> tuple[float, float, float]:
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
    except OSError as e:
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
        logger.warning("Insufficient disk space!")
        logger.warning("  Required: %.1f GB + %.1f GB buffer", required_gb, buffer_gb)
        logger.warning("  Available: %.1f GB", free_gb)
        logger.warning("  Total: %.1f GB, Used: %.1f GB", total_gb, used_gb)
        return False

    logger.info("Disk space check passed: %.1f GB available", free_gb)
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

    logger.info("Verifying %s...", dataset_name)

    try:
        if dataset_name == "cifar10":
            # Try loading CIFAR-10
            train_data = datasets.CIFAR10(root=data_path, train=True, download=False)
            test_data = datasets.CIFAR10(root=data_path, train=False, download=False)
            assert len(train_data) == 50000, f"Expected 50000 train images, got {len(train_data)}"
            assert len(test_data) == 10000, f"Expected 10000 test images, got {len(test_data)}"
            logger.info(
                "CIFAR-10 verified: %d train + %d test images", len(train_data), len(test_data)
            )
            return True

        elif dataset_name == "cifar100":
            # Try loading CIFAR-100
            train_data = datasets.CIFAR100(root=data_path, train=True, download=False)
            test_data = datasets.CIFAR100(root=data_path, train=False, download=False)
            assert len(train_data) == 50000, f"Expected 50000 train images, got {len(train_data)}"
            assert len(test_data) == 10000, f"Expected 10000 test images, got {len(test_data)}"
            logger.info(
                "CIFAR-100 verified: %d train + %d test images", len(train_data), len(test_data)
            )
            return True

        elif dataset_name == "stl10":
            # Try loading STL-10
            train_data = datasets.STL10(root=data_path, split="train", download=False)
            test_data = datasets.STL10(root=data_path, split="test", download=False)
            unlabeled_data = datasets.STL10(root=data_path, split="unlabeled", download=False)
            logger.info(
                "STL-10 verified: %d train + %d test + %d unlabeled images",
                len(train_data),
                len(test_data),
                len(unlabeled_data),
            )
            return True

        elif dataset_name in ["imagenet", "imagenet100"]:
            # Check for ImageNet directory structure
            train_dir = data_path / "train"
            val_dir = data_path / "val"

            if not train_dir.exists() or not val_dir.exists():
                logger.error("ImageNet not found at %s", data_path)
                logger.error("  Expected directories: %s and %s", train_dir, val_dir)
                return False

            # Count number of class directories
            train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
            val_classes = [d for d in val_dir.iterdir() if d.is_dir()]

            if len(train_classes) == 0 or len(val_classes) == 0:
                logger.error("ImageNet directories exist but appear empty")
                return False

            logger.info(
                "ImageNet found: %d train classes, %d val classes",
                len(train_classes),
                len(val_classes),
            )

            if dataset_name == "imagenet100":
                if len(train_classes) < 100:
                    logger.info(
                        "  Note: Found %d classes, expected >= 100 for ImageNet-100",
                        len(train_classes),
                    )
                    logger.info("  Will filter to 100 classes at runtime")

            return True

        else:
            logger.error("Unknown dataset: %s", dataset_name)
            return False

    except (OSError, ValueError, RuntimeError, AssertionError) as e:
        logger.error("Verification failed: %s", e)
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
        logger.error("Unknown dataset: %s", dataset_name)
        return False

    info = DATASET_INFO[dataset_name]

    logger.info("=" * 70)
    logger.info("Dataset: %s", info["name"])
    logger.info("Size: %.2f GB", info["size_gb"])
    logger.info("Images: %s", f"{info['num_images']:,}")
    logger.info("Classes: %d", info["num_classes"])
    logger.info("Resolution: %s", info["resolution"])
    logger.info("=" * 70)

    # Check if auto-download is supported
    if not info["auto_download"]:
        logger.warning("%s requires manual download", info["name"])
        print_manual_download_instructions(dataset_name)
        return False

    # Check disk space
    if not check_disk_space(data_path, info["size_gb"]):
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != "y":
            logger.info("Download cancelled")
            return False

    # Create data directory
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Downloading %s...", info["name"])

        if dataset_name == "cifar10":
            # Download CIFAR-10
            logger.info("Downloading training set...")
            datasets.CIFAR10(root=data_path, train=True, download=True)
            logger.info("Downloading test set...")
            datasets.CIFAR10(root=data_path, train=False, download=True)

        elif dataset_name == "cifar100":
            # Download CIFAR-100
            logger.info("Downloading training set...")
            datasets.CIFAR100(root=data_path, train=True, download=True)
            logger.info("Downloading test set...")
            datasets.CIFAR100(root=data_path, train=False, download=True)

        elif dataset_name == "stl10":
            # Download STL-10
            logger.info("Downloading training set...")
            datasets.STL10(root=data_path, split="train", download=True)
            logger.info("Downloading test set...")
            datasets.STL10(root=data_path, split="test", download=True)
            logger.info("Downloading unlabeled set...")
            datasets.STL10(root=data_path, split="unlabeled", download=True)

        logger.info("%s downloaded successfully!", info["name"])

        # Verify if requested
        if verify:
            if verify_dataset(dataset_name, data_path):
                logger.info("%s verified successfully!", info["name"])
            else:
                logger.warning("%s verification failed", info["name"])
                return False

        return True

    except URLError as e:
        logger.error("Download failed due to network error: %s", e)
        logger.error("Please check your internet connection and try again.")
        return False

    except OSError as e:
        logger.error("Download failed: %s", e)
        return False


def print_manual_download_instructions(dataset_name: str) -> None:
    """
    Print instructions for manually downloading a dataset.

    Args:
        dataset_name: Name of the dataset
    """
    dataset_name = dataset_name.lower()

    logger.info("=" * 70)
    logger.info("MANUAL DOWNLOAD INSTRUCTIONS: %s", DATASET_INFO[dataset_name]["name"])
    logger.info("=" * 70)

    if dataset_name == "imagenet":
        logger.info(
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
        logger.info(
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

    logger.info("=" * 70)


def print_dataset_summary() -> None:
    """Print a summary of all supported datasets."""
    logger.info("=" * 70)
    logger.info("SUPPORTED DATASETS FOR H-JEPA")
    logger.info("=" * 70)

    total_size = 0.0

    for name, info in DATASET_INFO.items():
        auto = "Auto" if info["auto_download"] else "Manual"
        logger.info(
            "%s %s %6.1f GB  %9s images  %4d classes",
            f"{info['name']:25}",
            f"{auto:12}",
            info["size_gb"],
            f"{info['num_images']:,}",
            info["num_classes"],
        )
        total_size += info["size_gb"]

    logger.info("-" * 70)
    logger.info("%-25s %19.1f GB", "Total (all datasets)", total_size)
    logger.info("=" * 70)

    logger.info("Recommended for quick start: CIFAR-10 or CIFAR-100 (auto-download)")
    logger.info("Recommended for research: ImageNet or ImageNet-100 (manual download)")
    logger.info("Use download_data.sh script or this module to download datasets.")


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
        logger.info("Current data path: %s", data_path)
        total_gb, used_gb, free_gb = get_disk_usage(data_path)
        logger.info("Disk space: %.1f GB available (%.1f GB total)", free_gb, total_gb)
        return

    # Process each dataset
    for dataset_name in args.datasets:
        dataset_name = dataset_name.lower()

        if dataset_name not in DATASET_INFO:
            logger.error("Unknown dataset: %s", dataset_name)
            logger.error("Supported datasets: %s", ", ".join(DATASET_INFO.keys()))
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
                logger.error("Failed to download %s", dataset_name)

    logger.info("Done!")


if __name__ == "__main__":
    main()

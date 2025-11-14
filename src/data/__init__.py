"""
Data loading and preprocessing utilities for H-JEPA.
"""

from .datasets import (
    JEPATransform,
    JEPAEvalTransform,
    ImageNetDataset,
    ImageNet100Dataset,
    CIFAR10Dataset,
    CIFAR100Dataset,
    STL10Dataset,
    build_dataset,
    build_dataloader,
)

from .download import (
    download_dataset,
    verify_dataset,
    print_dataset_summary,
    print_manual_download_instructions,
    DATASET_INFO,
)

__all__ = [
    # Transforms
    "JEPATransform",
    "JEPAEvalTransform",
    # Datasets
    "ImageNetDataset",
    "ImageNet100Dataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "STL10Dataset",
    # Builders
    "build_dataset",
    "build_dataloader",
    # Download utilities
    "download_dataset",
    "verify_dataset",
    "print_dataset_summary",
    "print_manual_download_instructions",
    "DATASET_INFO",
]

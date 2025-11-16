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

from .multi_dataset import (
    WeightedMultiDataset,
    BalancedMultiDataset,
    build_multi_dataset,
    create_foundation_model_dataset,
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
    # Multi-dataset (foundation models)
    "WeightedMultiDataset",
    "BalancedMultiDataset",
    "build_multi_dataset",
    "create_foundation_model_dataset",
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

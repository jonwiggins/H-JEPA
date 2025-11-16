"""
Data loading and preprocessing utilities for H-JEPA.
"""

from .datasets import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    ImageNet100Dataset,
    ImageNetDataset,
    JEPAEvalTransform,
    JEPATransform,
    STL10Dataset,
    build_dataloader,
    build_dataset,
)
from .download import (
    DATASET_INFO,
    download_dataset,
    print_dataset_summary,
    print_manual_download_instructions,
    verify_dataset,
)
from .multi_dataset import (
    BalancedMultiDataset,
    WeightedMultiDataset,
    build_multi_dataset,
    create_foundation_model_dataset,
)
from .multicrop_dataset import (
    MultiCropDataset,
    MultiCropDatasetRaw,
    build_multicrop_dataloader,
    build_multicrop_dataset,
    multicrop_collate_fn,
)
from .multicrop_transforms import (
    AdaptiveMultiCropTransform,
    MultiCropEvalTransform,
    MultiCropTransform,
    build_multicrop_transform,
)
from .transforms import (
    CutMix,
    DeiTIIIAugmentation,
    DeiTIIIEvalTransform,
    Mixup,
    MixupCutmix,
    RandAugment,
    RandomErasing,
    build_deit3_transform,
)

__all__ = [
    # Basic Transforms
    "JEPATransform",
    "JEPAEvalTransform",
    # DeiT III Transforms
    "RandAugment",
    "Mixup",
    "CutMix",
    "MixupCutmix",
    "RandomErasing",
    "DeiTIIIAugmentation",
    "DeiTIIIEvalTransform",
    "build_deit3_transform",
    # Multi-crop Transforms
    "MultiCropTransform",
    "MultiCropEvalTransform",
    "AdaptiveMultiCropTransform",
    "build_multicrop_transform",
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
    # Multi-crop datasets
    "MultiCropDataset",
    "MultiCropDatasetRaw",
    "multicrop_collate_fn",
    "build_multicrop_dataset",
    "build_multicrop_dataloader",
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

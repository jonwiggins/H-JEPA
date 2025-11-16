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

from .transforms import (
    RandAugment,
    Mixup,
    CutMix,
    MixupCutmix,
    RandomErasing,
    DeiTIIIAugmentation,
    DeiTIIIEvalTransform,
    build_deit3_transform,
)

from .multicrop_transforms import (
    MultiCropTransform,
    MultiCropEvalTransform,
    AdaptiveMultiCropTransform,
    build_multicrop_transform,
)

from .multicrop_dataset import (
    MultiCropDataset,
    MultiCropDatasetRaw,
    multicrop_collate_fn,
    build_multicrop_dataset,
    build_multicrop_dataloader,
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

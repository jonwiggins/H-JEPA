"""
Multi-crop dataset wrapper for H-JEPA.

This module provides a dataset wrapper that applies multi-crop transforms
to existing datasets, enabling multi-crop training with minimal code changes.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch.utils.data import DataLoader, Dataset

from .datasets import CIFAR10Dataset, CIFAR100Dataset, ImageNetDataset, STL10Dataset, build_dataset
from .multicrop_transforms import (
    MultiCropEvalTransform,
    MultiCropTransform,
    build_multicrop_transform,
)


class MultiCropDataset(Dataset[Union[List[torch.Tensor], Tuple[List[torch.Tensor], int]]]):
    """
    Dataset wrapper that applies multi-crop augmentation.

    Wraps any existing dataset and applies multi-crop transforms,
    returning multiple augmented views of each image.

    Args:
        base_dataset: Underlying dataset to wrap
        multicrop_transform: Multi-crop transform to apply
        return_labels: Whether to return labels (default: True)

    Example:
        >>> base_dataset = build_dataset('cifar10', '/data', split='train')
        >>> transform = MultiCropTransform(num_global_crops=2, num_local_crops=6)
        >>> multicrop_dataset = MultiCropDataset(base_dataset, transform)
        >>> crops, label = multicrop_dataset[0]
        >>> print(len(crops))  # 8 (2 global + 6 local)
    """

    def __init__(
        self,
        base_dataset: Dataset[Any],
        multicrop_transform: MultiCropTransform,
        return_labels: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.multicrop_transform = multicrop_transform
        self.return_labels = return_labels

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], int]]:
        """
        Get item with multi-crop augmentation.

        Args:
            idx: Index

        Returns:
            If return_labels=True: (crops, label) where crops is a list of tensors
            If return_labels=False: crops (list of tensors)
        """
        # Get base item
        item = self.base_dataset[idx]

        if isinstance(item, tuple):
            image, label = item
        else:
            image = item
            label = None

        # Apply multi-crop transform
        # Note: base_dataset may already have a transform applied
        # We need to get the raw image
        # For now, assume the image is already a PIL Image or tensor
        crops = self.multicrop_transform(image)

        if self.return_labels and label is not None:
            return crops, label
        else:
            return crops

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for adaptive transforms.

        Args:
            epoch: Current epoch
        """
        if hasattr(self.multicrop_transform, "set_epoch"):
            self.multicrop_transform.set_epoch(epoch)

    @property
    def classes(self) -> Optional[List[str]]:
        """Get classes from base dataset."""
        if hasattr(self.base_dataset, "classes"):
            return cast(List[str], self.base_dataset.classes)
        return None

    @property
    def num_global_crops(self) -> int:
        """Number of global crops."""
        return self.multicrop_transform.num_global_crops

    @property
    def num_local_crops(self) -> int:
        """Number of local crops."""
        return self.multicrop_transform.num_local_crops

    @property
    def total_crops(self) -> int:
        """Total number of crops per image."""
        return self.num_global_crops + self.num_local_crops


class MultiCropDatasetRaw(Dataset[Tuple[Union[List[torch.Tensor], torch.Tensor], int]]):
    """
    Multi-crop dataset that loads raw images without pre-transforms.

    This version wraps the base dataset classes directly and ensures
    that transforms are only applied once (in the multi-crop transform).

    Args:
        dataset_name: Name of dataset ('cifar10', 'cifar100', 'imagenet', etc.)
        data_path: Path to data directory
        split: 'train' or 'val'
        multicrop_config: Configuration dict for multi-crop transform
        download: Whether to download dataset if not present (default: True)

    Example:
        >>> config = {
        ...     'num_global_crops': 2,
        ...     'num_local_crops': 6,
        ...     'global_crop_size': 224,
        ...     'local_crop_size': 96,
        ... }
        >>> dataset = MultiCropDatasetRaw(
        ...     dataset_name='cifar10',
        ...     data_path='/data',
        ...     split='train',
        ...     multicrop_config=config,
        ... )
    """

    def __init__(
        self,
        dataset_name: str,
        data_path: Union[str, Path],
        split: str = "train",
        multicrop_config: Optional[Dict[str, Any]] = None,
        download: bool = True,
    ) -> None:
        self.dataset_name = dataset_name.lower()
        self.data_path = Path(data_path)
        self.split = split

        # Default multi-crop config
        if multicrop_config is None:
            multicrop_config = {
                "num_global_crops": 2,
                "num_local_crops": 6,
                "global_crop_size": 224,
                "local_crop_size": 96,
                "global_crop_scale": (0.4, 1.0),
                "local_crop_scale": (0.05, 0.4),
            }

        # Build multi-crop transform
        if split == "train":
            self.transform: Union[MultiCropTransform, MultiCropEvalTransform] = (
                build_multicrop_transform(**multicrop_config)
            )
        else:
            # For validation, use single-crop evaluation transform
            self.transform = MultiCropEvalTransform(
                crop_size=multicrop_config.get("global_crop_size", 224)
            )

        # Build base dataset WITHOUT transforms (we'll apply our own)
        self.base_dataset = build_dataset(
            dataset_name=dataset_name,
            data_path=data_path,
            split=split,
            image_size=multicrop_config.get("global_crop_size", 224),
            color_jitter=None,  # Handled by multicrop transform
            transform=None,  # No transform - we apply it ourselves
            download=download,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[Union[List[torch.Tensor], torch.Tensor], int]:
        """
        Get item with multi-crop transform applied.

        Args:
            idx: Index

        Returns:
            (crops, label) where crops is a list of tensors (train) or single tensor (val)
        """
        # Get raw image and label from base dataset
        # The base_dataset has a .dataset attribute that holds the actual dataset
        item = getattr(self.base_dataset, "dataset")[idx]

        if isinstance(item, tuple):
            image, label = item
        else:
            image = item
            label = -1

        # Apply transform (multi-crop for train, single crop for val)
        transformed = self.transform(image)

        return transformed, label

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for adaptive transforms."""
        if hasattr(self.transform, "set_epoch"):
            self.transform.set_epoch(epoch)

    @property
    def classes(self) -> List[str]:
        """Get classes from base dataset."""
        return cast(List[str], cast(Any, self.base_dataset).classes)

    @property
    def num_global_crops(self) -> int:
        """Number of global crops."""
        if self.split == "train" and isinstance(self.transform, MultiCropTransform):
            return self.transform.num_global_crops
        return 1

    @property
    def num_local_crops(self) -> int:
        """Number of local crops."""
        if self.split == "train" and isinstance(self.transform, MultiCropTransform):
            return self.transform.num_local_crops
        return 0


def multicrop_collate_fn(batch: List[Any]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Custom collate function for multi-crop datasets.

    Converts a batch of (crops_list, label) into batched crops.

    Args:
        batch: List of (crops, label) tuples from MultiCropDataset

    Returns:
        (batched_crops, labels) where:
        - batched_crops: List of tensors, one per crop type
          [global_0_batch, global_1_batch, local_0_batch, ...]
        - labels: Tensor of labels (batch_size,)

    Example:
        >>> # Input: [(crops_list_0, label_0), (crops_list_1, label_1), ...]
        >>> # Output: ([global_0_batch, global_1_batch, ...], labels_batch)
    """
    if len(batch) == 0:
        return [], torch.tensor([])

    # Check if items are (crops, label) or just crops
    if isinstance(batch[0], tuple):
        crops_batch, labels_tuple = zip(*batch)
        labels_tensor = torch.tensor(labels_tuple)
        crops_list: Any = crops_batch
    else:
        crops_list = batch
        labels_tensor = torch.tensor([])

    # Determine number of crops from first item
    num_crops = len(crops_list[0])

    # Stack each crop type across the batch
    batched_crops: List[torch.Tensor] = []
    for crop_idx in range(num_crops):
        crop_tensors = [item[crop_idx] for item in crops_list]
        batched_crops.append(torch.stack(crop_tensors))

    return batched_crops, labels_tensor


def build_multicrop_dataset(
    dataset_name: str,
    data_path: Union[str, Path],
    split: str = "train",
    num_global_crops: int = 2,
    num_local_crops: int = 6,
    global_crop_size: int = 224,
    local_crop_size: int = 96,
    global_crop_scale: Tuple[float, float] = (0.4, 1.0),
    local_crop_scale: Tuple[float, float] = (0.05, 0.4),
    global_color_jitter: float = 0.4,
    local_color_jitter: float = 0.4,
    adaptive: bool = False,
    download: bool = True,
    **kwargs: Any,
) -> MultiCropDatasetRaw:
    """
    Factory function to build multi-crop datasets.

    Args:
        dataset_name: Name of dataset
        data_path: Path to data directory
        split: 'train' or 'val'
        num_global_crops: Number of global crops
        num_local_crops: Number of local crops
        global_crop_size: Size of global crops
        local_crop_size: Size of local crops
        global_crop_scale: Scale range for global crops
        local_crop_scale: Scale range for local crops
        global_color_jitter: Color jitter for global crops
        local_color_jitter: Color jitter for local crops
        adaptive: Whether to use adaptive multi-crop
        download: Whether to download dataset
        **kwargs: Additional arguments

    Returns:
        MultiCropDatasetRaw instance

    Example:
        >>> dataset = build_multicrop_dataset(
        ...     'cifar10',
        ...     '/data',
        ...     split='train',
        ...     num_global_crops=2,
        ...     num_local_crops=6,
        ... )
    """
    multicrop_config = {
        "num_global_crops": num_global_crops,
        "num_local_crops": num_local_crops,
        "global_crop_size": global_crop_size,
        "local_crop_size": local_crop_size,
        "global_crop_scale": global_crop_scale,
        "local_crop_scale": local_crop_scale,
        "global_color_jitter": global_color_jitter,
        "local_color_jitter": local_color_jitter,
        "adaptive": adaptive,
        **kwargs,
    }

    return MultiCropDatasetRaw(
        dataset_name=dataset_name,
        data_path=data_path,
        split=split,
        multicrop_config=multicrop_config,
        download=download,
    )


def build_multicrop_dataloader(
    dataset: MultiCropDatasetRaw,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs: Any,
) -> DataLoader[Tuple[Union[List[torch.Tensor], torch.Tensor], int]]:
    """
    Build a DataLoader for multi-crop datasets.

    Args:
        dataset: Multi-crop dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle
        drop_last: Whether to drop last incomplete batch
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader with custom collate function

    Example:
        >>> dataset = build_multicrop_dataset('cifar10', '/data')
        >>> loader = build_multicrop_dataloader(dataset, batch_size=32)
        >>> for crops, labels in loader:
        ...     # crops is a list of tensors
        ...     print(len(crops))  # 8 (2 global + 6 local)
        ...     print(crops[0].shape)  # (32, 3, 224, 224)
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=multicrop_collate_fn,
        **kwargs,
    )


if __name__ == "__main__":
    # Demo of multi-crop dataset
    print("Multi-Crop Dataset Demo")
    print("=" * 70)

    # Build dataset
    print("\nBuilding CIFAR-10 multi-crop dataset...")
    dataset = build_multicrop_dataset(
        dataset_name="cifar10",
        data_path="/tmp/data",
        split="train",
        num_global_crops=2,
        num_local_crops=6,
        global_crop_size=224,
        local_crop_size=96,
        download=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of global crops: {dataset.num_global_crops}")
    print(f"Number of local crops: {dataset.num_local_crops}")

    # Get a sample
    print("\nSample item:")
    crops, label = dataset[0]
    print(f"  Number of crops: {len(crops)}")
    print(f"  Global crop shapes: {[crops[i].shape for i in range(2)]}")
    print(f"  Local crop shapes: {[crops[i].shape for i in range(2, 8)]}")
    print(f"  Label: {label}")

    # Build dataloader
    print("\nBuilding dataloader...")
    dataloader = build_multicrop_dataloader(
        dataset,
        batch_size=4,
        num_workers=0,  # Use 0 for demo
        shuffle=True,
    )

    print(f"Batches per epoch: {len(dataloader)}")

    # Get a batch
    print("\nSample batch:")
    batch_crops, batch_labels = next(iter(dataloader))
    print(f"  Number of crop types: {len(batch_crops)}")
    print(f"  Global crop 0 shape: {batch_crops[0].shape}")
    print(f"  Global crop 1 shape: {batch_crops[1].shape}")
    print(f"  Local crop 0 shape: {batch_crops[2].shape}")
    print(f"  Labels shape: {batch_labels.shape}")

    print("\n" + "=" * 70)
    print("Demo complete!")

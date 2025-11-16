"""
Dataset implementations for H-JEPA training.

This module provides dataset classes for various computer vision datasets with
custom transforms optimized for JEPA (Joint-Embedding Predictive Architecture).
Unlike traditional contrastive learning, JEPA does not require heavy augmentations.
"""

import os
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class JEPATransform:
    """
    Transform for JEPA training.

    Unlike contrastive learning (SimCLR, MoCo), JEPA doesn't need aggressive
    augmentations since it learns from masked prediction rather than instance discrimination.
    We use minimal preprocessing: resize, crop, normalize.
    """

    def __init__(
        self,
        image_size: int = 224,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        horizontal_flip: bool = True,
        color_jitter: Optional[float] = None,
    ):
        """
        Args:
            image_size: Target image size (square)
            crop_scale: Random crop scale range
            interpolation: Image interpolation mode
            mean: ImageNet normalization mean
            std: ImageNet normalization std
            horizontal_flip: Whether to apply random horizontal flip
            color_jitter: Color jitter strength (None to disable)
        """
        transform_list = []

        # Resize with some margin
        transform_list.append(transforms.Resize(image_size + 32, interpolation=interpolation))

        # Random crop
        transform_list.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=crop_scale,
                interpolation=interpolation,
            )
        )

        # Optional horizontal flip
        if horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        # Optional mild color jitter (much milder than contrastive learning)
        if color_jitter is not None and color_jitter > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter * 0.4,
                    contrast=color_jitter * 0.4,
                    saturation=color_jitter * 0.2,
                    hue=color_jitter * 0.1,
                )
            )

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalize
        transform_list.append(transforms.Normalize(mean=mean, std=std))

        self.transform = transforms.Compose(transform_list)

    def __call__(self, img):
        return self.transform(img)


class JEPAEvalTransform:
    """
    Transform for JEPA evaluation/validation.

    Simple center crop and normalize, no augmentations.
    """

    def __init__(
        self,
        image_size: int = 224,
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            image_size: Target image size
            interpolation: Image interpolation mode
            mean: Normalization mean
            std: Normalization std
        """
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14), interpolation=interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transform(img)


class ImageNetDataset(Dataset):
    """
    ImageNet dataset with JEPA transforms.

    This class wraps torchvision's ImageFolder for ImageNet data.
    ImageNet must be manually downloaded due to terms of service.

    Expected directory structure:
        data_path/
            train/
                n01440764/
                    n01440764_18.JPEG
                    ...
                n01443537/
                    ...
            val/
                n01440764/
                    ...
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        image_size: int = 224,
        color_jitter: Optional[float] = 0.4,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_path: Path to ImageNet directory
            split: 'train' or 'val'
            image_size: Target image size
            color_jitter: Color jitter strength (only for training)
            transform: Custom transform (overrides default JEPA transform)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size

        # Determine split directory
        split_dir = self.data_path / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"ImageNet {split} directory not found at {split_dir}.\n"
                f"Please download ImageNet manually from https://image-net.org/download.php\n"
                f"and organize it in the expected directory structure."
            )

        # Set up transform
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = JEPATransform(
                image_size=image_size,
                color_jitter=color_jitter,
            )
        else:
            self.transform = JEPAEvalTransform(image_size=image_size)

        # Use torchvision's ImageFolder
        self.dataset = datasets.ImageFolder(split_dir, transform=self.transform)

        print(
            f"Loaded ImageNet {split} split: {len(self.dataset)} images, "
            f"{len(self.dataset.classes)} classes"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx


class ImageNet100Dataset(ImageNetDataset):
    """
    ImageNet-100 subset for faster experimentation.

    Uses a subset of 100 classes from ImageNet. This is useful for
    quick prototyping and testing before scaling to full ImageNet.

    The classes are selected based on common ImageNet-100 benchmarks.
    """

    # Standard ImageNet-100 classes (from iNaturalist benchmark)
    IMAGENET100_CLASSES = [
        "n01498041",
        "n01537544",
        "n01580077",
        "n01592084",
        "n01632777",
        "n01644373",
        "n01665541",
        "n01675722",
        "n01688243",
        "n01729977",
        "n01775062",
        "n01818515",
        "n01843383",
        "n01883070",
        "n01950731",
        "n02002724",
        "n02013706",
        "n02092339",
        "n02093754",
        "n02096585",
        "n02097298",
        "n02098413",
        "n02099712",
        "n02106662",
        "n02110063",
        "n02110341",
        "n02111129",
        "n02114548",
        "n02116738",
        "n02120079",
        "n02123045",
        "n02124075",
        "n02125311",
        "n02129165",
        "n02132136",
        "n02165456",
        "n02190166",
        "n02206856",
        "n02279972",
        "n02317335",
        "n02326432",
        "n02342885",
        "n02363005",
        "n02391049",
        "n02395406",
        "n02398521",
        "n02410509",
        "n02415577",
        "n02423022",
        "n02437616",
        "n02445715",
        "n02454379",
        "n02483708",
        "n02486410",
        "n02504458",
        "n02509815",
        "n02666196",
        "n02669723",
        "n02699494",
        "n02769748",
        "n02788148",
        "n02791270",
        "n02793495",
        "n02795169",
        "n02802426",
        "n02808440",
        "n02814533",
        "n02814860",
        "n02815834",
        "n02823428",
        "n02837789",
        "n02841315",
        "n02843684",
        "n02883205",
        "n02906734",
        "n02909870",
        "n02917067",
        "n02927161",
        "n02948072",
        "n02950826",
        "n02963159",
        "n02977058",
        "n02988304",
        "n02999410",
        "n03014705",
        "n03026506",
        "n03042490",
        "n03085013",
        "n03089624",
        "n03100240",
        "n03126707",
        "n03160309",
        "n03179701",
        "n03220513",
        "n03347037",
        "n03388549",
        "n03476684",
        "n03535780",
        "n03584254",
        "n03627232",
    ]

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        image_size: int = 224,
        color_jitter: Optional[float] = 0.4,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_path: Path to ImageNet directory
            split: 'train' or 'val'
            image_size: Target image size
            color_jitter: Color jitter strength
            transform: Custom transform
        """
        # Initialize parent class
        super().__init__(
            data_path=data_path,
            split=split,
            image_size=image_size,
            color_jitter=color_jitter,
            transform=transform,
        )

        # Filter to only include ImageNet-100 classes
        self._filter_classes()

        print(
            f"Filtered to ImageNet-100: {len(self)} images, "
            f"{len(self.IMAGENET100_CLASSES)} classes"
        )

    def _filter_classes(self):
        """Filter dataset to only include ImageNet-100 classes."""
        # Get indices of samples that belong to ImageNet-100 classes
        valid_indices = []
        for idx in range(len(self.dataset)):
            path, _ = self.dataset.samples[idx]
            class_name = Path(path).parent.name
            if class_name in self.IMAGENET100_CLASSES:
                valid_indices.append(idx)

        # Store original dataset and create filtered version
        self._valid_indices = valid_indices
        self._original_dataset = self.dataset

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        original_idx = self._valid_indices[idx]
        return self._original_dataset[original_idx]


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset with JEPA transforms.

    Automatically downloads CIFAR-10 if not present.
    10 classes, 50k training images, 10k test images, 32x32 resolution.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        image_size: int = 224,
        color_jitter: Optional[float] = 0.4,
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        """
        Args:
            data_path: Path to data directory
            split: 'train' or 'val' (val uses test split)
            image_size: Target image size (images will be resized from 32x32)
            color_jitter: Color jitter strength
            transform: Custom transform
            download: Whether to download if not present
        """
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size

        # Set up transform
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = JEPATransform(
                image_size=image_size,
                color_jitter=color_jitter,
                crop_scale=(0.8, 1.0),  # Slightly different for small images
            )
        else:
            self.transform = JEPAEvalTransform(image_size=image_size)

        # Load CIFAR-10
        train = split == "train"
        self.dataset = datasets.CIFAR10(
            root=self.data_path,
            train=train,
            transform=self.transform,
            download=download,
        )

        print(f"Loaded CIFAR-10 {split} split: {len(self.dataset)} images")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @property
    def classes(self):
        return self.dataset.classes


class CIFAR100Dataset(Dataset):
    """
    CIFAR-100 dataset with JEPA transforms.

    Automatically downloads CIFAR-100 if not present.
    100 classes, 50k training images, 10k test images, 32x32 resolution.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        image_size: int = 224,
        color_jitter: Optional[float] = 0.4,
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        """
        Args:
            data_path: Path to data directory
            split: 'train' or 'val' (val uses test split)
            image_size: Target image size
            color_jitter: Color jitter strength
            transform: Custom transform
            download: Whether to download if not present
        """
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size

        # Set up transform
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = JEPATransform(
                image_size=image_size,
                color_jitter=color_jitter,
                crop_scale=(0.8, 1.0),
            )
        else:
            self.transform = JEPAEvalTransform(image_size=image_size)

        # Load CIFAR-100
        train = split == "train"
        self.dataset = datasets.CIFAR100(
            root=self.data_path,
            train=train,
            transform=self.transform,
            download=download,
        )

        print(f"Loaded CIFAR-100 {split} split: {len(self.dataset)} images")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @property
    def classes(self):
        return self.dataset.classes


class STL10Dataset(Dataset):
    """
    STL-10 dataset with JEPA transforms.

    Automatically downloads STL-10 if not present.
    10 classes, 5k training images, 8k test images, 100k unlabeled, 96x96 resolution.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        image_size: int = 224,
        color_jitter: Optional[float] = 0.4,
        transform: Optional[Callable] = None,
        download: bool = True,
        use_unlabeled: bool = True,
    ):
        """
        Args:
            data_path: Path to data directory
            split: 'train', 'val' (uses test), or 'unlabeled'
            image_size: Target image size
            color_jitter: Color jitter strength
            transform: Custom transform
            download: Whether to download if not present
            use_unlabeled: Whether to include unlabeled data for training
        """
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size

        # Set up transform
        if transform is not None:
            self.transform = transform
        elif split == "train" or split == "unlabeled":
            self.transform = JEPATransform(
                image_size=image_size,
                color_jitter=color_jitter,
            )
        else:
            self.transform = JEPAEvalTransform(image_size=image_size)

        # Map our split names to STL10 split names
        stl10_split = split
        if split == "val":
            stl10_split = "test"
        elif split == "unlabeled":
            stl10_split = "unlabeled"

        # Load STL-10
        self.dataset = datasets.STL10(
            root=self.data_path,
            split=stl10_split,
            transform=self.transform,
            download=download,
        )

        # If training and use_unlabeled, we might want to combine labeled + unlabeled
        # For now, we keep them separate

        print(f"Loaded STL-10 {split} split: {len(self.dataset)} images")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @property
    def classes(self):
        return self.dataset.classes if hasattr(self.dataset, "classes") else None


def build_dataset(
    dataset_name: str,
    data_path: Union[str, Path],
    split: str = "train",
    image_size: int = 224,
    color_jitter: Optional[float] = 0.4,
    download: bool = True,
    **kwargs,
) -> Dataset:
    """
    Factory function to build datasets.

    Args:
        dataset_name: Name of dataset ('imagenet', 'imagenet100', 'cifar10', 'cifar100', 'stl10')
        data_path: Path to data directory
        split: 'train' or 'val'
        image_size: Target image size
        color_jitter: Color jitter strength (for training only)
        download: Whether to download dataset if not present
        **kwargs: Additional dataset-specific arguments

    Returns:
        Dataset instance

    Examples:
        >>> train_dataset = build_dataset('cifar10', '/data', split='train')
        >>> val_dataset = build_dataset('cifar10', '/data', split='val')
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "imagenet":
        return ImageNetDataset(
            data_path=data_path,
            split=split,
            image_size=image_size,
            color_jitter=color_jitter if split == "train" else None,
            **kwargs,
        )
    elif dataset_name == "imagenet100":
        return ImageNet100Dataset(
            data_path=data_path,
            split=split,
            image_size=image_size,
            color_jitter=color_jitter if split == "train" else None,
            **kwargs,
        )
    elif dataset_name == "cifar10":
        return CIFAR10Dataset(
            data_path=data_path,
            split=split,
            image_size=image_size,
            color_jitter=color_jitter if split == "train" else None,
            download=download,
            **kwargs,
        )
    elif dataset_name == "cifar100":
        return CIFAR100Dataset(
            data_path=data_path,
            split=split,
            image_size=image_size,
            color_jitter=color_jitter if split == "train" else None,
            download=download,
            **kwargs,
        )
    elif dataset_name == "stl10":
        return STL10Dataset(
            data_path=data_path,
            split=split,
            image_size=image_size,
            color_jitter=color_jitter if split == "train" else None,
            download=download,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: imagenet, imagenet100, cifar10, cifar100, stl10"
        )


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Build a DataLoader from a dataset.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        **kwargs,
    )


if __name__ == "__main__":
    # Quick test of datasets
    print("Testing CIFAR-10...")
    dataset = build_dataset("cifar10", "/tmp/data", split="train", download=True)
    print(f"Dataset size: {len(dataset)}")
    img, label = dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")

    print("\nTesting CIFAR-100...")
    dataset = build_dataset("cifar100", "/tmp/data", split="train", download=True)
    print(f"Dataset size: {len(dataset)}")

    print("\nDataset tests completed!")

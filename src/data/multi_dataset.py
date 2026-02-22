"""
Multi-dataset support for foundation model training.

This module enables training on multiple datasets simultaneously,
which is essential for building foundation models with broad capabilities.
"""

import logging
import random
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, Dataset

logger = logging.getLogger(__name__)


class WeightedMultiDataset(Dataset[Any]):
    """
    Combines multiple datasets with optional sampling weights.

    Unlike simple ConcatDataset, this allows:
    1. Weighted sampling (e.g., oversample ImageNet, undersample CIFAR)
    2. Balancing datasets of different sizes
    3. Per-dataset statistics tracking

    This is how foundation models like CLIP and DINOv2 are trained.

    Args:
        datasets: List of Dataset objects
        weights: Sampling weights for each dataset (None for uniform)
        dataset_names: Names for logging/tracking
        temperature: Temperature for weight softmax (higher = more uniform)

    Example:
        >>> imagenet = build_dataset('imagenet', '/data', 'train')
        >>> coco = build_dataset('coco', '/data', 'train')
        >>> multi = WeightedMultiDataset(
        ...     datasets=[imagenet, coco],
        ...     weights=[0.7, 0.3],  # 70% ImageNet, 30% COCO
        ...     dataset_names=['imagenet', 'coco']
        ... )
    """

    def __init__(
        self,
        datasets: list[Dataset[Any]],
        weights: list[float] | None = None,
        dataset_names: list[str] | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.datasets = datasets
        self.num_datasets = len(datasets)

        # Dataset names for tracking
        if dataset_names is None:
            self.dataset_names = [f"dataset_{i}" for i in range(self.num_datasets)]
        else:
            self.dataset_names = dataset_names

        # Dataset sizes
        self.dataset_sizes: list[int] = [len(d) for d in datasets]  # type: ignore[arg-type]
        self.total_size = sum(self.dataset_sizes)

        # Compute sampling weights
        if weights is None:
            # Uniform sampling across datasets (not samples!)
            # This balances datasets: small dataset gets upsampled
            weights = [1.0 / self.num_datasets] * self.num_datasets
        else:
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

        # Apply temperature
        if temperature != 1.0:
            weights = [w ** (1.0 / temperature) for w in weights]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

        self.weights = weights

        # Create cumulative offsets for each dataset
        self.cumulative_sizes = []
        cumsum = 0
        for size in self.dataset_sizes:
            self.cumulative_sizes.append(cumsum)
            cumsum += size

        # Compute effective dataset size based on sampling
        # We oversample smaller datasets to match weights
        self.effective_size = self._compute_effective_size()

        logger.info("Created WeightedMultiDataset:")
        for i, (name, size, weight) in enumerate(
            zip(self.dataset_names, self.dataset_sizes, self.weights)
        ):
            logger.info(
                "  %s: %s images (%.1f%% sampling probability)", name, f"{size:,}", weight * 100
            )
        logger.info("Total images: %s", f"{self.total_size:,}")
        logger.info("Effective size (one epoch): %s", f"{self.effective_size:,}")

    def _compute_effective_size(self) -> int:
        """
        Compute effective dataset size for one epoch.

        With weighted sampling, we want one epoch to see a representative
        sample from all datasets according to their weights.
        """
        # Use total size as baseline, then adjust based on weights
        # This ensures we don't undersample any dataset too much
        return max(self.dataset_sizes)  # At least go through largest dataset

    def __len__(self) -> int:
        return self.effective_size

    def __getitem__(self, idx: int) -> Any:
        """
        Sample from datasets according to weights.

        Note: idx is ignored for weighted sampling. We randomly sample
        a dataset based on weights, then randomly sample from that dataset.
        """
        # Choose dataset based on weights
        dataset_idx = random.choices(range(self.num_datasets), weights=self.weights, k=1)[0]

        # Sample random item from chosen dataset
        item_idx = random.randint(0, self.dataset_sizes[dataset_idx] - 1)

        # Get item
        item = self.datasets[dataset_idx][item_idx]

        # Optionally add dataset index for tracking
        if isinstance(item, tuple):
            # Return (image, label, dataset_idx)
            return (*item, dataset_idx)
        else:
            return (item, dataset_idx)

    def get_dataset_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics about each dataset."""
        stats = {}
        for i, name in enumerate(self.dataset_names):
            stats[name] = {
                "size": self.dataset_sizes[i],
                "weight": self.weights[i],
                "expected_samples_per_epoch": int(self.effective_size * self.weights[i]),
            }
        return stats


class BalancedMultiDataset(Dataset[Any]):
    """
    Balanced sampling from multiple datasets.

    Ensures each dataset contributes equally in each epoch,
    regardless of size. Small datasets are oversampled,
    large datasets are undersampled.

    This is useful when you want equal representation from
    diverse sources (e.g., photos + art + medical images).

    Args:
        datasets: List of Dataset objects
        dataset_names: Names for logging/tracking
        samples_per_dataset: How many samples from each dataset per epoch
    """

    def __init__(
        self,
        datasets: list[Dataset[Any]],
        dataset_names: list[str] | None = None,
        samples_per_dataset: int | None = None,
    ) -> None:
        self.datasets = datasets
        self.num_datasets = len(datasets)

        if dataset_names is None:
            self.dataset_names = [f"dataset_{i}" for i in range(self.num_datasets)]
        else:
            self.dataset_names = dataset_names

        self.dataset_sizes: list[int] = [len(d) for d in datasets]  # type: ignore[arg-type]

        # Determine samples per dataset
        if samples_per_dataset is None:
            # Use size of smallest dataset
            samples_per_dataset = min(self.dataset_sizes)
        self.samples_per_dataset = samples_per_dataset

        # Total size is samples_per_dataset * num_datasets
        self._length = self.samples_per_dataset * self.num_datasets

        # Pre-generate sampling indices for reproducibility
        self.resample_indices()

        logger.info("Created BalancedMultiDataset:")
        for name, size in zip(self.dataset_names, self.dataset_sizes):
            logger.info(
                "  %s: %s images -> %s samples/epoch",
                name,
                f"{size:,}",
                f"{self.samples_per_dataset:,}",
            )
        logger.info("Total samples per epoch: %s", f"{self._length:,}")

    def resample_indices(self) -> None:
        """Generate new random sampling indices."""
        self.indices: list[tuple[int, int]] = []
        for dataset_idx, size in enumerate(self.dataset_sizes):
            # Sample with replacement if needed
            if size >= self.samples_per_dataset:
                # Undersample
                dataset_indices = random.sample(range(size), self.samples_per_dataset)
            else:
                # Oversample with replacement
                dataset_indices = random.choices(range(size), k=self.samples_per_dataset)

            # Store as (dataset_idx, item_idx) pairs
            for item_idx in dataset_indices:
                self.indices.append((dataset_idx, item_idx))

        # Shuffle so datasets are interleaved
        random.shuffle(self.indices)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Any:
        dataset_idx, item_idx = self.indices[idx]
        item = self.datasets[dataset_idx][item_idx]

        # Add dataset index
        if isinstance(item, tuple):
            return (*item, dataset_idx)
        else:
            return (item, dataset_idx)


def build_multi_dataset(
    dataset_configs: list[dict[str, Any]],
    data_path: str | Path,
    split: str = "train",
    sampling_strategy: str = "weighted",
    **kwargs: Any,
) -> Dataset[Any]:
    """
    Build a multi-dataset for foundation model training.

    Args:
        dataset_configs: List of dicts with 'name' and optional 'weight', 'path'
        data_path: Base path for datasets
        split: 'train' or 'val'
        sampling_strategy: 'weighted', 'balanced', or 'concat'
        **kwargs: Additional args passed to all datasets

    Returns:
        Multi-dataset instance

    Example:
        >>> configs = [
        ...     {'name': 'imagenet', 'weight': 0.7},
        ...     {'name': 'coco', 'weight': 0.2},
        ...     {'name': 'places365', 'weight': 0.1},
        ... ]
        >>> dataset = build_multi_dataset(configs, '/data', 'train', 'weighted')
    """
    from .datasets import build_dataset

    datasets = []
    names = []
    weights = []

    for config in dataset_configs:
        # Get dataset name
        name = config["name"]
        names.append(name)

        # Get dataset-specific path if provided
        dataset_path = config.get("path", data_path)

        # Get weight
        weight = config.get("weight", 1.0)
        weights.append(weight)

        # Build dataset
        dataset = build_dataset(
            dataset_name=name,
            data_path=dataset_path,
            split=split,
            **kwargs,
        )
        datasets.append(dataset)

    # Create multi-dataset based on strategy
    if sampling_strategy == "weighted":
        return WeightedMultiDataset(
            datasets=datasets,
            weights=weights,
            dataset_names=names,
        )
    elif sampling_strategy == "balanced":
        return BalancedMultiDataset(
            datasets=datasets,
            dataset_names=names,
        )
    elif sampling_strategy == "concat":
        # Simple concatenation (no weighting)
        return ConcatDataset(datasets)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")


def create_foundation_model_dataset(
    scale: str = "mini",  # mini, medium, large
    data_path: str | Path = "./data",
    split: str = "train",
    **kwargs: Any,
) -> Dataset[Any]:
    """
    Create a pre-configured foundation model dataset.

    Args:
        scale: Dataset scale ('mini', 'medium', 'large')
        data_path: Path to datasets
        split: 'train' or 'val'
        **kwargs: Additional args

    Returns:
        Multi-dataset configured for foundation model training

    Scales:
        mini: ~250K images (ImageNet-100 + STL-10) - 12-18 hours on M1 Max
        medium: ~1.4M images (ImageNet + STL-10) - 7-10 days on M1 Max
        large: Custom (requires specifying datasets)
    """
    if scale == "mini":
        # Good for M1 Max: ~250K images, ~18 hours training
        configs = [
            {"name": "imagenet100", "weight": 0.6},  # 126K images
            {"name": "stl10", "weight": 0.3},  # 100K images
            {"name": "cifar100", "weight": 0.1},  # 50K images (for diversity)
        ]
    elif scale == "medium":
        # Serious foundation model: ~1.4M images
        configs = [
            {"name": "imagenet", "weight": 0.9},  # 1.28M images
            {"name": "stl10", "weight": 0.1},  # 100K images
        ]
    elif scale == "large":
        raise NotImplementedError(
            "Large scale requires custom dataset configuration. "
            "Use build_multi_dataset() with your own configs."
        )
    else:
        raise ValueError(f"Unknown scale: {scale}")

    return build_multi_dataset(
        dataset_configs=configs,
        data_path=data_path,
        split=split,
        sampling_strategy="weighted",
        **kwargs,
    )

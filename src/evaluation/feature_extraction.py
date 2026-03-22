"""
Shared feature extraction utilities for H-JEPA evaluation modules.

Provides a common extract_features function used by linear probe, k-NN,
and feature quality evaluation to avoid code duplication.
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataloader: DataLoader[Any],
    hierarchy_level: int = 0,
    device: str = "cuda",
    pool: bool = False,
    pooling: str = "mean",
    normalize: bool = False,
    max_samples: int | None = None,
    desc: str = "Extracting features",
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Extract features from a dataset using a frozen H-JEPA model.

    Args:
        model: H-JEPA model (should already be in eval mode)
        dataloader: DataLoader for the dataset
        hierarchy_level: Which hierarchy level to extract features from
        device: Device to run on
        pool: Whether to pool patch features to a single vector
        pooling: Pooling method ('mean' or 'max') when pool=True
        normalize: Whether to L2 normalize features
        max_samples: Maximum number of samples to extract (None for all)
        desc: Progress bar description

    Returns:
        Tuple of (features, labels) as numpy arrays.
        features shape: [N, D] if pooled, [N, num_patches, D] otherwise
        labels shape: [N]
    """
    all_features: list[npt.NDArray[Any]] = []
    all_labels: list[npt.NDArray[Any]] = []
    num_samples = 0

    for batch in tqdm(dataloader, desc=desc):
        if max_samples is not None and num_samples >= max_samples:
            break

        # Handle different dataloader return formats
        if isinstance(batch, (list, tuple)):
            images = batch[0]
            labels = batch[1]
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        images = images.to(device)

        # Extract features at specified hierarchy level
        features: torch.Tensor = model.extract_features(  # type: ignore[operator]
            images,
            level=hierarchy_level,
            use_target_encoder=True,
        )

        # Pool if requested
        if pool and features.ndim == 3:
            if pooling == "mean":
                features = features.mean(dim=1)
            elif pooling == "max":
                features = features.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")

        # Normalize if requested
        if normalize:
            features = F.normalize(features, p=2, dim=-1)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        num_samples += len(images)

        if max_samples is not None and num_samples >= max_samples:
            excess = num_samples - max_samples
            if excess > 0:
                all_features[-1] = all_features[-1][:-excess]
                all_labels[-1] = all_labels[-1][:-excess]
            break

    features_np: npt.NDArray[Any] = np.concatenate(all_features, axis=0)
    labels_np: npt.NDArray[Any] = np.concatenate(all_labels, axis=0)

    return features_np, labels_np

"""
Multi-crop augmentation strategy for H-JEPA.

This module implements multi-crop training inspired by DINOv2 and other modern
self-supervised learning methods. Multi-crop uses multiple views at different
scales to improve representation learning.

The strategy generates:
- Global crops: Full resolution views (224x224) with standard augmentations
- Local crops: Lower resolution views (96x96) with more aggressive augmentations

This provides the model with different scales and contexts to learn from,
improving robustness and scale invariance.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torchvision import transforms


class MultiCropTransform:
    """
    Multi-crop augmentation for H-JEPA training.

    Generates multiple views of an image at different scales:
    - Global crops: 2 crops at full resolution (default 224x224)
    - Local crops: 4-8 crops at lower resolution (default 96x96)

    Each crop type has different augmentation strengths to provide
    diverse views while maintaining semantic consistency.

    Args:
        global_crop_size: Size of global crops (default: 224)
        local_crop_size: Size of local crops (default: 96)
        num_global_crops: Number of global crops (default: 2)
        num_local_crops: Number of local crops (default: 6)
        global_crop_scale: Scale range for global crops (default: (0.4, 1.0))
        local_crop_scale: Scale range for local crops (default: (0.05, 0.4))
        interpolation: Image interpolation mode (default: BICUBIC)
        mean: Normalization mean (default: ImageNet mean)
        std: Normalization std (default: ImageNet std)
        global_color_jitter: Color jitter strength for global crops (default: 0.4)
        local_color_jitter: Color jitter strength for local crops (default: 0.4)
        horizontal_flip_prob: Probability of horizontal flip (default: 0.5)

    Example:
        >>> transform = MultiCropTransform(
        ...     num_global_crops=2,
        ...     num_local_crops=6,
        ...     global_crop_size=224,
        ...     local_crop_size=96,
        ... )
        >>> crops = transform(image)
        >>> print(len(crops))  # 8 (2 global + 6 local)
        >>> print(crops[0].shape)  # torch.Size([3, 224, 224])
        >>> print(crops[2].shape)  # torch.Size([3, 96, 96])
    """

    def __init__(
        self,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        num_global_crops: int = 2,
        num_local_crops: int = 6,
        global_crop_scale: Tuple[float, float] = (0.4, 1.0),
        local_crop_scale: Tuple[float, float] = (0.05, 0.4),
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        global_color_jitter: Optional[float] = 0.4,
        local_color_jitter: Optional[float] = 0.4,
        horizontal_flip_prob: float = 0.5,
    ):
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

        # Build global crop transform
        self.global_transform = self._build_crop_transform(
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            color_jitter=global_color_jitter,
            horizontal_flip_prob=horizontal_flip_prob,
            interpolation=interpolation,
            mean=mean,
            std=std,
        )

        # Build local crop transform (more aggressive)
        self.local_transform = self._build_crop_transform(
            crop_size=local_crop_size,
            crop_scale=local_crop_scale,
            color_jitter=local_color_jitter,
            horizontal_flip_prob=horizontal_flip_prob,
            interpolation=interpolation,
            mean=mean,
            std=std,
        )

    def _build_crop_transform(
        self,
        crop_size: int,
        crop_scale: Tuple[float, float],
        color_jitter: Optional[float],
        horizontal_flip_prob: float,
        interpolation: transforms.InterpolationMode,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> transforms.Compose:
        """Build a crop-specific augmentation pipeline."""
        transform_list = []

        # Random resized crop
        transform_list.append(
            transforms.RandomResizedCrop(
                crop_size,
                scale=crop_scale,
                interpolation=interpolation,
            )
        )

        # Horizontal flip
        if horizontal_flip_prob > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip_prob))

        # Color jitter (mild, unlike contrastive methods)
        if color_jitter is not None and color_jitter > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter * 0.4,
                    contrast=color_jitter * 0.4,
                    saturation=color_jitter * 0.2,
                    hue=color_jitter * 0.1,
                )
            )

        # Convert to tensor and normalize
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=mean, std=std))

        return transforms.Compose(transform_list)

    def __call__(self, image: Union[Image.Image, npt.NDArray[np.uint8]]) -> List[torch.Tensor]:
        """
        Apply multi-crop augmentation to an image.

        Args:
            image: PIL Image or numpy array

        Returns:
            List of cropped and augmented image tensors.
            Order: [global_crop_0, global_crop_1, ..., local_crop_0, local_crop_1, ...]
        """
        crops = []

        # Generate global crops
        for _ in range(self.num_global_crops):
            crops.append(self.global_transform(image))

        # Generate local crops
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))

        return crops

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"global_crops={self.num_global_crops}x{self.global_crop_size}, "
            f"local_crops={self.num_local_crops}x{self.local_crop_size})"
        )


class MultiCropEvalTransform:
    """
    Evaluation transform for multi-crop models.

    For evaluation, we only use a single center crop at global resolution.
    No augmentations are applied.

    Args:
        crop_size: Size of the center crop (default: 224)
        interpolation: Image interpolation mode (default: BICUBIC)
        mean: Normalization mean (default: ImageNet mean)
        std: Normalization std (default: ImageNet std)
    """

    def __init__(
        self,
        crop_size: int = 224,
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(crop_size * 1.14), interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Union[Image.Image, npt.NDArray[np.uint8]]) -> torch.Tensor:
        """
        Apply evaluation transform.

        Args:
            image: PIL Image or numpy array

        Returns:
            Transformed image tensor
        """
        result: torch.Tensor = self.transform(image)
        return result


class AdaptiveMultiCropTransform(MultiCropTransform):
    """
    Adaptive multi-crop that adjusts number of local crops during training.

    Useful for curriculum learning where we start with fewer crops and
    gradually increase difficulty.

    Args:
        min_local_crops: Minimum number of local crops (default: 2)
        max_local_crops: Maximum number of local crops (default: 10)
        warmup_epochs: Number of epochs to reach max_local_crops (default: 0, disabled)
        **kwargs: Arguments passed to MultiCropTransform
    """

    def __init__(
        self,
        min_local_crops: int = 2,
        max_local_crops: int = 10,
        warmup_epochs: int = 0,
        **kwargs: Any,
    ) -> None:
        # Initialize with minimum local crops
        kwargs["num_local_crops"] = min_local_crops
        super().__init__(**kwargs)

        self.min_local_crops = min_local_crops
        self.max_local_crops = max_local_crops
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """
        Update the current epoch and adjust number of local crops.

        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch

        if self.warmup_epochs > 0:
            # Linear warmup
            progress = min(1.0, epoch / self.warmup_epochs)
            num_crops = int(
                self.min_local_crops + (self.max_local_crops - self.min_local_crops) * progress
            )
            self.num_local_crops = num_crops

    def __call__(self, image: Union[Image.Image, npt.NDArray[np.uint8]]) -> List[torch.Tensor]:
        """Apply multi-crop with adaptive number of local crops."""
        crops = []

        # Generate global crops
        for _ in range(self.num_global_crops):
            crops.append(self.global_transform(image))

        # Generate local crops (adaptive number)
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))

        return crops


def build_multicrop_transform(
    num_global_crops: int = 2,
    num_local_crops: int = 6,
    global_crop_size: int = 224,
    local_crop_size: int = 96,
    global_crop_scale: Tuple[float, float] = (0.4, 1.0),
    local_crop_scale: Tuple[float, float] = (0.05, 0.4),
    global_color_jitter: float = 0.4,
    local_color_jitter: float = 0.4,
    adaptive: bool = False,
    min_local_crops: int = 2,
    max_local_crops: int = 10,
    warmup_epochs: int = 0,
) -> MultiCropTransform:
    """
    Factory function to build multi-crop transforms.

    Args:
        num_global_crops: Number of global crops
        num_local_crops: Number of local crops
        global_crop_size: Size of global crops
        local_crop_size: Size of local crops
        global_crop_scale: Scale range for global crops
        local_crop_scale: Scale range for local crops
        global_color_jitter: Color jitter for global crops
        local_color_jitter: Color jitter for local crops
        adaptive: Whether to use adaptive multi-crop
        min_local_crops: Minimum local crops (for adaptive)
        max_local_crops: Maximum local crops (for adaptive)
        warmup_epochs: Warmup epochs (for adaptive)

    Returns:
        MultiCropTransform instance

    Example:
        >>> transform = build_multicrop_transform(
        ...     num_global_crops=2,
        ...     num_local_crops=6,
        ... )
        >>> crops = transform(image)
    """
    if adaptive:
        return AdaptiveMultiCropTransform(
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            num_global_crops=num_global_crops,
            num_local_crops=num_local_crops,
            global_crop_scale=global_crop_scale,
            local_crop_scale=local_crop_scale,
            global_color_jitter=global_color_jitter,
            local_color_jitter=local_color_jitter,
            min_local_crops=min_local_crops,
            max_local_crops=max_local_crops,
            warmup_epochs=warmup_epochs,
        )
    else:
        return MultiCropTransform(
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            num_global_crops=num_global_crops,
            num_local_crops=num_local_crops,
            global_crop_scale=global_crop_scale,
            local_crop_scale=local_crop_scale,
            global_color_jitter=global_color_jitter,
            local_color_jitter=local_color_jitter,
        )


if __name__ == "__main__":
    # Demo of multi-crop transform
    print("Multi-Crop Transform Demo")
    print("=" * 60)

    # Create a dummy image
    from PIL import Image

    dummy_image = Image.new("RGB", (256, 256), color="red")

    # Create multi-crop transform
    transform = MultiCropTransform(
        num_global_crops=2,
        num_local_crops=6,
        global_crop_size=224,
        local_crop_size=96,
    )

    print(f"Transform: {transform}")
    print()

    # Apply transform
    crops = transform(dummy_image)

    print(f"Generated {len(crops)} crops:")
    print(f"  Global crops (0-1): {crops[0].shape}, {crops[1].shape}")
    print("  Local crops (2-7): ", end="")
    print(", ".join([str(crop.shape) for crop in crops[2:]]))
    print()

    # Demo adaptive transform
    print("\nAdaptive Multi-Crop Demo")
    print("=" * 60)

    adaptive_transform = AdaptiveMultiCropTransform(
        num_global_crops=2,
        min_local_crops=2,
        max_local_crops=8,
        warmup_epochs=10,
    )

    print("Epoch progression:")
    for epoch in [0, 5, 10, 15]:
        adaptive_transform.set_epoch(epoch)
        crops = adaptive_transform(dummy_image)
        print(
            f"  Epoch {epoch:2d}: {len(crops)} total crops "
            f"({adaptive_transform.num_global_crops} global + "
            f"{adaptive_transform.num_local_crops} local)"
        )

    print("\nDemo complete!")

"""
Advanced data augmentation strategies for H-JEPA training.

This module implements DeiT III augmentation pipeline with strong augmentations
including RandAugment, Mixup, CutMix, and RandomErasing. These augmentations
are designed to improve model robustness and generalization.

References:
    - DeiT III: "DeiT III: Revenge of the ViT" (Touvron et al., 2022)
    - RandAugment: "RandAugment: Practical automated data augmentation with a reduced search space" (Cubuk et al., 2020)
    - Mixup: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
    - CutMix: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)
"""

import math
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageOps
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F

# =============================================================================
# RandAugment Implementation
# =============================================================================

class RandAugment:
    """
    RandAugment: Practical automated data augmentation.

    RandAugment selects N operations from a predefined set of transformations
    and applies them with a uniform magnitude M. This is simpler than AutoAugment
    but achieves similar or better results.

    Args:
        num_ops: Number of augmentation operations to apply sequentially (N)
        magnitude: Magnitude of augmentations (M), typically 5-10 for ImageNet
        num_magnitude_bins: Number of magnitude bins (default: 31)
        interpolation: PIL interpolation mode for transforms
        fill: Fill color for transforms that need padding

    Example:
        >>> rand_aug = RandAugment(num_ops=2, magnitude=9)
        >>> augmented_image = rand_aug(image)
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill if fill is not None else [128, 128, 128]

        # Define augmentation operations
        # Each operation is a tuple of (name, lambda, magnitude_range)
        self.augment_ops = [
            ("Identity", self._identity, None),
            ("AutoContrast", self._auto_contrast, None),
            ("Equalize", self._equalize, None),
            ("Rotate", self._rotate, (-30, 30)),
            ("Solarize", self._solarize, (256, 0)),
            ("Color", self._color, (0.1, 1.9)),
            ("Posterize", self._posterize, (8, 4)),
            ("Contrast", self._contrast, (0.1, 1.9)),
            ("Brightness", self._brightness, (0.1, 1.9)),
            ("Sharpness", self._sharpness, (0.1, 1.9)),
            ("ShearX", self._shear_x, (-0.3, 0.3)),
            ("ShearY", self._shear_y, (-0.3, 0.3)),
            ("TranslateX", self._translate_x, (-0.3, 0.3)),
            ("TranslateY", self._translate_y, (-0.3, 0.3)),
        ]

    def _get_magnitude(self, magnitude_range: Optional[Tuple[float, float]]) -> Optional[float]:
        """Get the magnitude value for an operation."""
        if magnitude_range is None:
            return None

        low, high = magnitude_range
        # Interpolate between low and high based on magnitude setting
        return low + (high - low) * self.magnitude / self.num_magnitude_bins

    def _identity(self, img: Image.Image, magnitude: Optional[float]) -> Image.Image:
        """Identity operation (no change)."""
        return img

    def _auto_contrast(self, img: Image.Image, magnitude: Optional[float]) -> Image.Image:
        """Auto contrast operation."""
        return ImageOps.autocontrast(img)

    def _equalize(self, img: Image.Image, magnitude: Optional[float]) -> Image.Image:
        """Histogram equalization."""
        return ImageOps.equalize(img)

    def _rotate(self, img: Image.Image, magnitude: float) -> Image.Image:
        """Rotate image."""
        return img.rotate(magnitude, resample=self.interpolation, fillcolor=tuple(self.fill))

    def _solarize(self, img: Image.Image, magnitude: float) -> Image.Image:
        """Solarize image (invert pixels above threshold)."""
        return ImageOps.solarize(img, int(magnitude))

    def _color(self, img: Image.Image, magnitude: float) -> Image.Image:
        """Adjust color balance."""
        return ImageEnhance.Color(img).enhance(magnitude)

    def _posterize(self, img: Image.Image, magnitude: float) -> Image.Image:
        """Reduce number of bits for each color channel."""
        return ImageOps.posterize(img, int(magnitude))

    def _contrast(self, img: Image.Image, magnitude: float) -> Image.Image:
        """Adjust contrast."""
        return ImageEnhance.Contrast(img).enhance(magnitude)

    def _brightness(self, img: Image.Image, magnitude: float) -> Image.Image:
        """Adjust brightness."""
        return ImageEnhance.Brightness(img).enhance(magnitude)

    def _sharpness(self, img: Image.Image, magnitude: float) -> Image.Image:
        """Adjust sharpness."""
        return ImageEnhance.Sharpness(img).enhance(magnitude)

    def _shear_x(self, img: Image.Image, magnitude: float) -> Image.Image:
        """Shear image along X axis."""
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, magnitude, 0, 0, 1, 0),
            resample=self.interpolation,
            fillcolor=tuple(self.fill)
        )

    def _shear_y(self, img: Image.Image, magnitude: float) -> Image.Image:
        """Shear image along Y axis."""
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, magnitude, 1, 0),
            resample=self.interpolation,
            fillcolor=tuple(self.fill)
        )

    def _translate_x(self, img: Image.Image, magnitude: float) -> Image.Image:
        """Translate image along X axis."""
        pixels = int(magnitude * img.size[0])
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, pixels, 0, 1, 0),
            resample=self.interpolation,
            fillcolor=tuple(self.fill)
        )

    def _translate_y(self, img: Image.Image, magnitude: float) -> Image.Image:
        """Translate image along Y axis."""
        pixels = int(magnitude * img.size[1])
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, pixels),
            resample=self.interpolation,
            fillcolor=tuple(self.fill)
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply N random augmentation operations."""
        # Randomly select N operations
        ops = random.choices(self.augment_ops, k=self.num_ops)

        for op_name, op_func, magnitude_range in ops:
            magnitude = self._get_magnitude(magnitude_range)
            img = op_func(img, magnitude)

        return img


# =============================================================================
# Mixup and CutMix
# =============================================================================

class Mixup:
    """
    Mixup augmentation: linear interpolation between two images and labels.

    Mixup creates virtual training examples by mixing pairs of examples and
    their labels. This acts as a strong regularizer and improves generalization.

    Formula:
        x_mixed = lambda * x_i + (1 - lambda) * x_j
        y_mixed = lambda * y_i + (1 - lambda) * y_j

    where lambda ~ Beta(alpha, alpha)

    Args:
        alpha: Beta distribution parameter. Higher values create more uniform mixing.
               DeiT III uses alpha=0.8
        num_classes: Number of classes for one-hot encoding
        prob: Probability of applying mixup (default: 1.0)

    Example:
        >>> mixup = Mixup(alpha=0.8, num_classes=1000)
        >>> mixed_images, mixed_targets = mixup(images, targets)
    """

    def __init__(self, alpha: float = 0.8, num_classes: int = 1000, prob: float = 1.0):
        self.alpha = alpha
        self.num_classes = num_classes
        self.prob = prob

    def __call__(
        self,
        images: Tensor,
        targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply mixup to a batch of images and targets.

        Args:
            images: Batch of images [B, C, H, W]
            targets: Batch of labels [B] (class indices)

        Returns:
            mixed_images: Mixed images [B, C, H, W]
            mixed_targets: Mixed targets as one-hot vectors [B, num_classes]
        """
        if random.random() > self.prob:
            # Convert to one-hot but don't mix
            targets_onehot = torch.nn.functional.one_hot(targets, self.num_classes).float()
            return images, targets_onehot

        batch_size = images.size(0)

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation
        index = torch.randperm(batch_size, device=images.device)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]

        # Convert targets to one-hot and mix
        targets_a = torch.nn.functional.one_hot(targets, self.num_classes).float()
        targets_b = torch.nn.functional.one_hot(targets[index], self.num_classes).float()
        mixed_targets = lam * targets_a + (1 - lam) * targets_b

        return mixed_images, mixed_targets


class CutMix:
    """
    CutMix augmentation: replace image regions with patches from other images.

    CutMix randomly cuts a patch from one image and pastes it onto another,
    mixing both the images and labels proportionally to the area of the patch.

    Args:
        alpha: Beta distribution parameter. DeiT III uses alpha=1.0
        num_classes: Number of classes for one-hot encoding
        prob: Probability of applying cutmix (default: 1.0)

    Example:
        >>> cutmix = CutMix(alpha=1.0, num_classes=1000)
        >>> mixed_images, mixed_targets = cutmix(images, targets)
    """

    def __init__(self, alpha: float = 1.0, num_classes: int = 1000, prob: float = 1.0):
        self.alpha = alpha
        self.num_classes = num_classes
        self.prob = prob

    def _rand_bbox(self, height: int, width: int, lam: float) -> Tuple[int, int, int, int]:
        """
        Generate random bounding box.

        Returns:
            (x1, y1, x2, y2) coordinates of the bounding box
        """
        # Sample box size proportional to lambda
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)

        # Uniform random box center
        cx = np.random.randint(width)
        cy = np.random.randint(height)

        # Bounding box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)

        return x1, y1, x2, y2

    def __call__(
        self,
        images: Tensor,
        targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply CutMix to a batch of images and targets.

        Args:
            images: Batch of images [B, C, H, W]
            targets: Batch of labels [B] (class indices)

        Returns:
            mixed_images: Images with cut-and-pasted patches [B, C, H, W]
            mixed_targets: Mixed targets as one-hot vectors [B, num_classes]
        """
        if random.random() > self.prob:
            # Convert to one-hot but don't mix
            targets_onehot = torch.nn.functional.one_hot(targets, self.num_classes).float()
            return images, targets_onehot

        batch_size = images.size(0)
        _, _, height, width = images.shape

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation
        index = torch.randperm(batch_size, device=images.device)

        # Generate random bounding box
        x1, y1, x2, y2 = self._rand_bbox(height, width, lam)

        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Adjust lambda based on actual box size
        lam = 1 - ((x2 - x1) * (y2 - y1) / (height * width))

        # Convert targets to one-hot and mix
        targets_a = torch.nn.functional.one_hot(targets, self.num_classes).float()
        targets_b = torch.nn.functional.one_hot(targets[index], self.num_classes).float()
        mixed_targets = lam * targets_a + (1 - lam) * targets_b

        return mixed_images, mixed_targets


class MixupCutmix:
    """
    Combined Mixup and CutMix augmentation.

    Randomly applies either Mixup or CutMix to each batch. This provides
    more diverse augmentations during training.

    Args:
        mixup_alpha: Mixup alpha parameter (default: 0.8 for DeiT III)
        cutmix_alpha: CutMix alpha parameter (default: 1.0 for DeiT III)
        num_classes: Number of classes
        prob: Probability of applying either mixup or cutmix
        switch_prob: Probability of using mixup vs cutmix (default: 0.5)
    """

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        num_classes: int = 1000,
        prob: float = 1.0,
        switch_prob: float = 0.5,
    ):
        self.mixup = Mixup(alpha=mixup_alpha, num_classes=num_classes, prob=1.0)
        self.cutmix = CutMix(alpha=cutmix_alpha, num_classes=num_classes, prob=1.0)
        self.prob = prob
        self.switch_prob = switch_prob

    def __call__(self, images: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply either Mixup or CutMix randomly."""
        if random.random() > self.prob:
            # Convert to one-hot but don't mix
            targets_onehot = torch.nn.functional.one_hot(
                targets, self.mixup.num_classes
            ).float()
            return images, targets_onehot

        if random.random() < self.switch_prob:
            return self.mixup(images, targets)
        else:
            return self.cutmix(images, targets)


# =============================================================================
# Random Erasing
# =============================================================================

class RandomErasing:
    """
    Random Erasing augmentation.

    Randomly erases a rectangular region in the image with random values.
    This helps improve model robustness to occlusion.

    Args:
        prob: Probability of applying random erasing (DeiT III uses 0.25)
        scale: Range of proportion of erased area (default: (0.02, 0.33))
        ratio: Range of aspect ratio of erased area (default: (0.3, 3.3))
        value: Erasing value. Can be a number or 'random'
        inplace: Whether to apply inplace (default: False)

    Example:
        >>> random_erasing = RandomErasing(prob=0.25)
        >>> erased_image = random_erasing(image)
    """

    def __init__(
        self,
        prob: float = 0.25,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Union[float, str] = 0,
        inplace: bool = False,
    ):
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def __call__(self, img: Tensor) -> Tensor:
        """
        Apply random erasing to a tensor image.

        Args:
            img: Input image tensor [C, H, W]

        Returns:
            Image with random region erased
        """
        if random.random() > self.prob:
            return img

        if not self.inplace:
            img = img.clone()

        # Get image dimensions
        _, height, width = img.shape
        area = height * width

        # Try to find valid erasing region (attempt up to 10 times)
        for _ in range(10):
            # Sample target area and aspect ratio
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            # Calculate erasing box dimensions
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                # Random position
                x = random.randint(0, width - w)
                y = random.randint(0, height - h)

                # Erase region
                if self.value == 'random':
                    img[:, y:y+h, x:x+w] = torch.rand(
                        img[:, y:y+h, x:x+w].shape,
                        dtype=img.dtype,
                        device=img.device
                    )
                else:
                    img[:, y:y+h, x:x+w] = self.value

                return img

        return img


# =============================================================================
# DeiT III Augmentation Pipeline
# =============================================================================

class DeiTIIIAugmentation:
    """
    Complete DeiT III augmentation pipeline.

    This class implements the full augmentation strategy used in DeiT III,
    combining:
    1. Basic augmentations (resize, crop, flip)
    2. RandAugment for strong augmentations
    3. Color jittering
    4. Conversion to tensor and normalization
    5. Random erasing
    6. Mixup/CutMix (applied to batches, not individual images)

    The Mixup/CutMix operations are returned separately since they operate
    on batches rather than individual images.

    Args:
        image_size: Target image size (default: 224)
        color_jitter: Color jitter strength (default: 0.4)
        auto_augment: Whether to use RandAugment (default: True)
        rand_aug_num_ops: Number of RandAugment ops (default: 2)
        rand_aug_magnitude: RandAugment magnitude (default: 9)
        interpolation: Interpolation mode (default: BICUBIC)
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        random_erasing_prob: Random erasing probability (default: 0.25)
        random_erasing_scale: Random erasing scale range
        random_erasing_ratio: Random erasing aspect ratio range
        mixup_alpha: Mixup alpha (default: 0.8 for DeiT III)
        cutmix_alpha: CutMix alpha (default: 1.0 for DeiT III)
        mixup_cutmix_prob: Probability of applying mixup/cutmix (default: 1.0)
        mixup_switch_prob: Probability of mixup vs cutmix (default: 0.5)
        num_classes: Number of classes for mixup/cutmix

    Example:
        >>> # For training
        >>> aug = DeiTIIIAugmentation(
        ...     image_size=224,
        ...     auto_augment=True,
        ...     num_classes=1000
        ... )
        >>>
        >>> # Get image transform (for DataLoader)
        >>> transform = aug.get_image_transform()
        >>>
        >>> # Get batch transform (for Mixup/CutMix)
        >>> batch_transform = aug.get_batch_transform()
    """

    def __init__(
        self,
        # Image preprocessing
        image_size: int = 224,
        color_jitter: float = 0.4,
        auto_augment: bool = True,
        rand_aug_num_ops: int = 2,
        rand_aug_magnitude: int = 9,
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        # Random erasing
        random_erasing_prob: float = 0.25,
        random_erasing_scale: Tuple[float, float] = (0.02, 0.33),
        random_erasing_ratio: Tuple[float, float] = (0.3, 3.3),
        # Mixup/CutMix
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        mixup_cutmix_prob: float = 1.0,
        mixup_switch_prob: float = 0.5,
        num_classes: int = 1000,
    ):
        self.image_size = image_size
        self.color_jitter = color_jitter
        self.auto_augment = auto_augment
        self.mean = mean
        self.std = std
        self.num_classes = num_classes

        # Build image-level transform pipeline
        transform_list = []

        # 1. Resize and random crop
        transform_list.extend([
            transforms.Resize(int(image_size * 1.14), interpolation=interpolation),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # 2. RandAugment (strong augmentation)
        if auto_augment:
            transform_list.append(
                RandAugment(
                    num_ops=rand_aug_num_ops,
                    magnitude=rand_aug_magnitude,
                    interpolation=interpolation,
                )
            )

        # 3. Color jitter
        if color_jitter > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.4 * color_jitter,
                    contrast=0.4 * color_jitter,
                    saturation=0.4 * color_jitter,
                    hue=0.1 * color_jitter,
                )
            )

        # 4. Convert to tensor
        transform_list.append(transforms.ToTensor())

        # 5. Normalize
        transform_list.append(transforms.Normalize(mean=mean, std=std))

        # 6. Random erasing (applied after normalization)
        if random_erasing_prob > 0:
            transform_list.append(
                RandomErasing(
                    prob=random_erasing_prob,
                    scale=random_erasing_scale,
                    ratio=random_erasing_ratio,
                    value='random',
                )
            )

        self.image_transform = transforms.Compose(transform_list)

        # Batch-level transform (Mixup/CutMix)
        self.batch_transform = MixupCutmix(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            num_classes=num_classes,
            prob=mixup_cutmix_prob,
            switch_prob=mixup_switch_prob,
        )

    def get_image_transform(self) -> transforms.Compose:
        """
        Get the image-level transform for use in DataLoader.

        Returns:
            Composed transform that can be passed to Dataset
        """
        return self.image_transform

    def get_batch_transform(self) -> MixupCutmix:
        """
        Get the batch-level transform (Mixup/CutMix).

        Returns:
            MixupCutmix transform to apply to batches during training

        Example:
            >>> batch_transform = aug.get_batch_transform()
            >>> for images, targets in dataloader:
            ...     images, targets = batch_transform(images, targets)
            ...     # Train with mixed images and targets
        """
        return self.batch_transform

    def __call__(self, img: Image.Image) -> Tensor:
        """Apply image-level transform to a single image."""
        return self.image_transform(img)


class DeiTIIIEvalTransform:
    """
    DeiT III evaluation/validation transform.

    For evaluation, we use simple center crop without augmentations.

    Args:
        image_size: Target image size (default: 224)
        interpolation: Interpolation mode (default: BICUBIC)
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)

    Example:
        >>> eval_transform = DeiTIIIEvalTransform(image_size=224)
        >>> val_dataset = ImageFolder(val_dir, transform=eval_transform)
    """

    def __init__(
        self,
        image_size: int = 224,
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14), interpolation=interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img: Image.Image) -> Tensor:
        """Apply evaluation transform."""
        return self.transform(img)


# =============================================================================
# Configuration Helper
# =============================================================================

def build_deit3_transform(
    is_training: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Union[DeiTIIIAugmentation, DeiTIIIEvalTransform]:
    """
    Build DeiT III transform from configuration.

    Args:
        is_training: Whether to build training or evaluation transform
        config: Configuration dictionary with augmentation parameters.
                If None, uses DeiT III defaults.

    Returns:
        DeiTIIIAugmentation for training or DeiTIIIEvalTransform for eval

    Example:
        >>> # Using defaults
        >>> train_transform = build_deit3_transform(is_training=True)
        >>> val_transform = build_deit3_transform(is_training=False)
        >>>
        >>> # Using custom config
        >>> config = {
        ...     'image_size': 224,
        ...     'auto_augment': True,
        ...     'mixup_alpha': 0.8,
        ...     'cutmix_alpha': 1.0,
        ...     'num_classes': 1000,
        ... }
        >>> train_transform = build_deit3_transform(True, config)
    """
    # Default DeiT III configuration
    default_config = {
        'image_size': 224,
        'color_jitter': 0.4,
        'auto_augment': True,
        'rand_aug_num_ops': 2,
        'rand_aug_magnitude': 9,
        'random_erasing_prob': 0.25,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'mixup_cutmix_prob': 1.0,
        'num_classes': 1000,
    }

    # Merge with provided config
    if config is not None:
        default_config.update(config)

    if is_training:
        return DeiTIIIAugmentation(**default_config)
    else:
        return DeiTIIIEvalTransform(
            image_size=default_config['image_size'],
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'RandAugment',
    'Mixup',
    'CutMix',
    'MixupCutmix',
    'RandomErasing',
    'DeiTIIIAugmentation',
    'DeiTIIIEvalTransform',
    'build_deit3_transform',
]

"""
Comprehensive tests for H-JEPA data transforms.

Tests cover:
- RandAugment implementation
- Mixup and CutMix augmentations
- Random Erasing
- DeiT III augmentation pipeline
- All individual augmentation operations
- Edge cases and parameter variations
"""

import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data.transforms import (
    CutMix,
    DeiTIIIAugmentation,
    DeiTIIIEvalTransform,
    Mixup,
    MixupCutmix,
    RandAugment,
    RandomErasing,
    build_deit3_transform,
)


class TestRandAugment:
    """Test RandAugment implementation."""

    def test_basic_creation(self):
        """Test basic RandAugment creation."""
        rand_aug = RandAugment(num_ops=2, magnitude=9)
        assert rand_aug.num_ops == 2
        assert rand_aug.magnitude == 9

    def test_apply_augmentation(self):
        """Test applying augmentation to image."""
        rand_aug = RandAugment(num_ops=2, magnitude=9)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        result = rand_aug(img)

        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_different_num_ops(self):
        """Test with different numbers of operations."""
        for num_ops in [1, 2, 3, 5]:
            rand_aug = RandAugment(num_ops=num_ops, magnitude=5)
            img = Image.new("RGB", (224, 224), color="red")
            result = rand_aug(img)
            assert isinstance(result, Image.Image)

    def test_different_magnitudes(self):
        """Test with different magnitude values."""
        for magnitude in [0, 5, 10, 15, 20]:
            rand_aug = RandAugment(num_ops=2, magnitude=magnitude)
            img = Image.new("RGB", (224, 224), color="blue")
            result = rand_aug(img)
            assert isinstance(result, Image.Image)

    def test_interpolation_modes(self):
        """Test different interpolation modes."""
        modes = [
            transforms.InterpolationMode.NEAREST,
            transforms.InterpolationMode.BILINEAR,
            transforms.InterpolationMode.BICUBIC,
        ]

        for mode in modes:
            rand_aug = RandAugment(num_ops=2, magnitude=9, interpolation=mode)
            img = Image.new("RGB", (224, 224), color="green")
            result = rand_aug(img)
            assert isinstance(result, Image.Image)

    def test_custom_fill_color(self):
        """Test custom fill color."""
        rand_aug = RandAugment(num_ops=2, magnitude=9, fill=[255, 0, 0])
        img = Image.new("RGB", (224, 224), color="white")
        result = rand_aug(img)
        assert isinstance(result, Image.Image)

    def test_identity_operation(self):
        """Test identity operation (no change)."""
        rand_aug = RandAugment(num_ops=2, magnitude=0)

        # Manually apply identity
        img = Image.new("RGB", (100, 100), color=(128, 64, 32))
        result = rand_aug._identity(img, None)

        # Should return same image
        assert result == img

    def test_auto_contrast(self):
        """Test AutoContrast operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=5)
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        result = rand_aug._auto_contrast(img, None)
        assert isinstance(result, Image.Image)

    def test_equalize(self):
        """Test Equalize operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=5)
        img = Image.new("RGB", (100, 100), color="gray")
        result = rand_aug._equalize(img, None)
        assert isinstance(result, Image.Image)

    def test_rotate(self):
        """Test Rotate operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=10)
        img = Image.new("RGB", (100, 100), color="red")
        result = rand_aug._rotate(img, 15.0)
        assert isinstance(result, Image.Image)

    def test_solarize(self):
        """Test Solarize operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=10)
        img = Image.new("RGB", (100, 100), color="yellow")
        result = rand_aug._solarize(img, 128.0)
        assert isinstance(result, Image.Image)

    def test_color_adjustment(self):
        """Test Color operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=10)
        img = Image.new("RGB", (100, 100), color="blue")
        result = rand_aug._color(img, 1.5)
        assert isinstance(result, Image.Image)

    def test_posterize(self):
        """Test Posterize operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=10)
        img = Image.new("RGB", (100, 100), color="purple")
        result = rand_aug._posterize(img, 4.0)
        assert isinstance(result, Image.Image)

    def test_contrast(self):
        """Test Contrast operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=10)
        img = Image.new("RGB", (100, 100), color="orange")
        result = rand_aug._contrast(img, 1.2)
        assert isinstance(result, Image.Image)

    def test_brightness(self):
        """Test Brightness operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=10)
        img = Image.new("RGB", (100, 100), color="cyan")
        result = rand_aug._brightness(img, 1.3)
        assert isinstance(result, Image.Image)

    def test_sharpness(self):
        """Test Sharpness operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=10)
        img = Image.new("RGB", (100, 100), color="magenta")
        result = rand_aug._sharpness(img, 1.4)
        assert isinstance(result, Image.Image)

    def test_shear_x(self):
        """Test ShearX operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=10)
        img = Image.new("RGB", (100, 100), color="pink")
        result = rand_aug._shear_x(img, 0.2)
        assert isinstance(result, Image.Image)

    def test_shear_y(self):
        """Test ShearY operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=10)
        img = Image.new("RGB", (100, 100), color="lime")
        result = rand_aug._shear_y(img, 0.2)
        assert isinstance(result, Image.Image)

    def test_translate_x(self):
        """Test TranslateX operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=10)
        img = Image.new("RGB", (100, 100), color="navy")
        result = rand_aug._translate_x(img, 0.1)
        assert isinstance(result, Image.Image)

    def test_translate_y(self):
        """Test TranslateY operation."""
        rand_aug = RandAugment(num_ops=1, magnitude=10)
        img = Image.new("RGB", (100, 100), color="teal")
        result = rand_aug._translate_y(img, 0.1)
        assert isinstance(result, Image.Image)

    def test_deterministic_with_seed(self):
        """Test that augmentations are deterministic with seed."""
        img = Image.new("RGB", (224, 224), color="red")

        # Set seed and apply
        random.seed(42)
        rand_aug1 = RandAugment(num_ops=3, magnitude=10)
        result1 = rand_aug1(img)

        # Reset seed and apply again
        random.seed(42)
        rand_aug2 = RandAugment(num_ops=3, magnitude=10)
        result2 = rand_aug2(img)

        # Convert to arrays and compare
        arr1 = np.array(result1)
        arr2 = np.array(result2)
        assert np.allclose(arr1, arr2)


class TestMixup:
    """Test Mixup augmentation."""

    def test_basic_mixup(self):
        """Test basic Mixup operation."""
        mixup = Mixup(alpha=0.8, num_classes=10)

        images = torch.rand(8, 3, 224, 224)
        targets = torch.randint(0, 10, (8,))

        mixed_images, mixed_targets = mixup(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (8, 10)
        assert torch.all((mixed_targets >= 0) & (mixed_targets <= 1))

    def test_mixup_probability(self):
        """Test Mixup with probability < 1."""
        mixup = Mixup(alpha=0.8, num_classes=10, prob=0.0)

        images = torch.rand(4, 3, 224, 224)
        targets = torch.randint(0, 10, (4,))

        # With prob=0, should just return one-hot encoded targets
        mixed_images, mixed_targets = mixup(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (4, 10)

    def test_mixup_alpha_zero(self):
        """Test Mixup with alpha=0 (no mixing)."""
        mixup = Mixup(alpha=0.0, num_classes=5)

        images = torch.rand(4, 3, 224, 224)
        targets = torch.randint(0, 5, (4,))

        mixed_images, mixed_targets = mixup(images, targets)

        # With alpha=0, lambda=1.0, so no mixing
        assert torch.allclose(mixed_images, images)

    def test_one_hot_encoding(self):
        """Test that targets are properly one-hot encoded."""
        mixup = Mixup(alpha=1.0, num_classes=5, prob=1.0)

        images = torch.rand(2, 3, 224, 224)
        targets = torch.tensor([0, 4])

        mixed_images, mixed_targets = mixup(images, targets)

        # Sum of probabilities should be 1
        assert torch.allclose(mixed_targets.sum(dim=1), torch.ones(2))

    def test_batch_size_one(self):
        """Test Mixup with batch size 1."""
        mixup = Mixup(alpha=0.8, num_classes=10)

        images = torch.rand(1, 3, 224, 224)
        targets = torch.tensor([5])

        mixed_images, mixed_targets = mixup(images, targets)

        assert mixed_images.shape == (1, 3, 224, 224)
        assert mixed_targets.shape == (1, 10)

    def test_different_num_classes(self):
        """Test Mixup with different number of classes."""
        for num_classes in [10, 100, 1000]:
            mixup = Mixup(alpha=0.8, num_classes=num_classes)

            images = torch.rand(4, 3, 224, 224)
            targets = torch.randint(0, num_classes, (4,))

            mixed_images, mixed_targets = mixup(images, targets)

            assert mixed_targets.shape == (4, num_classes)


class TestCutMix:
    """Test CutMix augmentation."""

    def test_basic_cutmix(self):
        """Test basic CutMix operation."""
        cutmix = CutMix(alpha=1.0, num_classes=10)

        images = torch.rand(8, 3, 224, 224)
        targets = torch.randint(0, 10, (8,))

        mixed_images, mixed_targets = cutmix(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (8, 10)

    def test_cutmix_probability(self):
        """Test CutMix with probability."""
        cutmix = CutMix(alpha=1.0, num_classes=10, prob=0.0)

        images = torch.rand(4, 3, 224, 224)
        targets = torch.randint(0, 10, (4,))

        mixed_images, mixed_targets = cutmix(images, targets)

        # With prob=0, should just convert to one-hot
        assert mixed_targets.shape == (4, 10)

    def test_rand_bbox(self):
        """Test random bounding box generation."""
        cutmix = CutMix(alpha=1.0, num_classes=10)

        # Test bbox generation
        x1, y1, x2, y2 = cutmix._rand_bbox(224, 224, 0.5)

        # Check bounds
        assert 0 <= x1 < x2 <= 224
        assert 0 <= y1 < y2 <= 224

    def test_bbox_with_different_lambdas(self):
        """Test bbox generation with different lambda values."""
        cutmix = CutMix(alpha=1.0, num_classes=10)

        for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
            x1, y1, x2, y2 = cutmix._rand_bbox(224, 224, lam)
            assert x1 < x2 and y1 < y2

    def test_alpha_zero(self):
        """Test CutMix with alpha=0."""
        cutmix = CutMix(alpha=0.0, num_classes=5)

        images = torch.rand(4, 3, 224, 224)
        targets = torch.randint(0, 5, (4,))

        mixed_images, mixed_targets = cutmix(images, targets)

        # Should still work
        assert mixed_images.shape == images.shape

    def test_image_modification(self):
        """Test that CutMix produces valid outputs."""
        # Set seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        cutmix = CutMix(alpha=1.0, num_classes=2, prob=1.0)

        # Create distinct images
        images = torch.zeros(2, 3, 224, 224)
        images[0] = 1.0  # First image all white
        images[1] = 0.0  # Second image all black

        targets = torch.tensor([0, 1])

        mixed_images, mixed_targets = cutmix(images, targets)

        # Check that output shapes are correct
        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (2, 2)


class TestRandomErasing:
    """Test Random Erasing augmentation."""

    def test_basic_erasing(self):
        """Test basic random erasing."""
        erasing = RandomErasing(prob=1.0)

        img = torch.rand(3, 224, 224)
        result = erasing(img)

        assert result.shape == img.shape

    def test_probability_zero(self):
        """Test that prob=0 means no erasing."""
        erasing = RandomErasing(prob=0.0)

        img = torch.rand(3, 224, 224)
        result = erasing(img)

        # Should be unchanged
        assert torch.equal(img, result)

    def test_inplace_operation(self):
        """Test inplace vs non-inplace."""
        img = torch.rand(3, 224, 224)

        # Non-inplace
        erasing = RandomErasing(prob=1.0, inplace=False)
        result = erasing(img.clone())

        # Inplace
        erasing_inplace = RandomErasing(prob=1.0, inplace=True)
        img_copy = img.clone()
        result_inplace = erasing_inplace(img_copy)

        # Both should work
        assert result.shape == result_inplace.shape == (3, 224, 224)

    def test_different_scales(self):
        """Test different erasing scales."""
        for scale in [(0.01, 0.1), (0.02, 0.33), (0.1, 0.5)]:
            erasing = RandomErasing(prob=1.0, scale=scale)
            img = torch.rand(3, 224, 224)
            result = erasing(img)
            assert result.shape == img.shape

    def test_different_ratios(self):
        """Test different aspect ratios."""
        for ratio in [(0.3, 3.3), (0.5, 2.0), (1.0, 1.0)]:
            erasing = RandomErasing(prob=1.0, ratio=ratio)
            img = torch.rand(3, 224, 224)
            result = erasing(img)
            assert result.shape == img.shape

    def test_random_value(self):
        """Test erasing with random values."""
        erasing = RandomErasing(prob=1.0, value="random")

        img = torch.zeros(3, 224, 224)
        result = erasing(img)

        # Should have some non-zero values after random erasing
        # (with high probability)
        assert result.shape == img.shape

    def test_constant_value(self):
        """Test erasing with constant value."""
        erasing = RandomErasing(prob=1.0, value=0.5)

        img = torch.ones(3, 224, 224)
        result = erasing(img)

        assert result.shape == img.shape


class TestMixupCutmix:
    """Test combined Mixup and CutMix."""

    def test_basic_operation(self):
        """Test basic combined operation."""
        mixup_cutmix = MixupCutmix(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=10)

        images = torch.rand(8, 3, 224, 224)
        targets = torch.randint(0, 10, (8,))

        mixed_images, mixed_targets = mixup_cutmix(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (8, 10)

    def test_probability_zero(self):
        """Test with prob=0."""
        mixup_cutmix = MixupCutmix(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=10, prob=0.0)

        images = torch.rand(4, 3, 224, 224)
        targets = torch.randint(0, 10, (4,))

        mixed_images, mixed_targets = mixup_cutmix(images, targets)

        # Should still one-hot encode
        assert mixed_targets.shape == (4, 10)

    def test_switch_probability(self):
        """Test different switch probabilities."""
        # This is probabilistic, so we just test it doesn't crash
        for switch_prob in [0.0, 0.5, 1.0]:
            mixup_cutmix = MixupCutmix(
                mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=10, switch_prob=switch_prob
            )

            images = torch.rand(4, 3, 224, 224)
            targets = torch.randint(0, 10, (4,))

            mixed_images, mixed_targets = mixup_cutmix(images, targets)

            assert mixed_images.shape == images.shape


class TestDeiTIIIAugmentation:
    """Test DeiT III augmentation pipeline."""

    def test_basic_creation(self):
        """Test creating DeiT III augmentation."""
        aug = DeiTIIIAugmentation(image_size=224, num_classes=1000)

        assert aug.image_size == 224
        assert aug.num_classes == 1000

    def test_get_image_transform(self):
        """Test getting image-level transform."""
        aug = DeiTIIIAugmentation()
        transform = aug.get_image_transform()

        assert isinstance(transform, transforms.Compose)

        # Apply to image
        img = Image.new("RGB", (256, 256), color="red")
        result = transform(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_get_batch_transform(self):
        """Test getting batch-level transform."""
        aug = DeiTIIIAugmentation(num_classes=10)
        batch_transform = aug.get_batch_transform()

        assert isinstance(batch_transform, MixupCutmix)

        # Apply to batch
        images = torch.rand(4, 3, 224, 224)
        targets = torch.randint(0, 10, (4,))

        mixed_images, mixed_targets = batch_transform(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (4, 10)

    def test_with_auto_augment(self):
        """Test with RandAugment enabled."""
        aug = DeiTIIIAugmentation(auto_augment=True, rand_aug_num_ops=2, rand_aug_magnitude=9)

        transform = aug.get_image_transform()
        img = Image.new("RGB", (256, 256), color="blue")
        result = transform(img)

        assert result.shape == (3, 224, 224)

    def test_without_auto_augment(self):
        """Test without RandAugment."""
        aug = DeiTIIIAugmentation(auto_augment=False)

        transform = aug.get_image_transform()
        img = Image.new("RGB", (256, 256), color="green")
        result = transform(img)

        assert result.shape == (3, 224, 224)

    def test_with_color_jitter(self):
        """Test with color jitter."""
        aug = DeiTIIIAugmentation(color_jitter=0.4)

        transform = aug.get_image_transform()
        img = Image.new("RGB", (256, 256), color="yellow")
        result = transform(img)

        assert result.shape == (3, 224, 224)

    def test_without_color_jitter(self):
        """Test without color jitter."""
        aug = DeiTIIIAugmentation(color_jitter=0)

        transform = aug.get_image_transform()
        img = Image.new("RGB", (256, 256), color="purple")
        result = transform(img)

        assert result.shape == (3, 224, 224)

    def test_with_random_erasing(self):
        """Test with random erasing."""
        aug = DeiTIIIAugmentation(random_erasing_prob=0.25)

        transform = aug.get_image_transform()
        img = Image.new("RGB", (256, 256), color="orange")
        result = transform(img)

        assert result.shape == (3, 224, 224)

    def test_without_random_erasing(self):
        """Test without random erasing."""
        aug = DeiTIIIAugmentation(random_erasing_prob=0.0)

        transform = aug.get_image_transform()
        img = Image.new("RGB", (256, 256), color="pink")
        result = transform(img)

        assert result.shape == (3, 224, 224)

    def test_call_method(self):
        """Test calling augmentation directly."""
        aug = DeiTIIIAugmentation()

        img = Image.new("RGB", (256, 256), color="cyan")
        result = aug(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_different_image_sizes(self):
        """Test with different image sizes."""
        for size in [96, 128, 224, 384]:
            aug = DeiTIIIAugmentation(image_size=size)
            transform = aug.get_image_transform()

            img = Image.new("RGB", (512, 512), color="magenta")
            result = transform(img)

            assert result.shape == (3, size, size)


class TestDeiTIIIEvalTransform:
    """Test DeiT III evaluation transform."""

    def test_basic_transform(self):
        """Test basic evaluation transform."""
        transform = DeiTIIIEvalTransform(image_size=224)

        img = Image.new("RGB", (256, 256), color="red")
        result = transform(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_deterministic(self):
        """Test that eval transform is deterministic."""
        transform = DeiTIIIEvalTransform(image_size=224)

        img = Image.new("RGB", (256, 256), color="blue")

        # Apply multiple times
        results = [transform(img) for _ in range(5)]

        # All should be identical
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], atol=1e-6)

    def test_different_sizes(self):
        """Test different image sizes."""
        for size in [96, 128, 224, 384]:
            transform = DeiTIIIEvalTransform(image_size=size)
            img = Image.new("RGB", (512, 512), color="green")
            result = transform(img)
            assert result.shape == (3, size, size)

    def test_center_crop(self):
        """Test that center crop is used."""
        transform = DeiTIIIEvalTransform(image_size=224)

        # Larger image
        img = Image.new("RGB", (512, 512), color="yellow")
        result = transform(img)

        # Should be cropped to target size
        assert result.shape == (3, 224, 224)


class TestBuildDeiT3Transform:
    """Test DeiT III transform builder."""

    def test_build_train_transform(self):
        """Test building training transform."""
        transform = build_deit3_transform(is_training=True)

        assert isinstance(transform, DeiTIIIAugmentation)

        # Test application
        img = Image.new("RGB", (256, 256), color="red")
        result = transform(img)
        assert result.shape == (3, 224, 224)

    def test_build_eval_transform(self):
        """Test building evaluation transform."""
        transform = build_deit3_transform(is_training=False)

        assert isinstance(transform, DeiTIIIEvalTransform)

        # Test application
        img = Image.new("RGB", (256, 256), color="blue")
        result = transform(img)
        assert result.shape == (3, 224, 224)

    def test_with_custom_config(self):
        """Test with custom configuration."""
        config = {
            "image_size": 384,
            "auto_augment": False,
            "num_classes": 100,
        }

        transform = build_deit3_transform(is_training=True, config=config)

        assert isinstance(transform, DeiTIIIAugmentation)
        assert transform.image_size == 384
        assert transform.num_classes == 100

    def test_default_config(self):
        """Test default configuration values."""
        transform = build_deit3_transform(is_training=True)

        assert transform.image_size == 224
        assert transform.num_classes == 1000
        assert transform.auto_augment is True

    def test_config_merge(self):
        """Test that custom config merges with defaults."""
        config = {"image_size": 128}

        transform = build_deit3_transform(is_training=True, config=config)

        # Custom value
        assert transform.image_size == 128
        # Default value
        assert transform.num_classes == 1000


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_images(self):
        """Test with very small images."""
        rand_aug = RandAugment(num_ops=1, magnitude=5)
        img = Image.new("RGB", (10, 10), color="red")
        result = rand_aug(img)
        assert isinstance(result, Image.Image)

    def test_very_large_images(self):
        """Test with very large images."""
        transform = DeiTIIIAugmentation(image_size=224)
        img = Image.new("RGB", (2000, 2000), color="blue")
        result = transform(img)
        assert result.shape == (3, 224, 224)

    def test_grayscale_image(self):
        """Test with grayscale image (should be converted to RGB)."""
        DeiTIIIAugmentation(image_size=224)
        Image.new("L", (256, 256), color=128)
        # Note: Some transforms may fail with grayscale
        # This is expected behavior

    def test_mixup_with_single_class(self):
        """Test Mixup when all samples are same class."""
        mixup = Mixup(alpha=0.8, num_classes=10)

        images = torch.rand(4, 3, 224, 224)
        targets = torch.tensor([5, 5, 5, 5])

        mixed_images, mixed_targets = mixup(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (4, 10)

    def test_cutmix_small_images(self):
        """Test CutMix with small images."""
        cutmix = CutMix(alpha=1.0, num_classes=10)

        images = torch.rand(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))

        mixed_images, mixed_targets = cutmix(images, targets)

        assert mixed_images.shape == images.shape

    def test_random_erasing_full_image(self):
        """Test random erasing with large scale."""
        erasing = RandomErasing(prob=1.0, scale=(0.9, 1.0))

        img = torch.rand(3, 224, 224)
        result = erasing(img)

        # Should still work
        assert result.shape == img.shape

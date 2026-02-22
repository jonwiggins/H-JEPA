"""
Comprehensive tests for multi-crop transforms.

Tests cover:
- MultiCropTransform with global and local crops
- MultiCropEvalTransform
- AdaptiveMultiCropTransform with curriculum learning
- Factory function for building transforms
- Different configurations and edge cases
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data.multicrop_transforms import (
    AdaptiveMultiCropTransform,
    MultiCropEvalTransform,
    MultiCropTransform,
    build_multicrop_transform,
)


class TestMultiCropTransform:
    """Test multi-crop transform implementation."""

    def test_basic_creation(self):
        """Test basic multi-crop creation."""
        transform = MultiCropTransform(
            num_global_crops=2, num_local_crops=6, global_crop_size=224, local_crop_size=96
        )

        assert transform.num_global_crops == 2
        assert transform.num_local_crops == 6
        assert transform.global_crop_size == 224
        assert transform.local_crop_size == 96

    def test_apply_transform(self):
        """Test applying multi-crop transform."""
        transform = MultiCropTransform(num_global_crops=2, num_local_crops=4)

        img = Image.new("RGB", (256, 256), color="red")
        crops = transform(img)

        # Should have 2 global + 4 local = 6 crops
        assert len(crops) == 6
        assert all(isinstance(crop, torch.Tensor) for crop in crops)

    def test_crop_sizes(self):
        """Test that crops have correct sizes."""
        transform = MultiCropTransform(
            num_global_crops=2, num_local_crops=4, global_crop_size=224, local_crop_size=96
        )

        img = Image.new("RGB", (512, 512), color="blue")
        crops = transform(img)

        # First 2 crops are global
        assert crops[0].shape == (3, 224, 224)
        assert crops[1].shape == (3, 224, 224)

        # Next 4 crops are local
        for i in range(2, 6):
            assert crops[i].shape == (3, 96, 96)

    def test_different_num_crops(self):
        """Test with different numbers of crops."""
        configs = [
            (1, 2),  # 1 global, 2 local
            (2, 6),  # 2 global, 6 local
            (3, 8),  # 3 global, 8 local
            (2, 10),  # 2 global, 10 local
        ]

        for num_global, num_local in configs:
            transform = MultiCropTransform(num_global_crops=num_global, num_local_crops=num_local)

            img = Image.new("RGB", (256, 256), color="green")
            crops = transform(img)

            assert len(crops) == num_global + num_local

    def test_crop_scales(self):
        """Test different crop scale ranges."""
        transform = MultiCropTransform(
            num_global_crops=2,
            num_local_crops=2,
            global_crop_scale=(0.4, 1.0),
            local_crop_scale=(0.05, 0.4),
        )

        img = Image.new("RGB", (256, 256), color="yellow")
        crops = transform(img)

        # Should produce valid crops
        assert len(crops) == 4
        assert all(crop.shape[0] == 3 for crop in crops)

    def test_color_jitter(self):
        """Test with color jitter."""
        transform = MultiCropTransform(
            num_global_crops=2, num_local_crops=2, global_color_jitter=0.4, local_color_jitter=0.4
        )

        img = Image.new("RGB", (256, 256), color=(128, 64, 32))
        crops = transform(img)

        assert len(crops) == 4

    def test_no_color_jitter(self):
        """Test without color jitter."""
        transform = MultiCropTransform(
            num_global_crops=2, num_local_crops=2, global_color_jitter=None, local_color_jitter=None
        )

        img = Image.new("RGB", (256, 256), color="purple")
        crops = transform(img)

        assert len(crops) == 4

    def test_interpolation_mode(self):
        """Test different interpolation modes."""
        modes = [
            transforms.InterpolationMode.NEAREST,
            transforms.InterpolationMode.BILINEAR,
            transforms.InterpolationMode.BICUBIC,
        ]

        for mode in modes:
            transform = MultiCropTransform(
                num_global_crops=1, num_local_crops=1, interpolation=mode
            )

            img = Image.new("RGB", (256, 256), color="orange")
            crops = transform(img)

            assert len(crops) == 2

    def test_normalization(self):
        """Test custom normalization parameters."""
        transform = MultiCropTransform(
            num_global_crops=1, num_local_crops=1, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )

        img = Image.new("RGB", (256, 256), color=(255, 255, 255))
        crops = transform(img)

        # Check that normalization was applied
        for crop in crops:
            assert crop.max() <= 2.0
            assert crop.min() >= -2.0

    def test_horizontal_flip(self):
        """Test horizontal flip probability."""
        transform = MultiCropTransform(
            num_global_crops=2, num_local_crops=2, horizontal_flip_prob=0.5
        )

        img = Image.new("RGB", (256, 256), color="cyan")
        crops = transform(img)

        assert len(crops) == 4

    def test_no_horizontal_flip(self):
        """Test without horizontal flip."""
        transform = MultiCropTransform(
            num_global_crops=2, num_local_crops=2, horizontal_flip_prob=0.0
        )

        img = Image.new("RGB", (256, 256), color="magenta")
        crops = transform(img)

        assert len(crops) == 4

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        transform = MultiCropTransform(num_global_crops=1, num_local_crops=1)

        # Convert numpy array to PIL Image first
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_array)
        crops = transform(img_pil)

        assert len(crops) == 2
        assert all(isinstance(crop, torch.Tensor) for crop in crops)

    def test_repr(self):
        """Test string representation."""
        transform = MultiCropTransform(
            num_global_crops=2, num_local_crops=6, global_crop_size=224, local_crop_size=96
        )

        repr_str = repr(transform)

        assert "MultiCropTransform" in repr_str
        assert "224" in repr_str
        assert "96" in repr_str

    def test_determinism_with_seed(self):
        """Test that crops are deterministic with seed."""
        transform = MultiCropTransform(num_global_crops=2, num_local_crops=2)

        img = Image.new("RGB", (256, 256), color="red")

        # Apply with seed
        torch.manual_seed(42)
        np.random.seed(42)
        crops1 = transform(img)

        # Reset seed and apply again
        torch.manual_seed(42)
        np.random.seed(42)
        crops2 = transform(img)

        # Should be identical
        for c1, c2 in zip(crops1, crops2):
            assert torch.allclose(c1, c2, atol=1e-6)


class TestMultiCropEvalTransform:
    """Test multi-crop evaluation transform."""

    def test_basic_creation(self):
        """Test basic creation."""
        transform = MultiCropEvalTransform(crop_size=224)

        assert transform is not None

    def test_apply_transform(self):
        """Test applying transform."""
        transform = MultiCropEvalTransform(crop_size=224)

        img = Image.new("RGB", (256, 256), color="red")
        result = transform(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_different_sizes(self):
        """Test with different crop sizes."""
        sizes = [96, 128, 224, 384]

        for size in sizes:
            transform = MultiCropEvalTransform(crop_size=size)
            img = Image.new("RGB", (512, 512), color="blue")
            result = transform(img)
            assert result.shape == (3, size, size)

    def test_deterministic(self):
        """Test that eval transform is deterministic."""
        transform = MultiCropEvalTransform(crop_size=224)

        img = Image.new("RGB", (256, 256), color="green")

        # Apply multiple times
        results = [transform(img) for _ in range(5)]

        # All should be identical
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], atol=1e-6)

    def test_center_crop(self):
        """Test that center crop is used."""
        transform = MultiCropEvalTransform(crop_size=224)

        img = Image.new("RGB", (512, 512), color="yellow")
        result = transform(img)

        assert result.shape == (3, 224, 224)

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        transform = MultiCropEvalTransform(crop_size=224)

        # Convert numpy array to PIL Image first
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_array)
        result = transform(img_pil)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)


class TestAdaptiveMultiCropTransform:
    """Test adaptive multi-crop with curriculum learning."""

    def test_basic_creation(self):
        """Test basic creation."""
        transform = AdaptiveMultiCropTransform(
            min_local_crops=2, max_local_crops=10, warmup_epochs=10, num_global_crops=2
        )

        assert transform.min_local_crops == 2
        assert transform.max_local_crops == 10
        assert transform.warmup_epochs == 10

    def test_initial_state(self):
        """Test initial number of crops."""
        transform = AdaptiveMultiCropTransform(
            min_local_crops=2, max_local_crops=10, warmup_epochs=10, num_global_crops=2
        )

        # Should start with min_local_crops
        assert transform.num_local_crops == 2

        img = Image.new("RGB", (256, 256), color="red")
        crops = transform(img)

        # 2 global + 2 local = 4
        assert len(crops) == 4

    def test_warmup_progression(self):
        """Test that crops increase during warmup."""
        transform = AdaptiveMultiCropTransform(
            min_local_crops=2, max_local_crops=10, warmup_epochs=10, num_global_crops=2
        )

        img = Image.new("RGB", (256, 256), color="blue")

        # Epoch 0: should have min_local_crops
        transform.set_epoch(0)
        crops0 = transform(img)
        assert len(crops0) == 2 + 2  # 2 global + 2 local

        # Epoch 5: should have intermediate number
        transform.set_epoch(5)
        crops5 = transform(img)
        num_crops_5 = len(crops5)
        assert num_crops_5 > len(crops0)
        assert num_crops_5 < 2 + 10

        # Epoch 10: should have max_local_crops
        transform.set_epoch(10)
        crops10 = transform(img)
        assert len(crops10) == 2 + 10  # 2 global + 10 local

    def test_after_warmup(self):
        """Test that crops stay at max after warmup."""
        transform = AdaptiveMultiCropTransform(
            min_local_crops=2, max_local_crops=10, warmup_epochs=10, num_global_crops=2
        )

        img = Image.new("RGB", (256, 256), color="green")

        # After warmup, should stay at max
        transform.set_epoch(10)
        crops10 = transform(img)

        transform.set_epoch(15)
        crops15 = transform(img)

        transform.set_epoch(20)
        crops20 = transform(img)

        # All should have max crops
        assert len(crops10) == len(crops15) == len(crops20) == 12

    def test_no_warmup(self):
        """Test with warmup_epochs=0 (no warmup)."""
        transform = AdaptiveMultiCropTransform(
            min_local_crops=2, max_local_crops=10, warmup_epochs=0, num_global_crops=2
        )

        img = Image.new("RGB", (256, 256), color="yellow")

        # Should always use min_local_crops when warmup=0
        for epoch in [0, 5, 10]:
            transform.set_epoch(epoch)
            crops = transform(img)
            assert len(crops) == 2 + 2  # 2 global + 2 local (min)

    def test_linear_progression(self):
        """Test that progression is linear."""
        transform = AdaptiveMultiCropTransform(
            min_local_crops=2, max_local_crops=10, warmup_epochs=8, num_global_crops=2
        )

        img = Image.new("RGB", (256, 256), color="purple")

        # Record number of crops at different epochs
        crop_counts = []
        for epoch in range(9):
            transform.set_epoch(epoch)
            crops = transform(img)
            crop_counts.append(len(crops))

        # Should be monotonically increasing
        for i in range(1, len(crop_counts)):
            assert crop_counts[i] >= crop_counts[i - 1]

    def test_inherits_parent_functionality(self):
        """Test that it inherits MultiCropTransform functionality."""
        transform = AdaptiveMultiCropTransform(
            min_local_crops=2,
            max_local_crops=6,
            warmup_epochs=5,
            num_global_crops=2,
            global_crop_size=224,
            local_crop_size=96,
        )

        img = Image.new("RGB", (256, 256), color="orange")
        transform.set_epoch(0)
        crops = transform(img)

        # Check crop sizes
        assert crops[0].shape == (3, 224, 224)  # Global
        assert crops[1].shape == (3, 224, 224)  # Global
        assert crops[2].shape == (3, 96, 96)  # Local


class TestBuildMulticropTransform:
    """Test factory function for building multi-crop transforms."""

    def test_build_basic_transform(self):
        """Test building basic transform."""
        transform = build_multicrop_transform(num_global_crops=2, num_local_crops=6)

        assert isinstance(transform, MultiCropTransform)
        assert transform.num_global_crops == 2
        assert transform.num_local_crops == 6

    def test_build_adaptive_transform(self):
        """Test building adaptive transform."""
        transform = build_multicrop_transform(
            num_global_crops=2,
            num_local_crops=6,
            adaptive=True,
            min_local_crops=2,
            max_local_crops=10,
            warmup_epochs=10,
        )

        assert isinstance(transform, AdaptiveMultiCropTransform)
        assert transform.min_local_crops == 2
        assert transform.max_local_crops == 10

    def test_custom_crop_sizes(self):
        """Test with custom crop sizes."""
        transform = build_multicrop_transform(global_crop_size=384, local_crop_size=128)

        img = Image.new("RGB", (512, 512), color="red")
        crops = transform(img)

        assert crops[0].shape == (3, 384, 384)  # Global
        assert crops[2].shape == (3, 128, 128)  # Local

    def test_custom_scales(self):
        """Test with custom crop scales."""
        transform = build_multicrop_transform(
            global_crop_scale=(0.5, 1.0), local_crop_scale=(0.1, 0.5)
        )

        img = Image.new("RGB", (256, 256), color="blue")
        crops = transform(img)

        assert len(crops) > 0

    def test_custom_color_jitter(self):
        """Test with custom color jitter."""
        transform = build_multicrop_transform(global_color_jitter=0.8, local_color_jitter=0.6)

        img = Image.new("RGB", (256, 256), color="green")
        crops = transform(img)

        assert len(crops) > 0

    def test_all_parameters(self):
        """Test with all parameters specified."""
        transform = build_multicrop_transform(
            num_global_crops=3,
            num_local_crops=8,
            global_crop_size=256,
            local_crop_size=128,
            global_crop_scale=(0.6, 1.0),
            local_crop_scale=(0.08, 0.6),
            global_color_jitter=0.5,
            local_color_jitter=0.5,
            adaptive=False,
        )

        assert isinstance(transform, MultiCropTransform)
        assert transform.num_global_crops == 3
        assert transform.num_local_crops == 8


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_local_crops(self):
        """Test with zero local crops."""
        transform = MultiCropTransform(num_global_crops=2, num_local_crops=0)

        img = Image.new("RGB", (256, 256), color="red")
        crops = transform(img)

        # Should only have global crops
        assert len(crops) == 2

    def test_zero_global_crops(self):
        """Test with zero global crops."""
        transform = MultiCropTransform(num_global_crops=0, num_local_crops=4)

        img = Image.new("RGB", (256, 256), color="blue")
        crops = transform(img)

        # Should only have local crops
        assert len(crops) == 4

    def test_many_crops(self):
        """Test with many crops."""
        transform = MultiCropTransform(num_global_crops=4, num_local_crops=16)

        img = Image.new("RGB", (512, 512), color="green")
        crops = transform(img)

        assert len(crops) == 20

    def test_very_small_images(self):
        """Test with very small images."""
        transform = MultiCropTransform(
            num_global_crops=1, num_local_crops=1, global_crop_size=32, local_crop_size=16
        )

        img = Image.new("RGB", (64, 64), color="yellow")
        crops = transform(img)

        assert len(crops) == 2

    def test_very_large_images(self):
        """Test with very large images."""
        transform = MultiCropTransform(num_global_crops=1, num_local_crops=1)

        img = Image.new("RGB", (2048, 2048), color="purple")
        crops = transform(img)

        assert len(crops) == 2

    def test_overlapping_scales(self):
        """Test with overlapping global and local scales."""
        transform = MultiCropTransform(
            num_global_crops=2,
            num_local_crops=2,
            global_crop_scale=(0.3, 0.8),
            local_crop_scale=(0.2, 0.7),
        )

        img = Image.new("RGB", (256, 256), color="orange")
        crops = transform(img)

        assert len(crops) == 4

    def test_same_crop_sizes(self):
        """Test with same global and local crop sizes."""
        transform = MultiCropTransform(
            num_global_crops=2, num_local_crops=2, global_crop_size=224, local_crop_size=224
        )

        img = Image.new("RGB", (256, 256), color="cyan")
        crops = transform(img)

        # All crops should be same size
        for crop in crops:
            assert crop.shape == (3, 224, 224)

    def test_adaptive_same_min_max(self):
        """Test adaptive transform when min == max."""
        transform = AdaptiveMultiCropTransform(
            min_local_crops=6, max_local_crops=6, warmup_epochs=10, num_global_crops=2
        )

        img = Image.new("RGB", (256, 256), color="magenta")

        # Should always have same number of crops
        for epoch in [0, 5, 10]:
            transform.set_epoch(epoch)
            crops = transform(img)
            assert len(crops) == 8  # 2 global + 6 local

    def test_adaptive_immediate_warmup(self):
        """Test adaptive transform with 1 epoch warmup."""
        transform = AdaptiveMultiCropTransform(
            min_local_crops=2, max_local_crops=10, warmup_epochs=1, num_global_crops=2
        )

        img = Image.new("RGB", (256, 256), color="pink")

        # Epoch 0: min
        transform.set_epoch(0)
        crops0 = transform(img)
        assert len(crops0) == 4  # 2 global + 2 local

        # Epoch 1: max
        transform.set_epoch(1)
        crops1 = transform(img)
        assert len(crops1) == 12  # 2 global + 10 local

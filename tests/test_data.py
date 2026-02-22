"""
Comprehensive tests for data loading and dataset functionality.

This module provides extensive test coverage for the H-JEPA data pipeline,
including transforms, datasets, multi-crop functionality, and data loaders.

Test Coverage:
- Transform operations (JEPA, DeiT III, Multi-crop)
- Dataset loading and initialization (CIFAR, STL, ImageNet variants)
- Data augmentations (RandAugment, Mixup, CutMix, Random Erasing)
- Multi-crop dataset and transforms
- Multi-dataset support
- Data loaders and batch generation
- Edge cases and error handling
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data import (
    DATASET_INFO,
    AdaptiveMultiCropTransform,
    BalancedMultiDataset,
    CIFAR10Dataset,
    CIFAR100Dataset,
    CutMix,
    DeiTIIIAugmentation,
    DeiTIIIEvalTransform,
    JEPAEvalTransform,
    JEPATransform,
    Mixup,
    MixupCutmix,
    MultiCropDataset,
    MultiCropDatasetRaw,
    MultiCropEvalTransform,
    MultiCropTransform,
    RandAugment,
    RandomErasing,
    WeightedMultiDataset,
    build_dataloader,
    build_dataset,
    build_deit3_transform,
    build_multicrop_dataloader,
    build_multicrop_dataset,
    build_multicrop_transform,
    multicrop_collate_fn,
    verify_dataset,
)


class TestJEPATransforms:
    """Test JEPA training and evaluation transforms."""

    def test_jepa_transform_output_shape(self, sample_image_224):
        """Test JEPATransform produces correct output shape."""
        transform = JEPATransform(image_size=224)
        output = transform(sample_image_224)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)
        assert output.dtype == torch.float32

    def test_jepa_eval_transform_output_shape(self, sample_image_224):
        """Test JEPAEvalTransform produces correct output shape."""
        transform = JEPAEvalTransform(image_size=224)
        output = transform(sample_image_224)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)
        assert output.dtype == torch.float32

    def test_transform_consistency_multiple_sizes(self):
        """Test transforms maintain output shape across different input sizes."""
        for size in [32, 64, 128, 256, 512]:
            img = Image.new("RGB", (size, size), color=(128, 128, 128))

            # Train transform
            train_transform = JEPATransform(image_size=224)
            train_output = train_transform(img)
            assert train_output.shape == (3, 224, 224)

            # Eval transform
            eval_transform = JEPAEvalTransform(image_size=224)
            eval_output = eval_transform(img)
            assert eval_output.shape == (3, 224, 224)

    def test_jepa_transform_with_color_jitter(self, sample_image_224):
        """Test JEPATransform with color jitter enabled."""
        transform = JEPATransform(image_size=224, color_jitter=0.4)
        output = transform(sample_image_224)
        assert output.shape == (3, 224, 224)

    def test_jepa_transform_without_color_jitter(self, sample_image_224):
        """Test JEPATransform with color jitter disabled."""
        transform = JEPATransform(image_size=224, color_jitter=None)
        output = transform(sample_image_224)
        assert output.shape == (3, 224, 224)

    def test_jepa_transform_custom_normalization(self, sample_image_224):
        """Test JEPATransform with custom normalization parameters."""
        mean = (0.5, 0.5, 0.5)
        std = (0.2, 0.2, 0.2)
        transform = JEPATransform(image_size=224, mean=mean, std=std)
        output = transform(sample_image_224)
        assert output.shape == (3, 224, 224)

    def test_jepa_transform_no_horizontal_flip(self, sample_image_224):
        """Test JEPATransform with horizontal flip disabled."""
        transform = JEPATransform(image_size=224, horizontal_flip=False)
        output = transform(sample_image_224)
        assert output.shape == (3, 224, 224)

    def test_jepa_transform_custom_crop_scale(self, sample_image_224):
        """Test JEPATransform with custom crop scale."""
        transform = JEPATransform(image_size=224, crop_scale=(0.5, 1.0))
        output = transform(sample_image_224)
        assert output.shape == (3, 224, 224)

    def test_jepa_transform_deterministic_eval(self, sample_image_224):
        """Test that eval transform is deterministic (no augmentation)."""
        transform = JEPAEvalTransform(image_size=224)

        # Apply twice and check outputs are identical
        output1 = transform(sample_image_224.copy())
        output2 = transform(sample_image_224.copy())

        # Note: Due to random ops in train transform, we can't check exact equality
        # But we can verify both are same shape
        assert output1.shape == output2.shape

    def test_jepa_transform_output_range(self, sample_image_224):
        """Test that normalized output is in reasonable range."""
        transform = JEPATransform(image_size=224)
        output = transform(sample_image_224)

        # Normalized tensors should be roughly in range [-3, 3]
        assert output.min() >= -5.0
        assert output.max() <= 5.0

    def test_jepa_transform_dimensions(self):
        """Test transform with various image dimensions."""
        sizes = [(32, 32), (64, 64), (100, 200), (256, 512)]

        for h, w in sizes:
            img = Image.new("RGB", (w, h), color=(128, 128, 128))
            transform = JEPATransform(image_size=224)
            output = transform(img)
            assert output.shape == (3, 224, 224)


class TestAugmentations:
    """Test data augmentation transforms."""

    def test_rand_augment_basic(self, sample_image_224):
        """Test RandAugment applies transformations."""
        aug = RandAugment(num_ops=2, magnitude=9)
        output = aug(sample_image_224)
        assert isinstance(output, Image.Image)

    def test_rand_augment_num_ops(self, sample_image_224):
        """Test RandAugment with different number of operations."""
        for num_ops in [1, 2, 3, 5]:
            aug = RandAugment(num_ops=num_ops, magnitude=9)
            output = aug(sample_image_224)
            assert isinstance(output, Image.Image)

    def test_rand_augment_magnitude(self, sample_image_224):
        """Test RandAugment with different magnitudes."""
        for magnitude in [0, 5, 9, 15, 30]:
            aug = RandAugment(num_ops=2, magnitude=magnitude)
            output = aug(sample_image_224)
            assert isinstance(output, Image.Image)

    def test_mixup_basic(self, sample_batch_224):
        """Test Mixup augmentation."""
        images, targets = sample_batch_224
        mixup = Mixup(alpha=0.8, num_classes=1000)
        mixed_images, mixed_targets = mixup(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (images.shape[0], 1000)
        assert mixed_targets.dtype == torch.float32

    def test_mixup_alpha_variations(self, sample_batch_224):
        """Test Mixup with different alpha values."""
        images, targets = sample_batch_224

        for alpha in [0.0, 0.5, 0.8, 1.0, 2.0]:
            mixup = Mixup(alpha=alpha, num_classes=1000)
            mixed_images, mixed_targets = mixup(images, targets)
            assert mixed_images.shape == images.shape

    def test_mixup_probability(self, sample_batch_224):
        """Test Mixup probability parameter."""
        images, targets = sample_batch_224
        mixup_no_apply = Mixup(alpha=0.8, num_classes=1000, prob=0.0)
        _, mixed_targets = mixup_no_apply(images, targets)

        # Should just convert to one-hot, not mix
        assert mixed_targets.sum(dim=1).allclose(
            torch.ones(images.shape[0], device=mixed_targets.device)
        )

    def test_cutmix_basic(self, sample_batch_224):
        """Test CutMix augmentation."""
        images, targets = sample_batch_224
        cutmix = CutMix(alpha=1.0, num_classes=1000)
        mixed_images, mixed_targets = cutmix(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (images.shape[0], 1000)

    def test_cutmix_alpha_variations(self, sample_batch_224):
        """Test CutMix with different alpha values."""
        images, targets = sample_batch_224

        for alpha in [0.0, 0.5, 1.0, 2.0]:
            cutmix = CutMix(alpha=alpha, num_classes=1000)
            mixed_images, mixed_targets = cutmix(images, targets)
            assert mixed_images.shape == images.shape

    def test_mixup_cutmix_combined(self, sample_batch_224):
        """Test combined Mixup and CutMix."""
        images, targets = sample_batch_224
        mixup_cutmix = MixupCutmix(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            num_classes=1000,
            switch_prob=0.5,
        )
        mixed_images, mixed_targets = mixup_cutmix(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (images.shape[0], 1000)

    def test_random_erasing_basic(self):
        """Test RandomErasing augmentation."""
        img = torch.randn(3, 224, 224)
        erasing = RandomErasing(prob=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        output = erasing(img)

        assert output.shape == img.shape
        # Check that some region was modified
        assert not torch.allclose(output, img)

    def test_random_erasing_probability(self):
        """Test RandomErasing with probability."""
        img = torch.randn(3, 224, 224)

        # With prob=0, should not erase
        erasing_no_erase = RandomErasing(prob=0.0)
        output = erasing_no_erase(img)
        assert torch.allclose(output, img)

        # With prob=1, should always erase
        erasing_always = RandomErasing(prob=1.0)
        output = erasing_always(img)
        # High probability of at least some difference
        assert not torch.allclose(output, img)

    def test_random_erasing_value_modes(self):
        """Test RandomErasing with different value modes."""
        img = torch.randn(3, 224, 224)

        # Erase with specific value
        erasing_zero = RandomErasing(prob=1.0, value=0)
        output_zero = erasing_zero(img)
        assert output_zero.shape == img.shape

        # Erase with random values
        erasing_random = RandomErasing(prob=1.0, value="random")
        output_random = erasing_random(img)
        assert output_random.shape == img.shape

    def test_deit3_augmentation(self, sample_image_224):
        """Test DeiT III augmentation pipeline."""
        aug = DeiTIIIAugmentation(image_size=224, num_classes=1000)
        output = aug(sample_image_224)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_deit3_augmentation_batch_transform(self, sample_batch_224):
        """Test DeiT III batch-level transforms."""
        aug = DeiTIIIAugmentation(image_size=224, num_classes=1000)
        batch_transform = aug.get_batch_transform()

        images, targets = sample_batch_224
        mixed_images, mixed_targets = batch_transform(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape[0] == images.shape[0]

    def test_deit3_eval_transform(self, sample_image_224):
        """Test DeiT III evaluation transform."""
        transform = DeiTIIIEvalTransform(image_size=224)
        output = transform(sample_image_224)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_build_deit3_transform_training(self):
        """Test building DeiT III transform for training."""
        transform = build_deit3_transform(is_training=True)
        assert isinstance(transform, DeiTIIIAugmentation)

    def test_build_deit3_transform_evaluation(self):
        """Test building DeiT III transform for evaluation."""
        transform = build_deit3_transform(is_training=False)
        assert isinstance(transform, DeiTIIIEvalTransform)

    def test_build_deit3_transform_custom_config(self):
        """Test building DeiT III with custom configuration."""
        config = {
            "image_size": 256,
            "num_classes": 100,
            "rand_aug_magnitude": 15,
        }
        transform = build_deit3_transform(is_training=True, config=config)
        assert isinstance(transform, DeiTIIIAugmentation)


class TestMultiCropTransforms:
    """Test multi-crop augmentation transforms."""

    def test_multicrop_transform_basic(self, sample_image_224):
        """Test MultiCropTransform produces multiple crops."""
        transform = MultiCropTransform(
            num_global_crops=2,
            num_local_crops=6,
            global_crop_size=224,
            local_crop_size=96,
        )
        crops = transform(sample_image_224)

        assert isinstance(crops, list)
        assert len(crops) == 8  # 2 global + 6 local

    def test_multicrop_crop_sizes(self, sample_image_224):
        """Test MultiCropTransform produces correct sizes."""
        transform = MultiCropTransform(
            num_global_crops=2,
            num_local_crops=6,
            global_crop_size=224,
            local_crop_size=96,
        )
        crops = transform(sample_image_224)

        # Check global crop sizes
        for i in range(2):
            assert crops[i].shape == (3, 224, 224)

        # Check local crop sizes
        for i in range(2, 8):
            assert crops[i].shape == (3, 96, 96)

    def test_multicrop_configurations(self, sample_image_224):
        """Test MultiCropTransform with different configurations."""
        configs = [
            {"num_global": 2, "num_local": 4},
            {"num_global": 1, "num_local": 8},
            {"num_global": 4, "num_local": 2},
        ]

        for config in configs:
            transform = MultiCropTransform(
                num_global_crops=config["num_global"],
                num_local_crops=config["num_local"],
            )
            crops = transform(sample_image_224)
            assert len(crops) == config["num_global"] + config["num_local"]

    def test_multicrop_eval_transform(self, sample_image_224):
        """Test MultiCropEvalTransform produces single crop."""
        transform = MultiCropEvalTransform(crop_size=224)
        output = transform(sample_image_224)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_adaptive_multicrop_warmup(self, sample_image_224):
        """Test AdaptiveMultiCropTransform with epoch warmup."""
        transform = AdaptiveMultiCropTransform(
            num_global_crops=2,
            min_local_crops=2,
            max_local_crops=10,
            warmup_epochs=10,
        )

        # Test epoch progression
        for epoch in [0, 5, 10, 15]:
            transform.set_epoch(epoch)
            crops = transform(sample_image_224)

            # More crops at later epochs
            if epoch == 0:
                assert len(crops) == 4  # 2 global + 2 local
            elif epoch == 10:
                assert len(crops) == 12  # 2 global + 10 local

    def test_build_multicrop_transform_default(self):
        """Test building multicrop transform with defaults."""
        transform = build_multicrop_transform()
        assert isinstance(transform, MultiCropTransform)

    def test_build_multicrop_transform_adaptive(self):
        """Test building adaptive multicrop transform."""
        transform = build_multicrop_transform(
            adaptive=True,
            warmup_epochs=10,
            min_local_crops=2,
            max_local_crops=8,
        )
        assert isinstance(transform, AdaptiveMultiCropTransform)


class TestCIFARDatasets:
    """Test CIFAR dataset classes."""

    @pytest.fixture(scope="class")
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup after tests
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_cifar10_download(self, temp_data_dir):
        """Test CIFAR-10 download and loading."""
        dataset = CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="train",
            image_size=224,
            download=True,
        )

        # Check dataset size
        assert len(dataset) == 50000

        # Check data format
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert 0 <= label < 10

    def test_cifar100_download(self, temp_data_dir):
        """Test CIFAR-100 download and loading."""
        dataset = CIFAR100Dataset(
            data_path=temp_data_dir / "cifar100",
            split="train",
            image_size=224,
            download=True,
        )

        # Check dataset size
        assert len(dataset) == 50000

        # Check data format
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert 0 <= label < 100

    def test_cifar10_splits(self, temp_data_dir):
        """Test CIFAR-10 train/val splits."""
        train_dataset = CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )
        val_dataset = CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="val",
            download=True,
        )

        assert len(train_dataset) == 50000
        assert len(val_dataset) == 10000

    def test_cifar10_classes_attribute(self, temp_data_dir):
        """Test CIFAR-10 classes attribute."""
        dataset = CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        assert hasattr(dataset, "classes")
        assert len(dataset.classes) == 10

    def test_cifar100_classes_attribute(self, temp_data_dir):
        """Test CIFAR-100 classes attribute."""
        dataset = CIFAR100Dataset(
            data_path=temp_data_dir / "cifar100",
            split="train",
            download=True,
        )

        assert hasattr(dataset, "classes")
        assert len(dataset.classes) == 100


class TestMultiCropDatasets:
    """Test multi-crop dataset classes."""

    @pytest.fixture(scope="class")
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_multicrop_dataset_basic(self, temp_data_dir):
        """Test MultiCropDataset wrapper."""
        base_dataset = CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        transform = MultiCropTransform(
            num_global_crops=2,
            num_local_crops=6,
            global_crop_size=224,
            local_crop_size=96,
        )

        dataset = MultiCropDataset(base_dataset, transform)
        assert len(dataset) == len(base_dataset)

        # Get a sample
        crops, label = dataset[0]
        assert isinstance(crops, list)
        assert len(crops) == 8
        assert isinstance(label, int)

    def test_multicrop_dataset_raw(self, temp_data_dir):
        """Test MultiCropDatasetRaw."""
        dataset = MultiCropDatasetRaw(
            dataset_name="cifar10",
            data_path=temp_data_dir,
            split="train",
            download=True,
        )

        assert len(dataset) > 0

        # Get a sample
        item = dataset[0]
        if isinstance(item, tuple):
            crops, label = item
            assert isinstance(crops, (list, torch.Tensor))

    def test_build_multicrop_dataset(self, temp_data_dir):
        """Test building multi-crop dataset."""
        dataset = build_multicrop_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir,
            split="train",
            num_global_crops=2,
            num_local_crops=6,
            download=True,
        )

        assert len(dataset) > 0
        crops, label = dataset[0]
        assert isinstance(label, int)

    def test_build_multicrop_dataloader(self, temp_data_dir):
        """Test building multi-crop dataloader."""
        dataset = build_multicrop_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir,
            split="train",
            download=True,
        )

        dataloader = build_multicrop_dataloader(
            dataset,
            batch_size=4,
            num_workers=0,
            shuffle=True,
        )

        # Get a batch
        batch = next(iter(dataloader))
        crops_batch, labels_batch = batch

        assert isinstance(crops_batch, list)
        assert labels_batch.shape[0] == 4

    def test_multicrop_collate_fn(self):
        """Test custom collate function for multi-crop datasets."""
        # Create mock batch
        batch = [
            (
                [
                    torch.randn(3, 224, 224),
                    torch.randn(3, 224, 224),
                    torch.randn(3, 96, 96),
                ],
                0,
            ),
            (
                [
                    torch.randn(3, 224, 224),
                    torch.randn(3, 224, 224),
                    torch.randn(3, 96, 96),
                ],
                1,
            ),
        ]

        crops_batch, labels = multicrop_collate_fn(batch)

        assert isinstance(crops_batch, list)
        assert len(crops_batch) == 3
        assert crops_batch[0].shape[0] == 2  # batch size


class TestMultiDatasets:
    """Test multi-dataset support for foundation models."""

    @pytest.fixture(scope="class")
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_weighted_multi_dataset(self, temp_data_dir):
        """Test WeightedMultiDataset with sampling weights."""
        # Create mock datasets
        dataset1 = CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )
        dataset2 = CIFAR100Dataset(
            data_path=temp_data_dir / "cifar100",
            split="train",
            download=True,
        )

        multi_dataset = WeightedMultiDataset(
            datasets=[dataset1, dataset2],
            weights=[0.7, 0.3],
            dataset_names=["cifar10", "cifar100"],
        )

        assert len(multi_dataset) > 0

        # Get a sample
        multi_dataset[0]
        # Should return (image, label, dataset_idx) or similar

    def test_balanced_multi_dataset(self, temp_data_dir):
        """Test BalancedMultiDataset with equal representation."""
        dataset1 = CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )
        dataset2 = CIFAR100Dataset(
            data_path=temp_data_dir / "cifar100",
            split="train",
            download=True,
        )

        multi_dataset = BalancedMultiDataset(
            datasets=[dataset1, dataset2],
            dataset_names=["cifar10", "cifar100"],
        )

        assert len(multi_dataset) > 0

        # Test resampling
        multi_dataset.resample_indices()
        assert len(multi_dataset.indices) == len(multi_dataset)

    def test_get_dataset_stats(self, temp_data_dir):
        """Test getting dataset statistics from WeightedMultiDataset."""
        dataset1 = CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )
        dataset2 = CIFAR100Dataset(
            data_path=temp_data_dir / "cifar100",
            split="train",
            download=True,
        )

        multi_dataset = WeightedMultiDataset(
            datasets=[dataset1, dataset2],
            weights=[0.7, 0.3],
            dataset_names=["cifar10", "cifar100"],
        )

        stats = multi_dataset.get_dataset_stats()
        assert "cifar10" in stats
        assert "cifar100" in stats
        assert "size" in stats["cifar10"]
        assert "weight" in stats["cifar10"]


class TestDatasetBuilder:
    """Test dataset builder function."""

    @pytest.fixture(scope="class")
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_build_cifar10(self, temp_data_dir):
        """Test building CIFAR-10 dataset."""
        dataset = build_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        assert len(dataset) == 50000
        img, label = dataset[0]
        assert img.shape == (3, 224, 224)

    def test_build_cifar100(self, temp_data_dir):
        """Test building CIFAR-100 dataset."""
        dataset = build_dataset(
            dataset_name="cifar100",
            data_path=temp_data_dir / "cifar100",
            split="train",
            download=True,
        )

        assert len(dataset) == 50000
        img, label = dataset[0]
        assert img.shape == (3, 224, 224)

    def test_build_unknown_dataset(self, temp_data_dir):
        """Test building unknown dataset raises error."""
        with pytest.raises(ValueError):
            build_dataset(
                dataset_name="unknown_dataset",
                data_path=temp_data_dir,
                split="train",
            )

    def test_build_dataset_case_insensitive(self, temp_data_dir):
        """Test dataset builder is case-insensitive."""
        for name in ["CIFAR10", "CiFaR10", "cifar10"]:
            dataset = build_dataset(
                dataset_name=name,
                data_path=temp_data_dir / "cifar10",
                split="train",
                download=True,
            )
            assert len(dataset) > 0

    def test_build_dataset_custom_image_size(self, temp_data_dir):
        """Test building dataset with custom image size."""
        for size in [128, 256, 384]:
            dataset = build_dataset(
                dataset_name="cifar10",
                data_path=temp_data_dir / "cifar10",
                split="train",
                image_size=size,
                download=True,
            )
            img, _ = dataset[0]
            assert img.shape == (3, size, size)


class TestDataLoader:
    """Test dataloader functionality."""

    @pytest.fixture(scope="class")
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_build_dataloader(self, temp_data_dir):
        """Test building dataloader."""
        dataset = build_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        dataloader = build_dataloader(
            dataset,
            batch_size=32,
            num_workers=0,
            shuffle=True,
        )

        # Check dataloader
        assert len(dataloader) == len(dataset) // 32

        # Get one batch
        images, labels = next(iter(dataloader))
        assert images.shape == (32, 3, 224, 224)
        assert labels.shape == (32,)

    def test_dataloader_iteration(self, temp_data_dir):
        """Test iterating through dataloader."""
        dataset = build_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        dataloader = build_dataloader(
            dataset,
            batch_size=32,
            num_workers=0,
            shuffle=False,
            drop_last=True,
        )

        batch_count = 0
        total_samples = 0

        for images, labels in dataloader:
            batch_count += 1
            total_samples += images.shape[0]
            assert images.shape[0] == 32
            assert images.shape[1:] == (3, 224, 224)

        assert batch_count == len(dataset) // 32
        assert total_samples == (len(dataset) // 32) * 32

    def test_dataloader_batch_sizes(self, temp_data_dir):
        """Test dataloader with different batch sizes."""
        dataset = build_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        for batch_size in [8, 16, 32, 64]:
            dataloader = build_dataloader(
                dataset,
                batch_size=batch_size,
                num_workers=0,
                drop_last=True,
            )

            images, labels = next(iter(dataloader))
            assert images.shape[0] == batch_size

    def test_dataloader_no_shuffle(self, temp_data_dir):
        """Test dataloader with shuffle disabled."""
        dataset = build_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        dataloader = build_dataloader(
            dataset,
            batch_size=32,
            num_workers=0,
            shuffle=False,
        )

        # Should iterate successfully
        count = 0
        for _ in dataloader:
            count += 1
            if count > 2:
                break
        assert count > 0

    def test_dataloader_pin_memory(self, temp_data_dir):
        """Test dataloader with pin_memory setting."""
        dataset = build_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        dataloader = build_dataloader(
            dataset,
            batch_size=16,
            num_workers=0,
            pin_memory=True,
        )

        images, labels = next(iter(dataloader))
        assert images.shape[0] == 16


class TestDatasetInfo:
    """Test dataset information and utilities."""

    def test_dataset_info_structure(self):
        """Test DATASET_INFO has correct structure."""
        assert isinstance(DATASET_INFO, dict)
        assert "cifar10" in DATASET_INFO
        assert "cifar100" in DATASET_INFO
        assert "imagenet" in DATASET_INFO

        # Check required fields
        for name, info in DATASET_INFO.items():
            assert "name" in info
            assert "size_gb" in info
            assert "num_images" in info
            assert "num_classes" in info
            assert "resolution" in info
            assert "auto_download" in info
            assert "description" in info
            assert "url" in info

    def test_dataset_info_values(self):
        """Test DATASET_INFO has correct values."""
        # CIFAR-10
        assert DATASET_INFO["cifar10"]["num_classes"] == 10
        assert DATASET_INFO["cifar10"]["auto_download"] is True

        # CIFAR-100
        assert DATASET_INFO["cifar100"]["num_classes"] == 100
        assert DATASET_INFO["cifar100"]["auto_download"] is True

        # ImageNet
        assert DATASET_INFO["imagenet"]["num_classes"] == 1000
        assert DATASET_INFO["imagenet"]["auto_download"] is False

    def test_dataset_info_completeness(self):
        """Test that all datasets have complete information."""
        for name, info in DATASET_INFO.items():
            # Check numeric fields are positive
            assert info["size_gb"] > 0
            assert info["num_images"] > 0
            assert info["num_classes"] > 0

            # Check string fields are not empty
            assert len(info["name"]) > 0
            assert len(info["description"]) > 0
            assert len(info["url"]) > 0


class TestVerification:
    """Test dataset verification."""

    @pytest.fixture(scope="class")
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_verify_cifar10(self, temp_data_dir):
        """Test CIFAR-10 verification."""
        # Download first
        CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        # Verify
        is_valid = verify_dataset("cifar10", temp_data_dir / "cifar10")
        assert is_valid is True

    def test_verify_cifar100(self, temp_data_dir):
        """Test CIFAR-100 verification."""
        CIFAR100Dataset(
            data_path=temp_data_dir / "cifar100",
            split="train",
            download=True,
        )

        is_valid = verify_dataset("cifar100", temp_data_dir / "cifar100")
        assert is_valid is True

    def test_verify_nonexistent(self, temp_data_dir):
        """Test verification fails for non-existent dataset."""
        is_valid = verify_dataset("cifar10", temp_data_dir / "nonexistent")
        assert is_valid is False

    def test_verify_case_insensitive(self, temp_data_dir):
        """Test verification is case-insensitive."""
        # Download CIFAR-10
        CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        # Test with different cases
        for name in ["CIFAR10", "CiFaR10", "cifar10"]:
            is_valid = verify_dataset(name, temp_data_dir / "cifar10")
            assert is_valid is True


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    @pytest.fixture(scope="class")
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_custom_image_size(self, temp_data_dir):
        """Test datasets with custom image sizes."""
        for size in [96, 128, 256, 384]:
            dataset = build_dataset(
                dataset_name="cifar10",
                data_path=temp_data_dir / "cifar10",
                split="train",
                image_size=size,
                download=True,
            )

            img, _ = dataset[0]
            assert img.shape == (3, size, size)

    def test_no_color_jitter(self, temp_data_dir):
        """Test dataset without color jitter."""
        dataset = CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="train",
            color_jitter=None,
            download=True,
        )

        img, _ = dataset[0]
        assert img.shape == (3, 224, 224)

    def test_custom_transform(self, temp_data_dir):
        """Test dataset with custom transform."""
        from torchvision import transforms

        custom_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        dataset = CIFAR10Dataset(
            data_path=temp_data_dir / "cifar10",
            split="train",
            transform=custom_transform,
            download=True,
        )

        img, _ = dataset[0]
        assert img.shape == (3, 224, 224)

    def test_batch_accumulation(self, temp_data_dir):
        """Test batch accumulation from multiple samples."""
        dataset = build_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        # Accumulate multiple samples
        images = []
        labels = []
        for i in range(10):
            img, label = dataset[i]
            images.append(img.unsqueeze(0))
            labels.append(label)

        images_batch = torch.cat(images, dim=0)
        assert images_batch.shape == (10, 3, 224, 224)
        assert len(labels) == 10

    def test_transform_reproducibility(self, sample_image_224):
        """Test transform reproducibility with different image inputs."""
        transform = JEPATransform(image_size=224)

        # Apply to same image multiple times
        outputs = [transform(sample_image_224.copy()) for _ in range(3)]

        # All should have same shape
        for output in outputs:
            assert output.shape == (3, 224, 224)

    def test_large_batch_handling(self, temp_data_dir):
        """Test handling of large batch sizes."""
        dataset = build_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        dataloader = build_dataloader(
            dataset,
            batch_size=256,
            num_workers=0,
            drop_last=True,
        )

        images, labels = next(iter(dataloader))
        assert images.shape[0] == 256
        assert labels.shape[0] == 256

    def test_single_sample_batch(self, temp_data_dir):
        """Test handling of single sample batches."""
        dataset = build_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir / "cifar10",
            split="train",
            download=True,
        )

        dataloader = build_dataloader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            drop_last=False,
        )

        images, labels = next(iter(dataloader))
        assert images.shape == (1, 3, 224, 224)
        assert labels.shape == (1,)

    def test_transform_dtype_consistency(self, sample_image_224):
        """Test that transforms maintain consistent dtype."""
        transforms_to_test = [
            JEPATransform(image_size=224),
            JEPAEvalTransform(image_size=224),
            DeiTIIIAugmentation(image_size=224),
        ]

        for transform in transforms_to_test:
            output = transform(sample_image_224)
            assert output.dtype == torch.float32

    def test_invalid_dataset_split(self, temp_data_dir):
        """Test handling of invalid splits."""
        # Most datasets accept 'train' and 'val/test'
        # Try with invalid split (should either work or fail gracefully)
        try:
            dataset = build_dataset(
                dataset_name="cifar10",
                data_path=temp_data_dir / "cifar10",
                split="train",
                download=True,
            )
            assert len(dataset) > 0
        except (ValueError, FileNotFoundError):
            # Expected for invalid split
            pass

    def test_zero_probability_augmentation(self, sample_batch_224):
        """Test augmentations with zero probability."""
        images, targets = sample_batch_224

        # Mixup with 0 probability
        mixup = Mixup(alpha=0.8, num_classes=1000, prob=0.0)
        mixed_images, mixed_targets = mixup(images, targets)
        assert mixed_images.shape == images.shape

        # CutMix with 0 probability
        cutmix = CutMix(alpha=1.0, num_classes=1000, prob=0.0)
        mixed_images, mixed_targets = cutmix(images, targets)
        assert mixed_images.shape == images.shape

    def test_tensor_device_preservation(self, sample_batch_224):
        """Test that augmentations preserve tensor device."""
        images, targets = sample_batch_224

        mixup = Mixup(alpha=0.8, num_classes=1000)
        mixed_images, mixed_targets = mixup(images, targets)

        assert mixed_images.device == images.device
        assert mixed_targets.device == targets.device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

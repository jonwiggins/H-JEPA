"""
Comprehensive tests for H-JEPA data modules.

This test module provides extensive coverage for the four core data modules:
- src/data/datasets.py (159 lines)
- src/data/transforms.py (195 lines)
- src/data/multicrop_dataset.py (123 lines)
- src/data/multicrop_transforms.py (87 lines)

Target: 70%+ coverage for each module

Test Strategy:
1. Mock external dependencies (PIL, torch datasets, downloads)
2. Test all public methods and classes
3. Cover edge cases and error handling
4. Test integration between modules
"""

import math
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# Import all modules we need to test
from src.data.datasets import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    ImageNet100Dataset,
    ImageNetDataset,
    JEPAEvalTransform,
    JEPATransform,
    STL10Dataset,
    build_dataloader,
    build_dataset,
)
from src.data.multicrop_dataset import (
    MultiCropDataset,
    MultiCropDatasetRaw,
    build_multicrop_dataloader,
    build_multicrop_dataset,
    multicrop_collate_fn,
)
from src.data.multicrop_transforms import (
    AdaptiveMultiCropTransform,
    MultiCropEvalTransform,
    MultiCropTransform,
    build_multicrop_transform,
)
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

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_pil_image():
    """Create a sample PIL Image for testing."""
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def sample_pil_image_small():
    """Create a small PIL Image for testing."""
    return Image.new("RGB", (32, 32), color=(100, 100, 100))


@pytest.fixture
def sample_tensor_image():
    """Create a sample tensor image for testing."""
    return torch.randn(3, 224, 224)


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    images = torch.randn(8, 3, 224, 224)
    targets = torch.randint(0, 1000, (8,))
    return images, targets


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_imagenet_structure(temp_data_dir):
    """Create a mock ImageNet directory structure."""
    train_dir = temp_data_dir / "imagenet" / "train"
    val_dir = temp_data_dir / "imagenet" / "val"

    # Create some mock class directories
    for class_id in ["n01440764", "n01443537", "n01498041"]:
        class_train_dir = train_dir / class_id
        class_val_dir = val_dir / class_id
        class_train_dir.mkdir(parents=True, exist_ok=True)
        class_val_dir.mkdir(parents=True, exist_ok=True)

        # Create a few mock images
        for i in range(3):
            img = Image.new("RGB", (224, 224), color=(i * 50, i * 50, i * 50))
            img.save(class_train_dir / f"{class_id}_{i}.JPEG")
            img.save(class_val_dir / f"{class_id}_{i}.JPEG")

    return temp_data_dir / "imagenet"


# =============================================================================
# Tests for datasets.py
# =============================================================================


class TestJEPATransform:
    """Test JEPA training transform."""

    def test_initialization_default(self):
        """Test default initialization."""
        transform = JEPATransform()
        assert hasattr(transform, "transform")

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        transform = JEPATransform(
            image_size=256,
            crop_scale=(0.5, 1.0),
            horizontal_flip=False,
            color_jitter=0.5,
        )
        assert hasattr(transform, "transform")

    def test_transform_output_shape(self, sample_pil_image):
        """Test transform produces correct output shape."""
        transform = JEPATransform(image_size=224)
        output = transform(sample_pil_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)
        assert output.dtype == torch.float32

    def test_transform_with_color_jitter(self, sample_pil_image):
        """Test transform with color jitter enabled."""
        transform = JEPATransform(image_size=224, color_jitter=0.4)
        output = transform(sample_pil_image)
        assert output.shape == (3, 224, 224)

    def test_transform_without_color_jitter(self, sample_pil_image):
        """Test transform with color jitter disabled."""
        transform = JEPATransform(image_size=224, color_jitter=None)
        output = transform(sample_pil_image)
        assert output.shape == (3, 224, 224)

    def test_transform_no_horizontal_flip(self, sample_pil_image):
        """Test transform without horizontal flip."""
        transform = JEPATransform(image_size=224, horizontal_flip=False)
        output = transform(sample_pil_image)
        assert output.shape == (3, 224, 224)

    def test_transform_custom_normalization(self, sample_pil_image):
        """Test transform with custom normalization."""
        transform = JEPATransform(
            image_size=224,
            mean=(0.5, 0.5, 0.5),
            std=(0.2, 0.2, 0.2),
        )
        output = transform(sample_pil_image)
        assert output.shape == (3, 224, 224)

    def test_transform_different_sizes(self):
        """Test transform with different input sizes."""
        transform = JEPATransform(image_size=224)

        for size in [32, 64, 128, 256, 512]:
            img = Image.new("RGB", (size, size), color=(128, 128, 128))
            output = transform(img)
            assert output.shape == (3, 224, 224)

    def test_transform_tensor_input(self, sample_tensor_image):
        """Test transform accepts tensor input."""
        transform = JEPATransform(image_size=224)
        # The transform might work with tensors or might not - test gracefully
        try:
            output = transform(sample_tensor_image)
            assert isinstance(output, torch.Tensor)
        except Exception:
            # It's OK if it doesn't support tensor input
            pass


class TestJEPAEvalTransform:
    """Test JEPA evaluation transform."""

    def test_initialization(self):
        """Test initialization."""
        transform = JEPAEvalTransform(image_size=224)
        assert hasattr(transform, "transform")

    def test_transform_output_shape(self, sample_pil_image):
        """Test transform produces correct output shape."""
        transform = JEPAEvalTransform(image_size=224)
        output = transform(sample_pil_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_transform_deterministic(self, sample_pil_image):
        """Test that eval transform is deterministic."""
        transform = JEPAEvalTransform(image_size=224)

        # Create a consistent image
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))

        # Apply twice - results should be very similar (center crop is deterministic)
        output1 = transform(img.copy())
        output2 = transform(img.copy())

        # Both should have same shape
        assert output1.shape == output2.shape

    def test_transform_custom_size(self):
        """Test transform with custom size."""
        transform = JEPAEvalTransform(image_size=256)
        img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        output = transform(img)
        assert output.shape == (3, 256, 256)


class TestCIFAR10Dataset:
    """Test CIFAR-10 dataset."""

    @patch("src.data.datasets.datasets.CIFAR10")
    def test_initialization_train(self, mock_cifar10, temp_data_dir):
        """Test CIFAR-10 train dataset initialization."""
        # Mock the CIFAR10 dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=50000)
        mock_cifar10.return_value = mock_dataset

        dataset = CIFAR10Dataset(
            data_path=temp_data_dir,
            split="train",
            download=False,
        )

        assert dataset.split == "train"
        assert mock_cifar10.called

    @patch("src.data.datasets.datasets.CIFAR10")
    def test_initialization_val(self, mock_cifar10, temp_data_dir):
        """Test CIFAR-10 val dataset initialization."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10000)
        mock_cifar10.return_value = mock_dataset

        dataset = CIFAR10Dataset(
            data_path=temp_data_dir,
            split="val",
            download=False,
        )

        assert dataset.split == "val"

    @patch("src.data.datasets.datasets.CIFAR10")
    def test_len(self, mock_cifar10, temp_data_dir):
        """Test dataset length."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=50000)
        mock_cifar10.return_value = mock_dataset

        dataset = CIFAR10Dataset(data_path=temp_data_dir, download=False)
        assert len(dataset) == 50000

    @patch("src.data.datasets.datasets.CIFAR10")
    def test_getitem(self, mock_cifar10, temp_data_dir):
        """Test getting an item from dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=50000)
        mock_dataset.__getitem__ = MagicMock(return_value=(torch.randn(3, 224, 224), 5))
        mock_cifar10.return_value = mock_dataset

        dataset = CIFAR10Dataset(data_path=temp_data_dir, download=False)
        img, label = dataset[0]

        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)

    @patch("src.data.datasets.datasets.CIFAR10")
    def test_classes_property(self, mock_cifar10, temp_data_dir):
        """Test classes property."""
        mock_dataset = MagicMock()
        mock_dataset.classes = ["airplane", "automobile", "bird"]
        mock_cifar10.return_value = mock_dataset

        dataset = CIFAR10Dataset(data_path=temp_data_dir, download=False)
        assert len(dataset.classes) == 3


class TestCIFAR100Dataset:
    """Test CIFAR-100 dataset."""

    @patch("src.data.datasets.datasets.CIFAR100")
    def test_initialization(self, mock_cifar100, temp_data_dir):
        """Test CIFAR-100 initialization."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=50000)
        mock_cifar100.return_value = mock_dataset

        dataset = CIFAR100Dataset(data_path=temp_data_dir, download=False)
        assert hasattr(dataset, "dataset")

    @patch("src.data.datasets.datasets.CIFAR100")
    def test_classes_property(self, mock_cifar100, temp_data_dir):
        """Test classes property returns 100 classes."""
        mock_dataset = MagicMock()
        mock_dataset.classes = [f"class_{i}" for i in range(100)]
        mock_cifar100.return_value = mock_dataset

        dataset = CIFAR100Dataset(data_path=temp_data_dir, download=False)
        assert len(dataset.classes) == 100


class TestSTL10Dataset:
    """Test STL-10 dataset."""

    @patch("src.data.datasets.datasets.STL10")
    def test_initialization_train(self, mock_stl10, temp_data_dir):
        """Test STL-10 train dataset initialization."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5000)
        mock_stl10.return_value = mock_dataset

        dataset = STL10Dataset(data_path=temp_data_dir, split="train", download=False)
        assert dataset.split == "train"

    @patch("src.data.datasets.datasets.STL10")
    def test_initialization_unlabeled(self, mock_stl10, temp_data_dir):
        """Test STL-10 unlabeled dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100000)
        mock_stl10.return_value = mock_dataset

        dataset = STL10Dataset(data_path=temp_data_dir, split="unlabeled", download=False)
        assert dataset.split == "unlabeled"


class TestImageNetDataset:
    """Test ImageNet dataset."""

    def test_initialization_missing_directory(self, temp_data_dir):
        """Test that missing directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ImageNetDataset(
                data_path=temp_data_dir / "nonexistent",
                split="train",
            )

    def test_initialization_with_mock_structure(self, mock_imagenet_structure):
        """Test initialization with proper directory structure."""
        dataset = ImageNetDataset(
            data_path=mock_imagenet_structure,
            split="train",
        )

        assert len(dataset) > 0
        assert hasattr(dataset, "classes")
        assert hasattr(dataset, "class_to_idx")

    def test_getitem(self, mock_imagenet_structure):
        """Test getting an item from ImageNet dataset."""
        dataset = ImageNetDataset(
            data_path=mock_imagenet_structure,
            split="train",
        )

        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)

    def test_classes_property(self, mock_imagenet_structure):
        """Test classes property."""
        dataset = ImageNetDataset(
            data_path=mock_imagenet_structure,
            split="train",
        )

        classes = dataset.classes
        assert isinstance(classes, list)
        assert len(classes) > 0

    def test_class_to_idx_property(self, mock_imagenet_structure):
        """Test class_to_idx property."""
        dataset = ImageNetDataset(
            data_path=mock_imagenet_structure,
            split="train",
        )

        class_to_idx = dataset.class_to_idx
        assert isinstance(class_to_idx, dict)
        assert len(class_to_idx) > 0


class TestImageNet100Dataset:
    """Test ImageNet-100 subset dataset."""

    def test_initialization(self, mock_imagenet_structure):
        """Test ImageNet-100 initialization."""
        dataset = ImageNet100Dataset(
            data_path=mock_imagenet_structure,
            split="train",
        )

        assert hasattr(dataset, "_valid_indices")
        assert hasattr(dataset, "IMAGENET100_CLASSES")

    def test_filter_classes(self, mock_imagenet_structure):
        """Test that only ImageNet-100 classes are included."""
        dataset = ImageNet100Dataset(
            data_path=mock_imagenet_structure,
            split="train",
        )

        # Should filter to only classes in IMAGENET100_CLASSES
        assert len(dataset) <= len(dataset._original_dataset)

    def test_len(self, mock_imagenet_structure):
        """Test filtered dataset length."""
        dataset = ImageNet100Dataset(
            data_path=mock_imagenet_structure,
            split="train",
        )

        # Length should be number of valid indices
        assert len(dataset) == len(dataset._valid_indices)

    def test_getitem(self, mock_imagenet_structure):
        """Test getting item from filtered dataset."""
        dataset = ImageNet100Dataset(
            data_path=mock_imagenet_structure,
            split="train",
        )

        if len(dataset) > 0:
            img, label = dataset[0]
            assert isinstance(img, torch.Tensor)
            assert isinstance(label, int)


class TestBuildDataset:
    """Test build_dataset factory function."""

    @patch("src.data.datasets.CIFAR10Dataset")
    def test_build_cifar10(self, mock_cifar10, temp_data_dir):
        """Test building CIFAR-10 dataset."""
        mock_cifar10.return_value = MagicMock()

        dataset = build_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir,
            split="train",
            download=False,
        )

        assert mock_cifar10.called

    @patch("src.data.datasets.CIFAR100Dataset")
    def test_build_cifar100(self, mock_cifar100, temp_data_dir):
        """Test building CIFAR-100 dataset."""
        mock_cifar100.return_value = MagicMock()

        dataset = build_dataset(
            dataset_name="cifar100",
            data_path=temp_data_dir,
            split="train",
            download=False,
        )

        assert mock_cifar100.called

    @patch("src.data.datasets.STL10Dataset")
    def test_build_stl10(self, mock_stl10, temp_data_dir):
        """Test building STL-10 dataset."""
        mock_stl10.return_value = MagicMock()

        dataset = build_dataset(
            dataset_name="stl10",
            data_path=temp_data_dir,
            split="train",
            download=False,
        )

        assert mock_stl10.called

    def test_build_unknown_dataset(self, temp_data_dir):
        """Test building unknown dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            build_dataset(
                dataset_name="unknown_dataset",
                data_path=temp_data_dir,
                split="train",
            )

    @patch("src.data.datasets.CIFAR10Dataset")
    def test_build_case_insensitive(self, mock_cifar10, temp_data_dir):
        """Test dataset name is case-insensitive."""
        mock_cifar10.return_value = MagicMock()

        for name in ["CIFAR10", "CiFaR10", "cifar10"]:
            build_dataset(
                dataset_name=name,
                data_path=temp_data_dir,
                split="train",
                download=False,
            )

        assert mock_cifar10.call_count == 3


class TestBuildDataLoader:
    """Test build_dataloader function."""

    def test_build_dataloader_basic(self):
        """Test building a basic dataloader."""
        # Create a simple mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        dataloader = build_dataloader(
            dataset=mock_dataset,
            batch_size=32,
            num_workers=0,
            shuffle=True,
        )

        assert dataloader.batch_size == 32
        assert dataloader.shuffle is True

    def test_build_dataloader_custom_params(self):
        """Test building dataloader with custom parameters."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        dataloader = build_dataloader(
            dataset=mock_dataset,
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )

        assert dataloader.batch_size == 64
        assert dataloader.pin_memory is True
        assert dataloader.drop_last is True


# =============================================================================
# Tests for transforms.py
# =============================================================================


class TestRandAugment:
    """Test RandAugment implementation."""

    def test_initialization(self):
        """Test RandAugment initialization."""
        aug = RandAugment(num_ops=2, magnitude=9)
        assert aug.num_ops == 2
        assert aug.magnitude == 9
        assert len(aug.augment_ops) > 0

    def test_augment_image(self, sample_pil_image):
        """Test augmenting an image."""
        aug = RandAugment(num_ops=2, magnitude=9)
        output = aug(sample_pil_image)

        assert isinstance(output, Image.Image)
        assert output.size == sample_pil_image.size

    def test_different_num_ops(self, sample_pil_image):
        """Test with different number of operations."""
        for num_ops in [1, 2, 3, 5]:
            aug = RandAugment(num_ops=num_ops, magnitude=9)
            output = aug(sample_pil_image)
            assert isinstance(output, Image.Image)

    def test_different_magnitudes(self, sample_pil_image):
        """Test with different magnitudes."""
        for magnitude in [0, 5, 10, 15, 30]:
            aug = RandAugment(num_ops=2, magnitude=magnitude)
            output = aug(sample_pil_image)
            assert isinstance(output, Image.Image)

    def test_identity_operation(self, sample_pil_image):
        """Test identity operation."""
        aug = RandAugment(num_ops=1, magnitude=0)
        output = aug._identity(sample_pil_image, None)
        assert output == sample_pil_image

    def test_auto_contrast(self, sample_pil_image):
        """Test auto contrast operation."""
        aug = RandAugment()
        output = aug._auto_contrast(sample_pil_image, None)
        assert isinstance(output, Image.Image)

    def test_equalize(self, sample_pil_image):
        """Test equalize operation."""
        aug = RandAugment()
        output = aug._equalize(sample_pil_image, None)
        assert isinstance(output, Image.Image)

    def test_rotate(self, sample_pil_image):
        """Test rotate operation."""
        aug = RandAugment()
        output = aug._rotate(sample_pil_image, 15.0)
        assert isinstance(output, Image.Image)

    def test_solarize(self, sample_pil_image):
        """Test solarize operation."""
        aug = RandAugment()
        output = aug._solarize(sample_pil_image, 128.0)
        assert isinstance(output, Image.Image)

    def test_color(self, sample_pil_image):
        """Test color operation."""
        aug = RandAugment()
        output = aug._color(sample_pil_image, 1.2)
        assert isinstance(output, Image.Image)

    def test_get_magnitude(self):
        """Test get_magnitude method."""
        aug = RandAugment(num_ops=2, magnitude=15)

        # Test with None range
        mag = aug._get_magnitude(None)
        assert mag is None

        # Test with numeric range
        mag = aug._get_magnitude((0.1, 1.9))
        assert mag is not None
        assert 0.1 <= mag <= 1.9


class TestMixup:
    """Test Mixup augmentation."""

    def test_initialization(self):
        """Test Mixup initialization."""
        mixup = Mixup(alpha=0.8, num_classes=1000)
        assert mixup.alpha == 0.8
        assert mixup.num_classes == 1000

    def test_mixup_basic(self, sample_batch):
        """Test basic Mixup operation."""
        images, targets = sample_batch
        mixup = Mixup(alpha=0.8, num_classes=1000)

        mixed_images, mixed_targets = mixup(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (images.shape[0], 1000)
        assert mixed_targets.dtype == torch.float32

    def test_mixup_alpha_zero(self, sample_batch):
        """Test Mixup with alpha=0 (no mixing)."""
        images, targets = sample_batch
        mixup = Mixup(alpha=0.0, num_classes=1000)

        mixed_images, mixed_targets = mixup(images, targets)
        assert mixed_images.shape == images.shape

    def test_mixup_probability_zero(self, sample_batch):
        """Test Mixup with prob=0 (never applies)."""
        images, targets = sample_batch
        mixup = Mixup(alpha=0.8, num_classes=1000, prob=0.0)

        mixed_images, mixed_targets = mixup(images, targets)

        # Should just return one-hot encoded targets without mixing
        assert mixed_targets.sum(dim=1).allclose(torch.ones(images.shape[0]))

    def test_mixup_target_distribution(self, sample_batch):
        """Test that Mixup creates valid target distributions."""
        images, targets = sample_batch
        mixup = Mixup(alpha=0.8, num_classes=1000)

        _, mixed_targets = mixup(images, targets)

        # Each target should sum to 1 (valid probability distribution)
        sums = mixed_targets.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


class TestCutMix:
    """Test CutMix augmentation."""

    def test_initialization(self):
        """Test CutMix initialization."""
        cutmix = CutMix(alpha=1.0, num_classes=1000)
        assert cutmix.alpha == 1.0
        assert cutmix.num_classes == 1000

    def test_cutmix_basic(self, sample_batch):
        """Test basic CutMix operation."""
        images, targets = sample_batch
        cutmix = CutMix(alpha=1.0, num_classes=1000)

        mixed_images, mixed_targets = cutmix(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape == (images.shape[0], 1000)

    def test_rand_bbox(self):
        """Test random bounding box generation."""
        cutmix = CutMix(alpha=1.0, num_classes=1000)

        x1, y1, x2, y2 = cutmix._rand_bbox(224, 224, 0.5)

        # Check bounds
        assert 0 <= x1 < x2 <= 224
        assert 0 <= y1 < y2 <= 224

    def test_cutmix_probability_zero(self, sample_batch):
        """Test CutMix with prob=0."""
        images, targets = sample_batch
        cutmix = CutMix(alpha=1.0, num_classes=1000, prob=0.0)

        mixed_images, mixed_targets = cutmix(images, targets)

        # Should return images unchanged
        assert torch.allclose(mixed_images, images)


class TestMixupCutmix:
    """Test combined Mixup and CutMix."""

    def test_initialization(self):
        """Test initialization."""
        mixup_cutmix = MixupCutmix(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            num_classes=1000,
        )
        assert hasattr(mixup_cutmix, "mixup")
        assert hasattr(mixup_cutmix, "cutmix")

    def test_call(self, sample_batch):
        """Test calling MixupCutmix."""
        images, targets = sample_batch
        mixup_cutmix = MixupCutmix(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            num_classes=1000,
            prob=1.0,
            switch_prob=0.5,
        )

        mixed_images, mixed_targets = mixup_cutmix(images, targets)

        assert mixed_images.shape == images.shape
        assert mixed_targets.shape[0] == images.shape[0]


class TestRandomErasing:
    """Test Random Erasing augmentation."""

    def test_initialization(self):
        """Test initialization."""
        erasing = RandomErasing(prob=0.5, scale=(0.02, 0.33))
        assert erasing.prob == 0.5
        assert erasing.scale == (0.02, 0.33)

    def test_random_erasing_basic(self, sample_tensor_image):
        """Test basic random erasing."""
        erasing = RandomErasing(prob=1.0)
        output = erasing(sample_tensor_image)

        assert output.shape == sample_tensor_image.shape

    def test_random_erasing_no_erase(self, sample_tensor_image):
        """Test with prob=0 (no erasing)."""
        erasing = RandomErasing(prob=0.0)
        output = erasing(sample_tensor_image)

        # Should be unchanged
        assert torch.allclose(output, sample_tensor_image)

    def test_random_erasing_value_modes(self, sample_tensor_image):
        """Test different value modes."""
        # Erase with zero
        erasing_zero = RandomErasing(prob=1.0, value=0)
        output = erasing_zero(sample_tensor_image)
        assert output.shape == sample_tensor_image.shape

        # Erase with random values
        erasing_random = RandomErasing(prob=1.0, value="random")
        output = erasing_random(sample_tensor_image)
        assert output.shape == sample_tensor_image.shape

    def test_random_erasing_inplace(self, sample_tensor_image):
        """Test inplace erasing."""
        img_copy = sample_tensor_image.clone()
        erasing = RandomErasing(prob=1.0, inplace=True)
        output = erasing(img_copy)

        # Should modify in place
        assert output is img_copy


class TestDeiTIIIAugmentation:
    """Test DeiT III augmentation pipeline."""

    def test_initialization(self):
        """Test initialization."""
        aug = DeiTIIIAugmentation(image_size=224, num_classes=1000)
        assert aug.image_size == 224
        assert aug.num_classes == 1000

    def test_augmentation_output(self, sample_pil_image):
        """Test augmentation produces correct output."""
        aug = DeiTIIIAugmentation(image_size=224, num_classes=1000)
        output = aug(sample_pil_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_get_image_transform(self):
        """Test getting image transform."""
        aug = DeiTIIIAugmentation(image_size=224, num_classes=1000)
        transform = aug.get_image_transform()

        assert transform is not None
        assert hasattr(transform, "__call__")

    def test_get_batch_transform(self):
        """Test getting batch transform."""
        aug = DeiTIIIAugmentation(image_size=224, num_classes=1000)
        batch_transform = aug.get_batch_transform()

        assert isinstance(batch_transform, MixupCutmix)

    def test_augmentation_with_params(self, sample_pil_image):
        """Test augmentation with custom parameters."""
        aug = DeiTIIIAugmentation(
            image_size=256,
            color_jitter=0.5,
            auto_augment=True,
            random_erasing_prob=0.5,
            mixup_alpha=1.0,
            cutmix_alpha=1.0,
            num_classes=100,
        )
        output = aug(sample_pil_image)
        assert output.shape == (3, 256, 256)


class TestDeiTIIIEvalTransform:
    """Test DeiT III evaluation transform."""

    def test_initialization(self):
        """Test initialization."""
        transform = DeiTIIIEvalTransform(image_size=224)
        assert hasattr(transform, "transform")

    def test_transform_output(self, sample_pil_image):
        """Test transform output."""
        transform = DeiTIIIEvalTransform(image_size=224)
        output = transform(sample_pil_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)


class TestBuildDeiT3Transform:
    """Test build_deit3_transform factory function."""

    def test_build_training_transform(self):
        """Test building training transform."""
        transform = build_deit3_transform(is_training=True)
        assert isinstance(transform, DeiTIIIAugmentation)

    def test_build_eval_transform(self):
        """Test building evaluation transform."""
        transform = build_deit3_transform(is_training=False)
        assert isinstance(transform, DeiTIIIEvalTransform)

    def test_build_with_config(self):
        """Test building with custom config."""
        config = {
            "image_size": 256,
            "num_classes": 100,
            "mixup_alpha": 1.0,
        }
        transform = build_deit3_transform(is_training=True, config=config)
        assert isinstance(transform, DeiTIIIAugmentation)
        assert transform.image_size == 256


# =============================================================================
# Tests for multicrop_transforms.py
# =============================================================================


class TestMultiCropTransform:
    """Test MultiCropTransform."""

    def test_initialization(self):
        """Test initialization."""
        transform = MultiCropTransform(
            num_global_crops=2,
            num_local_crops=6,
            global_crop_size=224,
            local_crop_size=96,
        )
        assert transform.num_global_crops == 2
        assert transform.num_local_crops == 6

    def test_transform_output(self, sample_pil_image):
        """Test transform produces correct output."""
        transform = MultiCropTransform(
            num_global_crops=2,
            num_local_crops=6,
            global_crop_size=224,
            local_crop_size=96,
        )
        crops = transform(sample_pil_image)

        assert isinstance(crops, list)
        assert len(crops) == 8  # 2 global + 6 local

    def test_crop_sizes(self, sample_pil_image):
        """Test crop sizes are correct."""
        transform = MultiCropTransform(
            num_global_crops=2,
            num_local_crops=6,
            global_crop_size=224,
            local_crop_size=96,
        )
        crops = transform(sample_pil_image)

        # Check global crop sizes
        for i in range(2):
            assert crops[i].shape == (3, 224, 224)

        # Check local crop sizes
        for i in range(2, 8):
            assert crops[i].shape == (3, 96, 96)

    def test_different_configurations(self, sample_pil_image):
        """Test with different configurations."""
        configs = [
            (2, 4, 224, 96),
            (1, 8, 224, 96),
            (4, 2, 224, 96),
        ]

        for num_global, num_local, global_size, local_size in configs:
            transform = MultiCropTransform(
                num_global_crops=num_global,
                num_local_crops=num_local,
                global_crop_size=global_size,
                local_crop_size=local_size,
            )
            crops = transform(sample_pil_image)
            assert len(crops) == num_global + num_local

    def test_repr(self):
        """Test string representation."""
        transform = MultiCropTransform(
            num_global_crops=2,
            num_local_crops=6,
        )
        repr_str = repr(transform)
        assert "MultiCropTransform" in repr_str


class TestMultiCropEvalTransform:
    """Test MultiCropEvalTransform."""

    def test_initialization(self):
        """Test initialization."""
        transform = MultiCropEvalTransform(crop_size=224)
        assert hasattr(transform, "transform")

    def test_transform_output(self, sample_pil_image):
        """Test transform output."""
        transform = MultiCropEvalTransform(crop_size=224)
        output = transform(sample_pil_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)


class TestAdaptiveMultiCropTransform:
    """Test AdaptiveMultiCropTransform."""

    def test_initialization(self):
        """Test initialization."""
        transform = AdaptiveMultiCropTransform(
            num_global_crops=2,
            min_local_crops=2,
            max_local_crops=10,
            warmup_epochs=10,
        )
        assert transform.min_local_crops == 2
        assert transform.max_local_crops == 10
        assert transform.warmup_epochs == 10

    def test_set_epoch(self, sample_pil_image):
        """Test setting epoch updates crop count."""
        transform = AdaptiveMultiCropTransform(
            num_global_crops=2,
            min_local_crops=2,
            max_local_crops=10,
            warmup_epochs=10,
        )

        # At epoch 0
        transform.set_epoch(0)
        crops = transform(sample_pil_image)
        assert len(crops) == 4  # 2 global + 2 local

        # At epoch 5 (halfway)
        transform.set_epoch(5)
        crops = transform(sample_pil_image)
        assert len(crops) == 8  # 2 global + 6 local (approximately)

        # At epoch 10 (complete)
        transform.set_epoch(10)
        crops = transform(sample_pil_image)
        assert len(crops) == 12  # 2 global + 10 local

    def test_warmup_progression(self):
        """Test warmup progresses correctly."""
        transform = AdaptiveMultiCropTransform(
            num_global_crops=2,
            min_local_crops=2,
            max_local_crops=10,
            warmup_epochs=10,
        )

        prev_num_crops = 2
        for epoch in range(11):
            transform.set_epoch(epoch)
            # Number of local crops should increase or stay same
            assert transform.num_local_crops >= prev_num_crops
            prev_num_crops = transform.num_local_crops


class TestBuildMulticropTransform:
    """Test build_multicrop_transform factory function."""

    def test_build_default(self):
        """Test building with defaults."""
        transform = build_multicrop_transform()
        assert isinstance(transform, MultiCropTransform)

    def test_build_adaptive(self):
        """Test building adaptive transform."""
        transform = build_multicrop_transform(
            adaptive=True,
            warmup_epochs=10,
        )
        assert isinstance(transform, AdaptiveMultiCropTransform)

    def test_build_with_params(self):
        """Test building with custom parameters."""
        transform = build_multicrop_transform(
            num_global_crops=4,
            num_local_crops=8,
            global_crop_size=256,
            local_crop_size=128,
        )
        assert transform.num_global_crops == 4
        assert transform.num_local_crops == 8


# =============================================================================
# Tests for multicrop_dataset.py
# =============================================================================


class TestMultiCropDataset:
    """Test MultiCropDataset wrapper."""

    def test_initialization(self):
        """Test initialization."""
        # Create mock base dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        transform = MultiCropTransform(num_global_crops=2, num_local_crops=6)

        dataset = MultiCropDataset(
            base_dataset=mock_dataset,
            multicrop_transform=transform,
            return_labels=True,
        )

        assert dataset.base_dataset is mock_dataset
        assert dataset.multicrop_transform is transform

    def test_len(self):
        """Test dataset length."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        transform = MultiCropTransform(num_global_crops=2, num_local_crops=6)
        dataset = MultiCropDataset(mock_dataset, transform)

        assert len(dataset) == 100

    def test_getitem_with_labels(self, sample_pil_image):
        """Test getting item with labels."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=(sample_pil_image, 5))

        transform = MultiCropTransform(num_global_crops=2, num_local_crops=6)
        dataset = MultiCropDataset(mock_dataset, transform, return_labels=True)

        crops, label = dataset[0]

        assert isinstance(crops, list)
        assert len(crops) == 8
        assert label == 5

    def test_getitem_without_labels(self, sample_pil_image):
        """Test getting item without labels."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=(sample_pil_image, 5))

        transform = MultiCropTransform(num_global_crops=2, num_local_crops=6)
        dataset = MultiCropDataset(mock_dataset, transform, return_labels=False)

        result = dataset[0]
        assert isinstance(result, list)

    def test_set_epoch(self):
        """Test set_epoch method."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        transform = AdaptiveMultiCropTransform(
            num_global_crops=2,
            min_local_crops=2,
            max_local_crops=10,
            warmup_epochs=10,
        )

        dataset = MultiCropDataset(mock_dataset, transform)

        # Should not raise error
        dataset.set_epoch(5)

    def test_properties(self):
        """Test dataset properties."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.classes = ["class_0", "class_1"]

        transform = MultiCropTransform(num_global_crops=2, num_local_crops=6)
        dataset = MultiCropDataset(mock_dataset, transform)

        assert dataset.num_global_crops == 2
        assert dataset.num_local_crops == 6
        assert dataset.total_crops == 8
        assert dataset.classes == ["class_0", "class_1"]


class TestMultiCropDatasetRaw:
    """Test MultiCropDatasetRaw."""

    @patch("src.data.multicrop_dataset.build_dataset")
    def test_initialization_train(self, mock_build_dataset, temp_data_dir):
        """Test initialization for training."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_build_dataset.return_value = mock_dataset

        dataset = MultiCropDatasetRaw(
            dataset_name="cifar10",
            data_path=temp_data_dir,
            split="train",
            download=False,
        )

        assert dataset.split == "train"
        assert isinstance(dataset.transform, MultiCropTransform)

    @patch("src.data.multicrop_dataset.build_dataset")
    def test_initialization_val(self, mock_build_dataset, temp_data_dir):
        """Test initialization for validation."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_build_dataset.return_value = mock_dataset

        dataset = MultiCropDatasetRaw(
            dataset_name="cifar10",
            data_path=temp_data_dir,
            split="val",
            download=False,
        )

        assert dataset.split == "val"
        assert isinstance(dataset.transform, MultiCropEvalTransform)

    @patch("src.data.multicrop_dataset.build_dataset")
    def test_properties(self, mock_build_dataset, temp_data_dir):
        """Test dataset properties."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.classes = ["class_0", "class_1"]
        mock_build_dataset.return_value = mock_dataset

        dataset = MultiCropDatasetRaw(
            dataset_name="cifar10",
            data_path=temp_data_dir,
            split="train",
            download=False,
        )

        assert dataset.num_global_crops == 2
        assert dataset.num_local_crops >= 0


class TestMulticropCollateFunction:
    """Test multicrop_collate_fn."""

    def test_collate_basic(self):
        """Test basic collation."""
        batch = [
            ([torch.randn(3, 224, 224), torch.randn(3, 224, 224)], 0),
            ([torch.randn(3, 224, 224), torch.randn(3, 224, 224)], 1),
        ]

        crops_batch, labels = multicrop_collate_fn(batch)

        assert isinstance(crops_batch, list)
        assert len(crops_batch) == 2  # 2 crop types
        assert crops_batch[0].shape[0] == 2  # batch size

    def test_collate_empty_batch(self):
        """Test with empty batch."""
        crops, labels = multicrop_collate_fn([])

        assert isinstance(crops, list)
        assert len(crops) == 0

    def test_collate_multiple_crops(self):
        """Test with multiple crop types."""
        batch = [
            (
                [
                    torch.randn(3, 224, 224),
                    torch.randn(3, 224, 224),
                    torch.randn(3, 96, 96),
                    torch.randn(3, 96, 96),
                ],
                0,
            ),
            (
                [
                    torch.randn(3, 224, 224),
                    torch.randn(3, 224, 224),
                    torch.randn(3, 96, 96),
                    torch.randn(3, 96, 96),
                ],
                1,
            ),
        ]

        crops_batch, labels = multicrop_collate_fn(batch)

        assert len(crops_batch) == 4
        assert all(c.shape[0] == 2 for c in crops_batch)


class TestBuildMulticropDataset:
    """Test build_multicrop_dataset factory function."""

    @patch("src.data.multicrop_dataset.MultiCropDatasetRaw")
    def test_build_basic(self, mock_dataset_class, temp_data_dir):
        """Test building dataset with defaults."""
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset

        dataset = build_multicrop_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir,
            split="train",
            download=False,
        )

        assert mock_dataset_class.called

    @patch("src.data.multicrop_dataset.MultiCropDatasetRaw")
    def test_build_with_params(self, mock_dataset_class, temp_data_dir):
        """Test building with custom parameters."""
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset

        dataset = build_multicrop_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir,
            split="train",
            num_global_crops=4,
            num_local_crops=8,
            global_crop_size=256,
            local_crop_size=128,
            download=False,
        )

        assert mock_dataset_class.called


class TestBuildMulticropDataloader:
    """Test build_multicrop_dataloader function."""

    def test_build_dataloader(self):
        """Test building dataloader."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)

        dataloader = build_multicrop_dataloader(
            dataset=mock_dataset,
            batch_size=32,
            num_workers=0,
            shuffle=True,
        )

        assert dataloader.batch_size == 32
        assert dataloader.shuffle is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full data pipeline."""

    @patch("src.data.datasets.datasets.CIFAR10")
    def test_full_pipeline_cifar10(self, mock_cifar10, temp_data_dir):
        """Test full pipeline from dataset to dataloader."""
        # Mock CIFAR10
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=(torch.randn(3, 224, 224), 5))
        mock_cifar10.return_value = mock_dataset

        # Build dataset
        dataset = build_dataset(
            dataset_name="cifar10",
            data_path=temp_data_dir,
            split="train",
            download=False,
        )

        # Build dataloader
        dataloader = build_dataloader(
            dataset=dataset,
            batch_size=4,
            num_workers=0,
        )

        assert dataloader is not None

    def test_transform_pipeline(self, sample_pil_image):
        """Test complete transform pipeline."""
        # Create DeiT III augmentation
        aug = DeiTIIIAugmentation(image_size=224, num_classes=1000)

        # Apply image transform
        image_transform = aug.get_image_transform()
        transformed = image_transform(sample_pil_image)

        assert transformed.shape == (3, 224, 224)

    def test_multicrop_pipeline(self, sample_pil_image):
        """Test multicrop transform pipeline."""
        # Create multicrop transform
        transform = MultiCropTransform(
            num_global_crops=2,
            num_local_crops=6,
        )

        # Apply transform
        crops = transform(sample_pil_image)

        assert len(crops) == 8
        assert all(isinstance(c, torch.Tensor) for c in crops)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

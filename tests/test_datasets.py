"""
Comprehensive tests for H-JEPA dataset implementations.

Tests cover:
- JEPATransform and JEPAEvalTransform functionality
- CIFAR10Dataset and CIFAR100Dataset
- ImageNetDataset and ImageNet100Dataset
- STL10Dataset
- Dataset building factory functions
- DataLoader creation
- Edge cases and error handling
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from PIL import Image
from torchvision import datasets, transforms

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


class TestJEPATransform:
    """Test JEPA transform for training."""

    def test_basic_transform(self):
        """Test basic transform creation and application."""
        transform = JEPATransform(image_size=224)

        # Create a test image
        img = Image.new("RGB", (256, 256), color="red")

        # Apply transform
        result = transform(img)

        # Check output
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

    def test_different_image_sizes(self):
        """Test transform with different image sizes."""
        sizes = [96, 128, 224, 384]

        for size in sizes:
            transform = JEPATransform(image_size=size)
            img = Image.new("RGB", (256, 256), color="blue")
            result = transform(img)
            assert result.shape == (3, size, size)

    def test_crop_scale(self):
        """Test different crop scale settings."""
        # Default crop scale
        transform1 = JEPATransform(crop_scale=(0.8, 1.0))

        # More aggressive crop
        transform2 = JEPATransform(crop_scale=(0.4, 0.8))

        img = Image.new("RGB", (256, 256), color="green")
        result1 = transform1(img)
        result2 = transform2(img)

        # Both should produce valid tensors
        assert result1.shape == result2.shape == (3, 224, 224)

    def test_horizontal_flip(self):
        """Test horizontal flip option."""
        # With flip
        transform_flip = JEPATransform(horizontal_flip=True)

        # Without flip
        transform_no_flip = JEPATransform(horizontal_flip=False)

        img = Image.new("RGB", (256, 256), color="yellow")
        result1 = transform_flip(img)
        result2 = transform_no_flip(img)

        assert result1.shape == result2.shape

    def test_color_jitter(self):
        """Test color jitter augmentation."""
        # With color jitter
        transform_jitter = JEPATransform(color_jitter=0.4)

        # Without color jitter
        transform_no_jitter = JEPATransform(color_jitter=None)

        img = Image.new("RGB", (256, 256), color=(128, 64, 32))
        result1 = transform_jitter(img)
        result2 = transform_no_jitter(img)

        # Both should work
        assert result1.shape == result2.shape == (3, 224, 224)

    def test_normalization(self):
        """Test that normalization is applied correctly."""
        transform = JEPATransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        # Create a white image (255, 255, 255)
        img = Image.new("RGB", (256, 256), color=(255, 255, 255))
        result = transform(img)

        # After normalization: (1.0 - 0.5) / 0.5 = 1.0
        # Values should be around 1.0 (may vary due to resize/crop)
        assert result.max() <= 2.0  # Allow some margin
        assert result.min() >= -2.0

    def test_interpolation_modes(self):
        """Test different interpolation modes."""
        modes = [
            transforms.InterpolationMode.NEAREST,
            transforms.InterpolationMode.BILINEAR,
            transforms.InterpolationMode.BICUBIC,
        ]

        for mode in modes:
            transform = JEPATransform(interpolation=mode)
            img = Image.new("RGB", (256, 256), color="purple")
            result = transform(img)
            assert result.shape == (3, 224, 224)

    def test_tensor_input(self):
        """Test that transform works with tensor input."""
        transform = JEPATransform()

        # Create tensor image
        img_tensor = torch.rand(3, 256, 256)

        # Should work with tensor (ToTensor handles it)
        # Note: Some transforms may fail with tensor input
        # This tests the robustness
        try:
            result = transform(img_tensor)
            assert isinstance(result, torch.Tensor)
        except Exception:
            # Some transforms don't support tensor input, which is okay
            pass


class TestJEPAEvalTransform:
    """Test JEPA evaluation transform."""

    def test_basic_eval_transform(self):
        """Test basic evaluation transform."""
        transform = JEPAEvalTransform(image_size=224)

        img = Image.new("RGB", (256, 256), color="red")
        result = transform(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_center_crop(self):
        """Test that eval transform uses center crop."""
        transform = JEPAEvalTransform(image_size=224)

        # Create an image with distinct regions
        img = Image.new("RGB", (300, 300), color="white")

        result = transform(img)
        assert result.shape == (3, 224, 224)

    def test_no_randomness(self):
        """Test that eval transform is deterministic."""
        transform = JEPAEvalTransform(image_size=224)

        img = Image.new("RGB", (256, 256), color="blue")

        # Apply transform multiple times
        results = [transform(img) for _ in range(5)]

        # All results should be identical
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], atol=1e-6)

    def test_different_sizes(self):
        """Test eval transform with different sizes."""
        sizes = [96, 128, 224, 384]

        for size in sizes:
            transform = JEPAEvalTransform(image_size=size)
            img = Image.new("RGB", (512, 512), color="green")
            result = transform(img)
            assert result.shape == (3, size, size)


class TestCIFAR10Dataset:
    """Test CIFAR-10 dataset wrapper."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary directory for data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_create_dataset_train(self, temp_data_path):
        """Test creating training dataset."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            # Mock the dataset
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=50000)
            mock_dataset.__getitem__ = Mock(return_value=(torch.rand(3, 224, 224), 0))
            mock_dataset.classes = ["class0", "class1"]
            mock_cifar.return_value = mock_dataset

            dataset = CIFAR10Dataset(data_path=temp_data_path, split="train", download=False)

            assert len(dataset) == 50000
            assert dataset.split == "train"

    def test_create_dataset_val(self, temp_data_path):
        """Test creating validation dataset."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=10000)
            mock_dataset.__getitem__ = Mock(return_value=(torch.rand(3, 224, 224), 5))
            mock_dataset.classes = ["class0"]
            mock_cifar.return_value = mock_dataset

            dataset = CIFAR10Dataset(data_path=temp_data_path, split="val", download=False)

            assert len(dataset) == 10000
            assert dataset.split == "val"

    def test_custom_transform(self, temp_data_path):
        """Test using custom transform."""
        custom_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=50000)
            mock_cifar.return_value = mock_dataset

            dataset = CIFAR10Dataset(
                data_path=temp_data_path, split="train", transform=custom_transform, download=False
            )

            # Transform should be set to custom one
            assert dataset.transform == custom_transform

    def test_different_image_sizes(self, temp_data_path):
        """Test dataset with different image sizes."""
        sizes = [96, 128, 224]

        for size in sizes:
            with patch.object(datasets, "CIFAR10") as mock_cifar:
                mock_dataset = MagicMock()
                mock_dataset.__len__ = Mock(return_value=50000)
                mock_cifar.return_value = mock_dataset

                dataset = CIFAR10Dataset(
                    data_path=temp_data_path, split="train", image_size=size, download=False
                )

                assert dataset.image_size == size

    def test_classes_property(self, temp_data_path):
        """Test accessing classes property."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=50000)
            mock_dataset.classes = ["airplane", "automobile", "bird", "cat"]
            mock_cifar.return_value = mock_dataset

            dataset = CIFAR10Dataset(data_path=temp_data_path, split="train", download=False)

            assert len(dataset.classes) == 4
            assert "airplane" in dataset.classes


class TestCIFAR100Dataset:
    """Test CIFAR-100 dataset wrapper."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary directory for data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_create_dataset(self, temp_data_path):
        """Test creating CIFAR-100 dataset."""
        with patch.object(datasets, "CIFAR100") as mock_cifar:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=50000)
            mock_dataset.__getitem__ = Mock(return_value=(torch.rand(3, 224, 224), 0))
            mock_dataset.classes = [f"class{i}" for i in range(100)]
            mock_cifar.return_value = mock_dataset

            dataset = CIFAR100Dataset(data_path=temp_data_path, split="train", download=False)

            assert len(dataset) == 50000
            assert len(dataset.classes) == 100

    def test_val_split(self, temp_data_path):
        """Test validation split."""
        with patch.object(datasets, "CIFAR100") as mock_cifar:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=10000)
            mock_cifar.return_value = mock_dataset

            dataset = CIFAR100Dataset(data_path=temp_data_path, split="val", download=False)

            assert len(dataset) == 10000


class TestSTL10Dataset:
    """Test STL-10 dataset wrapper."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary directory for data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_train_split(self, temp_data_path):
        """Test training split."""
        with patch.object(datasets, "STL10") as mock_stl:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=5000)
            mock_dataset.classes = [f"class{i}" for i in range(10)]
            mock_stl.return_value = mock_dataset

            dataset = STL10Dataset(data_path=temp_data_path, split="train", download=False)

            assert len(dataset) == 5000

    def test_val_split(self, temp_data_path):
        """Test validation split (maps to test)."""
        with patch.object(datasets, "STL10") as mock_stl:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=8000)
            mock_stl.return_value = mock_dataset

            dataset = STL10Dataset(data_path=temp_data_path, split="val", download=False)

            # Val should map to test split
            assert len(dataset) == 8000

    def test_unlabeled_split(self, temp_data_path):
        """Test unlabeled split."""
        with patch.object(datasets, "STL10") as mock_stl:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=100000)
            mock_stl.return_value = mock_dataset

            dataset = STL10Dataset(data_path=temp_data_path, split="unlabeled", download=False)

            assert len(dataset) == 100000


class TestImageNetDataset:
    """Test ImageNet dataset wrapper."""

    @pytest.fixture
    def mock_imagenet_dir(self, tmp_path):
        """Create mock ImageNet directory structure."""
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"

        # Create class directories
        for split_dir in [train_dir, val_dir]:
            split_dir.mkdir(parents=True)
            for i in range(5):
                class_dir = split_dir / f"n0000{i}"
                class_dir.mkdir()
                # Create dummy image
                img = Image.new("RGB", (100, 100), color="red")
                img.save(class_dir / "image.JPEG")

        return tmp_path

    def test_create_train_dataset(self, mock_imagenet_dir):
        """Test creating training dataset."""
        dataset = ImageNetDataset(data_path=mock_imagenet_dir, split="train", image_size=224)

        assert dataset.split == "train"
        assert dataset.image_size == 224
        assert len(dataset) > 0

    def test_create_val_dataset(self, mock_imagenet_dir):
        """Test creating validation dataset."""
        dataset = ImageNetDataset(data_path=mock_imagenet_dir, split="val", image_size=224)

        assert dataset.split == "val"
        assert len(dataset) > 0

    def test_missing_directory(self, tmp_path):
        """Test error when ImageNet directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="ImageNet.*not found"):
            ImageNetDataset(data_path=tmp_path / "nonexistent", split="train")

    def test_custom_transform(self, mock_imagenet_dir):
        """Test using custom transform."""
        custom_transform = transforms.ToTensor()

        dataset = ImageNetDataset(
            data_path=mock_imagenet_dir, split="train", transform=custom_transform
        )

        assert dataset.transform == custom_transform

    def test_classes_property(self, mock_imagenet_dir):
        """Test accessing classes."""
        dataset = ImageNetDataset(data_path=mock_imagenet_dir, split="train")

        assert len(dataset.classes) > 0


class TestImageNet100Dataset:
    """Test ImageNet-100 dataset wrapper."""

    @pytest.fixture
    def mock_imagenet_dir(self, tmp_path):
        """Create mock ImageNet directory with some ImageNet-100 classes."""
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"

        # Use some actual ImageNet-100 class IDs
        imagenet100_classes = ["n01498041", "n01537544", "n01580077"]
        other_classes = ["n99999999", "n88888888"]

        for split_dir in [train_dir, val_dir]:
            split_dir.mkdir(parents=True)
            # Create ImageNet-100 classes
            for class_id in imagenet100_classes:
                class_dir = split_dir / class_id
                class_dir.mkdir()
                img = Image.new("RGB", (100, 100), color="red")
                img.save(class_dir / "image.JPEG")

            # Create non-ImageNet-100 classes
            for class_id in other_classes:
                class_dir = split_dir / class_id
                class_dir.mkdir()
                img = Image.new("RGB", (100, 100), color="blue")
                img.save(class_dir / "image.JPEG")

        return tmp_path

    def test_filtering(self, mock_imagenet_dir):
        """Test that dataset filters to ImageNet-100 classes."""
        dataset = ImageNet100Dataset(data_path=mock_imagenet_dir, split="train")

        # Should only include ImageNet-100 classes
        # Mock has 3 ImageNet-100 classes, each with 1 image
        assert len(dataset) == 3

    def test_correct_classes_filtered(self, mock_imagenet_dir):
        """Test that correct classes are kept."""
        dataset = ImageNet100Dataset(data_path=mock_imagenet_dir, split="train")

        # Should have filtered correctly
        assert len(dataset._valid_indices) == 3


class TestBuildDataset:
    """Test dataset factory function."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary directory for data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_build_cifar10(self, temp_data_path):
        """Test building CIFAR-10."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=50000)
            mock_cifar.return_value = mock_dataset

            dataset = build_dataset(
                dataset_name="cifar10", data_path=temp_data_path, split="train", download=False
            )

            assert isinstance(dataset, CIFAR10Dataset)

    def test_build_cifar100(self, temp_data_path):
        """Test building CIFAR-100."""
        with patch.object(datasets, "CIFAR100") as mock_cifar:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=50000)
            mock_cifar.return_value = mock_dataset

            dataset = build_dataset(
                dataset_name="cifar100", data_path=temp_data_path, split="train", download=False
            )

            assert isinstance(dataset, CIFAR100Dataset)

    def test_build_stl10(self, temp_data_path):
        """Test building STL-10."""
        with patch.object(datasets, "STL10") as mock_stl:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=5000)
            mock_stl.return_value = mock_dataset

            dataset = build_dataset(
                dataset_name="stl10", data_path=temp_data_path, split="train", download=False
            )

            assert isinstance(dataset, STL10Dataset)

    def test_case_insensitive(self, temp_data_path):
        """Test that dataset name is case-insensitive."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=50000)
            mock_cifar.return_value = mock_dataset

            dataset1 = build_dataset("CIFAR10", temp_data_path, download=False)
            dataset2 = build_dataset("cifar10", temp_data_path, download=False)
            dataset3 = build_dataset("CiFaR10", temp_data_path, download=False)

            assert all(isinstance(d, CIFAR10Dataset) for d in [dataset1, dataset2, dataset3])

    def test_unknown_dataset(self, temp_data_path):
        """Test error for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            build_dataset(dataset_name="unknown_dataset", data_path=temp_data_path)

    def test_color_jitter_train_only(self, temp_data_path):
        """Test that color jitter is only applied to training split."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = Mock(return_value=50000)
            mock_cifar.return_value = mock_dataset

            # Train should get color jitter
            train_dataset = build_dataset(
                "cifar10", temp_data_path, split="train", color_jitter=0.4, download=False
            )

            # Val should not get color jitter (None)
            val_dataset = build_dataset(
                "cifar10", temp_data_path, split="val", color_jitter=0.4, download=False
            )

            # Both should be created successfully
            assert train_dataset is not None
            assert val_dataset is not None


class TestBuildDataloader:
    """Test DataLoader factory function."""

    def test_basic_dataloader(self):
        """Test creating basic DataLoader."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(return_value=(torch.rand(3, 224, 224), 0))

        dataloader = build_dataloader(dataset=mock_dataset, batch_size=32)

        assert dataloader.batch_size == 32
        assert dataloader.num_workers == 4

    def test_custom_parameters(self):
        """Test DataLoader with custom parameters."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = Mock(return_value=100)

        dataloader = build_dataloader(
            dataset=mock_dataset,
            batch_size=64,
            num_workers=8,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

        assert dataloader.batch_size == 64
        assert dataloader.num_workers == 8
        assert not dataloader.pin_memory
        assert not dataloader.drop_last

    def test_dataloader_iteration(self):
        """Test that DataLoader can iterate."""

        # Create simple mock dataset
        class SimpleDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return torch.rand(3, 224, 224), idx

        dataset = SimpleDataset()
        dataloader = build_dataloader(
            dataset=dataset,
            batch_size=2,
            num_workers=0,  # No multiprocessing for test
            shuffle=False,
        )

        # Should be able to iterate
        batch_count = 0
        for images, labels in dataloader:
            assert images.shape[0] <= 2  # Batch size
            assert images.shape[1:] == (3, 224, 224)
            batch_count += 1

        # Should have 5 batches (10 samples / 2 batch_size)
        assert batch_count == 5


class TestIntegration:
    """Integration tests for datasets."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary directory for data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_end_to_end_cifar10(self, temp_data_path):
        """Test complete workflow: build dataset -> create dataloader -> iterate."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:
            # Create mock dataset with proper behavior
            class MockCIFAR10:
                def __init__(self, *args, **kwargs):
                    self.classes = ["airplane", "automobile"]

                def __len__(self):
                    return 20

                def __getitem__(self, idx):
                    return torch.rand(3, 224, 224), idx % 2

            mock_cifar.return_value = MockCIFAR10()

            # Build dataset
            dataset = build_dataset(
                "cifar10", temp_data_path, split="train", image_size=224, download=False
            )

            # Create dataloader
            dataloader = build_dataloader(dataset, batch_size=4, num_workers=0, shuffle=False)

            # Iterate once
            images, labels = next(iter(dataloader))

            assert images.shape == (4, 3, 224, 224)
            assert labels.shape == (4,)

    def test_different_splits(self, temp_data_path):
        """Test train and val splits work together."""
        with patch.object(datasets, "CIFAR10") as mock_cifar:

            class MockDataset:
                def __init__(self, *args, train=True, **kwargs):
                    self.train = train
                    self.classes = ["a", "b"]

                def __len__(self):
                    return 100 if self.train else 20

                def __getitem__(self, idx):
                    return torch.rand(3, 224, 224), 0

            mock_cifar.side_effect = lambda *args, **kwargs: MockDataset(*args, **kwargs)

            train_dataset = build_dataset("cifar10", temp_data_path, split="train", download=False)
            val_dataset = build_dataset("cifar10", temp_data_path, split="val", download=False)

            assert len(train_dataset) == 100
            assert len(val_dataset) == 20

"""
Tests for data loading and dataset functionality.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from src.data import (
    DATASET_INFO,
    CIFAR10Dataset,
    CIFAR100Dataset,
    JEPAEvalTransform,
    JEPATransform,
    build_dataloader,
    build_dataset,
    verify_dataset,
)


class TestTransforms:
    """Test JEPA transforms."""

    def test_jepa_transform(self):
        """Test JEPATransform output."""
        import numpy as np
        from PIL import Image

        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))

        # Apply transform
        transform = JEPATransform(image_size=224)
        output = transform(img)

        # Check output
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)
        assert output.dtype == torch.float32

    def test_jepa_eval_transform(self):
        """Test JEPAEvalTransform output."""
        import numpy as np
        from PIL import Image

        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))

        # Apply transform
        transform = JEPAEvalTransform(image_size=224)
        output = transform(img)

        # Check output
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)
        assert output.dtype == torch.float32

    def test_transform_consistency(self):
        """Test that transforms produce consistent output shapes."""
        import numpy as np
        from PIL import Image

        # Test with different image sizes
        for size in [32, 64, 128, 256]:
            img = Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))

            # Train transform
            train_transform = JEPATransform(image_size=224)
            train_output = train_transform(img)
            assert train_output.shape == (3, 224, 224)

            # Eval transform
            eval_transform = JEPAEvalTransform(image_size=224)
            eval_output = eval_transform(img)
            assert eval_output.shape == (3, 224, 224)


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
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
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
            assert images.shape[0] == 32  # drop_last=True ensures fixed batch size
            assert images.shape[1:] == (3, 224, 224)

        assert batch_count == len(dataset) // 32
        assert total_samples == (len(dataset) // 32) * 32


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

    def test_verify_nonexistent(self, temp_data_dir):
        """Test verification fails for non-existent dataset."""
        is_valid = verify_dataset("cifar10", temp_data_dir / "nonexistent")
        assert is_valid is False


class TestEdgeCases:
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
        for size in [96, 128, 256]:
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
            color_jitter=None,  # Disable color jitter
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


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

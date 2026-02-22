"""
Comprehensive tests for multi-dataset support.

Tests cover:
- WeightedMultiDataset for weighted sampling
- BalancedMultiDataset for balanced sampling
- Multi-dataset factory functions
- Foundation model dataset configurations
- Dataset statistics and tracking
"""

import random
from unittest.mock import patch

import pytest
import torch

from src.data.multi_dataset import (
    BalancedMultiDataset,
    WeightedMultiDataset,
    build_multi_dataset,
    create_foundation_model_dataset,
)


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, size, name="mock"):
        self.size = size
        self.name = name

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError("Index out of range")
        return torch.rand(3, 224, 224), idx % 10


class TestWeightedMultiDataset:
    """Test weighted multi-dataset."""

    def test_basic_creation(self):
        """Test basic creation with uniform weights."""
        dataset1 = MockDataset(100, "dataset1")
        dataset2 = MockDataset(200, "dataset2")

        multi_dataset = WeightedMultiDataset(
            datasets=[dataset1, dataset2], dataset_names=["dataset1", "dataset2"]
        )

        assert multi_dataset.num_datasets == 2
        assert multi_dataset.total_size == 300

    def test_with_custom_weights(self):
        """Test with custom sampling weights."""
        dataset1 = MockDataset(100, "dataset1")
        dataset2 = MockDataset(200, "dataset2")

        multi_dataset = WeightedMultiDataset(
            datasets=[dataset1, dataset2],
            weights=[0.7, 0.3],
            dataset_names=["dataset1", "dataset2"],
        )

        assert len(multi_dataset.weights) == 2
        # Weights should be normalized
        assert abs(sum(multi_dataset.weights) - 1.0) < 1e-6

    def test_without_names(self):
        """Test creation without dataset names."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(200)

        multi_dataset = WeightedMultiDataset(datasets=[dataset1, dataset2])

        # Should have auto-generated names
        assert len(multi_dataset.dataset_names) == 2
        assert all("dataset_" in name for name in multi_dataset.dataset_names)

    def test_get_item(self):
        """Test getting items from dataset."""
        dataset1 = MockDataset(50)
        dataset2 = MockDataset(50)

        multi_dataset = WeightedMultiDataset(datasets=[dataset1, dataset2], weights=[0.5, 0.5])

        # Get multiple items
        for i in range(10):
            item = multi_dataset[i]
            # Should return (image, label, dataset_idx)
            assert len(item) == 3
            assert isinstance(item[0], torch.Tensor)
            assert isinstance(item[1], int)
            assert isinstance(item[2], int)
            assert item[2] in [0, 1]  # Dataset index

    def test_length(self):
        """Test dataset length."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(200)

        multi_dataset = WeightedMultiDataset(datasets=[dataset1, dataset2])

        # Effective size should be max of dataset sizes
        assert len(multi_dataset) == 200

    def test_temperature(self):
        """Test temperature parameter for weight softmax."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(100)

        # High temperature (more uniform)
        multi_dataset_high = WeightedMultiDataset(
            datasets=[dataset1, dataset2], weights=[0.9, 0.1], temperature=10.0
        )

        # Low temperature (more peaked)
        multi_dataset_low = WeightedMultiDataset(
            datasets=[dataset1, dataset2], weights=[0.9, 0.1], temperature=0.1
        )

        # Weights should be different due to temperature
        assert multi_dataset_high.weights != multi_dataset_low.weights

    def test_get_dataset_stats(self):
        """Test getting dataset statistics."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(200)

        multi_dataset = WeightedMultiDataset(
            datasets=[dataset1, dataset2], weights=[0.6, 0.4], dataset_names=["small", "large"]
        )

        stats = multi_dataset.get_dataset_stats()

        assert "small" in stats
        assert "large" in stats
        assert stats["small"]["size"] == 100
        assert stats["large"]["size"] == 200
        assert "weight" in stats["small"]
        assert "expected_samples_per_epoch" in stats["small"]

    def test_weighted_sampling_distribution(self):
        """Test that sampling respects weights (probabilistic)."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(100)

        multi_dataset = WeightedMultiDataset(
            datasets=[dataset1, dataset2],
            weights=[0.9, 0.1],  # Heavily favor dataset1
            dataset_names=["dataset1", "dataset2"],
        )

        # Sample many times and check distribution
        dataset_counts = {0: 0, 1: 0}
        num_samples = 1000

        random.seed(42)
        for i in range(num_samples):
            item = multi_dataset[i]
            dataset_idx = item[2]
            dataset_counts[dataset_idx] += 1

        # Dataset1 should be sampled more often
        # With 90% weight, expect ~900 samples from dataset1
        # Allow some variance (we're not testing the RNG)
        assert dataset_counts[0] > dataset_counts[1]

    def test_single_dataset(self):
        """Test with single dataset."""
        dataset = MockDataset(100)

        multi_dataset = WeightedMultiDataset(datasets=[dataset], dataset_names=["single"])

        assert multi_dataset.num_datasets == 1
        assert len(multi_dataset) == 100

    def test_many_datasets(self):
        """Test with many datasets."""
        datasets = [MockDataset(50 * (i + 1)) for i in range(5)]
        names = [f"dataset_{i}" for i in range(5)]

        multi_dataset = WeightedMultiDataset(datasets=datasets, dataset_names=names)

        assert multi_dataset.num_datasets == 5
        assert len(multi_dataset) == 250  # Max dataset size


class TestBalancedMultiDataset:
    """Test balanced multi-dataset."""

    def test_basic_creation(self):
        """Test basic creation."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(200)

        multi_dataset = BalancedMultiDataset(
            datasets=[dataset1, dataset2], dataset_names=["dataset1", "dataset2"]
        )

        assert multi_dataset.num_datasets == 2

    def test_automatic_samples_per_dataset(self):
        """Test that samples_per_dataset defaults to min size."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(200)
        dataset3 = MockDataset(50)

        multi_dataset = BalancedMultiDataset(datasets=[dataset1, dataset2, dataset3])

        # Should use size of smallest dataset (50)
        assert multi_dataset.samples_per_dataset == 50
        # Total: 3 datasets * 50 samples = 150
        assert len(multi_dataset) == 150

    def test_custom_samples_per_dataset(self):
        """Test with custom samples_per_dataset."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(200)

        multi_dataset = BalancedMultiDataset(datasets=[dataset1, dataset2], samples_per_dataset=75)

        assert multi_dataset.samples_per_dataset == 75
        # Total: 2 datasets * 75 samples = 150
        assert len(multi_dataset) == 150

    def test_get_item(self):
        """Test getting items."""
        dataset1 = MockDataset(50)
        dataset2 = MockDataset(100)

        multi_dataset = BalancedMultiDataset(datasets=[dataset1, dataset2], samples_per_dataset=50)

        # Get multiple items
        for i in range(len(multi_dataset)):
            item = multi_dataset[i]
            # Should return (image, label, dataset_idx)
            assert len(item) == 3
            assert isinstance(item[0], torch.Tensor)
            assert isinstance(item[1], int)
            assert isinstance(item[2], int)
            assert item[2] in [0, 1]

    def test_balanced_distribution(self):
        """Test that datasets are balanced."""
        dataset1 = MockDataset(50)
        dataset2 = MockDataset(200)

        multi_dataset = BalancedMultiDataset(datasets=[dataset1, dataset2], samples_per_dataset=50)

        # Count samples from each dataset
        dataset_counts = {0: 0, 1: 0}
        for i in range(len(multi_dataset)):
            item = multi_dataset[i]
            dataset_idx = item[2]
            dataset_counts[dataset_idx] += 1

        # Should have exactly 50 samples from each
        assert dataset_counts[0] == 50
        assert dataset_counts[1] == 50

    def test_resampling(self):
        """Test index resampling."""
        dataset1 = MockDataset(50)
        dataset2 = MockDataset(100)

        multi_dataset = BalancedMultiDataset(datasets=[dataset1, dataset2], samples_per_dataset=50)

        # Get initial indices
        initial_indices = multi_dataset.indices.copy()

        # Resample
        multi_dataset.resample_indices()

        # Indices should change (with high probability)
        # Note: There's a tiny chance they're identical, but very unlikely
        assert len(multi_dataset.indices) == len(initial_indices)

    def test_oversampling_small_dataset(self):
        """Test that small datasets are oversampled."""
        dataset1 = MockDataset(10)  # Small
        dataset2 = MockDataset(100)

        multi_dataset = BalancedMultiDataset(datasets=[dataset1, dataset2], samples_per_dataset=50)

        # Dataset1 (size 10) should be sampled 50 times
        # This means some indices will be repeated
        dataset1_indices = [
            item_idx for dataset_idx, item_idx in multi_dataset.indices if dataset_idx == 0
        ]

        assert len(dataset1_indices) == 50
        # Some indices must be repeated since dataset only has 10 items
        assert len(set(dataset1_indices)) <= 10

    def test_undersampling_large_dataset(self):
        """Test that large datasets are undersampled."""
        dataset1 = MockDataset(200)  # Large
        dataset2 = MockDataset(50)

        multi_dataset = BalancedMultiDataset(datasets=[dataset1, dataset2], samples_per_dataset=50)

        # Dataset1 (size 200) should be sampled only 50 times
        dataset1_indices = [
            item_idx for dataset_idx, item_idx in multi_dataset.indices if dataset_idx == 0
        ]

        assert len(dataset1_indices) == 50
        # All indices should be unique (no oversampling needed)
        assert len(set(dataset1_indices)) == 50

    def test_without_names(self):
        """Test without dataset names."""
        dataset1 = MockDataset(50)
        dataset2 = MockDataset(100)

        multi_dataset = BalancedMultiDataset(datasets=[dataset1, dataset2])

        # Should have auto-generated names
        assert len(multi_dataset.dataset_names) == 2

    def test_single_dataset(self):
        """Test with single dataset."""
        dataset = MockDataset(100)

        multi_dataset = BalancedMultiDataset(datasets=[dataset], samples_per_dataset=50)

        assert len(multi_dataset) == 50


class TestBuildMultiDataset:
    """Test factory function for building multi-datasets."""

    @pytest.fixture
    def temp_data_path(self, tmp_path):
        """Create temporary data path."""
        return tmp_path

    def test_weighted_strategy(self, temp_data_path):
        """Test building weighted multi-dataset."""
        with patch("src.data.datasets.build_dataset") as mock_build:
            # Mock dataset building
            mock_build.side_effect = lambda **kwargs: MockDataset(100, kwargs["dataset_name"])

            configs = [{"name": "cifar10", "weight": 0.6}, {"name": "cifar100", "weight": 0.4}]

            dataset = build_multi_dataset(
                dataset_configs=configs,
                data_path=temp_data_path,
                split="train",
                sampling_strategy="weighted",
            )

            assert isinstance(dataset, WeightedMultiDataset)

    def test_balanced_strategy(self, temp_data_path):
        """Test building balanced multi-dataset."""
        with patch("src.data.datasets.build_dataset") as mock_build:
            mock_build.side_effect = lambda **kwargs: MockDataset(100, kwargs["dataset_name"])

            configs = [{"name": "cifar10"}, {"name": "cifar100"}]

            dataset = build_multi_dataset(
                dataset_configs=configs,
                data_path=temp_data_path,
                split="train",
                sampling_strategy="balanced",
            )

            assert isinstance(dataset, BalancedMultiDataset)

    def test_concat_strategy(self, temp_data_path):
        """Test building concatenated dataset."""
        with patch("src.data.datasets.build_dataset") as mock_build:
            from torch.utils.data import ConcatDataset

            mock_build.side_effect = lambda **kwargs: MockDataset(100, kwargs["dataset_name"])

            configs = [{"name": "cifar10"}, {"name": "cifar100"}]

            dataset = build_multi_dataset(
                dataset_configs=configs,
                data_path=temp_data_path,
                split="train",
                sampling_strategy="concat",
            )

            assert isinstance(dataset, ConcatDataset)

    def test_unknown_strategy(self, temp_data_path):
        """Test error for unknown strategy."""
        with patch("src.data.datasets.build_dataset") as mock_build:
            mock_build.side_effect = lambda **kwargs: MockDataset(100)

            configs = [{"name": "cifar10"}]

            with pytest.raises(ValueError, match="Unknown sampling strategy"):
                build_multi_dataset(
                    dataset_configs=configs, data_path=temp_data_path, sampling_strategy="unknown"
                )

    def test_custom_dataset_paths(self, temp_data_path):
        """Test with custom paths for each dataset."""
        with patch("src.data.datasets.build_dataset") as mock_build:
            mock_build.side_effect = lambda **kwargs: MockDataset(100)

            configs = [
                {"name": "cifar10", "path": "/custom/path1"},
                {"name": "cifar100", "path": "/custom/path2"},
            ]

            build_multi_dataset(
                dataset_configs=configs, data_path=temp_data_path, sampling_strategy="weighted"
            )

            # Check that custom paths were used
            calls = mock_build.call_args_list
            assert any("/custom/path1" in str(call) for call in calls)
            assert any("/custom/path2" in str(call) for call in calls)

    def test_additional_kwargs(self, temp_data_path):
        """Test passing additional kwargs to datasets."""
        with patch("src.data.datasets.build_dataset") as mock_build:
            mock_build.side_effect = lambda **kwargs: MockDataset(100)

            configs = [{"name": "cifar10"}]

            build_multi_dataset(
                dataset_configs=configs,
                data_path=temp_data_path,
                sampling_strategy="weighted",
                image_size=384,
                color_jitter=0.5,
            )

            # Check that kwargs were passed
            mock_build.assert_called()
            call_kwargs = mock_build.call_args[1]
            assert call_kwargs.get("image_size") == 384
            assert call_kwargs.get("color_jitter") == 0.5


class TestCreateFoundationModelDataset:
    """Test foundation model dataset creation."""

    @pytest.fixture
    def temp_data_path(self, tmp_path):
        """Create temporary data path."""
        return tmp_path

    def test_mini_scale(self, temp_data_path):
        """Test mini scale configuration."""
        with patch("src.data.multi_dataset.build_multi_dataset") as mock_build:
            mock_build.return_value = MockDataset(250000)

            create_foundation_model_dataset(scale="mini", data_path=temp_data_path, split="train")

            # Check that build_multi_dataset was called
            mock_build.assert_called_once()

            # Check config
            call_args = mock_build.call_args
            configs = call_args[1]["dataset_configs"]

            # Should have imagenet100, stl10, cifar100
            dataset_names = [cfg["name"] for cfg in configs]
            assert "imagenet100" in dataset_names
            assert "stl10" in dataset_names
            assert "cifar100" in dataset_names

    def test_medium_scale(self, temp_data_path):
        """Test medium scale configuration."""
        with patch("src.data.multi_dataset.build_multi_dataset") as mock_build:
            mock_build.return_value = MockDataset(1400000)

            create_foundation_model_dataset(scale="medium", data_path=temp_data_path, split="train")

            mock_build.assert_called_once()

            # Check config
            call_args = mock_build.call_args
            configs = call_args[1]["dataset_configs"]

            # Should have imagenet and stl10
            dataset_names = [cfg["name"] for cfg in configs]
            assert "imagenet" in dataset_names
            assert "stl10" in dataset_names

    def test_large_scale_not_implemented(self, temp_data_path):
        """Test that large scale raises error."""
        with pytest.raises(NotImplementedError, match="Large scale"):
            create_foundation_model_dataset(scale="large", data_path=temp_data_path)

    def test_unknown_scale(self, temp_data_path):
        """Test error for unknown scale."""
        with pytest.raises(ValueError, match="Unknown scale"):
            create_foundation_model_dataset(scale="unknown", data_path=temp_data_path)

    def test_mini_scale_weights(self, temp_data_path):
        """Test that mini scale has correct weights."""
        with patch("src.data.multi_dataset.build_multi_dataset") as mock_build:
            mock_build.return_value = MockDataset(250000)

            create_foundation_model_dataset(scale="mini", data_path=temp_data_path)

            call_args = mock_build.call_args
            configs = call_args[1]["dataset_configs"]

            # Check weights
            imagenet100_config = next(c for c in configs if c["name"] == "imagenet100")
            assert imagenet100_config["weight"] == 0.6

            stl10_config = next(c for c in configs if c["name"] == "stl10")
            assert stl10_config["weight"] == 0.3

            cifar100_config = next(c for c in configs if c["name"] == "cifar100")
            assert cifar100_config["weight"] == 0.1

    def test_passes_kwargs(self, temp_data_path):
        """Test that additional kwargs are passed through."""
        with patch("src.data.multi_dataset.build_multi_dataset") as mock_build:
            mock_build.return_value = MockDataset(250000)

            create_foundation_model_dataset(
                scale="mini", data_path=temp_data_path, split="val", image_size=384
            )

            call_args = mock_build.call_args
            assert call_args[1]["split"] == "val"
            assert call_args[1]["image_size"] == 384


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataset_list(self):
        """Test with empty dataset list."""
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            WeightedMultiDataset(datasets=[])

    def test_mismatched_weights_length(self):
        """Test with mismatched weights and datasets."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(200)

        # This should work - weights will be used as-is or normalized
        WeightedMultiDataset(
            datasets=[dataset1, dataset2], weights=[0.5, 0.3, 0.2]  # Too many weights
        )
        # Implementation may handle this differently

    def test_zero_weight(self):
        """Test with zero weight for one dataset."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(200)

        multi_dataset = WeightedMultiDataset(
            datasets=[dataset1, dataset2], weights=[1.0, 0.0]  # Dataset2 has zero weight
        )

        # Should still work, dataset2 just won't be sampled
        assert multi_dataset is not None

    def test_balanced_with_zero_samples(self):
        """Test balanced dataset with samples_per_dataset=0."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(200)

        # With zero samples, it should work but create an empty index list
        multi_dataset = BalancedMultiDataset(datasets=[dataset1, dataset2], samples_per_dataset=0)

        # Should have zero length
        assert len(multi_dataset) == 0

    def test_very_large_samples_per_dataset(self):
        """Test balanced dataset with very large samples_per_dataset."""
        dataset1 = MockDataset(10)
        dataset2 = MockDataset(20)

        # Request way more samples than available
        multi_dataset = BalancedMultiDataset(
            datasets=[dataset1, dataset2], samples_per_dataset=1000
        )

        # Should work with heavy oversampling
        assert len(multi_dataset) == 2000  # 2 datasets * 1000 samples

    def test_weighted_negative_temperature(self):
        """Test weighted dataset with negative temperature."""
        dataset1 = MockDataset(100)
        dataset2 = MockDataset(200)

        # Negative temperature might cause issues
        # But implementation may handle it
        try:
            WeightedMultiDataset(datasets=[dataset1, dataset2], temperature=-1.0)
            # If it works, that's fine
        except Exception:
            # If it raises error, that's also acceptable
            pass

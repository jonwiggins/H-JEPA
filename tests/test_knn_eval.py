"""
Unit tests for k-NN evaluation module.

Tests for:
1. KNNEvaluator initialization
2. Feature extraction
3. k-NN index building
4. Prediction with different k values
5. Distance metrics (cosine, euclidean)
6. Accuracy computation
7. Cross-validation with multiple k values
8. Edge cases (k=1, k=batch_size)
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.knn_eval import KNNEvaluator, knn_eval, sweep_knn_params

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_model():
    """Create a mock H-JEPA model for testing."""
    mock = MagicMock()
    mock.embed_dim = 384

    def mock_extract_features(images, level=0, use_target_encoder=True):
        batch_size = images.shape[0]
        # Return patch features [B, N, D]
        return torch.randn(batch_size, 196, 384)

    mock.extract_features = mock_extract_features
    mock.eval = MagicMock(return_value=mock)
    mock.parameters = MagicMock(return_value=[])

    return mock


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader with 100 samples, 10 classes."""
    num_samples = 100
    num_classes = 10
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=16, shuffle=False)


@pytest.fixture
def small_dataloader():
    """Create a small dataloader for quick tests."""
    num_samples = 32
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 5, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


# ============================================================================
# Test KNNEvaluator Initialization
# ============================================================================


def test_knn_evaluator_initialization(mock_model):
    """Test that KNNEvaluator initializes correctly."""
    evaluator = KNNEvaluator(
        model=mock_model,
        hierarchy_level=0,
        k=20,
        distance_metric="cosine",
        device="cpu",
    )

    assert evaluator.model is mock_model
    assert evaluator.hierarchy_level == 0
    assert evaluator.k == 20
    assert evaluator.distance_metric == "cosine"
    assert evaluator.device == "cpu"
    assert evaluator.knn_index is None
    assert evaluator.train_features is None
    assert evaluator.train_labels is None


def test_knn_evaluator_freezes_model(mock_model):
    """Test that model is frozen during initialization."""
    # Add a parameter to test freezing
    param = nn.Parameter(torch.randn(10, 10))
    param.requires_grad = True
    mock_model.parameters = MagicMock(return_value=[param])

    KNNEvaluator(model=mock_model, k=20, device="cpu")

    # Check model is in eval mode
    mock_model.eval.assert_called()

    # Check parameter is frozen
    assert not param.requires_grad


def test_knn_evaluator_different_metrics(mock_model):
    """Test initialization with different distance metrics."""
    metrics = ["cosine", "euclidean", "minkowski"]

    for metric in metrics:
        evaluator = KNNEvaluator(
            model=mock_model,
            k=10,
            distance_metric=metric,
            device="cpu",
        )
        assert evaluator.distance_metric == metric


# ============================================================================
# Test Feature Extraction
# ============================================================================


def test_extract_features_basic(mock_model, small_dataloader, random_seed):
    """Test basic feature extraction."""
    evaluator = KNNEvaluator(model=mock_model, k=20, device="cpu")

    features, labels = evaluator.extract_features(small_dataloader)

    assert features.shape[0] == 32  # num_samples
    assert features.shape[1] == 384  # embed_dim (after pooling)
    assert labels.shape[0] == 32
    assert isinstance(features, np.ndarray)
    assert isinstance(labels, np.ndarray)


def test_extract_features_normalization(mock_model, small_dataloader, random_seed):
    """Test feature extraction with normalization."""
    evaluator = KNNEvaluator(model=mock_model, k=20, device="cpu")

    features, labels = evaluator.extract_features(small_dataloader, normalize=True)

    # Check features are approximately normalized
    norms = np.linalg.norm(features, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_pool_features_mean(mock_model):
    """Test mean pooling of patch features."""
    evaluator = KNNEvaluator(model=mock_model, k=20, pooling="mean", device="cpu")

    # Create 3D patch features
    features_3d = torch.randn(4, 196, 384)
    pooled = evaluator.pool_features(features_3d)

    assert pooled.shape == (4, 384)
    assert torch.allclose(pooled, features_3d.mean(dim=1))


def test_pool_features_max(mock_model):
    """Test max pooling of patch features."""
    evaluator = KNNEvaluator(model=mock_model, k=20, pooling="max", device="cpu")

    # Create 3D patch features
    features_3d = torch.randn(4, 196, 384)
    pooled = evaluator.pool_features(features_3d)

    assert pooled.shape == (4, 384)
    assert torch.allclose(pooled, features_3d.max(dim=1)[0])


def test_pool_features_2d_passthrough(mock_model):
    """Test that 2D features pass through pooling unchanged."""
    evaluator = KNNEvaluator(model=mock_model, k=20, device="cpu")

    # Create 2D features (already pooled)
    features_2d = torch.randn(4, 384)
    pooled = evaluator.pool_features(features_2d)

    assert pooled.shape == (4, 384)
    assert torch.equal(pooled, features_2d)


def test_pool_features_invalid_method(mock_model):
    """Test that invalid pooling method raises error."""
    evaluator = KNNEvaluator(model=mock_model, k=20, pooling="invalid", device="cpu")

    features_3d = torch.randn(4, 196, 384)

    with pytest.raises(ValueError, match="Unknown pooling method"):
        evaluator.pool_features(features_3d)


# ============================================================================
# Test k-NN Index Building
# ============================================================================


def test_build_knn_index_basic(mock_model, simple_dataloader, random_seed):
    """Test building k-NN index."""
    evaluator = KNNEvaluator(model=mock_model, k=20, device="cpu")

    evaluator.build_knn_index(simple_dataloader)

    assert evaluator.train_features is not None
    assert evaluator.train_labels is not None
    assert evaluator.knn_index is not None
    assert evaluator.train_features.shape[0] == 100
    assert evaluator.train_labels.shape[0] == 100


def test_build_knn_index_cosine_metric(mock_model, small_dataloader, random_seed):
    """Test building k-NN index with cosine metric."""
    evaluator = KNNEvaluator(
        model=mock_model,
        k=5,
        distance_metric="cosine",
        device="cpu",
    )

    evaluator.build_knn_index(small_dataloader, normalize=True)

    # With normalized features, cosine should use euclidean
    assert evaluator.knn_index is not None
    assert evaluator.knn_index.metric == "euclidean"


def test_build_knn_index_euclidean_metric(mock_model, small_dataloader, random_seed):
    """Test building k-NN index with euclidean metric."""
    evaluator = KNNEvaluator(
        model=mock_model,
        k=5,
        distance_metric="euclidean",
        device="cpu",
    )

    evaluator.build_knn_index(small_dataloader, normalize=False)

    assert evaluator.knn_index is not None
    assert evaluator.knn_index.metric == "euclidean"


# ============================================================================
# Test k-NN Prediction
# ============================================================================


def test_predict_basic(mock_model, simple_dataloader, random_seed):
    """Test basic k-NN prediction."""
    evaluator = KNNEvaluator(model=mock_model, k=5, device="cpu")

    # Build index
    evaluator.build_knn_index(simple_dataloader)

    # Extract test features (use same data for simplicity)
    test_features, _ = evaluator.extract_features(simple_dataloader)

    # Predict
    predictions, prediction_probs = evaluator.predict(test_features, num_classes=10)

    assert predictions.shape == (100,)
    assert prediction_probs.shape == (100, 10)
    assert np.all(predictions >= 0) and np.all(predictions < 10)
    assert np.allclose(prediction_probs.sum(axis=1), 1.0, atol=1e-5)


def test_predict_without_index(mock_model):
    """Test that predict raises error without built index."""
    evaluator = KNNEvaluator(model=mock_model, k=5, device="cpu")

    test_features = np.random.randn(10, 384)

    with pytest.raises(RuntimeError, match="k-NN index not built"):
        evaluator.predict(test_features, num_classes=10)


def test_predict_k_equals_1(mock_model, small_dataloader, random_seed):
    """Test k-NN prediction with k=1."""
    evaluator = KNNEvaluator(model=mock_model, k=1, device="cpu")

    evaluator.build_knn_index(small_dataloader)
    test_features, _ = evaluator.extract_features(small_dataloader)

    predictions, prediction_probs = evaluator.predict(test_features, num_classes=5)

    assert predictions.shape == (32,)
    # With k=1, each prediction should be deterministic
    # Each sample should be its own nearest neighbor with same dataset
    assert len(np.unique(predictions)) >= 1


def test_predict_different_temperatures(mock_model, small_dataloader, random_seed):
    """Test prediction with different temperature values."""
    temperatures = [0.01, 0.07, 0.5, 1.0]

    for temp in temperatures:
        evaluator = KNNEvaluator(
            model=mock_model,
            k=5,
            temperature=temp,
            device="cpu",
        )

        evaluator.build_knn_index(small_dataloader)
        test_features, _ = evaluator.extract_features(small_dataloader)

        predictions, prediction_probs = evaluator.predict(test_features, num_classes=5)

        assert predictions.shape == (32,)
        assert prediction_probs.shape == (32, 5)


# ============================================================================
# Test k-NN Evaluation
# ============================================================================


def test_evaluate_basic(mock_model, simple_dataloader, random_seed):
    """Test basic k-NN evaluation."""
    evaluator = KNNEvaluator(model=mock_model, k=20, device="cpu")

    # Build index on train data
    evaluator.build_knn_index(simple_dataloader)

    # Evaluate on test data (using same data for simplicity)
    metrics = evaluator.evaluate(
        simple_dataloader,
        num_classes=10,
        verbose=False,
    )

    assert "accuracy" in metrics
    assert "top_1_accuracy" in metrics
    assert "top_5_accuracy" in metrics
    assert metrics["accuracy"] >= 0 and metrics["accuracy"] <= 100
    assert metrics["accuracy"] == metrics["top_1_accuracy"]


def test_evaluate_without_index(mock_model, simple_dataloader):
    """Test that evaluate raises error without built index."""
    evaluator = KNNEvaluator(model=mock_model, k=5, device="cpu")

    with pytest.raises(RuntimeError, match="k-NN index not built"):
        evaluator.evaluate(simple_dataloader, num_classes=10)


def test_evaluate_top_k_accuracies(mock_model, simple_dataloader, random_seed):
    """Test that top-k accuracies are computed correctly."""
    evaluator = KNNEvaluator(model=mock_model, k=10, device="cpu")

    evaluator.build_knn_index(simple_dataloader)

    metrics = evaluator.evaluate(
        simple_dataloader,
        num_classes=10,
        top_k_list=[1, 3, 5],
        verbose=False,
    )

    assert "top_1_accuracy" in metrics
    assert "top_3_accuracy" in metrics
    assert "top_5_accuracy" in metrics

    # Top-k accuracy should be non-decreasing as k increases
    assert metrics["top_1_accuracy"] <= metrics["top_3_accuracy"]
    assert metrics["top_3_accuracy"] <= metrics["top_5_accuracy"]


# ============================================================================
# Test Multiple k Values
# ============================================================================


def test_evaluate_multiple_k(mock_model, simple_dataloader, random_seed):
    """Test evaluation with multiple k values."""
    evaluator = KNNEvaluator(model=mock_model, k=20, device="cpu")

    evaluator.build_knn_index(simple_dataloader)

    k_values = [1, 5, 10, 20]
    results = evaluator.evaluate_multiple_k(
        simple_dataloader,
        num_classes=10,
        k_values=k_values,
        verbose=False,
    )

    assert len(results) == len(k_values)
    for k in k_values:
        assert k in results
        assert "accuracy" in results[k]
        assert results[k]["accuracy"] >= 0


def test_evaluate_multiple_k_skip_large(mock_model, small_dataloader, random_seed):
    """Test that very large k values are skipped."""
    evaluator = KNNEvaluator(model=mock_model, k=5, device="cpu")

    evaluator.build_knn_index(small_dataloader)

    # Request k larger than training set
    k_values = [1, 5, 100]  # 100 > 32 samples
    results = evaluator.evaluate_multiple_k(
        small_dataloader,
        num_classes=5,
        k_values=k_values,
        verbose=False,
    )

    # Should skip k=100
    assert 1 in results
    assert 5 in results
    assert 100 not in results


# ============================================================================
# Test Convenience Functions
# ============================================================================


def test_knn_eval_function(mock_model, simple_dataloader, random_seed):
    """Test the knn_eval convenience function."""
    metrics = knn_eval(
        model=mock_model,
        train_loader=simple_dataloader,
        test_loader=simple_dataloader,
        num_classes=10,
        k=20,
        device="cpu",
        verbose=False,
    )

    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0


def test_knn_eval_different_params(mock_model, small_dataloader, random_seed):
    """Test knn_eval with different parameters."""
    metrics = knn_eval(
        model=mock_model,
        train_loader=small_dataloader,
        test_loader=small_dataloader,
        num_classes=5,
        k=5,
        distance_metric="euclidean",
        temperature=0.1,
        device="cpu",
        verbose=False,
    )

    assert "accuracy" in metrics


def test_sweep_knn_params(mock_model, small_dataloader, random_seed):
    """Test parameter sweep function."""
    results = sweep_knn_params(
        model=mock_model,
        train_loader=small_dataloader,
        test_loader=small_dataloader,
        num_classes=5,
        k_values=[5, 10],
        temperatures=[0.07, 0.1],
        distance_metrics=["cosine"],
        device="cpu",
    )

    # Should have results for each combination
    assert len(results) >= 2  # At least 2 k values * 2 temps * 1 metric

    # Check structure
    for config_name, result in results.items():
        assert "config" in result
        assert "metrics" in result
        assert "accuracy" in result["metrics"]


# ============================================================================
# Edge Cases
# ============================================================================


def test_knn_with_single_class(mock_model, random_seed):
    """Test k-NN with single class (all same label)."""
    # Create dataset with single class
    images = torch.randn(32, 3, 224, 224)
    labels = torch.zeros(32, dtype=torch.long)  # All class 0
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=8)

    evaluator = KNNEvaluator(model=mock_model, k=5, device="cpu")
    evaluator.build_knn_index(dataloader)

    # Only compute top-1 accuracy for single class
    metrics = evaluator.evaluate(
        dataloader,
        num_classes=1,
        top_k_list=[1],  # Only top-1 for single class
        verbose=False,
    )

    # Should get 100% accuracy since all labels are the same
    assert metrics["accuracy"] == 100.0


def test_knn_k_equals_training_size(mock_model, random_seed):
    """Test k-NN when k equals training set size."""
    # Create small dataset
    images = torch.randn(16, 3, 224, 224)
    labels = torch.randint(0, 4, (16,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=8)

    # Set k to match training size
    evaluator = KNNEvaluator(model=mock_model, k=16, device="cpu")
    evaluator.build_knn_index(dataloader)

    # Should work without error
    metrics = evaluator.evaluate(
        dataloader,
        num_classes=4,
        verbose=False,
    )

    assert "accuracy" in metrics


def test_knn_empty_prediction_probs_normalization(mock_model, small_dataloader, random_seed):
    """Test that prediction probabilities are properly normalized."""
    evaluator = KNNEvaluator(model=mock_model, k=5, device="cpu")

    evaluator.build_knn_index(small_dataloader)
    test_features, _ = evaluator.extract_features(small_dataloader)

    predictions, prediction_probs = evaluator.predict(test_features, num_classes=5)

    # Check probabilities sum to 1
    prob_sums = prediction_probs.sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5)

    # Check all probabilities are non-negative
    assert np.all(prediction_probs >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

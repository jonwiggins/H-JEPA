"""
Unit tests for linear probe evaluation module.

Tests for:
1. LinearProbe class initialization and methods
2. Feature pooling (mean, cls, max, attention)
3. LinearProbeEvaluator initialization
4. Feature extraction with frozen model
5. Training probe with different configurations
6. Evaluation metrics (accuracy, top-k, confusion matrix)
7. K-fold cross-validation
8. Convenience functions
9. Edge cases (single sample, empty data, different batch sizes)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.evaluation.linear_probe import LinearProbe, LinearProbeEvaluator, linear_probe_eval

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


@pytest.fixture
def tiny_dataloader():
    """Create a tiny dataloader for very quick tests."""
    num_samples = 16
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 4, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


# ============================================================================
# Test LinearProbe Initialization
# ============================================================================


def test_linear_probe_initialization():
    """Test that LinearProbe initializes correctly with default params."""
    probe = LinearProbe(input_dim=384, num_classes=10)

    assert isinstance(probe.classifier, nn.Linear)
    assert probe.classifier.in_features == 384
    assert probe.classifier.out_features == 10
    assert probe.pooling == "mean"
    assert probe.normalize is False


def test_linear_probe_different_pooling_methods():
    """Test initialization with different pooling methods."""
    pooling_methods = ["mean", "cls", "max", "attention"]

    for pooling in pooling_methods:
        probe = LinearProbe(
            input_dim=384,
            num_classes=10,
            pooling=pooling,
        )
        assert probe.pooling == pooling

        # Check attention pooling creates attention module
        if pooling == "attention":
            assert hasattr(probe, "attention")
            assert isinstance(probe.attention, nn.Sequential)


def test_linear_probe_invalid_pooling():
    """Test that invalid pooling method raises error."""
    with pytest.raises(ValueError, match="Invalid pooling method"):
        LinearProbe(
            input_dim=384,
            num_classes=10,
            pooling="invalid_pooling",
        )


def test_linear_probe_with_normalization():
    """Test LinearProbe with normalization enabled."""
    probe = LinearProbe(
        input_dim=384,
        num_classes=10,
        normalize=True,
    )

    assert probe.normalize is True


# ============================================================================
# Test LinearProbe Feature Pooling
# ============================================================================


def test_pool_features_mean_3d():
    """Test mean pooling of 3D patch features."""
    probe = LinearProbe(input_dim=384, num_classes=10, pooling="mean")

    # Create 3D features [B, N, D]
    features_3d = torch.randn(4, 196, 384)
    pooled = probe.pool_features(features_3d)

    assert pooled.shape == (4, 384)
    # Mean pooling should match manual computation
    expected = features_3d.mean(dim=1)
    assert torch.allclose(pooled, expected)


def test_pool_features_mean_2d():
    """Test that 2D features pass through mean pooling unchanged."""
    probe = LinearProbe(input_dim=384, num_classes=10, pooling="mean")

    # Create 2D features [B, D] (already pooled)
    features_2d = torch.randn(4, 384)
    pooled = probe.pool_features(features_2d)

    assert pooled.shape == (4, 384)
    assert torch.equal(pooled, features_2d)


def test_pool_features_max():
    """Test max pooling of patch features."""
    probe = LinearProbe(input_dim=384, num_classes=10, pooling="max")

    features_3d = torch.randn(4, 196, 384)
    pooled = probe.pool_features(features_3d)

    assert pooled.shape == (4, 384)
    expected = features_3d.max(dim=1)[0]
    assert torch.allclose(pooled, expected)


def test_pool_features_cls_fallback():
    """Test that CLS pooling falls back to mean (with warning)."""
    probe = LinearProbe(input_dim=384, num_classes=10, pooling="cls")

    features_3d = torch.randn(4, 196, 384)

    with pytest.warns(UserWarning, match="CLS pooling requested"):
        pooled = probe.pool_features(features_3d)

    # Should fall back to mean pooling
    assert pooled.shape == (4, 384)
    expected = features_3d.mean(dim=1)
    assert torch.allclose(pooled, expected)


def test_pool_features_attention():
    """Test attention pooling of patch features."""
    probe = LinearProbe(input_dim=384, num_classes=10, pooling="attention")

    features_3d = torch.randn(4, 196, 384)
    pooled = probe.pool_features(features_3d)

    assert pooled.shape == (4, 384)
    # Attention pooling should produce valid output
    assert not torch.isnan(pooled).any()
    assert not torch.isinf(pooled).any()


# ============================================================================
# Test LinearProbe Forward Pass
# ============================================================================


def test_linear_probe_forward_3d():
    """Test forward pass with 3D features."""
    probe = LinearProbe(input_dim=384, num_classes=10, pooling="mean")

    features_3d = torch.randn(4, 196, 384)
    logits = probe(features_3d)

    assert logits.shape == (4, 10)
    assert not torch.isnan(logits).any()


def test_linear_probe_forward_2d():
    """Test forward pass with 2D features."""
    probe = LinearProbe(input_dim=384, num_classes=10, pooling="mean")

    features_2d = torch.randn(4, 384)
    logits = probe(features_2d)

    assert logits.shape == (4, 10)


def test_linear_probe_forward_with_normalization():
    """Test forward pass with L2 normalization."""
    probe = LinearProbe(
        input_dim=384,
        num_classes=10,
        pooling="mean",
        normalize=True,
    )

    features = torch.randn(4, 196, 384)
    logits = probe(features)

    assert logits.shape == (4, 10)
    # Check that normalization was applied internally


def test_linear_probe_forward_single_sample():
    """Test forward pass with single sample."""
    probe = LinearProbe(input_dim=384, num_classes=10)

    features = torch.randn(1, 196, 384)
    logits = probe(features)

    assert logits.shape == (1, 10)


def test_linear_probe_forward_large_batch():
    """Test forward pass with large batch."""
    probe = LinearProbe(input_dim=384, num_classes=10)

    features = torch.randn(128, 196, 384)
    logits = probe(features)

    assert logits.shape == (128, 10)


# ============================================================================
# Test LinearProbeEvaluator Initialization
# ============================================================================


def test_evaluator_initialization(mock_model):
    """Test that LinearProbeEvaluator initializes correctly."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=10,
        input_dim=384,
        hierarchy_level=0,
        device="cpu",
    )

    assert evaluator.model is mock_model
    assert evaluator.device == "cpu"
    assert evaluator.hierarchy_level == 0
    assert isinstance(evaluator.probe, LinearProbe)


def test_evaluator_freezes_model(mock_model):
    """Test that model is frozen during initialization."""
    param = nn.Parameter(torch.randn(10, 10))
    param.requires_grad = True
    mock_model.parameters = MagicMock(return_value=[param])

    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=10,
        input_dim=384,
        device="cpu",
    )

    # Check model is in eval mode
    mock_model.eval.assert_called()

    # Check parameter is frozen
    assert not param.requires_grad


def test_evaluator_different_pooling(mock_model):
    """Test evaluator with different pooling methods."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=10,
        input_dim=384,
        pooling="max",
        device="cpu",
    )

    assert evaluator.probe.pooling == "max"


def test_evaluator_with_normalization(mock_model):
    """Test evaluator with normalization enabled."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=10,
        input_dim=384,
        normalize=True,
        device="cpu",
    )

    assert evaluator.probe.normalize is True


# ============================================================================
# Test Feature Extraction
# ============================================================================


def test_extract_features_basic(mock_model, small_dataloader, random_seed):
    """Test basic feature extraction."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=5,
        input_dim=384,
        device="cpu",
    )

    features, labels = evaluator.extract_features(small_dataloader)

    assert features.shape[0] == 32  # num_samples
    assert labels.shape[0] == 32
    assert isinstance(features, np.ndarray)
    assert isinstance(labels, np.ndarray)


def test_extract_features_different_hierarchy_levels(mock_model, tiny_dataloader, random_seed):
    """Test feature extraction at different hierarchy levels."""
    for level in [0, 1, 2]:
        evaluator = LinearProbeEvaluator(
            model=mock_model,
            num_classes=4,
            input_dim=384,
            hierarchy_level=level,
            device="cpu",
        )

        features, labels = evaluator.extract_features(tiny_dataloader)

        assert features.shape[0] == 16
        assert labels.shape[0] == 16


def test_extract_features_tuple_batch_format(mock_model, random_seed):
    """Test feature extraction with tuple batch format."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=5,
        input_dim=384,
        device="cpu",
    )

    # Create dataloader that returns tuples
    images = torch.randn(16, 3, 224, 224)
    labels = torch.randint(0, 5, (16,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4)

    features, extracted_labels = evaluator.extract_features(dataloader)

    assert features.shape[0] == 16
    assert extracted_labels.shape[0] == 16


# ============================================================================
# Test Training Probe
# ============================================================================


def test_train_probe_basic(mock_model, tiny_dataloader, random_seed):
    """Test basic probe training."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    history = evaluator.train_probe(
        train_loader=tiny_dataloader,
        epochs=2,
        lr=0.1,
        verbose=False,
    )

    assert "train_loss" in history
    assert "train_acc" in history
    assert len(history["train_loss"]) == 2
    assert len(history["train_acc"]) == 2


def test_train_probe_with_validation(mock_model, tiny_dataloader, random_seed):
    """Test probe training with validation."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    history = evaluator.train_probe(
        train_loader=tiny_dataloader,
        val_loader=tiny_dataloader,  # Use same for simplicity
        epochs=2,
        lr=0.1,
        verbose=False,
    )

    assert "val_loss" in history
    assert "val_acc" in history
    assert len(history["val_loss"]) == 2
    assert len(history["val_acc"]) == 2


def test_train_probe_different_optimizers_params(mock_model, tiny_dataloader, random_seed):
    """Test training with different optimizer parameters."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    history = evaluator.train_probe(
        train_loader=tiny_dataloader,
        epochs=2,
        lr=0.01,
        weight_decay=0.0001,
        momentum=0.95,
        verbose=False,
    )

    assert len(history["train_loss"]) == 2


def test_train_probe_cosine_scheduler(mock_model, tiny_dataloader, random_seed):
    """Test training with cosine annealing scheduler."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    history = evaluator.train_probe(
        train_loader=tiny_dataloader,
        epochs=3,
        lr=0.1,
        scheduler_type="cosine",
        verbose=False,
    )

    assert len(history["train_loss"]) == 3


def test_train_probe_step_scheduler(mock_model, tiny_dataloader, random_seed):
    """Test training with step scheduler."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    history = evaluator.train_probe(
        train_loader=tiny_dataloader,
        epochs=3,
        lr=0.1,
        scheduler_type="step",
        verbose=False,
    )

    assert len(history["train_loss"]) == 3


def test_train_probe_no_scheduler(mock_model, tiny_dataloader, random_seed):
    """Test training without scheduler."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    history = evaluator.train_probe(
        train_loader=tiny_dataloader,
        epochs=2,
        lr=0.1,
        scheduler_type=None,
        verbose=False,
    )

    assert len(history["train_loss"]) == 2


def test_train_probe_metrics_are_valid(mock_model, tiny_dataloader, random_seed):
    """Test that training produces valid metrics."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    history = evaluator.train_probe(
        train_loader=tiny_dataloader,
        epochs=2,
        lr=0.1,
        verbose=False,
    )

    # Check metrics are in valid ranges
    for loss in history["train_loss"]:
        assert loss >= 0
    for acc in history["train_acc"]:
        assert 0 <= acc <= 100


# ============================================================================
# Test Evaluation
# ============================================================================


def test_evaluate_basic(mock_model, tiny_dataloader, random_seed):
    """Test basic evaluation."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    # Train first
    evaluator.train_probe(
        train_loader=tiny_dataloader,
        epochs=1,
        lr=0.1,
        verbose=False,
    )

    # Evaluate
    metrics = evaluator.evaluate(
        dataloader=tiny_dataloader,
        verbose=False,
    )

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "top_5_accuracy" in metrics
    assert metrics["accuracy"] >= 0
    assert metrics["accuracy"] <= 100


def test_evaluate_with_confusion_matrix(mock_model, tiny_dataloader, random_seed):
    """Test evaluation with confusion matrix computation."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    evaluator.train_probe(
        train_loader=tiny_dataloader,
        epochs=1,
        lr=0.1,
        verbose=False,
    )

    metrics = evaluator.evaluate(
        dataloader=tiny_dataloader,
        compute_confusion=True,
        verbose=False,
    )

    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"].shape == (4, 4)


def test_evaluate_different_top_k(mock_model, tiny_dataloader, random_seed):
    """Test evaluation with different top-k values."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    evaluator.train_probe(
        train_loader=tiny_dataloader,
        epochs=1,
        lr=0.1,
        verbose=False,
    )

    metrics = evaluator.evaluate(
        dataloader=tiny_dataloader,
        top_k=3,
        verbose=False,
    )

    assert "top_3_accuracy" in metrics
    assert metrics["top_3_accuracy"] >= metrics["accuracy"]


def test_evaluate_untrained_probe(mock_model, tiny_dataloader, random_seed):
    """Test evaluating untrained probe (should still work)."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    # Don't train, just evaluate
    metrics = evaluator.evaluate(
        dataloader=tiny_dataloader,
        verbose=False,
    )

    # Should get low but valid accuracy
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0


# ============================================================================
# Test K-Fold Cross-Validation
# ============================================================================


def test_k_fold_cross_validation_basic(mock_model, random_seed):
    """Test basic k-fold cross-validation."""
    # Create a small dataset
    images = torch.randn(40, 3, 224, 224)
    labels = torch.randint(0, 4, (40,))
    dataset = TensorDataset(images, labels)

    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    results = evaluator.k_fold_cross_validation(
        dataset=dataset,
        k_folds=2,  # Use 2 folds for speed
        epochs=1,
        batch_size=8,
        lr=0.1,
        num_workers=0,
        verbose=False,
    )

    assert "fold_accuracies" in results
    assert "fold_losses" in results
    assert "mean_accuracy" in results
    assert "std_accuracy" in results
    assert len(results["fold_accuracies"]) == 2
    assert len(results["fold_losses"]) == 2


def test_k_fold_cross_validation_multiple_folds(mock_model, random_seed):
    """Test k-fold with multiple folds."""
    images = torch.randn(30, 3, 224, 224)
    labels = torch.randint(0, 3, (30,))
    dataset = TensorDataset(images, labels)

    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=3,
        input_dim=384,
        device="cpu",
    )

    results = evaluator.k_fold_cross_validation(
        dataset=dataset,
        k_folds=3,
        epochs=1,
        batch_size=5,
        lr=0.1,
        num_workers=0,
        verbose=False,
    )

    assert len(results["fold_accuracies"]) == 3


def test_k_fold_statistics_are_valid(mock_model, random_seed):
    """Test that k-fold statistics are computed correctly."""
    images = torch.randn(40, 3, 224, 224)
    labels = torch.randint(0, 4, (40,))
    dataset = TensorDataset(images, labels)

    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    results = evaluator.k_fold_cross_validation(
        dataset=dataset,
        k_folds=2,
        epochs=1,
        batch_size=8,
        lr=0.1,
        num_workers=0,
        verbose=False,
    )

    # Mean should be close to average of fold accuracies
    expected_mean = np.mean(results["fold_accuracies"])
    assert np.isclose(results["mean_accuracy"], expected_mean)

    # Std should be close to std of fold accuracies
    expected_std = np.std(results["fold_accuracies"])
    assert np.isclose(results["std_accuracy"], expected_std)


# ============================================================================
# Test Convenience Function
# ============================================================================


def test_linear_probe_eval_function(mock_model, tiny_dataloader, random_seed):
    """Test the linear_probe_eval convenience function."""
    metrics = linear_probe_eval(
        model=mock_model,
        train_loader=tiny_dataloader,
        val_loader=tiny_dataloader,
        num_classes=4,
        hierarchy_level=0,
        epochs=2,
        lr=0.1,
        device="cpu",
        verbose=False,
    )

    assert "accuracy" in metrics
    assert "loss" in metrics
    assert "confusion_matrix" in metrics


def test_linear_probe_eval_different_params(mock_model, tiny_dataloader, random_seed):
    """Test convenience function with different parameters."""
    metrics = linear_probe_eval(
        model=mock_model,
        train_loader=tiny_dataloader,
        val_loader=tiny_dataloader,
        num_classes=4,
        hierarchy_level=1,
        epochs=1,
        lr=0.01,
        device="cpu",
        verbose=False,
    )

    assert "accuracy" in metrics


# ============================================================================
# Edge Cases
# ============================================================================


def test_single_sample_batch(mock_model, random_seed):
    """Test with single sample batches."""
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 2, (4,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=1)

    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=2,
        input_dim=384,
        device="cpu",
    )

    history = evaluator.train_probe(
        train_loader=dataloader,
        epochs=1,
        lr=0.1,
        verbose=False,
    )

    assert len(history["train_loss"]) == 1


def test_binary_classification(mock_model, random_seed):
    """Test with binary classification (2 classes)."""
    images = torch.randn(16, 3, 224, 224)
    labels = torch.randint(0, 2, (16,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4)

    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=2,
        input_dim=384,
        device="cpu",
    )

    history = evaluator.train_probe(
        train_loader=dataloader,
        epochs=1,
        lr=0.1,
        verbose=False,
    )

    metrics = evaluator.evaluate(dataloader, verbose=False)

    assert "accuracy" in metrics


def test_many_classes(mock_model, random_seed):
    """Test with many classes (100 classes)."""
    images = torch.randn(200, 3, 224, 224)
    labels = torch.randint(0, 100, (200,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=20)

    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=100,
        input_dim=384,
        device="cpu",
    )

    # Should not crash
    history = evaluator.train_probe(
        train_loader=dataloader,
        epochs=1,
        lr=0.1,
        verbose=False,
    )

    assert len(history["train_loss"]) == 1


def test_unbalanced_classes(mock_model, random_seed):
    """Test with highly unbalanced classes."""
    images = torch.randn(32, 3, 224, 224)
    # Create unbalanced labels: mostly class 0
    labels = torch.zeros(32, dtype=torch.long)
    labels[0:2] = 1  # Only 2 samples of class 1
    labels[2:3] = 2  # Only 1 sample of class 2

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=8)

    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=3,
        input_dim=384,
        device="cpu",
    )

    # Should handle unbalanced data
    history = evaluator.train_probe(
        train_loader=dataloader,
        epochs=1,
        lr=0.1,
        verbose=False,
    )

    metrics = evaluator.evaluate(dataloader, verbose=False)
    assert "accuracy" in metrics


def test_different_input_dimensions(mock_model, tiny_dataloader, random_seed):
    """Test with different input feature dimensions."""
    for dim in [128, 256, 512, 768]:
        # Update mock model to return correct dimension
        original_extract = mock_model.extract_features

        def mock_extract_custom_dim(images, level=0, use_target_encoder=True):
            batch_size = images.shape[0]
            return torch.randn(batch_size, 196, dim)

        mock_model.extract_features = mock_extract_custom_dim

        evaluator = LinearProbeEvaluator(
            model=mock_model,
            num_classes=4,
            input_dim=dim,
            device="cpu",
        )

        history = evaluator.train_probe(
            train_loader=tiny_dataloader,
            epochs=1,
            lr=0.1,
            verbose=False,
        )

        assert len(history["train_loss"]) == 1

        # Restore original
        mock_model.extract_features = original_extract


def test_empty_validation_loader(mock_model, tiny_dataloader, random_seed):
    """Test training with None validation loader."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    history = evaluator.train_probe(
        train_loader=tiny_dataloader,
        val_loader=None,
        epochs=2,
        lr=0.1,
        verbose=False,
    )

    # Should have empty validation lists
    assert history["val_loss"] == []
    assert history["val_acc"] == []


def test_probe_weights_initialization():
    """Test that probe weights are initialized properly."""
    probe = LinearProbe(input_dim=384, num_classes=10)

    # Check that weights are not all zeros
    assert not torch.allclose(probe.classifier.weight, torch.zeros_like(probe.classifier.weight))

    # Check that bias is zero
    assert torch.allclose(probe.classifier.bias, torch.zeros_like(probe.classifier.bias))


def test_probe_reset_weights(mock_model, tiny_dataloader, random_seed):
    """Test that probe weights can be reset between folds."""
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=4,
        input_dim=384,
        device="cpu",
    )

    # Get initial weights
    initial_weights = evaluator.probe.classifier.weight.clone()

    # Train
    evaluator.train_probe(
        train_loader=tiny_dataloader,
        epochs=1,
        lr=0.1,
        verbose=False,
    )

    # Weights should have changed
    assert not torch.allclose(initial_weights, evaluator.probe.classifier.weight)

    # Reset weights
    evaluator.probe._init_weights()

    # Weights should be different from trained weights
    # (though not necessarily same as initial due to random init)
    assert not torch.allclose(
        evaluator.probe.classifier.weight,
        torch.zeros_like(evaluator.probe.classifier.weight),
    )


def test_top_k_accuracy_with_few_classes(mock_model, random_seed):
    """Test top-k accuracy when k > num_classes."""
    images = torch.randn(16, 3, 224, 224)
    labels = torch.randint(0, 3, (16,))  # Only 3 classes
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4)

    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=3,
        input_dim=384,
        device="cpu",
    )

    evaluator.train_probe(
        train_loader=dataloader,
        epochs=1,
        lr=0.1,
        verbose=False,
    )

    # Request top-5 but only have 3 classes
    metrics = evaluator.evaluate(
        dataloader,
        top_k=5,
        verbose=False,
    )

    # Should still work (sklearn handles this)
    assert "top_5_accuracy" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for validation pipeline fixes.

Tests for:
1. HJEPA.encode_context() method
2. LinearProbe dataloader tuple unpacking robustness
"""

from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.linear_probe import LinearProbe, LinearProbeEvaluator
from src.models.hjepa import HJEPA, create_hjepa

# ============================================================================
# Test encode_context() Method
# ============================================================================


@pytest.fixture
def small_hjepa_model():
    """Create a small H-JEPA model for testing."""
    model = create_hjepa(
        encoder_type="vit_small_patch16_224",
        img_size=224,
        embed_dim=384,
        predictor_depth=4,
        predictor_num_heads=6,
        num_hierarchies=3,
        use_flash_attention=False,  # Disable for CPU testing
    )
    model.eval()
    return model


def test_encode_context_exists(small_hjepa_model):
    """Test that encode_context() method exists."""
    assert hasattr(small_hjepa_model, "encode_context")
    assert callable(small_hjepa_model.encode_context)


def test_encode_context_without_mask(small_hjepa_model):
    """Test encode_context() without mask (full image encoding)."""
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    # Encode without mask
    features = small_hjepa_model.encode_context(images)

    # Verify output shape
    num_patches = small_hjepa_model.get_num_patches()
    expected_shape = (batch_size, num_patches + 1, small_hjepa_model.embed_dim)
    assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"


def test_encode_context_with_mask(small_hjepa_model):
    """Test encode_context() with mask (masked encoding)."""
    batch_size = 2
    num_patches = small_hjepa_model.get_num_patches()
    images = torch.randn(batch_size, 3, 224, 224)

    # Create mask (50% masked)
    mask = torch.zeros(batch_size, num_patches)
    mask[:, : num_patches // 2] = 1

    # Encode with mask
    features = small_hjepa_model.encode_context(images, mask=mask)

    # Verify output shape
    expected_shape = (batch_size, num_patches + 1, small_hjepa_model.embed_dim)
    assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"


def test_encode_context_with_none_mask(small_hjepa_model):
    """Test encode_context() with explicit None mask."""
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    # Encode with explicit None mask
    features = small_hjepa_model.encode_context(images, mask=None)

    # Verify output shape
    num_patches = small_hjepa_model.get_num_patches()
    expected_shape = (batch_size, num_patches + 1, small_hjepa_model.embed_dim)
    assert features.shape == expected_shape


def test_encode_context_no_grad(small_hjepa_model):
    """Test that encode_context() operates in no_grad mode."""
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    # Encode should not require gradients
    features = small_hjepa_model.encode_context(images)

    # Verify no gradient tracking
    assert not features.requires_grad


def test_encode_context_batch_consistency(small_hjepa_model):
    """Test that encode_context() produces consistent results."""
    images = torch.randn(1, 3, 224, 224)

    # Encode twice
    features1 = small_hjepa_model.encode_context(images)
    features2 = small_hjepa_model.encode_context(images)

    # Should be identical (no dropout in eval mode)
    assert torch.allclose(features1, features2, atol=1e-5)


def test_encode_context_different_batch_sizes(small_hjepa_model):
    """Test encode_context() with different batch sizes."""
    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        images = torch.randn(batch_size, 3, 224, 224)
        features = small_hjepa_model.encode_context(images)

        num_patches = small_hjepa_model.get_num_patches()
        expected_shape = (batch_size, num_patches + 1, small_hjepa_model.embed_dim)
        assert features.shape == expected_shape


def test_encode_context_vs_context_encoder(small_hjepa_model):
    """Test that encode_context() matches direct context_encoder call."""
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    # Encode using both methods
    features_via_method = small_hjepa_model.encode_context(images)
    features_direct = small_hjepa_model.context_encoder(images)

    # Should be identical
    assert torch.allclose(features_via_method, features_direct, atol=1e-5)


# ============================================================================
# Test Linear Probe Tuple Unpacking
# ============================================================================


def create_mock_model():
    """Create a mock model for linear probe testing."""
    mock_model = MagicMock()
    mock_model.embed_dim = 384

    def mock_extract_features(images, level=0, use_target_encoder=True):
        batch_size = images.shape[0]
        return torch.randn(batch_size, 196, 384)  # Fake features

    mock_model.extract_features = mock_extract_features
    return mock_model


def test_linear_probe_with_standard_dataloader():
    """Test linear probe with standard (images, labels) dataloader."""
    # Create mock model
    mock_model = create_mock_model()

    # Create linear probe evaluator
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=10,
        input_dim=384,
        hierarchy_level=0,
        device="cpu",
    )

    # Create standard dataloader (images, labels)
    images = torch.randn(32, 3, 224, 224)
    labels = torch.randint(0, 10, (32,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=8)

    # Test extract_features
    features, extracted_labels = evaluator.extract_features(dataloader, desc="Test")

    # Verify shapes
    assert features.shape[0] == 32
    assert extracted_labels.shape[0] == 32
    assert extracted_labels.shape == (32,)


def test_linear_probe_with_metadata_dataloader():
    """Test linear probe with (images, labels, metadata) dataloader."""
    # Create mock model
    mock_model = create_mock_model()

    # Create linear probe evaluator
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=10,
        input_dim=384,
        hierarchy_level=0,
        device="cpu",
    )

    # Create dataloader with metadata
    images = torch.randn(32, 3, 224, 224)
    labels = torch.randint(0, 10, (32,))
    metadata = torch.randint(0, 100, (32,))  # Extra metadata
    dataset = TensorDataset(images, labels, metadata)
    dataloader = DataLoader(dataset, batch_size=8)

    # Test extract_features - should handle 3-tuple gracefully
    features, extracted_labels = evaluator.extract_features(dataloader, desc="Test")

    # Verify shapes
    assert features.shape[0] == 32
    assert extracted_labels.shape[0] == 32


def test_linear_probe_train_with_standard_dataloader():
    """Test linear probe training with standard dataloader."""
    # Create mock model
    mock_model = create_mock_model()

    # Create linear probe evaluator
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=10,
        input_dim=384,
        hierarchy_level=0,
        device="cpu",
    )

    # Create standard dataloader
    images = torch.randn(32, 3, 224, 224)
    labels = torch.randint(0, 10, (32,))
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=8)

    # Test training
    history = evaluator.train_probe(
        train_loader=train_loader,
        val_loader=None,
        epochs=2,
        lr=0.1,
        verbose=False,
    )

    # Verify history structure
    assert "train_loss" in history
    assert "train_acc" in history
    assert len(history["train_loss"]) == 2
    assert len(history["train_acc"]) == 2


def test_linear_probe_train_with_metadata_dataloader():
    """Test linear probe training with metadata dataloader."""
    # Create mock model
    mock_model = create_mock_model()

    # Create linear probe evaluator
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=10,
        input_dim=384,
        hierarchy_level=0,
        device="cpu",
    )

    # Create dataloader with metadata
    images = torch.randn(32, 3, 224, 224)
    labels = torch.randint(0, 10, (32,))
    metadata = torch.randint(0, 100, (32,))
    dataset = TensorDataset(images, labels, metadata)
    train_loader = DataLoader(dataset, batch_size=8)

    # Test training - should handle 3-tuple gracefully
    history = evaluator.train_probe(
        train_loader=train_loader,
        val_loader=None,
        epochs=2,
        lr=0.1,
        verbose=False,
    )

    # Verify history structure
    assert "train_loss" in history
    assert "train_acc" in history
    assert len(history["train_loss"]) == 2


def test_linear_probe_evaluate_with_standard_dataloader():
    """Test linear probe evaluation with standard dataloader."""
    # Create mock model
    mock_model = create_mock_model()

    # Create linear probe evaluator
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=10,
        input_dim=384,
        hierarchy_level=0,
        device="cpu",
    )

    # Train briefly first
    images = torch.randn(32, 3, 224, 224)
    # Ensure all classes are represented
    labels = torch.cat([torch.arange(10), torch.randint(0, 10, (22,))])
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=8)
    evaluator.train_probe(train_loader, epochs=1, verbose=False)

    # Test evaluation
    eval_loader = DataLoader(dataset, batch_size=8)
    metrics = evaluator.evaluate(eval_loader, verbose=False)

    # Verify metrics structure
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "top_5_accuracy" in metrics


def test_linear_probe_evaluate_with_metadata_dataloader():
    """Test linear probe evaluation with metadata dataloader."""
    # Create mock model
    mock_model = create_mock_model()

    # Create linear probe evaluator
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=10,
        input_dim=384,
        hierarchy_level=0,
        device="cpu",
    )

    # Train briefly first
    images = torch.randn(32, 3, 224, 224)
    # Ensure all classes are represented
    labels = torch.cat([torch.arange(10), torch.randint(0, 10, (22,))])
    dataset_train = TensorDataset(images, labels)
    train_loader = DataLoader(dataset_train, batch_size=8)
    evaluator.train_probe(train_loader, epochs=1, verbose=False)

    # Test evaluation with metadata
    metadata = torch.randint(0, 100, (32,))
    dataset_eval = TensorDataset(images, labels, metadata)
    eval_loader = DataLoader(dataset_eval, batch_size=8)
    metrics = evaluator.evaluate(eval_loader, verbose=False)

    # Verify metrics structure
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_linear_probe_invalid_batch_type():
    """Test that invalid batch types raise appropriate errors."""
    # Create mock model
    mock_model = create_mock_model()

    # Create linear probe evaluator
    evaluator = LinearProbeEvaluator(
        model=mock_model,
        num_classes=10,
        input_dim=384,
        hierarchy_level=0,
        device="cpu",
    )

    # Create dataloader that returns single tensors (invalid)
    class InvalidDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return torch.randn(3, 224, 224)  # Returns single tensor

    invalid_loader = DataLoader(InvalidDataset(), batch_size=4)

    # Should raise ValueError
    with pytest.raises(ValueError, match="Unexpected batch type"):
        evaluator.extract_features(invalid_loader, desc="Test")


# ============================================================================
# Integration Tests
# ============================================================================


def test_encode_context_integration_with_linear_probe(small_hjepa_model):
    """Integration test: encode_context() with linear probe."""
    # Create dummy data
    images = torch.randn(16, 3, 224, 224)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=4)

    # Create linear probe evaluator
    evaluator = LinearProbeEvaluator(
        model=small_hjepa_model,
        num_classes=10,
        input_dim=small_hjepa_model.embed_dim,
        hierarchy_level=0,
        device="cpu",
    )

    # Test that encode_context works with linear probe
    features, extracted_labels = evaluator.extract_features(dataloader)

    # Verify extraction worked
    assert features.shape[0] == 16
    assert extracted_labels.shape[0] == 16


def test_full_validation_pipeline(small_hjepa_model):
    """Full integration test of validation pipeline."""
    # Create training and validation data
    train_images = torch.randn(32, 3, 224, 224)
    # Ensure all classes are represented
    train_labels = torch.cat([torch.arange(10), torch.randint(0, 10, (22,))])
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8)

    val_images = torch.randn(16, 3, 224, 224)
    # Ensure all classes are represented in validation too
    val_labels = torch.cat([torch.arange(10), torch.randint(0, 10, (6,))])
    val_dataset = TensorDataset(val_images, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Create linear probe evaluator
    evaluator = LinearProbeEvaluator(
        model=small_hjepa_model,
        num_classes=10,
        input_dim=small_hjepa_model.embed_dim,
        hierarchy_level=0,
        device="cpu",
    )

    # Train probe
    history = evaluator.train_probe(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        lr=0.1,
        verbose=False,
    )

    # Verify training worked
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2

    # Evaluate
    metrics = evaluator.evaluate(val_loader, verbose=False)

    # Verify evaluation worked
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

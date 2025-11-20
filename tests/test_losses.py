"""
Test suite for loss functions in H-JEPA.

This module tests the core loss functions:
- HJEPALoss
"""

import pytest
import torch
import torch.nn as nn

from src.losses.hjepa_loss import HJEPALoss


class TestHJEPALoss:
    """Test suite for HJEPALoss."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        self.batch_size = 4
        self.num_patches = 16
        self.embed_dim = 128
        self.num_hierarchies = 3

        # Create sample predictions and targets for 3 hierarchy levels
        self.predictions = [
            torch.randn(self.batch_size, self.num_patches, self.embed_dim)
            for _ in range(self.num_hierarchies)
        ]
        self.targets = [
            torch.randn(self.batch_size, self.num_patches, self.embed_dim)
            for _ in range(self.num_hierarchies)
        ]

        yield

    def test_initialization_default(self):
        """Test default initialization of HJEPALoss."""
        loss_fn = HJEPALoss()
        assert loss_fn.loss_type == "mse"
        assert loss_fn.num_hierarchies == 3
        assert loss_fn.reduction == "mean"
        assert loss_fn.normalize_embeddings == False
        assert len(loss_fn.hierarchy_weights) == 3
        assert all(w == 1.0 for w in loss_fn.hierarchy_weights)

    def test_initialization_custom(self):
        """Test custom initialization of HJEPALoss."""
        weights = [1.0, 0.8, 0.6]
        loss_fn = HJEPALoss(
            loss_type="smoothl1",
            hierarchy_weights=weights,
            num_hierarchies=3,
            normalize_embeddings=True,
        )
        assert loss_fn.loss_type == "smoothl1"
        assert loss_fn.normalize_embeddings == True
        assert loss_fn.hierarchy_weights == weights

    def test_forward_mse_loss(self, setup):
        """Test forward pass with MSE loss."""
        loss_fn = HJEPALoss(loss_type="mse")
        loss = loss_fn(self.predictions, self.targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar output
        assert loss.item() >= 0  # Loss should be non-negative
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_smoothl1_loss(self, setup):
        """Test forward pass with Smooth L1 loss."""
        loss_fn = HJEPALoss(loss_type="smoothl1")
        loss = loss_fn(self.predictions, self.targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_forward_huber_loss(self, setup):
        """Test forward pass with Huber loss."""
        loss_fn = HJEPALoss(loss_type="huber", huber_delta=2.0)
        loss = loss_fn(self.predictions, self.targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_hierarchy_weights(self, setup):
        """Test that hierarchy weights are applied correctly."""
        # Test with equal weights
        loss_fn_equal = HJEPALoss(hierarchy_weights=1.0)
        loss_equal = loss_fn_equal(self.predictions, self.targets)

        # Test with different weights
        loss_fn_weighted = HJEPALoss(hierarchy_weights=[2.0, 1.0, 0.5])
        loss_weighted = loss_fn_weighted(self.predictions, self.targets)

        # Losses should be different due to different weights
        assert not torch.allclose(loss_equal, loss_weighted)

    def test_normalization(self, setup):
        """Test embedding normalization."""
        loss_fn = HJEPALoss(normalize_embeddings=True)
        loss_normalized = loss_fn(self.predictions, self.targets)

        loss_fn_no_norm = HJEPALoss(normalize_embeddings=False)
        loss_no_norm = loss_fn_no_norm(self.predictions, self.targets)

        # Losses should generally be different
        assert loss_normalized.item() >= 0
        assert loss_no_norm.item() >= 0

    def test_reduction_modes(self, setup):
        """Test different reduction modes."""
        # Mean reduction
        loss_fn_mean = HJEPALoss(reduction="mean")
        loss_mean = loss_fn_mean(self.predictions, self.targets)
        assert loss_mean.dim() == 0

        # Sum reduction
        loss_fn_sum = HJEPALoss(reduction="sum")
        loss_sum = loss_fn_sum(self.predictions, self.targets)
        assert loss_sum.dim() == 0

        # Sum should be larger than mean for batch_size > 1
        assert loss_sum.item() > loss_mean.item()

    def test_backward_gradient_flow(self, setup):
        """Test that gradients flow correctly through the loss."""
        loss_fn = HJEPALoss()

        # Make predictions require gradients
        predictions = [p.requires_grad_(True) for p in self.predictions]

        loss = loss_fn(predictions, self.targets)
        loss.backward()

        # Check that gradients are computed
        for pred in predictions:
            assert pred.grad is not None
            assert not torch.isnan(pred.grad).any()
            assert not torch.isinf(pred.grad).any()

    def test_mismatched_hierarchies(self):
        """Test error handling for mismatched hierarchy levels."""
        loss_fn = HJEPALoss(num_hierarchies=3)

        # Predictions with wrong number of hierarchies
        predictions = [torch.randn(2, 10, 128) for _ in range(2)]  # Only 2 levels
        targets = [torch.randn(2, 10, 128) for _ in range(3)]  # 3 levels

        with pytest.raises((AssertionError, ValueError)):
            loss_fn(predictions, targets)

    def test_edge_cases(self, setup):
        """Test edge cases."""
        loss_fn = HJEPALoss()

        # Test with zero targets (perfect prediction)
        zero_targets = [torch.zeros_like(p) for p in self.predictions]
        zero_predictions = [torch.zeros_like(p) for p in self.predictions]
        loss = loss_fn(zero_predictions, zero_targets)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

        # Test with very large values
        large_predictions = [p * 1000 for p in self.predictions]
        loss = loss_fn(large_predictions, self.targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_with_different_patch_counts(self):
        """Test with different patch counts at different hierarchy levels."""
        batch_size = 4
        embed_dim = 128

        # Different number of patches per hierarchy level (simulating pooling)
        predictions = [
            torch.randn(batch_size, 196, embed_dim),  # Level 0: 14x14 patches
            torch.randn(batch_size, 49, embed_dim),  # Level 1: 7x7 patches
            torch.randn(batch_size, 16, embed_dim),  # Level 2: 4x4 patches
        ]
        targets = [
            torch.randn(batch_size, 196, embed_dim),
            torch.randn(batch_size, 49, embed_dim),
            torch.randn(batch_size, 16, embed_dim),
        ]

        loss_fn = HJEPALoss()
        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_loss_values_reasonable_range(self):
        """Test that loss values are in reasonable ranges."""
        batch_size = 4
        embed_dim = 128

        # Create normalized inputs (typical after model forward pass)
        predictions = [
            torch.nn.functional.normalize(torch.randn(batch_size, 16, embed_dim), dim=-1)
            for _ in range(3)
        ]
        targets = [
            torch.nn.functional.normalize(torch.randn(batch_size, 16, embed_dim), dim=-1)
            for _ in range(3)
        ]

        loss_fn = HJEPALoss(normalize_embeddings=True)
        loss = loss_fn(predictions, targets)

        # For normalized embeddings, MSE loss should be in [0, 4] range
        # (max distance between two unit vectors is 2)
        assert 0 <= loss.item() <= 10  # Allow some margin

    def test_deterministic_loss(self):
        """Test that loss computation is deterministic."""
        torch.manual_seed(42)

        predictions = [torch.randn(2, 10, 64) for _ in range(3)]
        targets = [torch.randn(2, 10, 64) for _ in range(3)]

        loss_fn = HJEPALoss()
        loss1 = loss_fn(predictions, targets)
        loss2 = loss_fn(predictions, targets)

        assert torch.allclose(loss1, loss2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Test suite for VICReg loss functions.

This module tests the VICReg (Variance-Invariance-Covariance Regularization)
loss implementation, which prevents representation collapse through three
complementary regularization terms.
"""

import pytest
import torch
import torch.nn as nn

from src.losses.vicreg import AdaptiveVICRegLoss, VICRegLoss


class TestVICRegLoss:
    """Test suite for VICRegLoss."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        self.batch_size = 32
        self.num_patches = 16
        self.embed_dim = 128

        # Create sample representations for two views
        self.z_a = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        self.z_b = torch.randn(self.batch_size, self.num_patches, self.embed_dim)

        yield

    def test_initialization_default(self):
        """Test default initialization of VICRegLoss."""
        loss_fn = VICRegLoss()
        assert loss_fn.invariance_weight == 25.0
        assert loss_fn.variance_weight == 25.0
        assert loss_fn.covariance_weight == 1.0
        assert loss_fn.variance_threshold == 1.0
        assert loss_fn.eps == 1e-4
        assert loss_fn.flatten_patches == True

    def test_initialization_custom(self):
        """Test custom initialization of VICRegLoss."""
        loss_fn = VICRegLoss(
            invariance_weight=10.0,
            variance_weight=15.0,
            covariance_weight=2.0,
            variance_threshold=0.5,
            eps=1e-6,
            flatten_patches=False,
        )
        assert loss_fn.invariance_weight == 10.0
        assert loss_fn.variance_weight == 15.0
        assert loss_fn.covariance_weight == 2.0
        assert loss_fn.variance_threshold == 0.5
        assert loss_fn.eps == 1e-6
        assert loss_fn.flatten_patches == False

    def test_initialization_invalid_weights(self):
        """Test that invalid weights raise assertions."""
        with pytest.raises(AssertionError, match="invariance_weight must be >= 0"):
            VICRegLoss(invariance_weight=-1.0)

        with pytest.raises(AssertionError, match="variance_weight must be >= 0"):
            VICRegLoss(variance_weight=-1.0)

        with pytest.raises(AssertionError, match="covariance_weight must be >= 0"):
            VICRegLoss(covariance_weight=-1.0)

        with pytest.raises(AssertionError, match="variance_threshold must be > 0"):
            VICRegLoss(variance_threshold=0.0)

    def test_invariance_loss_component(self, setup):
        """Test the invariance loss component (MSE between representations)."""
        loss_fn = VICRegLoss(
            invariance_weight=1.0,
            variance_weight=0.0,
            covariance_weight=0.0,
        )

        # Test with identical representations - should have zero invariance loss
        z_identical = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        loss_dict = loss_fn(z_identical, z_identical)

        assert "invariance_loss" in loss_dict
        assert torch.allclose(loss_dict["invariance_loss"], torch.tensor(0.0), atol=1e-6)

        # Test with different representations - should have non-zero invariance loss
        loss_dict = loss_fn(self.z_a, self.z_b)
        assert loss_dict["invariance_loss"].item() > 0

    def test_variance_loss_component(self, setup):
        """Test the variance loss component (prevents collapse)."""
        loss_fn = VICRegLoss(
            invariance_weight=0.0,
            variance_weight=1.0,
            covariance_weight=0.0,
            variance_threshold=1.0,
        )

        # Create representations with low variance - should have high variance loss
        z_low_var = torch.ones(self.batch_size, self.num_patches, self.embed_dim) * 0.01
        z_low_var += torch.randn_like(z_low_var) * 0.1  # Add small noise
        loss_dict = loss_fn(z_low_var, z_low_var)

        assert "variance_loss" in loss_dict
        assert loss_dict["variance_loss"].item() > 0

        # Create representations with high variance - should have low variance loss
        z_high_var = torch.randn(self.batch_size, self.num_patches, self.embed_dim) * 10
        loss_dict_high = loss_fn(z_high_var, z_high_var)

        # High variance representations should have lower variance loss
        assert loss_dict_high["variance_loss"].item() < loss_dict["variance_loss"].item()

    def test_covariance_loss_component(self, setup):
        """Test the covariance loss component (decorrelates dimensions)."""
        loss_fn = VICRegLoss(
            invariance_weight=0.0,
            variance_weight=0.0,
            covariance_weight=1.0,
        )

        # Create representations with correlated dimensions
        B, N, D = self.batch_size, self.num_patches, self.embed_dim
        z_base = torch.randn(B * N, 1)
        z_correlated = z_base.repeat(1, D)  # All dimensions are identical
        z_correlated = z_correlated.reshape(B, N, D)

        loss_dict_corr = loss_fn(z_correlated, z_correlated)

        # Correlated dimensions should have high covariance loss
        assert "covariance_loss" in loss_dict_corr
        assert loss_dict_corr["covariance_loss"].item() > 0

        # Create uncorrelated representations
        z_uncorr = torch.randn(B, N, D)
        loss_dict_uncorr = loss_fn(z_uncorr, z_uncorr)

        # Random uncorrelated dimensions should have lower covariance loss
        assert loss_dict_uncorr["covariance_loss"].item() < loss_dict_corr["covariance_loss"].item()

    def test_combined_loss(self, setup):
        """Test combined VICReg loss with all three components."""
        loss_fn = VICRegLoss(
            invariance_weight=25.0,
            variance_weight=25.0,
            covariance_weight=1.0,
        )

        loss_dict = loss_fn(self.z_a, self.z_b)

        # Check all expected keys are present
        assert "loss" in loss_dict
        assert "invariance_loss" in loss_dict
        assert "variance_loss" in loss_dict
        assert "covariance_loss" in loss_dict
        assert "variance_loss_a" in loss_dict
        assert "variance_loss_b" in loss_dict
        assert "covariance_loss_a" in loss_dict
        assert "covariance_loss_b" in loss_dict

        # Check that total loss is weighted sum of components
        expected_loss = (
            25.0 * loss_dict["invariance_loss"]
            + 25.0 * loss_dict["variance_loss"]
            + 1.0 * loss_dict["covariance_loss"]
        )
        assert torch.allclose(loss_dict["loss"], expected_loss, atol=1e-5)

    def test_weight_combinations(self, setup):
        """Test different weight combinations for the three components."""
        configs = [
            (1.0, 0.0, 0.0),  # Only invariance
            (0.0, 1.0, 0.0),  # Only variance
            (0.0, 0.0, 1.0),  # Only covariance
            (1.0, 1.0, 1.0),  # All equal
            (10.0, 5.0, 1.0),  # Different weights
        ]

        for inv_w, var_w, cov_w in configs:
            loss_fn = VICRegLoss(
                invariance_weight=inv_w,
                variance_weight=var_w,
                covariance_weight=cov_w,
            )
            loss_dict = loss_fn(self.z_a, self.z_b)

            # Check that loss is computed correctly
            assert "loss" in loss_dict
            assert not torch.isnan(loss_dict["loss"])
            assert not torch.isinf(loss_dict["loss"])
            assert loss_dict["loss"].item() >= 0

    def test_batch_size_variations(self):
        """Test with different batch sizes."""
        embed_dim = 128
        num_patches = 16

        batch_sizes = [1, 4, 16, 64]
        loss_fn = VICRegLoss()

        for batch_size in batch_sizes:
            z_a = torch.randn(batch_size, num_patches, embed_dim)
            z_b = torch.randn(batch_size, num_patches, embed_dim)

            loss_dict = loss_fn(z_a, z_b)
            assert "loss" in loss_dict
            assert not torch.isnan(loss_dict["loss"])
            assert loss_dict["loss"].item() >= 0

    def test_dimension_variations(self):
        """Test with different embedding dimensions."""
        batch_size = 32
        num_patches = 16

        dimensions = [64, 128, 256, 512, 1024]
        loss_fn = VICRegLoss()

        for embed_dim in dimensions:
            z_a = torch.randn(batch_size, num_patches, embed_dim)
            z_b = torch.randn(batch_size, num_patches, embed_dim)

            loss_dict = loss_fn(z_a, z_b)
            assert "loss" in loss_dict
            assert not torch.isnan(loss_dict["loss"])
            assert loss_dict["loss"].item() >= 0

    def test_gradient_flow(self, setup):
        """Test that gradients flow through all components."""
        loss_fn = VICRegLoss()

        # Make inputs require gradients
        z_a = self.z_a.clone().requires_grad_(True)
        z_b = self.z_b.clone().requires_grad_(True)

        loss_dict = loss_fn(z_a, z_b)
        loss = loss_dict["loss"]
        loss.backward()

        # Check that gradients are computed
        assert z_a.grad is not None
        assert z_b.grad is not None
        assert not torch.isnan(z_a.grad).any()
        assert not torch.isnan(z_b.grad).any()
        assert not torch.isinf(z_a.grad).any()
        assert not torch.isinf(z_b.grad).any()

    def test_edge_case_identical_features(self, setup):
        """Test edge case with identical features."""
        loss_fn = VICRegLoss()

        # Test with identical representations
        z_identical = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        loss_dict = loss_fn(z_identical, z_identical)

        # Invariance loss should be zero
        assert torch.allclose(loss_dict["invariance_loss"], torch.tensor(0.0), atol=1e-6)

        # Total loss should be non-zero (due to variance and covariance terms)
        assert loss_dict["loss"].item() > 0

    def test_edge_case_zero_features(self, setup):
        """Test edge case with zero features."""
        loss_fn = VICRegLoss()

        # Test with zero representations
        z_zero = torch.zeros(self.batch_size, self.num_patches, self.embed_dim)
        loss_dict = loss_fn(z_zero, z_zero)

        # Should not produce NaN or Inf
        assert not torch.isnan(loss_dict["loss"])
        assert not torch.isinf(loss_dict["loss"])

        # Invariance loss should be zero
        assert torch.allclose(loss_dict["invariance_loss"], torch.tensor(0.0), atol=1e-6)

        # Variance loss should be high (variance is zero)
        assert loss_dict["variance_loss"].item() > 0

    def test_single_input_concatenated_views(self):
        """Test forward pass with single concatenated input."""
        batch_size = 32
        num_patches = 16
        embed_dim = 128

        # Create concatenated views (first half = view A, second half = view B)
        z_concatenated = torch.randn(batch_size * 2, num_patches, embed_dim)

        loss_fn = VICRegLoss()
        loss_dict = loss_fn(z_concatenated)

        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])
        assert loss_dict["loss"].item() >= 0

    def test_single_input_odd_batch_size(self):
        """Test that odd batch size raises error with single input."""
        # Odd batch size should fail (can't split into two views)
        z_odd = torch.randn(31, 16, 128)  # Odd batch size

        loss_fn = VICRegLoss()
        with pytest.raises(AssertionError, match="batch size must be even"):
            loss_fn(z_odd)

    def test_mismatched_shapes(self):
        """Test error handling for mismatched shapes."""
        loss_fn = VICRegLoss()

        z_a = torch.randn(32, 16, 128)
        z_b = torch.randn(32, 16, 64)  # Different embedding dimension

        with pytest.raises(AssertionError, match="must have the same shape"):
            loss_fn(z_a, z_b)

    def test_2d_input(self):
        """Test with 2D input [B, D] without patch dimension."""
        batch_size = 32
        embed_dim = 128

        z_a = torch.randn(batch_size, embed_dim)
        z_b = torch.randn(batch_size, embed_dim)

        loss_fn = VICRegLoss()
        loss_dict = loss_fn(z_a, z_b)

        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])
        assert loss_dict["loss"].item() >= 0

    def test_3d_input_with_flattening(self):
        """Test 3D input with patch flattening enabled."""
        batch_size = 16
        num_patches = 25
        embed_dim = 128

        z_a = torch.randn(batch_size, num_patches, embed_dim)
        z_b = torch.randn(batch_size, num_patches, embed_dim)

        loss_fn = VICRegLoss(flatten_patches=True)
        loss_dict = loss_fn(z_a, z_b)

        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])
        assert loss_dict["loss"].item() >= 0

    def test_3d_input_without_flattening(self):
        """Test 3D input with patch flattening disabled."""
        batch_size = 16
        num_patches = 25
        embed_dim = 128

        z_a = torch.randn(batch_size, num_patches, embed_dim)
        z_b = torch.randn(batch_size, num_patches, embed_dim)

        loss_fn = VICRegLoss(flatten_patches=False)
        loss_dict = loss_fn(z_a, z_b)

        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])
        assert loss_dict["loss"].item() >= 0

    def test_variance_loss_separate_views(self, setup):
        """Test that variance loss is computed separately for both views."""
        loss_fn = VICRegLoss()
        loss_dict = loss_fn(self.z_a, self.z_b)

        assert "variance_loss_a" in loss_dict
        assert "variance_loss_b" in loss_dict

        # Average of the two should equal the reported variance loss
        avg_var_loss = (loss_dict["variance_loss_a"] + loss_dict["variance_loss_b"]) / 2
        assert torch.allclose(loss_dict["variance_loss"], avg_var_loss, atol=1e-6)

    def test_covariance_loss_separate_views(self, setup):
        """Test that covariance loss is computed separately for both views."""
        loss_fn = VICRegLoss()
        loss_dict = loss_fn(self.z_a, self.z_b)

        assert "covariance_loss_a" in loss_dict
        assert "covariance_loss_b" in loss_dict

        # Average of the two should equal the reported covariance loss
        avg_cov_loss = (loss_dict["covariance_loss_a"] + loss_dict["covariance_loss_b"]) / 2
        assert torch.allclose(loss_dict["covariance_loss"], avg_cov_loss, atol=1e-6)

    def test_deterministic_computation(self):
        """Test that loss computation is deterministic."""
        torch.manual_seed(42)

        batch_size = 32
        num_patches = 16
        embed_dim = 128

        z_a = torch.randn(batch_size, num_patches, embed_dim)
        z_b = torch.randn(batch_size, num_patches, embed_dim)

        loss_fn = VICRegLoss()
        loss_dict1 = loss_fn(z_a, z_b)
        loss_dict2 = loss_fn(z_a, z_b)

        assert torch.allclose(loss_dict1["loss"], loss_dict2["loss"])
        assert torch.allclose(loss_dict1["invariance_loss"], loss_dict2["invariance_loss"])
        assert torch.allclose(loss_dict1["variance_loss"], loss_dict2["variance_loss"])
        assert torch.allclose(loss_dict1["covariance_loss"], loss_dict2["covariance_loss"])

    def test_extra_repr(self):
        """Test the string representation."""
        loss_fn = VICRegLoss(
            invariance_weight=10.0,
            variance_weight=15.0,
            covariance_weight=2.0,
            variance_threshold=0.8,
            flatten_patches=False,
        )

        repr_str = loss_fn.extra_repr()
        assert "inv_weight=10.0" in repr_str
        assert "var_weight=15.0" in repr_str
        assert "cov_weight=2.0" in repr_str
        assert "var_threshold=0.8" in repr_str
        assert "flatten_patches=False" in repr_str


class TestAdaptiveVICRegLoss:
    """Test suite for AdaptiveVICRegLoss."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        self.batch_size = 32
        self.num_patches = 16
        self.embed_dim = 128

        self.z_a = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        self.z_b = torch.randn(self.batch_size, self.num_patches, self.embed_dim)

        yield

    def test_initialization_non_adaptive(self):
        """Test initialization without adaptive weights."""
        loss_fn = AdaptiveVICRegLoss(adaptive_weights=False)

        assert loss_fn.adaptive_weights == False
        assert loss_fn.invariance_weight == 25.0
        assert loss_fn.variance_weight == 25.0
        assert loss_fn.covariance_weight == 1.0

        # Weights should be floats, not parameters
        assert isinstance(loss_fn.invariance_weight, float)
        assert isinstance(loss_fn.variance_weight, float)
        assert isinstance(loss_fn.covariance_weight, float)

    def test_initialization_adaptive(self):
        """Test initialization with adaptive weights."""
        loss_fn = AdaptiveVICRegLoss(
            invariance_weight=10.0,
            variance_weight=15.0,
            covariance_weight=2.0,
            adaptive_weights=True,
        )

        assert loss_fn.adaptive_weights == True

        # Weights should be parameters
        assert isinstance(loss_fn.invariance_weight, nn.Parameter)
        assert isinstance(loss_fn.variance_weight, nn.Parameter)
        assert isinstance(loss_fn.covariance_weight, nn.Parameter)

        # Check initial values
        assert torch.allclose(loss_fn.invariance_weight, torch.tensor(10.0))
        assert torch.allclose(loss_fn.variance_weight, torch.tensor(15.0))
        assert torch.allclose(loss_fn.covariance_weight, torch.tensor(2.0))

    def test_forward_adaptive(self, setup):
        """Test forward pass with adaptive weights."""
        loss_fn = AdaptiveVICRegLoss(adaptive_weights=True)
        loss_dict = loss_fn(self.z_a, self.z_b)

        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])
        assert loss_dict["loss"].item() >= 0

    def test_update_weights_non_adaptive(self, setup):
        """Test that update_weights does nothing when adaptive_weights=False."""
        loss_fn = AdaptiveVICRegLoss(adaptive_weights=False)
        loss_dict = loss_fn(self.z_a, self.z_b)

        # Store original weights
        orig_inv = loss_fn.invariance_weight
        orig_var = loss_fn.variance_weight
        orig_cov = loss_fn.covariance_weight

        # Update weights (should not change anything)
        loss_fn.update_weights(loss_dict)

        assert loss_fn.invariance_weight == orig_inv
        assert loss_fn.variance_weight == orig_var
        assert loss_fn.covariance_weight == orig_cov

    def test_update_weights_adaptive(self, setup):
        """Test that update_weights modifies weights when adaptive_weights=True."""
        loss_fn = AdaptiveVICRegLoss(
            adaptive_weights=True,
            weight_momentum=0.9,
        )
        loss_dict = loss_fn(self.z_a, self.z_b)

        # Store original weights
        orig_inv = loss_fn.invariance_weight.data.clone()
        orig_var = loss_fn.variance_weight.data.clone()
        orig_cov = loss_fn.covariance_weight.data.clone()

        # Update weights
        loss_fn.update_weights(loss_dict)

        # Weights should have changed (with high probability)
        # Using high tolerance since momentum is 0.9 and changes are small
        assert not torch.allclose(loss_fn.invariance_weight.data, orig_inv, atol=1e-3, rtol=1e-2)

    def test_gradient_flow_adaptive(self, setup):
        """Test gradient flow with adaptive weights."""
        loss_fn = AdaptiveVICRegLoss(adaptive_weights=True)

        # Make inputs require gradients
        z_a = self.z_a.clone().requires_grad_(True)
        z_b = self.z_b.clone().requires_grad_(True)

        loss_dict = loss_fn(z_a, z_b)
        loss = loss_dict["loss"]
        loss.backward()

        # Check gradients on inputs
        assert z_a.grad is not None
        assert z_b.grad is not None
        assert not torch.isnan(z_a.grad).any()
        assert not torch.isnan(z_b.grad).any()

        # Check gradients on weight parameters
        assert loss_fn.invariance_weight.grad is not None
        assert loss_fn.variance_weight.grad is not None
        assert loss_fn.covariance_weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Comprehensive tests for SIGReg loss implementation.

Tests cover:
- Epps-Pulley test correctness
- SIGReg loss computation
- Hybrid VICReg/SIGReg loss
- Configuration and factory creation
- Edge cases and error handling
"""

import pytest
import torch
import torch.nn as nn
from src.losses import (
    SIGRegLoss,
    HybridVICRegSIGRegLoss,
    EppsPulleyTest,
    VICRegLoss,
    create_loss_from_config,
)


class TestEppsPulleyTest:
    """Tests for Epps-Pulley statistical test"""

    def test_initialization(self):
        """Test EP test can be initialized"""
        test = EppsPulleyTest(num_points=17)
        assert test.num_points == 17
        assert test.reference_points.shape == (17,)

    def test_standard_gaussian(self):
        """Test EP statistic is low for standard Gaussian"""
        test = EppsPulleyTest(num_points=17)

        # Standard Gaussian should have low test statistic
        gaussian = torch.randn(1000)
        stat = test(gaussian)

        assert stat.item() < 1.0, "Standard Gaussian should have low EP statistic"

    def test_non_gaussian(self):
        """Test EP statistic is higher for non-Gaussian"""
        test = EppsPulleyTest(num_points=17)

        # Uniform distribution should have higher statistic
        uniform = torch.rand(1000) * 2 - 1
        stat_uniform = test(uniform)

        # Gaussian for comparison
        gaussian = torch.randn(1000)
        stat_gaussian = test(gaussian)

        assert stat_uniform > stat_gaussian, \
            "Non-Gaussian should have higher EP statistic"

    def test_batched_input(self):
        """Test EP test works with batched input"""
        test = EppsPulleyTest(num_points=17)

        # Batched input [B, N]
        batched = torch.randn(4, 1000)
        stats = test(batched)

        assert stats.shape == (4,), "Should return one stat per batch"
        assert all(s < 1.0 for s in stats), "All should be low for Gaussian"

    def test_deterministic(self):
        """Test EP test is deterministic for same input"""
        test = EppsPulleyTest(num_points=17)

        x = torch.randn(1000)
        stat1 = test(x)
        stat2 = test(x)

        assert torch.allclose(stat1, stat2), "Should be deterministic"


class TestSIGRegLoss:
    """Tests for SIGReg loss"""

    def test_initialization(self):
        """Test SIGReg loss can be initialized"""
        loss_fn = SIGRegLoss(
            num_slices=1024,
            num_test_points=17,
            invariance_weight=25.0,
            sigreg_weight=25.0,
        )

        assert loss_fn.num_slices == 1024
        assert loss_fn.invariance_weight == 25.0
        assert loss_fn.sigreg_weight == 25.0

    def test_forward_pass(self):
        """Test SIGReg loss forward pass"""
        loss_fn = SIGRegLoss(num_slices=256)  # Fewer slices for speed

        # Two views
        B, N, D = 8, 49, 192
        z_a = torch.randn(B, N, D)
        z_b = torch.randn(B, N, D)

        # Compute loss
        loss_dict = loss_fn(z_a, z_b)

        # Check outputs
        assert 'loss' in loss_dict
        assert 'invariance_loss' in loss_dict
        assert 'sigreg_loss' in loss_dict
        assert 'sigreg_loss_a' in loss_dict
        assert 'sigreg_loss_b' in loss_dict

        # Check loss is positive
        assert loss_dict['loss'] > 0

    def test_single_input(self):
        """Test SIGReg with single concatenated input"""
        loss_fn = SIGRegLoss(num_slices=256)

        # Concatenated views [2*B, N, D]
        B, N, D = 8, 49, 192
        z = torch.randn(2 * B, N, D)

        # Should split automatically
        loss_dict = loss_fn(z)

        assert 'loss' in loss_dict
        assert loss_dict['loss'] > 0

    def test_2d_input(self):
        """Test SIGReg with 2D input [B, D]"""
        loss_fn = SIGRegLoss(num_slices=256)

        # 2D input
        B, D = 32, 768
        z_a = torch.randn(B, D)
        z_b = torch.randn(B, D)

        loss_dict = loss_fn(z_a, z_b)

        assert 'loss' in loss_dict
        assert loss_dict['loss'] > 0

    def test_backward_pass(self):
        """Test SIGReg loss backward pass"""
        loss_fn = SIGRegLoss(num_slices=256)

        # Requires gradient
        B, N, D = 8, 49, 192
        z_a = torch.randn(B, N, D, requires_grad=True)
        z_b = torch.randn(B, N, D, requires_grad=True)

        # Forward
        loss_dict = loss_fn(z_a, z_b)
        total_loss = loss_dict['loss']

        # Backward
        total_loss.backward()

        # Check gradients exist
        assert z_a.grad is not None
        assert z_b.grad is not None
        assert not torch.isnan(z_a.grad).any()

    def test_fixed_slices(self):
        """Test fixed slices produce same results"""
        loss_fn = SIGRegLoss(
            num_slices=256,
            fixed_slices=True,
        )

        B, N, D = 8, 49, 192
        z_a = torch.randn(B, N, D)
        z_b = torch.randn(B, N, D)

        # Two forward passes should give same SIGReg loss
        loss1 = loss_fn(z_a, z_b)['sigreg_loss']
        loss2 = loss_fn(z_a, z_b)['sigreg_loss']

        assert torch.allclose(loss1, loss2), \
            "Fixed slices should be deterministic"

    def test_invariance_term(self):
        """Test invariance loss is MSE"""
        loss_fn = SIGRegLoss(
            num_slices=256,
            invariance_weight=1.0,
            sigreg_weight=0.0,  # Only invariance
        )

        B, N, D = 8, 49, 192
        z_a = torch.randn(B, N, D)
        z_b = z_a.clone()  # Identical

        loss_dict = loss_fn(z_a, z_b)

        # Should be near zero for identical inputs
        assert loss_dict['invariance_loss'] < 1e-6

    def test_different_num_slices(self):
        """Test different numbers of slices"""
        B, N, D = 8, 49, 192
        z_a = torch.randn(B, N, D)
        z_b = torch.randn(B, N, D)

        for num_slices in [128, 256, 512, 1024]:
            loss_fn = SIGRegLoss(num_slices=num_slices)
            loss_dict = loss_fn(z_a, z_b)

            assert loss_dict['loss'] > 0
            assert not torch.isnan(loss_dict['loss'])

    def test_shape_mismatch_error(self):
        """Test error on shape mismatch"""
        loss_fn = SIGRegLoss(num_slices=256)

        z_a = torch.randn(8, 49, 192)
        z_b = torch.randn(8, 50, 192)  # Wrong number of patches

        with pytest.raises(AssertionError):
            loss_fn(z_a, z_b)

    def test_invalid_parameters(self):
        """Test error on invalid parameters"""
        with pytest.raises(AssertionError):
            SIGRegLoss(num_slices=0)  # Invalid

        with pytest.raises(AssertionError):
            SIGRegLoss(invariance_weight=-1.0)  # Invalid


class TestHybridVICRegSIGRegLoss:
    """Tests for hybrid VICReg/SIGReg loss"""

    def test_initialization(self):
        """Test hybrid loss initialization"""
        loss_fn = HybridVICRegSIGRegLoss(
            vicreg_weight=1.0,
            sigreg_weight=1.0,
            num_slices=256,
        )

        assert loss_fn.vicreg_weight == 1.0
        assert loss_fn.sigreg_weight == 1.0

    def test_forward_pass(self):
        """Test hybrid loss forward pass"""
        loss_fn = HybridVICRegSIGRegLoss(
            vicreg_weight=0.5,
            sigreg_weight=0.5,
            num_slices=256,
        )

        B, N, D = 8, 49, 192
        z_a = torch.randn(B, N, D)
        z_b = torch.randn(B, N, D)

        loss_dict = loss_fn(z_a, z_b)

        # Check all components present
        assert 'loss' in loss_dict
        assert 'vicreg_loss' in loss_dict
        assert 'sigreg_loss' in loss_dict
        assert 'invariance_loss' in loss_dict
        assert 'variance_loss' in loss_dict
        assert 'covariance_loss' in loss_dict
        assert 'sigreg_regularization' in loss_dict

    def test_vicreg_only(self):
        """Test hybrid with only VICReg (no SIGReg)"""
        loss_fn = HybridVICRegSIGRegLoss(
            vicreg_weight=1.0,
            sigreg_weight=0.0,
            num_slices=256,
        )

        B, N, D = 8, 49, 192
        z_a = torch.randn(B, N, D)
        z_b = torch.randn(B, N, D)

        loss_dict = loss_fn(z_a, z_b)

        # Should be dominated by VICReg
        assert loss_dict['vicreg_loss'] > 0
        assert loss_dict['loss'] > 0

    def test_sigreg_only(self):
        """Test hybrid with only SIGReg (no VICReg)"""
        loss_fn = HybridVICRegSIGRegLoss(
            vicreg_weight=0.0,
            sigreg_weight=1.0,
            num_slices=256,
        )

        B, N, D = 8, 49, 192
        z_a = torch.randn(B, N, D)
        z_b = torch.randn(B, N, D)

        loss_dict = loss_fn(z_a, z_b)

        # Should be dominated by SIGReg
        assert loss_dict['sigreg_loss'] > 0
        assert loss_dict['loss'] > 0

    def test_weight_adjustment(self):
        """Test dynamic weight adjustment"""
        loss_fn = HybridVICRegSIGRegLoss(
            vicreg_weight=1.0,
            sigreg_weight=0.0,
            num_slices=256,
        )

        B, N, D = 8, 49, 192
        z_a = torch.randn(B, N, D)
        z_b = torch.randn(B, N, D)

        # Initial weights
        loss1 = loss_fn(z_a, z_b)['loss']

        # Adjust weights
        loss_fn.vicreg_weight = 0.5
        loss_fn.sigreg_weight = 0.5

        # Should give different total loss
        loss2 = loss_fn(z_a, z_b)['loss']

        # Losses should differ
        assert not torch.allclose(loss1, loss2)


class TestLossFactory:
    """Tests for loss factory creation"""

    def test_create_sigreg_from_config(self):
        """Test creating SIGReg from config"""
        config = {
            'type': 'sigreg',
            'sigreg_num_slices': 512,
            'sigreg_weight': 25.0,
            'sigreg_invariance_weight': 25.0,
        }

        loss_fn = create_loss_from_config(config)

        assert isinstance(loss_fn, SIGRegLoss)
        assert loss_fn.num_slices == 512

    def test_create_sigreg_with_defaults(self):
        """Test creating SIGReg with default parameters"""
        config = {
            'type': 'sigreg',
        }

        loss_fn = create_loss_from_config(config)

        assert isinstance(loss_fn, SIGRegLoss)
        assert loss_fn.num_slices == 1024  # Default

    def test_sigreg_config_variations(self):
        """Test various SIGReg configurations"""
        configs = [
            {'type': 'sigreg', 'sigreg_num_slices': 256},
            {'type': 'sigreg', 'sigreg_fixed_slices': True},
            {'type': 'sigreg', 'sigreg_num_test_points': 11},
        ]

        for config in configs:
            loss_fn = create_loss_from_config(config)
            assert isinstance(loss_fn, SIGRegLoss)


class TestComparison:
    """Comparison tests between VICReg and SIGReg"""

    def test_vicreg_vs_sigreg_output_format(self):
        """Test both losses return compatible output format"""
        vicreg = VICRegLoss()
        sigreg = SIGRegLoss(num_slices=256)

        B, N, D = 8, 49, 192
        z_a = torch.randn(B, N, D)
        z_b = torch.randn(B, N, D)

        vicreg_dict = vicreg(z_a, z_b)
        sigreg_dict = sigreg(z_a, z_b)

        # Both should have 'loss'
        assert 'loss' in vicreg_dict
        assert 'loss' in sigreg_dict

        # Both should have 'invariance_loss'
        assert 'invariance_loss' in vicreg_dict
        assert 'invariance_loss' in sigreg_dict

    def test_similar_invariance(self):
        """Test invariance terms are similar"""
        vicreg = VICRegLoss(
            invariance_weight=1.0,
            variance_weight=0.0,
            covariance_weight=0.0,
        )
        sigreg = SIGRegLoss(
            num_slices=256,
            invariance_weight=1.0,
            sigreg_weight=0.0,
        )

        B, N, D = 8, 49, 192
        z_a = torch.randn(B, N, D)
        z_b = torch.randn(B, N, D)

        vicreg_inv = vicreg(z_a, z_b)['invariance_loss']
        sigreg_inv = sigreg(z_a, z_b)['invariance_loss']

        # Should be identical (both are MSE)
        assert torch.allclose(vicreg_inv, sigreg_inv)


class TestEdgeCases:
    """Edge case and error handling tests"""

    def test_small_batch(self):
        """Test with very small batch"""
        loss_fn = SIGRegLoss(num_slices=256)

        # Single sample
        z_a = torch.randn(1, 49, 192)
        z_b = torch.randn(1, 49, 192)

        loss_dict = loss_fn(z_a, z_b)
        assert loss_dict['loss'] > 0

    def test_large_embedding_dim(self):
        """Test with large embedding dimension"""
        loss_fn = SIGRegLoss(num_slices=256)

        # Large dimension
        z_a = torch.randn(4, 49, 2048)
        z_b = torch.randn(4, 49, 2048)

        loss_dict = loss_fn(z_a, z_b)
        assert loss_dict['loss'] > 0

    def test_zero_weights(self):
        """Test with zero weights"""
        loss_fn = SIGRegLoss(
            num_slices=256,
            invariance_weight=0.0,
            sigreg_weight=0.0,
        )

        z_a = torch.randn(8, 49, 192)
        z_b = torch.randn(8, 49, 192)

        loss_dict = loss_fn(z_a, z_b)

        # Total loss should be zero
        assert torch.allclose(loss_dict['loss'], torch.tensor(0.0))

    def test_nan_handling(self):
        """Test behavior with NaN inputs"""
        loss_fn = SIGRegLoss(num_slices=256)

        z_a = torch.randn(8, 49, 192)
        z_b = torch.randn(8, 49, 192)

        # Add some NaN
        z_a[0, 0, 0] = float('nan')

        loss_dict = loss_fn(z_a, z_b)

        # Loss should be NaN
        assert torch.isnan(loss_dict['loss'])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

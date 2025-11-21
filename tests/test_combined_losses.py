"""
Test suite for combined loss functions in H-JEPA.

This module tests the combined loss functions:
- CombinedLoss (JEPA + VICReg)
- HierarchicalCombinedLoss (with hierarchy-specific configs)
- create_loss_from_config factory function
"""

import pytest
import torch
import torch.nn as nn

from src.losses.combined import CombinedLoss, HierarchicalCombinedLoss, create_loss_from_config
from src.losses.hjepa_loss import HJEPALoss
from src.losses.vicreg import VICRegLoss


class TestCombinedLoss:
    """Test suite for CombinedLoss."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        self.batch_size = 8  # Need even batch for VICReg splitting
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
        """Test default initialization of CombinedLoss."""
        loss_fn = CombinedLoss()
        assert loss_fn.num_hierarchies == 3
        assert loss_fn.apply_vicreg_per_level == True
        assert loss_fn.vicreg_on_targets == False
        assert len(loss_fn.vicreg_weights) == 3
        assert all(w == 0.1 for w in loss_fn.vicreg_weights)
        assert isinstance(loss_fn.jepa_loss, HJEPALoss)
        assert isinstance(loss_fn.vicreg_loss, VICRegLoss)

    def test_initialization_custom_scalar_weights(self):
        """Test initialization with scalar VICReg weight."""
        vicreg_weight = 0.5
        loss_fn = CombinedLoss(
            num_hierarchies=3,
            vicreg_weight=vicreg_weight,
        )
        assert len(loss_fn.vicreg_weights) == 3
        assert all(w == vicreg_weight for w in loss_fn.vicreg_weights)

    def test_initialization_custom_list_weights(self):
        """Test initialization with list of VICReg weights per hierarchy."""
        vicreg_weights = [0.1, 0.2, 0.3]
        loss_fn = CombinedLoss(
            num_hierarchies=3,
            vicreg_weight=vicreg_weights,
        )
        assert loss_fn.vicreg_weights == vicreg_weights

    def test_initialization_weight_mismatch(self):
        """Test error when VICReg weight list length doesn't match hierarchies."""
        with pytest.raises(AssertionError):
            CombinedLoss(
                num_hierarchies=3,
                vicreg_weight=[0.1, 0.2],  # Only 2 weights for 3 hierarchies
            )

    def test_forward_basic(self, setup):
        """Test basic forward pass with default settings."""
        loss_fn = CombinedLoss(num_hierarchies=3)
        loss_dict = loss_fn(self.predictions, self.targets)

        # Check required keys
        assert "loss" in loss_dict
        assert "jepa_loss" in loss_dict
        assert "vicreg_loss" in loss_dict

        # Check loss properties
        total_loss = loss_dict["loss"]
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0  # Scalar
        assert total_loss.item() >= 0
        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)

        # Verify total loss is sum of components
        expected_total = loss_dict["jepa_loss"] + loss_dict["vicreg_loss"]
        assert torch.allclose(total_loss, expected_total, rtol=1e-5)

    def test_forward_per_level_vicreg(self, setup):
        """Test forward pass with per-level VICReg application."""
        loss_fn = CombinedLoss(
            num_hierarchies=3,
            apply_vicreg_per_level=True,
        )
        loss_dict = loss_fn(self.predictions, self.targets)

        # Check per-level VICReg losses exist
        for i in range(3):
            assert f"vicreg_h{i}" in loss_dict
            assert f"vicreg_invariance_h{i}" in loss_dict
            assert f"vicreg_variance_h{i}" in loss_dict
            assert f"vicreg_covariance_h{i}" in loss_dict

            # Check values are valid
            assert loss_dict[f"vicreg_h{i}"].item() >= 0
            assert not torch.isnan(loss_dict[f"vicreg_h{i}"])

    def test_forward_single_level_vicreg(self, setup):
        """Test forward pass with VICReg only at last level."""
        loss_fn = CombinedLoss(
            num_hierarchies=3,
            apply_vicreg_per_level=False,
        )
        loss_dict = loss_fn(self.predictions, self.targets)

        # VICReg should be applied only at last level
        assert "vicreg_loss" in loss_dict
        assert "vicreg_invariance_loss" in loss_dict
        assert "vicreg_variance_loss" in loss_dict
        assert "vicreg_covariance_loss" in loss_dict

        # Should not have per-level VICReg keys
        assert "vicreg_h0" not in loss_dict
        assert "vicreg_h1" not in loss_dict

    def test_forward_vicreg_on_targets(self, setup):
        """Test forward pass with VICReg applied to targets."""
        loss_fn = CombinedLoss(
            num_hierarchies=3,
            vicreg_on_targets=True,
        )
        loss_dict = loss_fn(self.predictions, self.targets)

        assert "loss" in loss_dict
        assert loss_dict["loss"].item() >= 0

    def test_forward_different_vicreg_weights(self, setup):
        """Test that different VICReg weights produce different losses."""
        # Equal weights
        loss_fn_equal = CombinedLoss(vicreg_weight=0.1)
        loss_dict_equal = loss_fn_equal(self.predictions, self.targets)

        # Different weights per hierarchy
        loss_fn_weighted = CombinedLoss(vicreg_weight=[0.5, 0.3, 0.1])
        loss_dict_weighted = loss_fn_weighted(self.predictions, self.targets)

        # Total losses should be different
        assert not torch.allclose(
            loss_dict_equal["loss"],
            loss_dict_weighted["loss"],
        )

    def test_forward_single_tensor_input(self):
        """Test forward pass with single tensor instead of list."""
        batch_size = 8
        num_patches = 16
        embed_dim = 128

        prediction = torch.randn(batch_size, num_patches, embed_dim)
        target = torch.randn(batch_size, num_patches, embed_dim)

        loss_fn = CombinedLoss(num_hierarchies=1)
        loss_dict = loss_fn(prediction, target)

        assert "loss" in loss_dict
        assert isinstance(loss_dict["loss"], torch.Tensor)
        assert loss_dict["loss"].dim() == 0

    def test_forward_with_masks(self, setup):
        """Test forward pass with masking."""
        masks = [
            torch.randint(0, 2, (self.batch_size, self.num_patches)).float()
            for _ in range(self.num_hierarchies)
        ]

        loss_fn = CombinedLoss(num_hierarchies=3)
        loss_dict = loss_fn(self.predictions, self.targets, masks)

        assert "loss" in loss_dict
        assert loss_dict["loss"].item() >= 0

    def test_backward_gradient_flow(self, setup):
        """Test that gradients flow through both loss components."""
        loss_fn = CombinedLoss(num_hierarchies=3, vicreg_weight=0.1)

        # Make predictions require gradients
        predictions = [p.clone().requires_grad_(True) for p in self.predictions]

        loss_dict = loss_fn(predictions, self.targets)
        loss = loss_dict["loss"]
        loss.backward()

        # Check gradients exist and are valid
        for pred in predictions:
            assert pred.grad is not None
            assert not torch.isnan(pred.grad).any()
            assert not torch.isinf(pred.grad).any()
            # Gradient should be non-zero (both JEPA and VICReg contribute)
            assert pred.grad.abs().sum() > 0

    def test_vicreg_weight_zero(self, setup):
        """Test that zero VICReg weight reduces to pure JEPA loss."""
        loss_fn = CombinedLoss(vicreg_weight=0.0)
        loss_dict = loss_fn(self.predictions, self.targets)

        # Total loss should equal JEPA loss when VICReg weight is 0
        assert torch.allclose(
            loss_dict["loss"],
            loss_dict["jepa_loss"],
            rtol=1e-5,
        )
        # VICReg loss should still be computed but not contribute
        assert loss_dict["vicreg_loss"].item() == 0.0

    def test_different_jepa_loss_types(self, setup):
        """Test with different JEPA loss types."""
        for loss_type in ["mse", "smoothl1", "huber"]:
            loss_fn = CombinedLoss(
                jepa_loss_type=loss_type,
                num_hierarchies=3,
            )
            loss_dict = loss_fn(self.predictions, self.targets)

            assert "loss" in loss_dict
            assert loss_dict["loss"].item() >= 0
            assert not torch.isnan(loss_dict["loss"])

    def test_small_batch_size(self):
        """Test with batch size = 1 (edge case for VICReg)."""
        batch_size = 1
        predictions = [torch.randn(batch_size, 16, 128) for _ in range(3)]
        targets = [torch.randn(batch_size, 16, 128) for _ in range(3)]

        loss_fn = CombinedLoss(num_hierarchies=3)
        loss_dict = loss_fn(predictions, targets)

        # Should handle single batch gracefully
        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])

    def test_get_loss_summary(self, setup):
        """Test loss summary generation."""
        loss_fn = CombinedLoss(num_hierarchies=3)
        loss_dict = loss_fn(self.predictions, self.targets)

        summary = loss_fn.get_loss_summary(loss_dict)

        # Check that summary contains expected strings
        assert "Loss Summary:" in summary
        assert "Total Loss:" in summary
        assert "JEPA Loss:" in summary
        assert "VICReg Loss:" in summary
        assert "Level 0:" in summary
        assert "Level 1:" in summary
        assert "Level 2:" in summary

    def test_extra_repr(self):
        """Test string representation."""
        loss_fn = CombinedLoss(
            num_hierarchies=3,
            vicreg_weight=[0.1, 0.2, 0.3],
            apply_vicreg_per_level=True,
        )
        repr_str = loss_fn.extra_repr()

        assert "num_hierarchies=3" in repr_str
        assert "vicreg_weights=[0.1, 0.2, 0.3]" in repr_str
        assert "apply_vicreg_per_level=True" in repr_str

    def test_deterministic_output(self):
        """Test that loss computation is deterministic."""
        torch.manual_seed(42)

        predictions = [torch.randn(4, 16, 64) for _ in range(3)]
        targets = [torch.randn(4, 16, 64) for _ in range(3)]

        loss_fn = CombinedLoss(num_hierarchies=3)

        loss_dict1 = loss_fn(predictions, targets)
        loss_dict2 = loss_fn(predictions, targets)

        assert torch.allclose(loss_dict1["loss"], loss_dict2["loss"])

    def test_vicreg_single_level_with_targets(self):
        """Test VICReg at single level with targets flag."""
        batch_size = 8
        predictions = [torch.randn(batch_size, 16, 64) for _ in range(3)]
        targets = [torch.randn(batch_size, 16, 64) for _ in range(3)]

        loss_fn = CombinedLoss(
            num_hierarchies=3,
            apply_vicreg_per_level=False,
            vicreg_on_targets=True,
        )
        loss_dict = loss_fn(predictions, targets)

        assert "loss" in loss_dict
        assert "vicreg_loss" in loss_dict
        assert loss_dict["loss"].item() >= 0

    def test_vicreg_single_level_small_batch(self):
        """Test VICReg at single level with small batch."""
        batch_size = 1
        predictions = [torch.randn(batch_size, 16, 64) for _ in range(3)]
        targets = [torch.randn(batch_size, 16, 64) for _ in range(3)]

        loss_fn = CombinedLoss(
            num_hierarchies=3,
            apply_vicreg_per_level=False,
        )
        loss_dict = loss_fn(predictions, targets)

        # Should use pred-target pair when batch too small
        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])


class TestHierarchicalCombinedLoss:
    """Test suite for HierarchicalCombinedLoss."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        self.batch_size = 8
        self.num_patches = 16
        self.embed_dim = 128
        self.num_hierarchies = 3

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
        """Test default initialization."""
        loss_fn = HierarchicalCombinedLoss(num_hierarchies=3)
        assert loss_fn.num_hierarchies == 3
        assert hasattr(loss_fn, "vicreg_losses")
        assert len(loss_fn.vicreg_losses) == 3

    def test_initialization_with_custom_configs(self):
        """Test initialization with hierarchy-specific VICReg configs."""
        vicreg_configs = [
            {
                "invariance_weight": 25.0,
                "variance_weight": 25.0,
                "covariance_weight": 1.0,
            },
            {
                "invariance_weight": 15.0,
                "variance_weight": 15.0,
                "covariance_weight": 0.5,
            },
            {
                "invariance_weight": 10.0,
                "variance_weight": 10.0,
                "covariance_weight": 0.25,
            },
        ]

        loss_fn = HierarchicalCombinedLoss(
            num_hierarchies=3,
            vicreg_configs=vicreg_configs,
        )

        assert len(loss_fn.vicreg_losses) == 3
        # Check that different VICReg instances were created
        for i, config in enumerate(vicreg_configs):
            vicreg = loss_fn.vicreg_losses[i]
            assert vicreg.invariance_weight == config["invariance_weight"]
            assert vicreg.variance_weight == config["variance_weight"]
            assert vicreg.covariance_weight == config["covariance_weight"]

    def test_initialization_config_length_mismatch(self):
        """Test error when config list length doesn't match hierarchies."""
        vicreg_configs = [
            {"invariance_weight": 25.0},
            {"invariance_weight": 15.0},
        ]  # Only 2 configs

        with pytest.raises(AssertionError):
            HierarchicalCombinedLoss(
                num_hierarchies=3,
                vicreg_configs=vicreg_configs,
            )

    def test_forward_basic(self, setup):
        """Test basic forward pass."""
        loss_fn = HierarchicalCombinedLoss(num_hierarchies=3)
        loss_dict = loss_fn(self.predictions, self.targets)

        assert "loss" in loss_dict
        assert "jepa_loss" in loss_dict
        assert "vicreg_loss" in loss_dict

        # Check per-level losses
        for i in range(3):
            assert f"vicreg_h{i}" in loss_dict

        assert loss_dict["loss"].item() >= 0

    def test_forward_with_different_configs(self, setup):
        """Test that different configs produce different results."""
        # Config 1: Strong regularization
        configs_strong = [{"invariance_weight": 50.0, "variance_weight": 50.0} for _ in range(3)]
        loss_fn_strong = HierarchicalCombinedLoss(
            num_hierarchies=3,
            vicreg_configs=configs_strong,
        )

        # Config 2: Weak regularization
        configs_weak = [{"invariance_weight": 5.0, "variance_weight": 5.0} for _ in range(3)]
        loss_fn_weak = HierarchicalCombinedLoss(
            num_hierarchies=3,
            vicreg_configs=configs_weak,
        )

        loss_dict_strong = loss_fn_strong(self.predictions, self.targets)
        loss_dict_weak = loss_fn_weak(self.predictions, self.targets)

        # Strong regularization should produce higher VICReg loss
        assert loss_dict_strong["vicreg_loss"] > loss_dict_weak["vicreg_loss"]

    def test_backward_gradient_flow(self, setup):
        """Test gradient flow through hierarchical loss."""
        loss_fn = HierarchicalCombinedLoss(num_hierarchies=3)

        predictions = [p.clone().requires_grad_(True) for p in self.predictions]

        loss_dict = loss_fn(predictions, self.targets)
        loss = loss_dict["loss"]
        loss.backward()

        for pred in predictions:
            assert pred.grad is not None
            assert not torch.isnan(pred.grad).any()

    def test_forward_with_single_batch(self):
        """Test with batch size = 1 (edge case)."""
        batch_size = 1
        predictions = [torch.randn(batch_size, 16, 64) for _ in range(3)]
        targets = [torch.randn(batch_size, 16, 64) for _ in range(3)]

        loss_fn = HierarchicalCombinedLoss(num_hierarchies=3)
        loss_dict = loss_fn(predictions, targets)

        # Should handle single batch by using pred-target pairs
        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])

    def test_forward_tensor_conversion(self):
        """Test tensor to list conversion in forward."""
        batch_size = 4
        # Pass single tensors instead of lists
        prediction = torch.randn(batch_size, 16, 64)
        target = torch.randn(batch_size, 16, 64)

        loss_fn = HierarchicalCombinedLoss(num_hierarchies=1)
        loss_dict = loss_fn(prediction, target)

        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])


class TestCreateLossFromConfig:
    """Test suite for create_loss_from_config factory function."""

    def test_create_hjepa_loss(self):
        """Test creating HJEPA loss from config."""
        config = {
            "type": "hjepa",
            "jepa_loss_type": "smoothl1",
            "hierarchy_weights": [1.0, 0.5, 0.25],
            "num_hierarchies": 3,
        }

        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, HJEPALoss)

    def test_create_hjepa_with_contrastive(self):
        """Test creating HJEPA with contrastive loss enabled."""
        config = {
            "type": "hjepa",
            "use_contrastive": True,
            "contrastive_weight": 0.1,
            "contrastive_temperature": 0.1,
            "num_hierarchies": 3,
        }

        loss_fn = create_loss_from_config(config)
        # Should return ContrastiveJEPALoss wrapping HJEPALoss
        from src.losses.contrastive import ContrastiveJEPALoss

        assert isinstance(loss_fn, ContrastiveJEPALoss)

    def test_create_vicreg_loss(self):
        """Test creating VICReg loss from config."""
        config = {
            "type": "vicreg",
            "vicreg_invariance_weight": 25.0,
            "vicreg_variance_weight": 25.0,
            "vicreg_covariance_weight": 1.0,
        }

        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, VICRegLoss)

    def test_create_combined_loss(self):
        """Test creating combined loss from config."""
        config = {
            "type": "combined",
            "jepa_loss_type": "smoothl1",
            "hierarchy_weights": [1.0, 0.5, 0.25],
            "num_hierarchies": 3,
            "vicreg_weight": 0.1,
        }

        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, CombinedLoss)

    def test_create_hierarchical_combined_loss(self):
        """Test creating hierarchical combined loss from config."""
        config = {
            "type": "hierarchical_combined",
            "jepa_loss_type": "smoothl1",
            "hierarchy_weights": [1.0, 0.5, 0.25],
            "num_hierarchies": 3,
            "vicreg_weight": 0.1,
            "vicreg_configs": [
                {"invariance_weight": 25.0},
                {"invariance_weight": 15.0},
                {"invariance_weight": 10.0},
            ],
        }

        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, HierarchicalCombinedLoss)

    def test_create_loss_with_nested_config(self):
        """Test creating loss with nested config structure."""
        config = {
            "loss": {
                "type": "combined",
                "jepa_loss_type": "smoothl1",
                "vicreg_weight": 0.1,
            },
            "model": {
                "num_hierarchies": 3,
            },
        }

        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, CombinedLoss)
        assert loss_fn.num_hierarchies == 3

    def test_create_loss_unknown_type(self):
        """Test error handling for unknown loss type."""
        config = {
            "type": "unknown_loss_type",
        }

        with pytest.raises(ValueError, match="Unknown loss type"):
            create_loss_from_config(config)

    def test_create_loss_with_mse_type(self):
        """Test creating loss with 'mse' type."""
        config = {
            "type": "mse",
            "num_hierarchies": 3,
        }

        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, HJEPALoss)
        assert loss_fn.loss_type == "mse"

    def test_create_loss_with_smoothl1_type(self):
        """Test creating loss with 'smoothl1' type."""
        config = {
            "type": "smoothl1",
            "num_hierarchies": 3,
        }

        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, HJEPALoss)
        assert loss_fn.loss_type == "smoothl1"

    def test_vicreg_warning_on_hjepa(self):
        """Test that warning is raised when VICReg fields are in HJEPA config."""
        import warnings

        config = {
            "type": "hjepa",
            "vicreg_weight": 0.1,  # This should trigger warning
            "num_hierarchies": 3,
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loss_fn = create_loss_from_config(config)

            # Check that a warning was raised
            assert len(w) >= 1
            assert "VICReg" in str(w[0].message)
            assert isinstance(loss_fn, HJEPALoss)

    def test_create_sigreg_loss(self):
        """Test creating SIGReg loss from config."""
        config = {
            "type": "sigreg",
            "sigreg_num_slices": 1024,
            "sigreg_invariance_weight": 25.0,
            "sigreg_weight": 25.0,
        }

        from src.losses.sigreg import SIGRegLoss

        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, SIGRegLoss)

    def test_create_cjepa_loss(self):
        """Test creating C-JEPA (contrastive JEPA) loss from config."""
        config = {
            "type": "cjepa",
            "jepa_loss_type": "smoothl1",
            "contrastive_weight": 0.1,
            "contrastive_temperature": 0.1,
            "num_hierarchies": 3,
        }

        from src.losses.contrastive import ContrastiveJEPALoss

        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, ContrastiveJEPALoss)

    def test_create_contrastive_jepa_loss(self):
        """Test creating contrastive_jepa loss from config (alternative name)."""
        config = {
            "type": "contrastive_jepa",
            "num_hierarchies": 3,
        }

        from src.losses.contrastive import ContrastiveJEPALoss

        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, ContrastiveJEPALoss)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nan_handling(self):
        """Test that NaN in inputs is handled correctly."""
        batch_size = 4
        predictions = [torch.randn(batch_size, 16, 64) for _ in range(3)]
        targets = [torch.randn(batch_size, 16, 64) for _ in range(3)]

        # Introduce NaN
        predictions[0][0, 0, 0] = float("nan")

        loss_fn = CombinedLoss(num_hierarchies=3)
        loss_dict = loss_fn(predictions, targets)

        # Loss should contain NaN
        assert torch.isnan(loss_dict["loss"])

    def test_inf_handling(self):
        """Test handling of infinite values."""
        batch_size = 4
        predictions = [torch.randn(batch_size, 16, 64) for _ in range(3)]
        targets = [torch.randn(batch_size, 16, 64) for _ in range(3)]

        # Introduce Inf
        predictions[0][0, 0, 0] = float("inf")

        loss_fn = CombinedLoss(num_hierarchies=3)
        loss_dict = loss_fn(predictions, targets)

        # Loss might be inf or nan
        assert torch.isnan(loss_dict["loss"]) or torch.isinf(loss_dict["loss"])

    def test_zero_predictions_and_targets(self):
        """Test with all-zero predictions and targets."""
        batch_size = 4
        predictions = [torch.zeros(batch_size, 16, 64) for _ in range(3)]
        targets = [torch.zeros(batch_size, 16, 64) for _ in range(3)]

        loss_fn = CombinedLoss(num_hierarchies=3)
        loss_dict = loss_fn(predictions, targets)

        # JEPA loss should be near zero, VICReg might have variance penalty
        assert loss_dict["jepa_loss"].item() < 1e-5
        assert not torch.isnan(loss_dict["loss"])

    def test_very_large_values(self):
        """Test with very large values."""
        batch_size = 4
        predictions = [torch.randn(batch_size, 16, 64) * 1000 for _ in range(3)]
        targets = [torch.randn(batch_size, 16, 64) * 1000 for _ in range(3)]

        loss_fn = CombinedLoss(num_hierarchies=3)
        loss_dict = loss_fn(predictions, targets)

        # Should handle large values without overflow
        assert not torch.isnan(loss_dict["loss"])

    def test_different_hierarchy_patch_counts(self):
        """Test with different patch counts at different hierarchy levels."""
        batch_size = 8
        embed_dim = 128

        # Simulate hierarchical pooling with decreasing spatial resolution
        predictions = [
            torch.randn(batch_size, 196, embed_dim),  # 14x14
            torch.randn(batch_size, 49, embed_dim),  # 7x7
            torch.randn(batch_size, 16, embed_dim),  # 4x4
        ]
        targets = [
            torch.randn(batch_size, 196, embed_dim),
            torch.randn(batch_size, 49, embed_dim),
            torch.randn(batch_size, 16, embed_dim),
        ]

        loss_fn = CombinedLoss(num_hierarchies=3)
        loss_dict = loss_fn(predictions, targets)

        assert "loss" in loss_dict
        assert loss_dict["loss"].item() >= 0
        assert not torch.isnan(loss_dict["loss"])

    def test_normalize_embeddings_effect(self):
        """Test effect of normalize_embeddings parameter."""
        batch_size = 8
        predictions = [torch.randn(batch_size, 16, 64) * 10 for _ in range(3)]
        targets = [torch.randn(batch_size, 16, 64) * 10 for _ in range(3)]

        loss_fn_norm = CombinedLoss(normalize_embeddings=True)
        loss_fn_no_norm = CombinedLoss(normalize_embeddings=False)

        loss_dict_norm = loss_fn_norm(predictions, targets)
        loss_dict_no_norm = loss_fn_no_norm(predictions, targets)

        # Normalized version should have different loss
        assert not torch.allclose(
            loss_dict_norm["jepa_loss"],
            loss_dict_no_norm["jepa_loss"],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

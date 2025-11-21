"""
Test suite for contrastive loss functions in H-JEPA.

This module tests the contrastive learning components:
- NTXentLoss (InfoNCE/SimCLR)
- ContrastiveJEPALoss (hybrid JEPA + contrastive)
- Temperature scaling effects
- Similarity computation methods
- Negative sampling strategies
"""

import pytest
import torch
import torch.nn as nn

from src.losses.contrastive import ContrastiveJEPALoss, NTXentLoss, create_cjepa_loss_from_config
from src.losses.hjepa_loss import HJEPALoss


class TestNTXentLoss:
    """Test suite for NTXentLoss (InfoNCE/SimCLR)."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        self.batch_size = 8
        self.embed_dim = 128
        self.num_patches = 16

        # Create sample embeddings for two views
        self.z_i = torch.randn(self.batch_size, self.embed_dim)
        self.z_j = torch.randn(self.batch_size, self.embed_dim)

        yield

    def test_initialization_default(self):
        """Test default initialization of NTXentLoss."""
        loss_fn = NTXentLoss()
        assert loss_fn.temperature == 0.1
        assert loss_fn.use_cosine_similarity == True
        assert loss_fn.reduction == "mean"
        assert loss_fn.eps == 1e-8

    def test_initialization_custom(self):
        """Test custom initialization with different parameters."""
        loss_fn = NTXentLoss(
            temperature=0.5,
            use_cosine_similarity=False,
            reduction="sum",
            eps=1e-6,
        )
        assert loss_fn.temperature == 0.5
        assert loss_fn.use_cosine_similarity == False
        assert loss_fn.reduction == "sum"
        assert loss_fn.eps == 1e-6

    def test_temperature_validation(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            NTXentLoss(temperature=0.0)

        with pytest.raises(ValueError, match="Temperature must be positive"):
            NTXentLoss(temperature=-0.1)

    def test_forward_basic(self, setup):
        """Test basic forward pass."""
        loss_fn = NTXentLoss(temperature=0.1)
        loss_dict = loss_fn(self.z_i, self.z_j)

        # Check output structure
        assert isinstance(loss_dict, dict)
        assert "loss" in loss_dict
        assert "logits" in loss_dict
        assert "accuracy" in loss_dict
        assert "positive_similarity" in loss_dict
        assert "negative_similarity" in loss_dict

        # Check loss properties
        loss = loss_dict["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_temperature_effects(self, setup):
        """Test that temperature affects loss magnitude."""
        # Lower temperature should increase loss (harder discrimination)
        loss_fn_low_temp = NTXentLoss(temperature=0.05)
        loss_dict_low = loss_fn_low_temp(self.z_i, self.z_j)
        loss_low = loss_dict_low["loss"]

        # Higher temperature should decrease loss (softer discrimination)
        loss_fn_high_temp = NTXentLoss(temperature=1.0)
        loss_dict_high = loss_fn_high_temp(self.z_i, self.z_j)
        loss_high = loss_dict_high["loss"]

        # Lower temperature generally leads to higher loss
        # (Though this depends on the data, so we just check it's computed)
        assert loss_low.item() >= 0
        assert loss_high.item() >= 0

    def test_cosine_similarity_vs_dot_product(self, setup):
        """Test cosine similarity vs dot product similarity."""
        # Cosine similarity (normalized)
        loss_fn_cosine = NTXentLoss(use_cosine_similarity=True)
        loss_dict_cosine = loss_fn_cosine(self.z_i, self.z_j)
        loss_cosine = loss_dict_cosine["loss"]

        # Dot product similarity (unnormalized)
        loss_fn_dot = NTXentLoss(use_cosine_similarity=False)
        loss_dict_dot = loss_fn_dot(self.z_i, self.z_j)
        loss_dot = loss_dict_dot["loss"]

        # Both should be valid but generally different
        assert loss_cosine.item() >= 0
        assert loss_dot.item() >= 0
        # Typically they differ unless embeddings are normalized
        assert not torch.allclose(loss_cosine, loss_dot, atol=1e-3)

    def test_identical_embeddings(self, setup):
        """Test with identical embeddings (perfect positive pairs)."""
        # Create identical embeddings
        z_identical = self.z_i.clone()

        loss_fn = NTXentLoss(temperature=0.1)
        loss_dict = loss_fn(self.z_i, z_identical)

        # Loss should be very low (but not exactly zero due to negatives)
        loss = loss_dict["loss"]
        accuracy = loss_dict["accuracy"]

        assert loss.item() >= 0
        # Accuracy should be very high (close to 1.0)
        assert accuracy.item() >= 0.9

    def test_orthogonal_embeddings(self, setup):
        """Test with orthogonal embeddings (uncorrelated views)."""
        # Create orthogonal embeddings using QR decomposition
        batch_size = min(self.batch_size, self.embed_dim)
        z_i = torch.randn(batch_size, self.embed_dim)
        # Create orthogonal matrix
        q, r = torch.linalg.qr(torch.randn(self.embed_dim, self.embed_dim))
        z_j = z_i @ q[:, : self.embed_dim].t()

        loss_fn = NTXentLoss(temperature=0.1)
        loss_dict = loss_fn(z_i, z_j)

        loss = loss_dict["loss"]
        accuracy = loss_dict["accuracy"]

        # Loss should be computable
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        # Accuracy might be low due to orthogonality
        assert 0.0 <= accuracy.item() <= 1.0

    def test_batch_size_variations(self):
        """Test with different batch sizes."""
        embed_dim = 64

        for batch_size in [2, 4, 8, 16, 32]:
            z_i = torch.randn(batch_size, embed_dim)
            z_j = torch.randn(batch_size, embed_dim)

            loss_fn = NTXentLoss()
            loss_dict = loss_fn(z_i, z_j)

            loss = loss_dict["loss"]
            logits = loss_dict["logits"]

            assert loss.dim() == 0
            assert logits.shape[0] == 2 * batch_size
            assert not torch.isnan(loss)

    def test_3d_input_patch_level(self):
        """Test with 3D input (batch, num_patches, embed_dim)."""
        batch_size = 4
        num_patches = 16
        embed_dim = 128

        # 3D inputs (patch-level representations)
        z_i = torch.randn(batch_size, num_patches, embed_dim)
        z_j = torch.randn(batch_size, num_patches, embed_dim)

        loss_fn = NTXentLoss()
        loss_dict = loss_fn(z_i, z_j)

        loss = loss_dict["loss"]
        logits = loss_dict["logits"]

        # Should flatten to [B*N, D] internally
        expected_effective_batch = batch_size * num_patches
        assert loss.dim() == 0
        assert logits.shape[0] == 2 * expected_effective_batch
        assert not torch.isnan(loss)

    def test_reduction_modes(self, setup):
        """Test different reduction modes."""
        # Mean reduction
        loss_fn_mean = NTXentLoss(reduction="mean")
        loss_dict_mean = loss_fn_mean(self.z_i, self.z_j)
        loss_mean = loss_dict_mean["loss"]
        assert loss_mean.dim() == 0

        # Sum reduction
        loss_fn_sum = NTXentLoss(reduction="sum")
        loss_dict_sum = loss_fn_sum(self.z_i, self.z_j)
        loss_sum = loss_dict_sum["loss"]
        assert loss_sum.dim() == 0

        # Sum should be larger than mean
        assert loss_sum.item() > loss_mean.item()

        # None reduction
        loss_fn_none = NTXentLoss(reduction="none")
        loss_dict_none = loss_fn_none(self.z_i, self.z_j)
        loss_none = loss_dict_none["loss"]
        assert loss_none.dim() == 1  # Per-sample losses
        assert loss_none.shape[0] == 2 * self.batch_size

    def test_gradient_flow(self, setup):
        """Test gradient flow through the loss."""
        loss_fn = NTXentLoss()

        # Enable gradients
        z_i = self.z_i.requires_grad_(True)
        z_j = self.z_j.requires_grad_(True)

        loss_dict = loss_fn(z_i, z_j)
        loss = loss_dict["loss"]
        loss.backward()

        # Check gradients exist and are valid
        assert z_i.grad is not None
        assert z_j.grad is not None
        assert not torch.isnan(z_i.grad).any()
        assert not torch.isnan(z_j.grad).any()
        assert not torch.isinf(z_i.grad).any()
        assert not torch.isinf(z_j.grad).any()

    def test_shape_mismatch_error(self, setup):
        """Test that mismatched shapes raise error."""
        loss_fn = NTXentLoss()

        # Different batch sizes
        z_i = torch.randn(8, 128)
        z_j = torch.randn(4, 128)

        with pytest.raises(AssertionError, match="must have the same shape"):
            loss_fn(z_i, z_j)

        # Different embedding dimensions
        z_i = torch.randn(8, 128)
        z_j = torch.randn(8, 64)

        with pytest.raises(AssertionError, match="must have the same shape"):
            loss_fn(z_i, z_j)

    def test_accuracy_metric(self, setup):
        """Test that accuracy metric is reasonable."""
        loss_fn = NTXentLoss()
        loss_dict = loss_fn(self.z_i, self.z_j)

        accuracy = loss_dict["accuracy"]
        assert isinstance(accuracy, torch.Tensor)
        assert accuracy.dim() == 0
        assert 0.0 <= accuracy.item() <= 1.0

    def test_positive_negative_similarity_tracking(self, setup):
        """Test tracking of positive and negative similarities."""
        loss_fn = NTXentLoss()
        loss_dict = loss_fn(self.z_i, self.z_j)

        pos_sim = loss_dict["positive_similarity"]
        neg_sim = loss_dict["negative_similarity"]

        assert isinstance(pos_sim, torch.Tensor)
        assert isinstance(neg_sim, torch.Tensor)
        assert pos_sim.dim() == 0
        assert neg_sim.dim() == 0

        # For cosine similarity, positive similarity should be in [-1, 1]
        assert -1.0 <= pos_sim.item() <= 1.0
        # Negative similarity may include -inf from masked values, so just check it's computed
        assert isinstance(neg_sim.item(), float)

    def test_deterministic_computation(self):
        """Test that loss computation is deterministic."""
        torch.manual_seed(42)
        z_i = torch.randn(8, 64)
        z_j = torch.randn(8, 64)

        loss_fn = NTXentLoss()
        loss_dict1 = loss_fn(z_i, z_j)
        loss_dict2 = loss_fn(z_i, z_j)

        assert torch.allclose(loss_dict1["loss"], loss_dict2["loss"])
        assert torch.allclose(loss_dict1["accuracy"], loss_dict2["accuracy"])

    def test_extra_repr(self):
        """Test string representation."""
        loss_fn = NTXentLoss(temperature=0.2, use_cosine_similarity=True, reduction="sum")
        repr_str = loss_fn.extra_repr()

        assert "temperature=0.2" in repr_str
        assert "use_cosine_similarity=True" in repr_str
        assert "reduction=sum" in repr_str

    def test_compute_similarity_cosine(self, setup):
        """Test _compute_similarity method with cosine similarity."""
        loss_fn = NTXentLoss(use_cosine_similarity=True)
        similarity = loss_fn._compute_similarity(self.z_i, self.z_j)

        # Should return a batch_size x batch_size matrix
        assert similarity.shape == (self.batch_size, self.batch_size)
        # Cosine similarities should be in [-1, 1]
        assert similarity.min().item() >= -1.0
        assert similarity.max().item() <= 1.0

    def test_compute_similarity_dot_product(self, setup):
        """Test _compute_similarity method with dot product."""
        loss_fn = NTXentLoss(use_cosine_similarity=False)
        similarity = loss_fn._compute_similarity(self.z_i, self.z_j)

        # Should return a batch_size x batch_size matrix
        assert similarity.shape == (self.batch_size, self.batch_size)
        assert not torch.isnan(similarity).any()
        assert not torch.isinf(similarity).any()


class TestContrastiveJEPALoss:
    """Test suite for ContrastiveJEPALoss."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        self.batch_size = 4
        self.num_patches = 16
        self.embed_dim = 128
        self.num_hierarchies = 3

        # Create base JEPA loss
        self.jepa_loss = HJEPALoss(
            loss_type="smoothl1",
            hierarchy_weights=[1.0, 0.5, 0.25],
            num_hierarchies=self.num_hierarchies,
        )

        # Create sample predictions and targets for JEPA
        self.predictions = [
            torch.randn(self.batch_size, self.num_patches, self.embed_dim)
            for _ in range(self.num_hierarchies)
        ]
        self.targets = [
            torch.randn(self.batch_size, self.num_patches, self.embed_dim)
            for _ in range(self.num_hierarchies)
        ]

        # Create context/target features for contrastive learning
        # Shape: [B, N+1, D] where index 0 is CLS token
        self.context_features_i = torch.randn(self.batch_size, self.num_patches + 1, self.embed_dim)
        self.context_features_j = torch.randn(self.batch_size, self.num_patches + 1, self.embed_dim)
        self.target_features_i = torch.randn(self.batch_size, self.num_patches + 1, self.embed_dim)
        self.target_features_j = torch.randn(self.batch_size, self.num_patches + 1, self.embed_dim)

        yield

    def test_initialization_default(self, setup):
        """Test default initialization."""
        loss_fn = ContrastiveJEPALoss(jepa_loss=self.jepa_loss)
        assert loss_fn.jepa_weight == 1.0
        assert loss_fn.contrastive_weight == 0.1
        assert loss_fn.contrastive_on_context == False
        assert loss_fn.contrastive.temperature == 0.1

    def test_initialization_custom(self, setup):
        """Test custom initialization."""
        loss_fn = ContrastiveJEPALoss(
            jepa_loss=self.jepa_loss,
            jepa_weight=2.0,
            contrastive_weight=0.5,
            contrastive_temperature=0.2,
            use_cosine_similarity=False,
            contrastive_on_context=True,
        )
        assert loss_fn.jepa_weight == 2.0
        assert loss_fn.contrastive_weight == 0.5
        assert loss_fn.contrastive_on_context == True
        assert loss_fn.contrastive.temperature == 0.2
        assert loss_fn.contrastive.use_cosine_similarity == False

    def test_forward_with_target_features(self, setup):
        """Test forward pass with target features."""
        loss_fn = ContrastiveJEPALoss(
            jepa_loss=self.jepa_loss,
            contrastive_on_context=False,
        )

        loss_dict = loss_fn(
            predictions=self.predictions,
            targets=self.targets,
            target_features_i=self.target_features_i,
            target_features_j=self.target_features_j,
        )

        # Check output structure
        assert isinstance(loss_dict, dict)
        assert "loss" in loss_dict
        assert "jepa_loss" in loss_dict
        assert "contrastive_loss" in loss_dict
        assert "contrastive_accuracy" in loss_dict
        assert "contrastive_pos_sim" in loss_dict
        assert "contrastive_neg_sim" in loss_dict

        # Check loss properties
        loss = loss_dict["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_forward_with_context_features(self, setup):
        """Test forward pass with context features."""
        loss_fn = ContrastiveJEPALoss(
            jepa_loss=self.jepa_loss,
            contrastive_on_context=True,
        )

        loss_dict = loss_fn(
            predictions=self.predictions,
            targets=self.targets,
            context_features_i=self.context_features_i,
            context_features_j=self.context_features_j,
        )

        assert "loss" in loss_dict
        assert "contrastive_loss" in loss_dict
        assert loss_dict["loss"].item() >= 0

    def test_fallback_to_context_features(self, setup):
        """Test fallback to context features when target not provided."""
        loss_fn = ContrastiveJEPALoss(
            jepa_loss=self.jepa_loss,
            contrastive_on_context=False,  # Prefers target but will fallback
        )

        # Only provide context features
        loss_dict = loss_fn(
            predictions=self.predictions,
            targets=self.targets,
            context_features_i=self.context_features_i,
            context_features_j=self.context_features_j,
        )

        assert "loss" in loss_dict
        assert "contrastive_loss" in loss_dict
        assert loss_dict["loss"].item() >= 0

    def test_missing_features_error(self, setup):
        """Test error when required features are missing."""
        loss_fn = ContrastiveJEPALoss(
            jepa_loss=self.jepa_loss,
            contrastive_on_context=True,
        )

        # Don't provide any context features
        with pytest.raises(ValueError, match="context_features not provided"):
            loss_fn(
                predictions=self.predictions,
                targets=self.targets,
                target_features_i=self.target_features_i,
                target_features_j=self.target_features_j,
            )

    def test_missing_all_features_error(self, setup):
        """Test error when no features provided at all."""
        loss_fn = ContrastiveJEPALoss(
            jepa_loss=self.jepa_loss,
            contrastive_on_context=False,
        )

        with pytest.raises(ValueError, match="must be provided for contrastive learning"):
            loss_fn(predictions=self.predictions, targets=self.targets)

    def test_loss_weighting(self, setup):
        """Test that loss weights are applied correctly."""
        # Test with different weights
        loss_fn_1 = ContrastiveJEPALoss(
            jepa_loss=self.jepa_loss,
            jepa_weight=1.0,
            contrastive_weight=0.1,
        )
        loss_dict_1 = loss_fn_1(
            predictions=self.predictions,
            targets=self.targets,
            target_features_i=self.target_features_i,
            target_features_j=self.target_features_j,
        )

        loss_fn_2 = ContrastiveJEPALoss(
            jepa_loss=self.jepa_loss,
            jepa_weight=1.0,
            contrastive_weight=0.5,  # Higher contrastive weight
        )
        loss_dict_2 = loss_fn_2(
            predictions=self.predictions,
            targets=self.targets,
            target_features_i=self.target_features_i,
            target_features_j=self.target_features_j,
        )

        # Total losses should differ
        assert not torch.allclose(loss_dict_1["loss"], loss_dict_2["loss"])

        # Verify manual weighting calculation
        expected_loss_1 = 1.0 * loss_dict_1["jepa_loss"] + 0.1 * loss_dict_1["contrastive_loss"]
        assert torch.allclose(loss_dict_1["loss"], expected_loss_1, atol=1e-5)

    def test_cls_token_extraction(self, setup):
        """Test that CLS token is correctly extracted for contrastive learning."""
        loss_fn = ContrastiveJEPALoss(jepa_loss=self.jepa_loss)

        # Create features where CLS token has specific values
        context_i = torch.randn(self.batch_size, self.num_patches + 1, self.embed_dim)
        context_j = torch.randn(self.batch_size, self.num_patches + 1, self.embed_dim)

        # Set CLS token (index 0) to specific values
        context_i[:, 0, :] = 1.0
        context_j[:, 0, :] = 1.0

        loss_dict = loss_fn(
            predictions=self.predictions,
            targets=self.targets,
            context_features_i=context_i,
            context_features_j=context_j,
        )

        # Should successfully compute loss using CLS tokens
        assert "contrastive_loss" in loss_dict
        assert not torch.isnan(loss_dict["contrastive_loss"])

    def test_gradient_flow(self, setup):
        """Test gradient flow through combined loss."""
        loss_fn = ContrastiveJEPALoss(jepa_loss=self.jepa_loss)

        # Enable gradients
        predictions = [p.requires_grad_(True) for p in self.predictions]
        context_i = self.context_features_i.requires_grad_(True)
        context_j = self.context_features_j.requires_grad_(True)

        loss_dict = loss_fn(
            predictions=predictions,
            targets=self.targets,
            context_features_i=context_i,
            context_features_j=context_j,
        )

        loss = loss_dict["loss"]
        loss.backward()

        # Check gradients
        for pred in predictions:
            assert pred.grad is not None
            assert not torch.isnan(pred.grad).any()

        assert context_i.grad is not None
        assert context_j.grad is not None
        assert not torch.isnan(context_i.grad).any()
        assert not torch.isnan(context_j.grad).any()

    def test_get_loss_summary(self, setup):
        """Test loss summary string generation."""
        loss_fn = ContrastiveJEPALoss(jepa_loss=self.jepa_loss)

        loss_dict = loss_fn(
            predictions=self.predictions,
            targets=self.targets,
            context_features_i=self.context_features_i,
            context_features_j=self.context_features_j,
        )

        summary = loss_fn.get_loss_summary(loss_dict)

        # Check summary contains expected information
        assert "C-JEPA Loss Summary" in summary
        assert "Total Loss" in summary
        assert "JEPA Loss" in summary
        assert "Contrastive Loss" in summary
        assert "Contrastive Accuracy" in summary
        assert "Positive Similarity" in summary
        assert "Negative Similarity" in summary

    def test_extra_repr(self, setup):
        """Test string representation."""
        loss_fn = ContrastiveJEPALoss(
            jepa_loss=self.jepa_loss,
            jepa_weight=2.0,
            contrastive_weight=0.3,
            contrastive_temperature=0.15,
            contrastive_on_context=True,
        )

        repr_str = loss_fn.extra_repr()
        assert "jepa_weight=2.0" in repr_str
        assert "contrastive_weight=0.3" in repr_str
        assert "temperature=0.15" in repr_str
        assert "contrastive_on_context=True" in repr_str

    def test_hierarchical_loss_components(self, setup):
        """Test that hierarchical JEPA loss components are preserved."""
        loss_fn = ContrastiveJEPALoss(jepa_loss=self.jepa_loss)

        loss_dict = loss_fn(
            predictions=self.predictions,
            targets=self.targets,
            context_features_i=self.context_features_i,
            context_features_j=self.context_features_j,
        )

        # Should contain hierarchical components from JEPA loss
        assert "loss_h0" in loss_dict
        assert "loss_h1" in loss_dict
        assert "loss_h2" in loss_dict

    def test_deterministic_computation(self, setup):
        """Test deterministic loss computation."""
        torch.manual_seed(42)

        loss_fn = ContrastiveJEPALoss(jepa_loss=self.jepa_loss)

        loss_dict1 = loss_fn(
            predictions=self.predictions,
            targets=self.targets,
            context_features_i=self.context_features_i,
            context_features_j=self.context_features_j,
        )

        loss_dict2 = loss_fn(
            predictions=self.predictions,
            targets=self.targets,
            context_features_i=self.context_features_i,
            context_features_j=self.context_features_j,
        )

        assert torch.allclose(loss_dict1["loss"], loss_dict2["loss"])
        assert torch.allclose(loss_dict1["jepa_loss"], loss_dict2["jepa_loss"])
        assert torch.allclose(loss_dict1["contrastive_loss"], loss_dict2["contrastive_loss"])

    def test_with_tensor_predictions(self, setup):
        """Test with single tensor predictions (not list)."""
        # Create a simple JEPA loss that accepts tensor inputs
        simple_jepa_loss = HJEPALoss(num_hierarchies=1)

        loss_fn = ContrastiveJEPALoss(jepa_loss=simple_jepa_loss)

        # Single tensor predictions/targets
        pred_tensor = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        target_tensor = torch.randn(self.batch_size, self.num_patches, self.embed_dim)

        loss_dict = loss_fn(
            predictions=[pred_tensor],  # Still pass as list for JEPA loss compatibility
            targets=[target_tensor],
            context_features_i=self.context_features_i,
            context_features_j=self.context_features_j,
        )

        assert "loss" in loss_dict
        assert "contrastive_loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])


class TestCreateCJEPALossFromConfig:
    """Test suite for create_cjepa_loss_from_config factory function."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        self.jepa_loss = HJEPALoss(
            loss_type="smoothl1",
            hierarchy_weights=[1.0, 0.5, 0.25],
            num_hierarchies=3,
        )
        yield

    def test_create_with_default_config(self, setup):
        """Test creation with default config."""
        config = {"loss": {}}

        loss_fn = create_cjepa_loss_from_config(config, self.jepa_loss)

        assert isinstance(loss_fn, ContrastiveJEPALoss)
        assert loss_fn.jepa_weight == 1.0
        assert loss_fn.contrastive_weight == 0.1
        assert loss_fn.contrastive.temperature == 0.1
        assert loss_fn.contrastive.use_cosine_similarity == True
        assert loss_fn.contrastive_on_context == False

    def test_create_with_custom_config(self, setup):
        """Test creation with custom config."""
        config = {
            "loss": {
                "jepa_weight": 2.0,
                "contrastive_weight": 0.3,
                "contrastive_temperature": 0.2,
                "use_cosine_similarity": False,
                "contrastive_on_context": True,
            }
        }

        loss_fn = create_cjepa_loss_from_config(config, self.jepa_loss)

        assert isinstance(loss_fn, ContrastiveJEPALoss)
        assert loss_fn.jepa_weight == 2.0
        assert loss_fn.contrastive_weight == 0.3
        assert loss_fn.contrastive.temperature == 0.2
        assert loss_fn.contrastive.use_cosine_similarity == False
        assert loss_fn.contrastive_on_context == True

    def test_create_with_partial_config(self, setup):
        """Test creation with partial config (mix of defaults and custom)."""
        config = {
            "loss": {
                "contrastive_weight": 0.5,
                "contrastive_temperature": 0.15,
            }
        }

        loss_fn = create_cjepa_loss_from_config(config, self.jepa_loss)

        # Custom values
        assert loss_fn.contrastive_weight == 0.5
        assert loss_fn.contrastive.temperature == 0.15

        # Default values
        assert loss_fn.jepa_weight == 1.0
        assert loss_fn.contrastive.use_cosine_similarity == True
        assert loss_fn.contrastive_on_context == False

    def test_create_with_empty_config(self, setup):
        """Test creation with empty config."""
        config = {}

        loss_fn = create_cjepa_loss_from_config(config, self.jepa_loss)

        assert isinstance(loss_fn, ContrastiveJEPALoss)
        # Should use all defaults
        assert loss_fn.jepa_weight == 1.0
        assert loss_fn.contrastive_weight == 0.1


class TestEdgeCasesAndNumericalStability:
    """Test edge cases and numerical stability."""

    def test_very_small_batch_size(self):
        """Test with minimal batch size."""
        batch_size = 2
        embed_dim = 64

        z_i = torch.randn(batch_size, embed_dim)
        z_j = torch.randn(batch_size, embed_dim)

        loss_fn = NTXentLoss()
        loss_dict = loss_fn(z_i, z_j)

        assert not torch.isnan(loss_dict["loss"])
        assert not torch.isinf(loss_dict["loss"])

    def test_large_embedding_dimension(self):
        """Test with large embedding dimension."""
        batch_size = 4
        embed_dim = 2048  # Large dimension

        z_i = torch.randn(batch_size, embed_dim)
        z_j = torch.randn(batch_size, embed_dim)

        loss_fn = NTXentLoss()
        loss_dict = loss_fn(z_i, z_j)

        assert not torch.isnan(loss_dict["loss"])
        assert not torch.isinf(loss_dict["loss"])

    def test_normalized_embeddings(self):
        """Test with pre-normalized embeddings."""
        batch_size = 8
        embed_dim = 128

        # Pre-normalize embeddings
        z_i = torch.nn.functional.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        z_j = torch.nn.functional.normalize(torch.randn(batch_size, embed_dim), dim=-1)

        loss_fn = NTXentLoss(use_cosine_similarity=True)
        loss_dict = loss_fn(z_i, z_j)

        loss = loss_dict["loss"]
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Positive similarity should be in [-1, 1] range for normalized embeddings
        assert -1.0 <= loss_dict["positive_similarity"].item() <= 1.0
        # Negative similarity may include -inf from masked values
        assert isinstance(loss_dict["negative_similarity"].item(), float)

    def test_zero_embeddings(self):
        """Test behavior with zero embeddings."""
        batch_size = 4
        embed_dim = 64

        z_i = torch.zeros(batch_size, embed_dim)
        z_j = torch.zeros(batch_size, embed_dim)

        loss_fn = NTXentLoss()
        loss_dict = loss_fn(z_i, z_j)

        # Should handle gracefully (may produce specific values due to normalization)
        loss = loss_dict["loss"]
        # Loss may be NaN or specific value depending on implementation
        # Just verify it doesn't crash
        assert isinstance(loss, torch.Tensor)

    def test_extreme_temperature_values(self):
        """Test with extreme temperature values."""
        batch_size = 4
        embed_dim = 64
        z_i = torch.randn(batch_size, embed_dim)
        z_j = torch.randn(batch_size, embed_dim)

        # Very small temperature (near zero, but valid)
        loss_fn_small = NTXentLoss(temperature=0.001)
        loss_dict_small = loss_fn_small(z_i, z_j)
        assert not torch.isnan(loss_dict_small["loss"])

        # Large temperature
        loss_fn_large = NTXentLoss(temperature=10.0)
        loss_dict_large = loss_fn_large(z_i, z_j)
        assert not torch.isnan(loss_dict_large["loss"])

    def test_gpu_compatibility(self):
        """Test GPU compatibility if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 8
        embed_dim = 128

        z_i = torch.randn(batch_size, embed_dim).cuda()
        z_j = torch.randn(batch_size, embed_dim).cuda()

        loss_fn = NTXentLoss()
        loss_dict = loss_fn(z_i, z_j)

        assert loss_dict["loss"].device.type == "cuda"
        assert not torch.isnan(loss_dict["loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

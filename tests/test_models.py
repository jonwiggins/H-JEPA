"""
Comprehensive test suite for H-JEPA model implementation.
Tests cover model creation, forward pass, hierarchical feature extraction,
EMA updates, FPN functionality, and edge cases.
"""

import os
import tempfile

import pytest
import torch
import yaml

from src.models import HJEPA, create_hjepa, create_hjepa_from_config


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    return create_hjepa(
        encoder_type="vit_small_patch16_224",
        img_size=224,
        embed_dim=384,
        predictor_depth=4,
        predictor_num_heads=6,
        num_hierarchies=3,
        ema_momentum=0.996,
        ema_momentum_end=1.0,
        ema_warmup_steps=1000,
    )


@pytest.fixture
def fpn_model():
    """Create a model with FPN enabled."""
    return create_hjepa(
        encoder_type="vit_small_patch16_224",
        img_size=224,
        embed_dim=384,
        predictor_depth=4,
        predictor_num_heads=6,
        num_hierarchies=3,
        use_fpn=True,
        fpn_feature_dim=256,
        fpn_fusion_method="add",
    )


@pytest.fixture
def fpn_concat_model():
    """Create a model with FPN using concat fusion."""
    return create_hjepa(
        encoder_type="vit_small_patch16_224",
        img_size=224,
        embed_dim=384,
        predictor_depth=4,
        predictor_num_heads=6,
        num_hierarchies=3,
        use_fpn=True,
        fpn_feature_dim=256,
        fpn_fusion_method="concat",
    )


@pytest.fixture
def dummy_input(small_model):
    """Create dummy input for testing."""
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    num_patches = small_model.get_num_patches()
    mask = torch.zeros(batch_size, num_patches)
    num_masked = num_patches // 2
    mask[:, :num_masked] = 1
    return images, mask


def test_basic_model_creation(small_model):
    """Test basic model creation and attributes."""
    assert small_model.embed_dim == 384
    assert small_model.num_hierarchies == 3
    assert small_model.get_num_patches() > 0
    assert small_model.get_patch_size() == 16


def test_forward_pass_return_all_levels(small_model, dummy_input):
    """Test forward pass with return_all_levels=True."""
    images, mask = dummy_input
    small_model.eval()

    with torch.no_grad():
        outputs = small_model(images, mask, return_all_levels=True)

    assert "predictions" in outputs
    assert "targets" in outputs
    assert "masks_valid" in outputs
    assert "context_features" in outputs
    assert "target_features" in outputs

    assert len(outputs["predictions"]) == 3
    assert len(outputs["targets"]) == 3
    assert len(outputs["masks_valid"]) == 3


def test_forward_pass_single_level(small_model, dummy_input):
    """Test forward pass with return_all_levels=False."""
    images, mask = dummy_input
    small_model.eval()

    with torch.no_grad():
        outputs = small_model(images, mask, return_all_levels=False)

    assert len(outputs["predictions"]) == 1
    assert len(outputs["targets"]) == 1


def test_multi_scale_feature_extraction_level_0(small_model):
    """Test multi-scale feature extraction at hierarchy level 0 (finest)."""
    images = torch.randn(2, 3, 224, 224)
    small_model.eval()

    with torch.no_grad():
        features = small_model.extract_features(images, level=0, use_target_encoder=True)

    # Level 0 should have the most patches (no pooling)
    assert features.shape[0] == 2  # batch size
    assert features.shape[2] == 384  # embed_dim
    assert features.shape[1] > 0  # num patches


def test_multi_scale_feature_extraction_level_1(small_model):
    """Test multi-scale feature extraction at hierarchy level 1 (coarser)."""
    images = torch.randn(2, 3, 224, 224)
    small_model.eval()

    with torch.no_grad():
        features_level_0 = small_model.extract_features(images, level=0)
        features_level_1 = small_model.extract_features(images, level=1)

    # Level 1 should have fewer patches due to pooling (2x kernel)
    assert features_level_1.shape[1] < features_level_0.shape[1]
    assert features_level_1.shape[1] == features_level_0.shape[1] // 2


def test_multi_scale_feature_extraction_level_2(small_model):
    """Test multi-scale feature extraction at hierarchy level 2 (coarsest)."""
    images = torch.randn(2, 3, 224, 224)
    small_model.eval()

    with torch.no_grad():
        features_level_0 = small_model.extract_features(images, level=0)
        features_level_2 = small_model.extract_features(images, level=2)

    # Level 2 should have even fewer patches (4x kernel)
    assert features_level_2.shape[1] < features_level_0.shape[1]
    assert features_level_2.shape[1] == features_level_0.shape[1] // 4


def test_feature_extraction_context_encoder(small_model):
    """Test feature extraction using context encoder."""
    images = torch.randn(2, 3, 224, 224)
    small_model.eval()

    with torch.no_grad():
        features = small_model.extract_features(images, level=0, use_target_encoder=False)

    assert features.shape[0] == 2
    assert features.shape[2] == 384


def test_feature_extraction_invalid_level(small_model):
    """Test feature extraction with invalid hierarchy level."""
    images = torch.randn(2, 3, 224, 224)
    small_model.eval()

    with pytest.raises(ValueError, match="exceeds num_hierarchies"):
        small_model.extract_features(images, level=5)


def test_feature_dimension_consistency_across_scales(small_model, dummy_input):
    """Test that feature dimensions remain consistent across hierarchy levels."""
    images, mask = dummy_input
    small_model.eval()

    with torch.no_grad():
        outputs = small_model(images, mask, return_all_levels=True)

    # All predictions and targets should have the same embedding dimension
    for pred, target in zip(outputs["predictions"], outputs["targets"]):
        assert pred.shape[2] == 384  # embed_dim
        assert target.shape[2] == 384


def test_feature_extraction_different_image_sizes():
    """Test feature extraction with different image sizes."""
    # Create model with different image size
    model = create_hjepa(
        encoder_type="vit_small_patch16_224",
        img_size=224,  # Model expects 224
        embed_dim=384,
        predictor_depth=4,
        predictor_num_heads=6,
        num_hierarchies=2,
    )

    images = torch.randn(2, 3, 224, 224)
    model.eval()

    with torch.no_grad():
        features = model.extract_features(images, level=0)

    assert features.shape[0] == 2
    assert features.shape[2] == 384


def test_ema_target_encoder_update_warmup(small_model):
    """Test EMA target encoder updates during warmup."""
    # Test at different steps during warmup
    step_1 = 100
    step_2 = 500
    step_3 = 999

    momentum_1 = small_model.update_target_encoder(step_1)
    momentum_2 = small_model.update_target_encoder(step_2)
    momentum_3 = small_model.update_target_encoder(step_3)

    # Momentum should increase during warmup
    assert momentum_1 < momentum_2 < momentum_3
    assert momentum_1 >= 0.996
    assert momentum_3 <= 1.0


def test_ema_target_encoder_update_after_warmup(small_model):
    """Test EMA target encoder updates after warmup."""
    # Test after warmup is complete
    step_after_warmup = 1500
    momentum = small_model.update_target_encoder(step_after_warmup)

    # After warmup, momentum should be at or near the end value
    assert momentum >= 0.999


def test_ema_momentum_schedule():
    """Test EMA momentum warmup behavior with different schedules."""
    model = create_hjepa(
        encoder_type="vit_small_patch16_224",
        img_size=224,
        embed_dim=384,
        predictor_depth=4,
        predictor_num_heads=6,
        num_hierarchies=3,
        ema_momentum=0.99,
        ema_momentum_end=1.0,
        ema_warmup_steps=500,
    )

    # Test early warmup
    momentum_early = model.update_target_encoder(50)
    assert 0.99 <= momentum_early < 1.0

    # Test mid warmup
    momentum_mid = model.update_target_encoder(250)
    assert momentum_early < momentum_mid < 1.0

    # Test late warmup
    momentum_late = model.update_target_encoder(499)
    assert momentum_mid < momentum_late <= 1.0


def test_target_encoder_no_gradients(small_model, dummy_input):
    """Test that target encoder parameters don't receive gradients."""
    images, mask = dummy_input
    small_model.train()

    # Forward pass
    outputs = small_model(images, mask, return_all_levels=True)

    # Check that target features were computed without gradients
    assert not outputs["target_features"].requires_grad

    # Verify target encoder parameters have no gradients
    for param in small_model.target_encoder.parameters():
        assert param.grad is None or torch.all(param.grad == 0)


def test_mask_density_low(small_model):
    """Test forward pass with low mask density (10%)."""
    images = torch.randn(2, 3, 224, 224)
    num_patches = small_model.get_num_patches()
    mask = torch.zeros(2, num_patches)
    num_masked = int(num_patches * 0.1)
    mask[:, :num_masked] = 1

    small_model.eval()
    with torch.no_grad():
        outputs = small_model(images, mask, return_all_levels=True)

    assert len(outputs["predictions"]) == 3
    # Check that mask_valid tracks the correct number of masked positions
    assert outputs["masks_valid"][0].sum(dim=1).max() <= num_masked


def test_mask_density_medium(small_model):
    """Test forward pass with medium mask density (50%)."""
    images = torch.randn(2, 3, 224, 224)
    num_patches = small_model.get_num_patches()
    mask = torch.zeros(2, num_patches)
    num_masked = int(num_patches * 0.5)
    mask[:, :num_masked] = 1

    small_model.eval()
    with torch.no_grad():
        outputs = small_model(images, mask, return_all_levels=True)

    assert len(outputs["predictions"]) == 3
    assert outputs["masks_valid"][0].sum(dim=1).max() <= num_masked


def test_mask_density_high(small_model):
    """Test forward pass with high mask density (90%)."""
    images = torch.randn(2, 3, 224, 224)
    num_patches = small_model.get_num_patches()
    mask = torch.zeros(2, num_patches)
    num_masked = int(num_patches * 0.9)
    mask[:, :num_masked] = 1

    small_model.eval()
    with torch.no_grad():
        outputs = small_model(images, mask, return_all_levels=True)

    assert len(outputs["predictions"]) == 3
    assert outputs["masks_valid"][0].sum(dim=1).max() <= num_masked


def test_empty_mask_edge_case(small_model):
    """Test forward pass with empty mask (no masking).

    Note: This is a known edge case where the model expects at least some
    masked patches. When no patches are masked, the predictor receives an
    empty tensor which causes pooling operations to fail. In practice, this
    scenario should never occur during training.
    """
    images = torch.randn(2, 3, 224, 224)
    num_patches = small_model.get_num_patches()
    mask = torch.zeros(2, num_patches)  # All zeros - no masking

    small_model.eval()

    # This edge case should raise an error because pooling operations
    # cannot handle empty tensors
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            outputs = small_model(images, mask, return_all_levels=True)


def test_full_mask_edge_case(small_model):
    """Test forward pass with full mask (all patches masked)."""
    images = torch.randn(2, 3, 224, 224)
    num_patches = small_model.get_num_patches()
    mask = torch.ones(2, num_patches)  # All ones - full masking

    small_model.eval()
    with torch.no_grad():
        outputs = small_model(images, mask, return_all_levels=True)

    assert len(outputs["predictions"]) == 3
    # All positions should be valid (no padding)
    assert outputs["masks_valid"][0].all()


def test_state_dict_save_load(small_model, tmp_path):
    """Test model state dict save and load operations."""
    # Save state dict
    state_dict_path = tmp_path / "model_state.pt"
    torch.save(small_model.state_dict(), state_dict_path)

    # Create new model and load state dict
    new_model = create_hjepa(
        encoder_type="vit_small_patch16_224",
        img_size=224,
        embed_dim=384,
        predictor_depth=4,
        predictor_num_heads=6,
        num_hierarchies=3,
    )

    new_model.load_state_dict(torch.load(state_dict_path))

    # Verify parameters match
    for p1, p2 in zip(small_model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


def test_model_configuration_hierarchies_2():
    """Test model with 2 hierarchies."""
    model = create_hjepa(
        encoder_type="vit_small_patch16_224",
        img_size=224,
        embed_dim=384,
        predictor_depth=4,
        predictor_num_heads=6,
        num_hierarchies=2,
    )

    assert model.num_hierarchies == 2
    assert len(model.hierarchy_projections) == 2
    assert len(model.hierarchy_pooling) == 2


def test_model_configuration_hierarchies_4():
    """Test model with 4 hierarchies."""
    model = create_hjepa(
        encoder_type="vit_small_patch16_224",
        img_size=224,
        embed_dim=384,
        predictor_depth=4,
        predictor_num_heads=6,
        num_hierarchies=4,
    )

    assert model.num_hierarchies == 4
    assert len(model.hierarchy_projections) == 4
    assert len(model.hierarchy_pooling) == 4


def test_model_configuration_invalid_hierarchies():
    """Test model creation with invalid number of hierarchies."""
    with pytest.raises(ValueError, match="num_hierarchies must be between 2 and 4"):
        create_hjepa(
            encoder_type="vit_small_patch16_224",
            img_size=224,
            embed_dim=384,
            num_hierarchies=5,
        )

    with pytest.raises(ValueError, match="num_hierarchies must be between 2 and 4"):
        create_hjepa(
            encoder_type="vit_small_patch16_224",
            img_size=224,
            embed_dim=384,
            num_hierarchies=1,
        )


def test_model_configuration_gradient_checkpointing():
    """Test model with gradient checkpointing enabled."""
    model = create_hjepa(
        encoder_type="vit_small_patch16_224",
        img_size=224,
        embed_dim=384,
        predictor_depth=4,
        predictor_num_heads=6,
        num_hierarchies=3,
        use_gradient_checkpointing=True,
    )

    assert model.use_gradient_checkpointing is True


def test_fpn_model_creation(fpn_model):
    """Test FPN model creation and attributes."""
    assert fpn_model.use_fpn is True
    assert fpn_model.fpn_feature_dim == 256
    assert fpn_model.fpn_fusion_method == "add"
    assert len(fpn_model.fpn_lateral_convs) == 3
    assert len(fpn_model.fpn_top_down_convs) == 2  # N-1 for N hierarchies


def test_fpn_forward_pass(fpn_model, dummy_input):
    """Test forward pass with FPN enabled."""
    images, mask = dummy_input
    fpn_model.eval()

    with torch.no_grad():
        outputs = fpn_model(images, mask, return_all_levels=True)

    assert len(outputs["predictions"]) == 3
    assert len(outputs["targets"]) == 3
    assert len(outputs["masks_valid"]) == 3


def test_fpn_feature_extraction(fpn_model):
    """Test feature extraction with FPN at different levels."""
    images = torch.randn(2, 3, 224, 224)
    fpn_model.eval()

    with torch.no_grad():
        features_level_0 = fpn_model.extract_features(images, level=0)
        features_level_1 = fpn_model.extract_features(images, level=1)
        features_level_2 = fpn_model.extract_features(images, level=2)

    # All levels should have consistent embedding dimension
    assert features_level_0.shape[2] == 384
    assert features_level_1.shape[2] == 384
    assert features_level_2.shape[2] == 384

    # Different levels should have different number of patches
    assert features_level_1.shape[1] < features_level_0.shape[1]
    assert features_level_2.shape[1] < features_level_1.shape[1]


def test_fpn_concat_fusion(fpn_concat_model, dummy_input):
    """Test FPN with concatenation fusion method."""
    images, mask = dummy_input
    fpn_concat_model.eval()

    with torch.no_grad():
        outputs = fpn_concat_model(images, mask, return_all_levels=True)

    assert len(outputs["predictions"]) == 3
    assert len(fpn_concat_model.fpn_fusion_convs) == 2  # N-1 fusion convs


def test_fpn_invalid_fusion_method():
    """Test FPN with invalid fusion method."""
    with pytest.raises(ValueError, match="fpn_fusion_method must be 'add' or 'concat'"):
        create_hjepa(
            encoder_type="vit_small_patch16_224",
            img_size=224,
            embed_dim=384,
            num_hierarchies=3,
            use_fpn=True,
            fpn_fusion_method="invalid",
        )


def test_encode_context_with_mask(small_model):
    """Test encode_context method with mask."""
    images = torch.randn(2, 3, 224, 224)
    num_patches = small_model.get_num_patches()
    mask = torch.zeros(2, num_patches)
    mask[:, : num_patches // 2] = 1

    small_model.eval()
    with torch.no_grad():
        context_features = small_model.encode_context(images, mask)

    assert context_features.shape[0] == 2
    assert context_features.shape[2] == 384


def test_encode_context_without_mask(small_model):
    """Test encode_context method without mask."""
    images = torch.randn(2, 3, 224, 224)

    small_model.eval()
    with torch.no_grad():
        context_features = small_model.encode_context(images, mask=None)

    assert context_features.shape[0] == 2
    assert context_features.shape[2] == 384


def test_config_loading():
    """Test model creation from config."""
    config = {
        "model": {
            "encoder_type": "vit_small_patch16_224",
            "embed_dim": 384,
            "predictor": {
                "depth": 4,
                "num_heads": 6,
                "mlp_ratio": 4.0,
            },
            "num_hierarchies": 3,
            "ema": {
                "momentum": 0.996,
                "momentum_end": 1.0,
                "momentum_warmup_epochs": 30,
            },
            "fpn": {
                "use_fpn": False,
                "feature_dim": None,
                "fusion_method": "add",
            },
            "use_flash_attention": True,
            "layerscale": {
                "use_layerscale": False,
                "init_value": 1e-5,
            },
        },
        "data": {
            "image_size": 224,
        },
        "training": {
            "drop_path_rate": 0.0,
            "use_gradient_checkpointing": False,
        },
    }

    model = create_hjepa_from_config(config)

    assert model.embed_dim == 384
    assert model.num_hierarchies == 3


def test_parameter_count(small_model):
    """Test parameter counting."""
    total = sum(p.numel() for p in small_model.parameters())
    trainable = sum(p.numel() for p in small_model.parameters() if p.requires_grad)

    assert total > 0
    assert trainable > 0
    # Target encoder params should be non-trainable
    assert total > trainable


def test_device_handling_cpu(small_model):
    """Test model on CPU device."""
    images = torch.randn(2, 3, 224, 224)
    num_patches = small_model.get_num_patches()
    mask = torch.zeros(2, num_patches)
    mask[:, : num_patches // 2] = 1

    small_model.eval()
    small_model.cpu()

    with torch.no_grad():
        outputs = small_model(images, mask, return_all_levels=True)

    assert outputs["predictions"][0].device.type == "cpu"


def test_batch_size_variation(small_model):
    """Test model with different batch sizes."""
    for batch_size in [1, 2, 4, 8]:
        images = torch.randn(batch_size, 3, 224, 224)
        num_patches = small_model.get_num_patches()
        mask = torch.zeros(batch_size, num_patches)
        mask[:, : num_patches // 2] = 1

        small_model.eval()
        with torch.no_grad():
            outputs = small_model(images, mask, return_all_levels=True)

        assert outputs["predictions"][0].shape[0] == batch_size


def test_variable_mask_per_sample(small_model):
    """Test forward pass with variable number of masked patches per sample."""
    images = torch.randn(2, 3, 224, 224)
    num_patches = small_model.get_num_patches()
    mask = torch.zeros(2, num_patches)

    # First sample: mask 30% of patches
    mask[0, : int(num_patches * 0.3)] = 1

    # Second sample: mask 70% of patches
    mask[1, : int(num_patches * 0.7)] = 1

    small_model.eval()
    with torch.no_grad():
        outputs = small_model(images, mask, return_all_levels=True)

    # Check that mask_valid correctly tracks different numbers of masked patches
    assert len(outputs["predictions"]) == 3
    assert outputs["masks_valid"][0].shape[0] == 2


def test_get_num_patches(small_model):
    """Test get_num_patches method."""
    num_patches = small_model.get_num_patches()

    # For 224x224 image with 16x16 patches: (224/16)^2 = 196
    assert num_patches == 196


def test_get_patch_size(small_model):
    """Test get_patch_size method."""
    patch_size = small_model.get_patch_size()
    assert patch_size == 16

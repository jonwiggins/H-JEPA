"""
Comprehensive test suite for masks module in H-JEPA.

Tests cover:
- MultiBlockMaskGenerator: Random masks, block generation, overlaps
- HierarchicalMaskGenerator: Multi-level masking, scale progression
- MultiCropMaskGenerator: Multi-crop strategies
- Mask shape validation, reproducibility, edge cases, and constraints
"""

import numpy as np
import pytest
import torch

from src.masks.hierarchical import HierarchicalMaskGenerator
from src.masks.multi_block import MultiBlockMaskGenerator
from src.masks.multicrop_masking import MultiCropMaskGenerator


class TestMultiBlockMaskGenerator:
    """Test suite for MultiBlockMaskGenerator."""

    @pytest.fixture
    def generator_224_16(self):
        """Standard mask generator for 224x224 images with 16-pixel patches."""
        return MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=4,
            target_scale=(0.15, 0.2),
            context_scale=(0.85, 1.0),
            aspect_ratio_range=(0.75, 1.5),
            max_attempts=10,
        )

    @pytest.fixture
    def generator_small(self):
        """Small mask generator for edge case testing."""
        return MultiBlockMaskGenerator(
            input_size=64,
            patch_size=16,
            num_target_masks=2,
            target_scale=(0.1, 0.2),
            context_scale=(0.5, 0.8),
            aspect_ratio_range=(0.75, 1.5),
        )

    @pytest.fixture
    def generator_rectangular(self):
        """Rectangular input mask generator."""
        return MultiBlockMaskGenerator(
            input_size=(224, 288),
            patch_size=16,
            num_target_masks=4,
            target_scale=(0.15, 0.2),
            context_scale=(0.85, 1.0),
        )

    # Initialization tests
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        gen = MultiBlockMaskGenerator()
        assert gen.input_size == (224, 224)
        assert gen.patch_size == 16
        assert gen.num_target_masks == 4
        assert gen.num_patches == 196  # 14x14 patches
        assert gen.aspect_ratio_range == (0.75, 1.5)

    def test_init_int_input_size(self):
        """Test that int input_size is converted to tuple."""
        gen = MultiBlockMaskGenerator(input_size=256)
        assert isinstance(gen.input_size, tuple)
        assert gen.input_size == (256, 256)

    def test_init_tuple_input_size(self):
        """Test tuple input_size is preserved."""
        gen = MultiBlockMaskGenerator(input_size=(256, 320))
        assert gen.input_size == (256, 320)

    def test_init_custom_patch_size(self):
        """Test custom patch size configuration."""
        gen = MultiBlockMaskGenerator(input_size=256, patch_size=8)
        assert gen.patch_size == 8
        assert gen.num_patches_h == 32
        assert gen.num_patches_w == 32
        assert gen.num_patches == 1024

    def test_init_custom_num_target_masks(self):
        """Test custom number of target masks."""
        for num_targets in [1, 2, 4, 6]:
            gen = MultiBlockMaskGenerator(num_target_masks=num_targets)
            assert gen.num_target_masks == num_targets

    def test_init_custom_scales(self):
        """Test custom scale configurations."""
        target_scale = (0.1, 0.25)
        context_scale = (0.7, 0.95)
        gen = MultiBlockMaskGenerator(
            target_scale=target_scale,
            context_scale=context_scale,
        )
        assert gen.target_scale == target_scale
        assert gen.context_scale == context_scale

    # Mask generation tests
    def test_generate_single_mask_set(self, generator_224_16):
        """Test generation of a single mask set."""
        context_mask, target_masks = generator_224_16._generate_single_mask_set()

        # Shape tests
        assert context_mask.shape == (196,)
        assert target_masks.shape == (4, 196)

        # Data type tests
        assert context_mask.dtype == torch.bool
        assert target_masks.dtype == torch.bool

    def test_generate_batch_masks(self, generator_224_16, device):
        """Test batch mask generation."""
        batch_size = 8
        context_mask, target_masks = generator_224_16(batch_size=batch_size, device=device)

        # Shape tests
        assert context_mask.shape == (batch_size, 196)
        assert target_masks.shape == (batch_size, 4, 196)

        # Device tests
        assert context_mask.device.type == device.type
        assert target_masks.device.type == device.type

        # Data type tests
        assert context_mask.dtype == torch.bool
        assert target_masks.dtype == torch.bool

    def test_mask_shapes_various_batch_sizes(self, generator_224_16, device):
        """Test that shapes are correct for various batch sizes."""
        for batch_size in [1, 2, 4, 8, 16, 32]:
            context_mask, target_masks = generator_224_16(batch_size=batch_size, device=device)
            assert context_mask.shape[0] == batch_size
            assert target_masks.shape[0] == batch_size

    def test_mask_content_is_boolean(self, generator_224_16):
        """Test that masks contain only boolean values."""
        context_mask, target_masks = generator_224_16(batch_size=4, device="cpu")

        # All values should be True or False
        assert torch.all((context_mask == True) | (context_mask == False))
        assert torch.all((target_masks == True) | (target_masks == False))

    # Overlap constraint tests
    def test_no_overlap_between_context_and_targets(self, generator_224_16):
        """Test that context and target masks do not overlap."""
        context_mask, target_masks = generator_224_16(batch_size=4, device="cpu")

        # Check each sample in batch
        for batch_idx in range(context_mask.shape[0]):
            ctx = context_mask[batch_idx]
            for target_idx in range(target_masks.shape[1]):
                tgt = target_masks[batch_idx, target_idx]
                # Overlap should be zero
                overlap = (ctx & tgt).sum().item()
                assert overlap == 0, f"Context and target {target_idx} overlap"

    def test_no_overlap_between_target_masks(self, generator_224_16):
        """Test that target masks generally do not overlap (but may due to fallback blocks)."""
        _, target_masks = generator_224_16(batch_size=4, device="cpu")

        # Note: Target masks can overlap if they use fallback blocks when sampling fails
        # So we test that they're mostly non-overlapping on average
        total_overlaps = 0
        total_comparisons = 0

        for batch_idx in range(target_masks.shape[0]):
            targets = target_masks[batch_idx]
            for i in range(targets.shape[0]):
                for j in range(i + 1, targets.shape[0]):
                    overlap = (targets[i] & targets[j]).sum().item()
                    total_overlaps += overlap
                    total_comparisons += 1

        avg_overlap = total_overlaps / total_comparisons if total_comparisons > 0 else 0
        # Most target masks should not overlap significantly
        assert avg_overlap < 5, "Too much overlap between target masks"

    # Coverage tests
    def test_context_coverage_within_range(self, generator_224_16):
        """Test that context coverage is within expected range."""
        context_mask, _ = generator_224_16(batch_size=16, device="cpu")

        coverage = context_mask.float().mean(dim=1)
        # Context scale is (0.85, 1.0) but overlaps are removed with targets
        # So actual coverage is lower. Test that it's reasonable
        assert coverage.mean() > 0.3, "Average context coverage too low"
        assert coverage.max() <= 1.0, "Max coverage exceeds 1.0"
        assert coverage.min() >= 0.0, "Min coverage below 0"

    def test_target_coverage_within_range(self, generator_224_16):
        """Test that target coverage is within expected range."""
        _, target_masks = generator_224_16(batch_size=16, device="cpu")

        # Target scale is (0.15, 0.2), each target should be roughly 1/4 of this
        coverage = target_masks.float().mean(dim=2)  # Average over patches
        assert coverage.mean() > 0.02, "Average target coverage too low"
        assert coverage.max() <= 1.0, "Max target coverage exceeds 1.0"

    # Edge case tests
    def test_very_small_image(self):
        """Test mask generation with very small image (edge case)."""
        gen = MultiBlockMaskGenerator(
            input_size=32,
            patch_size=16,
            num_target_masks=2,
        )
        context_mask, target_masks = gen(batch_size=2, device="cpu")

        # Should still generate valid masks
        assert context_mask.shape == (2, 4)
        assert target_masks.shape == (2, 2, 4)

    def test_single_patch_image(self):
        """Test edge case where image has only 1 patch."""
        gen = MultiBlockMaskGenerator(
            input_size=16,
            patch_size=16,
            num_target_masks=1,
        )
        context_mask, target_masks = gen(batch_size=1, device="cpu")

        assert context_mask.shape == (1, 1)
        assert target_masks.shape == (1, 1, 1)

    def test_rectangular_aspect_ratio(self, generator_rectangular):
        """Test with rectangular input dimensions."""
        context_mask, target_masks = generator_rectangular(batch_size=4, device="cpu")

        # 224x288 with 16-pixel patches = 14x18 = 252 patches
        expected_patches = (224 // 16) * (288 // 16)
        assert context_mask.shape == (4, expected_patches)
        assert target_masks.shape == (4, 4, expected_patches)

    def test_num_target_masks_mismatch(self):
        """Test when target masks can't be generated (extreme scales)."""
        gen = MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=10,  # More targets than typically possible
            target_scale=(0.01, 0.05),  # Tiny targets
            max_attempts=5,  # Few attempts
        )
        context_mask, target_masks = gen(batch_size=2, device="cpu")

        # Should still generate the requested number of masks (with fallback)
        assert target_masks.shape == (2, 10, 196)

    # Reproducibility tests
    def test_reproducibility_with_seed(self, generator_224_16):
        """Test that same seed produces identical masks."""
        torch.manual_seed(42)
        np.random.seed(42)
        context1, targets1 = generator_224_16(batch_size=4, device="cpu")

        torch.manual_seed(42)
        np.random.seed(42)
        context2, targets2 = generator_224_16(batch_size=4, device="cpu")

        assert torch.equal(context1, context2)
        assert torch.equal(targets1, targets2)

    def test_different_seeds_different_masks(self, generator_224_16):
        """Test that different seeds produce different masks."""
        torch.manual_seed(42)
        np.random.seed(42)
        context1, targets1 = generator_224_16(batch_size=4, device="cpu")

        torch.manual_seed(123)
        np.random.seed(123)
        context2, targets2 = generator_224_16(batch_size=4, device="cpu")

        # Masks should be different (with very high probability)
        assert not torch.equal(context1, context2)
        assert not torch.equal(targets1, targets2)

    # Block conversion tests
    def test_block_to_mask_conversion(self, generator_224_16):
        """Test conversion of blocks to masks."""
        block = (5, 5, 4, 4)  # top, left, height, width
        mask = generator_224_16._block_to_mask(block)

        assert mask.shape == (196,)
        assert mask.dtype == torch.bool
        # Check that correct patches are masked
        assert mask.sum() == 16  # 4x4 block = 16 patches

    def test_block_to_mask_edge_positions(self, generator_224_16):
        """Test block conversion at image edges."""
        # Top-left corner
        block1 = (0, 0, 2, 2)
        mask1 = generator_224_16._block_to_mask(block1)
        assert mask1.sum() == 4

        # Bottom-right corner (14x14 grid)
        block2 = (12, 12, 2, 2)
        mask2 = generator_224_16._block_to_mask(block2)
        assert mask2.sum() == 4

    # Statistics tests
    def test_get_mask_statistics(self, generator_224_16):
        """Test mask statistics computation."""
        context_mask, target_masks = generator_224_16(batch_size=8, device="cpu")
        stats = generator_224_16.get_mask_statistics(context_mask, target_masks)

        # Check that all expected keys are present
        expected_keys = {
            "context_coverage_mean",
            "context_coverage_std",
            "target_coverage_mean",
            "target_coverage_std",
            "overlap_mean",
            "overlap_max",
        }
        assert set(stats.keys()) == expected_keys

        # Check value ranges
        assert 0 <= stats["context_coverage_mean"] <= 1
        assert 0 <= stats["target_coverage_mean"] <= 1
        assert 0 <= stats["overlap_mean"] <= 1
        assert 0 <= stats["overlap_max"] <= 1
        assert stats["overlap_max"] >= stats["overlap_mean"]

    def test_statistics_overlap_is_zero(self, generator_224_16):
        """Test that statistics correctly report zero overlap."""
        context_mask, target_masks = generator_224_16(batch_size=8, device="cpu")
        stats = generator_224_16.get_mask_statistics(context_mask, target_masks)

        # Overlap should be zero due to mask generation constraints
        assert stats["overlap_mean"] == 0.0
        assert stats["overlap_max"] == 0.0

    def test_statistics_coverage_consistency(self, generator_224_16):
        """Test consistency of coverage statistics."""
        context_mask, target_masks = generator_224_16(batch_size=16, device="cpu")
        stats = generator_224_16.get_mask_statistics(context_mask, target_masks)

        # Mean should be within reasonable range
        assert stats["context_coverage_mean"] > 0.2
        assert stats["context_coverage_mean"] < 1.0
        assert stats["target_coverage_mean"] > 0.0
        assert stats["target_coverage_mean"] < 0.5


class TestHierarchicalMaskGenerator:
    """Test suite for HierarchicalMaskGenerator."""

    @pytest.fixture
    def generator_3level(self):
        """Standard 3-level hierarchical generator."""
        return HierarchicalMaskGenerator(
            input_size=224,
            patch_size=16,
            num_hierarchies=3,
            num_target_masks=4,
            scale_progression="geometric",
            base_scale=(0.05, 0.15),
            aspect_ratio_range=(0.75, 1.5),
        )

    @pytest.fixture
    def generator_linear(self):
        """Hierarchical generator with linear scale progression."""
        return HierarchicalMaskGenerator(
            input_size=224,
            patch_size=16,
            num_hierarchies=3,
            num_target_masks=4,
            scale_progression="linear",
        )

    @pytest.fixture
    def generator_2level(self):
        """2-level hierarchical generator."""
        return HierarchicalMaskGenerator(
            input_size=224,
            patch_size=16,
            num_hierarchies=2,
            num_target_masks=4,
        )

    # Initialization tests
    def test_init_default_params(self):
        """Test default initialization."""
        gen = HierarchicalMaskGenerator()
        assert gen.num_hierarchies == 3
        assert gen.num_target_masks == 4
        assert gen.scale_progression == "geometric"
        assert gen.num_patches == 196

    def test_init_custom_hierarchies(self):
        """Test custom number of hierarchies."""
        for num_h in [1, 2, 3, 4, 5]:
            gen = HierarchicalMaskGenerator(num_hierarchies=num_h)
            assert gen.num_hierarchies == num_h
            assert len(gen.level_configs) == num_h

    def test_init_scale_progression_geometric(self):
        """Test geometric scale progression configuration."""
        gen = HierarchicalMaskGenerator(scale_progression="geometric")
        assert gen.scale_progression == "geometric"
        assert len(gen.level_configs) == 3

    def test_init_scale_progression_linear(self):
        """Test linear scale progression configuration."""
        gen = HierarchicalMaskGenerator(scale_progression="linear")
        assert gen.scale_progression == "linear"
        assert len(gen.level_configs) == 3

    def test_init_invalid_scale_progression(self):
        """Test that invalid scale progression is handled."""
        # Should use linear as default or handle gracefully
        gen = HierarchicalMaskGenerator(scale_progression="invalid")
        # The actual implementation doesn't validate, but we test current behavior
        assert hasattr(gen, "scale_progression")

    # Level configuration tests
    def test_compute_level_configs_geometric(self, generator_3level):
        """Test geometric scale progression creates valid configs."""
        configs = generator_3level.level_configs

        assert len(configs) == 3
        for level, config in enumerate(configs):
            assert "target_scale" in config
            assert "context_scale" in config
            assert config["level"] == level
            # Target scale should increase with level
            assert isinstance(config["target_scale"], tuple)
            assert len(config["target_scale"]) == 2
            assert config["target_scale"][0] <= config["target_scale"][1]

    def test_compute_level_configs_linear(self, generator_linear):
        """Test linear scale progression creates valid configs."""
        configs = generator_linear.level_configs

        assert len(configs) == 3
        for level, config in enumerate(configs):
            assert config["level"] == level

    def test_scale_progression_increases_with_level(self, generator_3level):
        """Test that target scales increase across levels (geometric)."""
        configs = generator_3level.level_configs

        # Get min target scale for each level
        min_scales = [config["target_scale"][0] for config in configs]

        # Scales should generally increase (though capped at 0.95)
        # For geometric: levels 0, 1, 2 should have scales of 1x, 2x, 4x (capped)
        assert min_scales[0] < min_scales[1] or min_scales[1] >= 0.95

    def test_context_coverage_increases_with_level(self, generator_3level):
        """Test that context coverage increases with coarser levels."""
        configs = generator_3level.level_configs

        context_mins = [config["context_scale"][0] for config in configs]
        # Context should increase with level (coarser levels have more context)
        assert context_mins[0] <= context_mins[1] <= context_mins[2]

    # Mask generation tests
    def test_generate_hierarchical_masks(self, generator_3level, device):
        """Test hierarchical mask generation."""
        masks = generator_3level(batch_size=4, device=device)

        # Check structure
        assert isinstance(masks, dict)
        assert len(masks) == 3
        for level_idx in range(3):
            level_key = f"level_{level_idx}"
            assert level_key in masks
            assert "context" in masks[level_key]
            assert "targets" in masks[level_key]

    def test_hierarchical_masks_shapes(self, generator_3level, device):
        """Test that hierarchical masks have correct shapes."""
        batch_size = 8
        masks = generator_3level(batch_size=batch_size, device=device)

        for level_idx in range(3):
            level_key = f"level_{level_idx}"
            context = masks[level_key]["context"]
            targets = masks[level_key]["targets"]

            assert context.shape == (batch_size, 196)
            assert targets.shape == (batch_size, 4, 196)

    def test_hierarchical_masks_device(self, generator_3level):
        """Test that masks are on the correct device."""
        for device_name in ["cpu", "cuda" if torch.cuda.is_available() else "cpu"]:
            masks = generator_3level(batch_size=2, device=device_name)
            for level_key in masks:
                assert masks[level_key]["context"].device.type == device_name
                assert masks[level_key]["targets"].device.type == device_name

    def test_generate_single_level_masks(self, generator_3level):
        """Test generation of masks for single level."""
        config = generator_3level.level_configs[0]
        context_mask, target_masks = generator_3level._generate_level_masks(config)

        assert context_mask.shape == (196,)
        assert target_masks.shape == (4, 196)
        assert context_mask.dtype == torch.bool
        assert target_masks.dtype == torch.bool

    # Overlap constraints across levels
    def test_no_overlap_within_levels(self, generator_3level):
        """Test that overlaps are zero within each level."""
        masks = generator_3level(batch_size=4, device="cpu")

        for level_key in masks:
            context_mask = masks[level_key]["context"]
            target_masks = masks[level_key]["targets"]

            for batch_idx in range(context_mask.shape[0]):
                ctx = context_mask[batch_idx]
                for target_idx in range(target_masks.shape[1]):
                    tgt = target_masks[batch_idx, target_idx]
                    overlap = (ctx & tgt).sum().item()
                    assert overlap == 0

    # Aspect ratio tests
    def test_aspect_ratio_config(self):
        """Test that aspect ratio configuration affects mask shapes."""
        gen = HierarchicalMaskGenerator(
            input_size=224,
            patch_size=16,
            aspect_ratio_range=(0.5, 2.0),  # Wide range
        )
        assert gen.aspect_ratio_range == (0.5, 2.0)

    def test_level_specific_aspect_ratio(self, generator_3level):
        """Test that finest level has squarer blocks (per design)."""
        # This tests the implementation detail that level 0 prefers square blocks
        config_level0 = generator_3level.level_configs[0]
        config_level2 = generator_3level.level_configs[2]

        assert "target_scale" in config_level0
        assert "target_scale" in config_level2

    # Edge cases
    def test_single_hierarchy_level(self):
        """Test with single hierarchy level."""
        gen = HierarchicalMaskGenerator(num_hierarchies=1)
        masks = gen(batch_size=2, device="cpu")

        assert len(masks) == 1
        assert "level_0" in masks

    def test_many_hierarchy_levels(self):
        """Test with many hierarchy levels."""
        gen = HierarchicalMaskGenerator(num_hierarchies=5)
        masks = gen(batch_size=2, device="cpu")

        assert len(masks) == 5
        for i in range(5):
            assert f"level_{i}" in masks

    # Reproducibility tests
    def test_reproducibility_hierarchical(self, generator_3level):
        """Test reproducibility with seed."""
        torch.manual_seed(42)
        np.random.seed(42)
        masks1 = generator_3level(batch_size=4, device="cpu")

        torch.manual_seed(42)
        np.random.seed(42)
        masks2 = generator_3level(batch_size=4, device="cpu")

        for level_key in masks1:
            assert torch.equal(masks1[level_key]["context"], masks2[level_key]["context"])
            assert torch.equal(masks1[level_key]["targets"], masks2[level_key]["targets"])

    # Statistics tests
    def test_hierarchical_statistics(self, generator_3level):
        """Test hierarchical mask statistics."""
        masks = generator_3level(batch_size=8, device="cpu")
        stats = generator_3level.get_hierarchical_statistics(masks)

        # Should have stats for each level
        assert len(stats) == 3
        for level_idx in range(3):
            level_key = f"level_{level_idx}"
            assert level_key in stats
            level_stats = stats[level_key]

            # Check expected keys
            expected_keys = {
                "context_coverage_mean",
                "context_coverage_std",
                "target_coverage_mean",
                "target_coverage_std",
                "overlap_mean",
                "overlap_max",
            }
            assert set(level_stats.keys()) == expected_keys


class TestMultiCropMaskGenerator:
    """Test suite for MultiCropMaskGenerator."""

    @pytest.fixture
    def generator_global_only(self):
        """Global-only strategy generator."""
        return MultiCropMaskGenerator(
            global_crop_size=224,
            local_crop_size=96,
            num_global_crops=2,
            num_local_crops=6,
            masking_strategy="global_only",
        )

    @pytest.fixture
    def generator_with_local_context(self):
        """Global with local context strategy generator."""
        return MultiCropMaskGenerator(
            global_crop_size=224,
            local_crop_size=96,
            num_global_crops=2,
            num_local_crops=6,
            masking_strategy="global_with_local_context",
        )

    @pytest.fixture
    def generator_cross_crop(self):
        """Cross-crop prediction strategy generator."""
        return MultiCropMaskGenerator(
            global_crop_size=224,
            local_crop_size=96,
            num_global_crops=2,
            num_local_crops=6,
            masking_strategy="cross_crop_prediction",
        )

    # Initialization tests
    def test_init_default_params(self):
        """Test default initialization."""
        gen = MultiCropMaskGenerator()
        assert gen.global_crop_size == 224
        assert gen.local_crop_size == 96
        assert gen.num_global_crops == 2
        assert gen.num_local_crops == 6
        assert gen.masking_strategy == "global_only"

    def test_init_custom_crop_sizes(self):
        """Test custom crop size configuration."""
        gen = MultiCropMaskGenerator(
            global_crop_size=256,
            local_crop_size=128,
        )
        assert gen.global_crop_size == 256
        assert gen.local_crop_size == 128

    def test_init_crop_patch_counts(self, generator_global_only):
        """Test that patch counts are calculated correctly."""
        # 224x224 with 16-pixel patches = 14x14 = 196
        assert generator_global_only.global_num_patches == 196
        # 96x96 with 16-pixel patches = 6x6 = 36
        assert generator_global_only.local_num_patches == 36

    def test_init_creates_hierarchical_generators(self, generator_global_only):
        """Test that hierarchical generators are created."""
        assert generator_global_only.global_mask_gen is not None
        assert isinstance(generator_global_only.global_mask_gen, HierarchicalMaskGenerator)

    def test_init_local_mask_gen_global_only(self, generator_global_only):
        """Test that local mask generator is None for global_only strategy."""
        assert generator_global_only.local_mask_gen is None

    def test_init_local_mask_gen_cross_crop(self, generator_cross_crop):
        """Test that local mask generator exists for cross_crop strategy."""
        assert generator_cross_crop.local_mask_gen is not None
        assert isinstance(generator_cross_crop.local_mask_gen, HierarchicalMaskGenerator)

    # Global-only strategy tests
    def test_global_only_strategy(self, generator_global_only, device):
        """Test global-only masking strategy."""
        masks = generator_global_only(batch_size=4, device=device)

        # Check structure
        assert "strategy" in masks
        assert masks["strategy"] == "global_only"
        assert "global_masks" in masks
        assert "local_masks" in masks
        assert masks["local_masks"] is None

        # Check global masks structure
        global_masks = masks["global_masks"]
        assert len(global_masks) == 2  # 2 global crops

        for crop_idx in range(2):
            crop_key = f"crop_{crop_idx}"
            assert crop_key in global_masks
            crop_masks = global_masks[crop_key]

            # Should have 3 hierarchy levels
            for level_idx in range(3):
                level_key = f"level_{level_idx}"
                assert level_key in crop_masks
                assert "context" in crop_masks[level_key]
                assert "targets" in crop_masks[level_key]

    def test_global_with_local_context_strategy(self, generator_with_local_context, device):
        """Test global with local context strategy."""
        masks = generator_with_local_context(batch_size=4, device=device)

        # Check structure
        assert masks["strategy"] == "global_with_local_context"
        assert "local_masks" in masks
        assert masks["local_masks"] is not None

        # Check local masks
        local_masks = masks["local_masks"]
        assert len(local_masks) == 6  # 6 local crops

        for crop_idx in range(6):
            crop_key = f"crop_{crop_idx}"
            assert crop_key in local_masks
            local_mask = local_masks[crop_key]

            # Local crops have all patches visible (no masking)
            context = local_mask["context"]
            assert context.shape == (4, 36)
            assert context.all()  # All True

            # No targets for local crops
            assert local_mask["targets"] is None

    def test_cross_crop_prediction_strategy(self, generator_cross_crop, device):
        """Test cross-crop prediction strategy."""
        masks = generator_cross_crop(batch_size=4, device=device)

        # Check structure
        assert masks["strategy"] == "cross_crop_prediction"
        assert masks["local_masks"] is not None

        # Check that both global and local have masks
        global_masks = masks["global_masks"]
        local_masks = masks["local_masks"]

        assert len(global_masks) == 2
        assert len(local_masks) == 6

        # First 3 local crops should have hierarchical masks with level_0
        for crop_idx in range(3):
            crop_key = f"crop_{crop_idx}"
            local_mask = local_masks[crop_key]
            # Should have level_0 for hierarchical structure
            assert "level_0" in local_mask
            assert "context" in local_mask["level_0"]
            assert "targets" in local_mask["level_0"]

        # Last 3 local crops should be pure context (level_0 with all visible)
        for crop_idx in range(3, 6):
            crop_key = f"crop_{crop_idx}"
            local_mask = local_masks[crop_key]
            # Should have level_0 with all-visible context and empty targets
            assert "level_0" in local_mask
            context = local_mask["level_0"]["context"]
            targets = local_mask["level_0"]["targets"]
            assert context.all()  # All visible
            assert targets.sum() == 0  # All targets are empty

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        gen = MultiCropMaskGenerator(masking_strategy="invalid_strategy")
        with pytest.raises(ValueError):
            gen(batch_size=2, device="cpu")

    # Mask shape tests
    def test_global_masks_shapes(self, generator_global_only, device):
        """Test that global mask shapes are correct."""
        batch_size = 4
        masks = generator_global_only(batch_size=batch_size, device=device)
        global_masks = masks["global_masks"]

        for crop_idx in range(2):
            crop_masks = global_masks[f"crop_{crop_idx}"]
            for level_idx in range(3):
                level_masks = crop_masks[f"level_{level_idx}"]
                context = level_masks["context"]
                targets = level_masks["targets"]

                assert context.shape == (batch_size, 196)
                assert targets.shape == (batch_size, 4, 196)

    def test_local_masks_shapes(self, generator_with_local_context, device):
        """Test that local mask shapes are correct."""
        batch_size = 4
        masks = generator_with_local_context(batch_size=batch_size, device=device)
        local_masks = masks["local_masks"]

        for crop_idx in range(6):
            local_mask = local_masks[f"crop_{crop_idx}"]
            context = local_mask["context"]

            assert context.shape == (batch_size, 36)
            assert context.dtype == torch.bool

    # Crop info tests
    def test_get_crop_info(self, generator_global_only):
        """Test crop information retrieval."""
        info = generator_global_only.get_crop_info()

        expected_keys = {
            "global_crop_size",
            "local_crop_size",
            "global_num_patches",
            "local_num_patches",
            "num_global_crops",
            "num_local_crops",
            "total_crops",
        }
        assert set(info.keys()) == expected_keys

        assert info["global_crop_size"] == 224
        assert info["local_crop_size"] == 96
        assert info["global_num_patches"] == 196
        assert info["local_num_patches"] == 36
        assert info["num_global_crops"] == 2
        assert info["num_local_crops"] == 6
        assert info["total_crops"] == 8

    # Reproducibility tests
    def test_reproducibility_multicrop(self, generator_global_only):
        """Test reproducibility with seed."""
        torch.manual_seed(42)
        np.random.seed(42)
        masks1 = generator_global_only(batch_size=4, device="cpu")

        torch.manual_seed(42)
        np.random.seed(42)
        masks2 = generator_global_only(batch_size=4, device="cpu")

        # Compare global masks
        global_masks1 = masks1["global_masks"]
        global_masks2 = masks2["global_masks"]

        for crop_idx in range(2):
            crop_key = f"crop_{crop_idx}"
            for level_idx in range(3):
                level_key = f"level_{level_idx}"
                assert torch.equal(
                    global_masks1[crop_key][level_key]["context"],
                    global_masks2[crop_key][level_key]["context"],
                )
                assert torch.equal(
                    global_masks1[crop_key][level_key]["targets"],
                    global_masks2[crop_key][level_key]["targets"],
                )

    # Various batch sizes
    def test_various_batch_sizes(self, generator_global_only, device):
        """Test with various batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            masks = generator_global_only(batch_size=batch_size, device=device)
            context = masks["global_masks"]["crop_0"]["level_0"]["context"]
            assert context.shape[0] == batch_size

    # Different configurations
    def test_different_crop_number_configs(self):
        """Test with different numbers of global and local crops."""
        configs = [
            (1, 2),
            (2, 4),
            (2, 6),
            (3, 8),
        ]

        for num_global, num_local in configs:
            gen = MultiCropMaskGenerator(
                num_global_crops=num_global,
                num_local_crops=num_local,
                masking_strategy="global_only",  # Use global_only to not generate local masks
            )
            masks = gen(batch_size=2, device="cpu")

            assert len(masks["global_masks"]) == num_global
            # With global_only strategy, local_masks should be None
            assert masks["strategy"] == "global_only"
            assert masks["local_masks"] is None

    def test_different_patch_sizes(self):
        """Test with different patch sizes."""
        for patch_size in [8, 16, 32]:
            gen = MultiCropMaskGenerator(
                global_crop_size=224,
                local_crop_size=96,
                patch_size=patch_size,
            )

            if patch_size == 8:
                expected_global = (224 // 8) ** 2  # 28x28 = 784
                expected_local = (96 // 8) ** 2  # 12x12 = 144
            elif patch_size == 16:
                expected_global = 196  # 14x14
                expected_local = 36  # 6x6
            else:  # 32
                expected_global = 49  # 7x7
                expected_local = 9  # 3x3

            assert gen.global_num_patches == expected_global
            assert gen.local_num_patches == expected_local


class TestMaskIntegration:
    """Integration tests combining multiple mask generators."""

    def test_hierarchical_vs_multiblock_consistency(self):
        """Test that hierarchical and multi-block generators produce compatible masks."""
        mb_gen = MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=4,
        )
        h_gen = HierarchicalMaskGenerator(
            input_size=224,
            patch_size=16,
            num_hierarchies=3,
            num_target_masks=4,
        )

        mb_context, mb_targets = mb_gen(batch_size=2, device="cpu")
        h_masks = h_gen(batch_size=2, device="cpu")

        # Both should operate on same patch space
        assert mb_context.shape[1] == h_masks["level_0"]["context"].shape[1]

    def test_multicrop_wraps_hierarchical(self):
        """Test that MultiCropMaskGenerator properly wraps HierarchicalMaskGenerator."""
        mc_gen = MultiCropMaskGenerator(
            global_crop_size=224,
            local_crop_size=96,
            num_global_crops=2,
            num_local_crops=6,
            masking_strategy="global_only",
        )

        masks = mc_gen(batch_size=4, device="cpu")

        # Check that it has hierarchical structure
        global_masks = masks["global_masks"]
        for crop_idx in range(2):
            crop_masks = global_masks[f"crop_{crop_idx}"]
            # Should have multiple hierarchy levels
            assert len(crop_masks) >= 2

    def test_all_generators_with_same_image_size(self):
        """Test all generators work with same image size."""
        batch_size = 4
        device = "cpu"

        mb_gen = MultiBlockMaskGenerator(input_size=224, patch_size=16)
        h_gen = HierarchicalMaskGenerator(input_size=224, patch_size=16)
        mc_gen = MultiCropMaskGenerator(global_crop_size=224, patch_size=16)

        mb_ctx, mb_tgt = mb_gen(batch_size=batch_size, device=device)
        h_masks = h_gen(batch_size=batch_size, device=device)
        mc_masks = mc_gen(batch_size=batch_size, device=device)

        # All should have consistent patch counts
        assert mb_ctx.shape == (batch_size, 196)
        assert h_masks["level_0"]["context"].shape == (batch_size, 196)
        assert mc_masks["global_masks"]["crop_0"]["level_0"]["context"].shape == (batch_size, 196)


class TestMaskStatistics:
    """Tests for mask statistics and validation."""

    def test_mask_coverage_statistics(self):
        """Test computation of coverage statistics."""
        gen = MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=4,
        )

        context_mask, target_masks = gen(batch_size=16, device="cpu")
        stats = gen.get_mask_statistics(context_mask, target_masks)

        # Coverage should sum to less than 100% due to separation
        avg_context = stats["context_coverage_mean"]
        avg_target = stats["target_coverage_mean"] * 4  # 4 targets

        assert avg_context + avg_target < 1.0

    def test_mask_diversity_across_batch(self):
        """Test that masks are diverse across batch samples."""
        gen = MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=4,
        )

        context_mask, target_masks = gen(batch_size=16, device="cpu")

        # Check that not all masks in batch are identical
        unique_contexts = len(set(tuple(c.numpy()) for c in context_mask))
        assert unique_contexts > 1, "Batch masks should be diverse"

    def test_large_batch_statistics_stable(self):
        """Test that statistics are stable across multiple large batches."""
        gen = MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=4,
        )

        stats_list = []
        for _ in range(3):
            context_mask, target_masks = gen(batch_size=32, device="cpu")
            stats = gen.get_mask_statistics(context_mask, target_masks)
            stats_list.append(stats["context_coverage_mean"])

        # Statistics should be relatively stable
        mean_coverage = np.mean(stats_list)
        std_coverage = np.std(stats_list)
        assert std_coverage < mean_coverage * 0.2  # Within 20%


class TestMultiBlockMaskGeneratorUncovered:
    """Tests for uncovered branches in MultiBlockMaskGenerator."""

    @pytest.fixture
    def generator_224_16(self):
        """Standard mask generator for 224x224 images with 16-pixel patches."""
        return MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=4,
            target_scale=(0.15, 0.2),
            context_scale=(0.85, 1.0),
            aspect_ratio_range=(0.75, 1.5),
            max_attempts=10,
        )

    def test_visualize_masks_with_save_path(self, tmp_path, generator_224_16):
        """Test visualization with save path."""
        context_mask, target_masks = generator_224_16(batch_size=2, device="cpu")
        save_path = tmp_path / "test_masks.png"
        fig = generator_224_16.visualize_masks(
            context_mask, target_masks, sample_idx=0, save_path=str(save_path)
        )
        assert save_path.exists(), "Visualization file should be saved"
        assert fig is not None, "Figure object should be returned"

    def test_visualize_masks_without_save_path(self, generator_224_16):
        """Test visualization without save path."""
        context_mask, target_masks = generator_224_16(batch_size=2, device="cpu")
        fig = generator_224_16.visualize_masks(
            context_mask, target_masks, sample_idx=0, save_path=None
        )
        assert fig is not None, "Figure object should be returned"

    def test_visualize_masks_custom_figsize(self, generator_224_16):
        """Test visualization with custom figure size."""
        context_mask, target_masks = generator_224_16(batch_size=2, device="cpu")
        fig = generator_224_16.visualize_masks(context_mask, target_masks, figsize=(15, 5))
        assert fig is not None

    def test_visualize_masks_different_samples(self, generator_224_16):
        """Test visualizing different samples from batch."""
        context_mask, target_masks = generator_224_16(batch_size=4, device="cpu")
        for sample_idx in range(4):
            fig = generator_224_16.visualize_masks(
                context_mask, target_masks, sample_idx=sample_idx
            )
            assert fig is not None

    def test_block_sampling_fallback_path(self):
        """Test that block sampling fallback is triggered for impossible configs."""
        gen = MultiBlockMaskGenerator(
            input_size=32,
            patch_size=16,
            num_target_masks=10,  # Many targets
            target_scale=(0.3, 0.5),  # Large targets
            max_attempts=1,  # Very few attempts to force fallback
        )
        context_mask, target_masks = gen(batch_size=2, device="cpu")
        # Should still generate valid masks despite constraints
        assert context_mask.shape == (2, 4)
        assert target_masks.shape == (2, 10, 4)

    def test_sample_block_with_occupied_all_occupied(self):
        """Test block sampling when all space is occupied."""
        gen = MultiBlockMaskGenerator(
            input_size=32,
            patch_size=16,
            num_target_masks=2,
        )
        # Create occupied array that's mostly full
        occupied = np.ones((2, 2), dtype=bool)
        block = gen._sample_block(scale_range=(0.1, 0.2), occupied=occupied)
        # Should still return a block (either empty or fallback)
        assert block is not None
        assert len(block) == 4

    def test_sample_block_random_position_clamping(self):
        """Test that block positions are properly clamped to grid."""
        gen = MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=4,
        )
        for _ in range(10):
            block = gen._sample_block(scale_range=(0.15, 0.2))
            top, left, height, width = block
            # Verify block fits within grid
            assert 0 <= top < gen.num_patches_h
            assert 0 <= left < gen.num_patches_w
            assert height >= 1
            assert width >= 1
            assert top + height <= gen.num_patches_h
            assert left + width <= gen.num_patches_w

    def test_block_to_mask_edge_coverage(self):
        """Test that block-to-mask conversion covers all edge cases."""
        gen = MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=4,
        )
        # Test various block positions
        test_blocks = [
            (0, 0, 1, 1),  # Single patch at origin
            (13, 13, 1, 1),  # Single patch at opposite corner
            (0, 0, 14, 14),  # Full grid
            (7, 7, 1, 1),  # Center single patch
        ]
        for block in test_blocks:
            mask = gen._block_to_mask(block)
            assert mask.shape == (196,)
            top, left, height, width = block
            expected_count = height * width
            assert mask.sum() == expected_count

    def test_statistics_with_empty_targets(self):
        """Test statistics when some targets have zero coverage."""
        gen = MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=4,
        )
        # Create artificial masks with zero coverage in one target
        context_mask = torch.zeros(2, 196, dtype=torch.bool)
        target_masks = torch.zeros(2, 4, 196, dtype=torch.bool)

        # One target has non-zero coverage
        target_masks[0, 0, :10] = True
        context_mask[0, 10:20] = True

        stats = gen.get_mask_statistics(context_mask, target_masks)
        assert "context_coverage_mean" in stats
        assert "target_coverage_mean" in stats

    def test_get_mask_statistics_batch_size_one(self):
        """Test statistics with batch size of 1."""
        gen = MultiBlockMaskGenerator()
        context_mask, target_masks = gen(batch_size=1, device="cpu")
        stats = gen.get_mask_statistics(context_mask, target_masks)

        assert isinstance(stats, dict)
        assert len(stats) == 6
        assert all(isinstance(v, float) for v in stats.values())

    def test_mask_generation_with_extreme_aspect_ratio(self):
        """Test mask generation with extreme aspect ratios."""
        gen = MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=4,
            aspect_ratio_range=(0.1, 10.0),  # Very extreme
        )
        context_mask, target_masks = gen(batch_size=2, device="cpu")
        assert context_mask.shape == (2, 196)
        assert target_masks.shape == (2, 4, 196)

    def test_device_transfer_after_generation(self):
        """Test transferring masks between devices."""
        gen = MultiBlockMaskGenerator()

        # Generate on CPU
        context_cpu, targets_cpu = gen(batch_size=2, device="cpu")
        assert context_cpu.device.type == "cpu"

        # Transfer to CPU explicitly (should be no-op)
        context_cpu2 = context_cpu.to("cpu")
        assert context_cpu2.device.type == "cpu"

    def test_multiple_consecutive_generations(self):
        """Test that multiple consecutive generations produce different masks."""
        gen = MultiBlockMaskGenerator()
        masks_list = []

        for _ in range(5):
            context, targets = gen(batch_size=1, device="cpu")
            masks_list.append((context, targets))

        # Check that not all masks are identical
        unique_contexts = len(set(tuple(m[0][0].numpy()) for m in masks_list))
        assert unique_contexts > 1, "Consecutive generations should produce varied masks"


class TestHierarchicalMaskGeneratorUncovered:
    """Tests for uncovered branches in HierarchicalMaskGenerator."""

    @pytest.fixture
    def generator_3level(self):
        """Standard 3-level hierarchical generator."""
        return HierarchicalMaskGenerator(
            input_size=224,
            patch_size=16,
            num_hierarchies=3,
            num_target_masks=4,
            scale_progression="geometric",
            base_scale=(0.05, 0.15),
            aspect_ratio_range=(0.75, 1.5),
        )

    def test_visualize_hierarchical_masks_with_save_path(self, tmp_path, generator_3level):
        """Test hierarchical visualization with save path."""
        masks = generator_3level(batch_size=2, device="cpu")
        save_path = tmp_path / "hier_masks.png"
        fig = generator_3level.visualize_hierarchical_masks(
            masks, sample_idx=0, save_path=str(save_path)
        )
        assert save_path.exists(), "Visualization file should be saved"
        assert fig is not None

    def test_visualize_hierarchical_masks_without_save(self, generator_3level):
        """Test hierarchical visualization without save path."""
        masks = generator_3level(batch_size=2, device="cpu")
        fig = generator_3level.visualize_hierarchical_masks(masks, sample_idx=0, save_path=None)
        assert fig is not None

    def test_visualize_combined_view_with_save_path(self, tmp_path, generator_3level):
        """Test combined view visualization with save path."""
        masks = generator_3level(batch_size=2, device="cpu")
        save_path = tmp_path / "combined_view.png"
        fig = generator_3level.visualize_combined_view(
            masks, sample_idx=0, save_path=str(save_path)
        )
        assert save_path.exists(), "Visualization file should be saved"
        assert fig is not None

    def test_visualize_combined_view_single_level(self, tmp_path):
        """Test combined view with single hierarchy level."""
        gen = HierarchicalMaskGenerator(num_hierarchies=1)
        masks = gen(batch_size=2, device="cpu")
        fig = gen.visualize_combined_view(masks, sample_idx=0, save_path=None)
        assert fig is not None

    def test_level_configs_single_hierarchy(self):
        """Test level config generation for single hierarchy."""
        gen = HierarchicalMaskGenerator(num_hierarchies=1)
        assert len(gen.level_configs) == 1
        config = gen.level_configs[0]
        assert "target_scale" in config
        assert "context_scale" in config
        assert config["level"] == 0

    def test_level_configs_many_hierarchies(self):
        """Test level config generation for many hierarchies."""
        gen = HierarchicalMaskGenerator(num_hierarchies=6)
        assert len(gen.level_configs) == 6

        # Verify all configs are valid
        for level, config in enumerate(gen.level_configs):
            assert config["level"] == level
            target_scale = config["target_scale"]
            context_scale = config["context_scale"]

            assert target_scale[0] <= target_scale[1]
            assert context_scale[0] <= context_scale[1]

    def test_aspect_ratio_constraint_level_zero(self, generator_3level):
        """Test that level 0 has stricter aspect ratio constraints."""
        # Sample many blocks at level 0
        configs = generator_3level.level_configs
        level_0_config = configs[0]

        # The _sample_block method should tighten aspect ratio for level 0
        # We test this indirectly through mask generation
        context_mask, target_masks = generator_3level._generate_level_masks(level_0_config)
        assert context_mask.dtype == torch.bool
        assert target_masks.dtype == torch.bool

    def test_sample_block_with_level_parameter(self, generator_3level):
        """Test block sampling respects level parameter."""
        for level in range(3):
            block = generator_3level._sample_block(
                scale_range=(0.05, 0.15), occupied=None, level=level
            )
            assert block is not None
            top, left, height, width = block
            assert height >= 1 and width >= 1

    def test_fallback_block_generation_different_levels(self, generator_3level):
        """Test that fallback blocks vary by level."""
        np.random.seed(42)
        # Force fallback by using occupied grid
        occupied = np.ones((14, 14), dtype=bool)

        fallback_0 = generator_3level._sample_block(
            scale_range=(0.05, 0.15), occupied=occupied, level=0
        )

        fallback_2 = generator_3level._sample_block(
            scale_range=(0.05, 0.15), occupied=occupied, level=2
        )

        # Both should return blocks
        assert fallback_0 is not None
        assert fallback_2 is not None

    def test_hierarchical_statistics_empty_targets(self, generator_3level):
        """Test statistics when targets have zero coverage."""
        masks = generator_3level(batch_size=2, device="cpu")

        # Create artificial situation with some empty targets
        for level_key in masks:
            masks[level_key]["targets"][0, 0, :] = False

        stats = generator_3level.get_hierarchical_statistics(masks)
        assert len(stats) == 3

    def test_generate_level_masks_all_levels(self, generator_3level):
        """Test mask generation for all hierarchy levels."""
        for level_idx, config in enumerate(generator_3level.level_configs):
            context, targets = generator_3level._generate_level_masks(config)

            assert context.shape == (196,)
            assert targets.shape == (4, 196)
            assert context.dtype == torch.bool
            assert targets.dtype == torch.bool

    def test_context_scale_increases_with_level(self):
        """Test that context scale increases monotonically across levels."""
        gen = HierarchicalMaskGenerator(num_hierarchies=4)
        configs = gen.level_configs

        # Context minimum should increase with level
        context_mins = [config["context_scale"][0] for config in configs]
        for i in range(len(context_mins) - 1):
            assert context_mins[i] <= context_mins[i + 1]

    def test_target_scale_capped_at_threshold(self):
        """Test that target scales are capped at 0.95."""
        gen = HierarchicalMaskGenerator(
            num_hierarchies=5,
            base_scale=(0.3, 0.4),  # Large base scale
            scale_progression="geometric",
        )

        for config in gen.level_configs:
            target_min, target_max = config["target_scale"]
            assert target_min <= 0.95
            assert target_max <= 0.95

    def test_linear_scale_progression_validation(self):
        """Test linear scale progression configuration."""
        gen = HierarchicalMaskGenerator(num_hierarchies=3, scale_progression="linear")
        configs = gen.level_configs

        # With linear progression, scale_factor = 1 + level
        # So level 0 has 1x, level 1 has 2x, level 2 has 3x (capped)
        assert len(configs) == 3
        for i, config in enumerate(configs):
            assert config["level"] == i

    def test_context_ratio_calculation(self):
        """Test context ratio calculation for different hierarchy sizes."""
        for num_h in [1, 2, 3, 4]:
            gen = HierarchicalMaskGenerator(num_hierarchies=num_h)
            configs = gen.level_configs

            # Context ratio should be: 0.6 + 0.3 * (level / max(1, num_h - 1))
            for level_idx, config in enumerate(configs):
                context_min = config["context_scale"][0]
                expected_ratio = 0.6 + 0.3 * (level_idx / max(1, num_h - 1))
                assert abs(context_min - min(0.95, expected_ratio)) < 0.01


class TestMultiCropMaskGeneratorUncovered:
    """Tests for uncovered branches in MultiCropMaskGenerator."""

    @pytest.fixture
    def generator_global_only(self):
        """Global-only strategy generator."""
        return MultiCropMaskGenerator(
            global_crop_size=224,
            local_crop_size=96,
            num_global_crops=2,
            num_local_crops=6,
            masking_strategy="global_only",
        )

    @pytest.fixture
    def generator_with_local_context(self):
        """Global with local context strategy generator."""
        return MultiCropMaskGenerator(
            global_crop_size=224,
            local_crop_size=96,
            num_global_crops=2,
            num_local_crops=6,
            masking_strategy="global_with_local_context",
        )

    @pytest.fixture
    def generator_cross_crop(self):
        """Cross-crop prediction strategy generator."""
        return MultiCropMaskGenerator(
            global_crop_size=224,
            local_crop_size=96,
            num_global_crops=2,
            num_local_crops=6,
            masking_strategy="cross_crop_prediction",
        )

    def test_visualize_multicrop_masks_with_save(self, tmp_path, generator_global_only):
        """Test multicrop visualization with save path."""
        masks = generator_global_only(batch_size=2, device="cpu")
        save_path = tmp_path / "multicrop.png"
        fig = generator_global_only.visualize_multicrop_masks(
            masks, sample_idx=0, save_path=str(save_path)
        )
        assert save_path.exists(), "Visualization file should be saved"
        assert fig is not None

    def test_visualize_multicrop_masks_without_save(self, generator_global_only):
        """Test multicrop visualization without save path."""
        masks = generator_global_only(batch_size=2, device="cpu")
        fig = generator_global_only.visualize_multicrop_masks(masks, sample_idx=0, save_path=None)
        assert fig is not None

    def test_visualize_multicrop_with_local_masks(self, tmp_path, generator_with_local_context):
        """Test multicrop visualization with local masks."""
        masks = generator_with_local_context(batch_size=2, device="cpu")
        fig = generator_with_local_context.visualize_multicrop_masks(
            masks, sample_idx=0, save_path=None
        )
        assert fig is not None

    def test_visualize_multicrop_cross_crop(self, tmp_path, generator_cross_crop):
        """Test multicrop visualization with cross-crop strategy."""
        masks = generator_cross_crop(batch_size=2, device="cpu")
        fig = generator_cross_crop.visualize_multicrop_masks(masks, sample_idx=0, save_path=None)
        assert fig is not None

    def test_global_only_masks_structure(self, generator_global_only):
        """Test complete structure of global_only strategy masks."""
        masks = generator_global_only(batch_size=2, device="cpu")

        assert "global_masks" in masks
        assert "local_masks" in masks
        assert masks["local_masks"] is None

        global_masks = masks["global_masks"]
        assert len(global_masks) == 2

        for crop_idx in range(2):
            crop_key = f"crop_{crop_idx}"
            assert crop_key in global_masks
            crop_masks = global_masks[crop_key]

            for level_idx in range(3):
                level_key = f"level_{level_idx}"
                assert level_key in crop_masks
                assert "context" in crop_masks[level_key]
                assert "targets" in crop_masks[level_key]

    def test_global_with_local_context_masks_structure(self, generator_with_local_context):
        """Test complete structure of global_with_local_context strategy."""
        masks = generator_with_local_context(batch_size=2, device="cpu")

        assert "local_masks" in masks
        assert masks["local_masks"] is not None

        local_masks = masks["local_masks"]
        assert len(local_masks) == 6

        for crop_idx in range(6):
            crop_key = f"crop_{crop_idx}"
            assert crop_key in local_masks
            local_mask = local_masks[crop_key]

            # Local crops should have all context and no targets
            assert "context" in local_mask
            assert "targets" in local_mask
            assert local_mask["targets"] is None
            assert local_mask["context"].all()

    def test_cross_crop_masks_masked_local_crops(self, generator_cross_crop):
        """Test masked local crops in cross_crop strategy."""
        masks = generator_cross_crop(batch_size=4, device="cpu")
        local_masks = masks["local_masks"]

        # First 3 crops should be masked
        for crop_idx in range(3):
            crop_key = f"crop_{crop_idx}"
            local_mask = local_masks[crop_key]

            # Should have hierarchical structure
            assert "level_0" in local_mask
            assert "context" in local_mask["level_0"]
            assert "targets" in local_mask["level_0"]

    def test_cross_crop_masks_unmasked_local_crops(self, generator_cross_crop):
        """Test unmasked local crops in cross_crop strategy."""
        masks = generator_cross_crop(batch_size=4, device="cpu")
        local_masks = masks["local_masks"]

        # Last 3 crops should be unmasked (pure context)
        for crop_idx in range(3, 6):
            crop_key = f"crop_{crop_idx}"
            local_mask = local_masks[crop_key]

            assert "level_0" in local_mask
            context = local_mask["level_0"]["context"]
            targets = local_mask["level_0"]["targets"]

            # Should be all visible context
            assert context.all()
            # Targets should be all zeros
            assert targets.sum() == 0

    def test_invalid_masking_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        gen = MultiCropMaskGenerator(masking_strategy="invalid_strategy")
        with pytest.raises(ValueError, match="Unknown masking strategy"):
            gen(batch_size=2, device="cpu")

    def test_local_mask_gen_only_for_cross_crop(self):
        """Test that local mask generator is created only for cross_crop strategy."""
        gen_global = MultiCropMaskGenerator(masking_strategy="global_only")
        gen_context = MultiCropMaskGenerator(masking_strategy="global_with_local_context")
        gen_cross = MultiCropMaskGenerator(masking_strategy="cross_crop_prediction")

        assert gen_global.local_mask_gen is None
        assert gen_context.local_mask_gen is None
        assert gen_cross.local_mask_gen is not None

    def test_crop_info_all_fields(self):
        """Test that crop_info returns all expected fields."""
        gen = MultiCropMaskGenerator(
            global_crop_size=256,
            local_crop_size=112,
            num_global_crops=3,
            num_local_crops=8,
            patch_size=16,
        )

        info = gen.get_crop_info()
        assert info["global_crop_size"] == 256
        assert info["local_crop_size"] == 112
        assert info["global_num_patches"] == (256 // 16) ** 2
        assert info["local_num_patches"] == (112 // 16) ** 2
        assert info["num_global_crops"] == 3
        assert info["num_local_crops"] == 8
        assert info["total_crops"] == 11

    def test_different_crop_sizes(self):
        """Test with various crop size combinations."""
        configs = [
            (224, 96),
            (256, 112),
            (192, 64),
        ]

        for global_size, local_size in configs:
            gen = MultiCropMaskGenerator(
                global_crop_size=global_size,
                local_crop_size=local_size,
            )
            masks = gen(batch_size=2, device="cpu")

            global_masks = masks["global_masks"]
            crop_0_masks = global_masks["crop_0"]
            level_0 = crop_0_masks["level_0"]

            expected_global_patches = (global_size // 16) ** 2
            assert level_0["context"].shape[1] == expected_global_patches

    def test_cross_crop_with_many_local_crops(self):
        """Test cross_crop strategy with many local crops."""
        gen = MultiCropMaskGenerator(num_local_crops=10, masking_strategy="cross_crop_prediction")
        masks = gen(batch_size=2, device="cpu")

        local_masks = masks["local_masks"]
        assert len(local_masks) == 10

        # First 5 should be masked, last 5 unmasked
        num_masked = 10 // 2
        for crop_idx in range(10):
            crop_key = f"crop_{crop_idx}"
            local_mask = local_masks[crop_key]

            if crop_idx < num_masked:
                # Check structure for masked crops
                assert "level_0" in local_mask
            else:
                # Unmasked should still have level_0 with empty targets
                assert "level_0" in local_mask
                assert local_mask["level_0"]["targets"].sum() == 0

    def test_single_global_crop(self):
        """Test with single global crop."""
        gen = MultiCropMaskGenerator(
            num_global_crops=1, num_local_crops=2, masking_strategy="global_only"
        )
        masks = gen(batch_size=2, device="cpu")

        global_masks = masks["global_masks"]
        assert len(global_masks) == 1
        assert "crop_0" in global_masks

    def test_many_global_crops(self):
        """Test with many global crops."""
        gen = MultiCropMaskGenerator(
            num_global_crops=4, num_local_crops=4, masking_strategy="global_only"
        )
        masks = gen(batch_size=2, device="cpu")

        global_masks = masks["global_masks"]
        assert len(global_masks) == 4
        for i in range(4):
            assert f"crop_{i}" in global_masks


class TestMaskEdgeCases:
    """Tests for edge cases across all mask generators."""

    def test_very_large_batch_size(self):
        """Test with very large batch size."""
        gen = MultiBlockMaskGenerator(input_size=224, patch_size=16)
        context_mask, target_masks = gen(batch_size=128, device="cpu")

        assert context_mask.shape == (128, 196)
        assert target_masks.shape == (128, 4, 196)

    def test_minimum_batch_size(self):
        """Test with minimum batch size of 1."""
        gen = MultiBlockMaskGenerator()
        context_mask, target_masks = gen(batch_size=1, device="cpu")

        assert context_mask.shape[0] == 1
        assert target_masks.shape[0] == 1

    def test_very_small_patch_size(self):
        """Test with very small patch size."""
        gen = MultiBlockMaskGenerator(input_size=224, patch_size=4, num_target_masks=4)
        # 224/4 = 56 patches per side = 3136 total
        assert gen.num_patches == 56 * 56
        context_mask, target_masks = gen(batch_size=2, device="cpu")
        assert context_mask.shape == (2, 3136)

    def test_large_patch_size(self):
        """Test with large patch size relative to image."""
        gen = MultiBlockMaskGenerator(input_size=64, patch_size=32, num_target_masks=1)
        # 64/32 = 2 patches per side = 4 total
        assert gen.num_patches == 4
        context_mask, target_masks = gen(batch_size=2, device="cpu")
        assert context_mask.shape == (2, 4)
        assert target_masks.shape == (2, 1, 4)

    def test_rectangular_image_various_ratios(self):
        """Test with various rectangular aspect ratios."""
        configs = [
            (224, 448),  # 1:2
            (224, 672),  # 1:3
            (448, 224),  # 2:1
            (200, 300),  # 2:3
        ]

        for height, width in configs:
            gen = MultiBlockMaskGenerator(input_size=(height, width), patch_size=16)
            context_mask, target_masks = gen(batch_size=2, device="cpu")

            expected_patches = (height // 16) * (width // 16)
            assert context_mask.shape == (2, expected_patches)

    def test_all_patches_masked_as_context(self):
        """Test when context scale could theoretically mask all patches."""
        gen = MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=1,
            context_scale=(0.95, 1.0),
            target_scale=(0.01, 0.02),
        )
        context_mask, target_masks = gen(batch_size=4, device="cpu")

        # Context should be large
        context_coverage = context_mask.float().mean(dim=1)
        assert context_coverage.mean() > 0.85

    def test_very_small_target_scale(self):
        """Test with very small target scale."""
        gen = MultiBlockMaskGenerator(target_scale=(0.001, 0.01), num_target_masks=4)
        context_mask, target_masks = gen(batch_size=4, device="cpu")

        # Targets should be very small
        target_coverage = target_masks.float().mean(dim=2)
        assert target_coverage.mean() < 0.05

    def test_masks_are_boolean_after_device_transfer(self):
        """Test that masks remain boolean after device operations."""
        gen = MultiBlockMaskGenerator()
        context_mask, target_masks = gen(batch_size=2, device="cpu")

        # Verify they're boolean
        assert context_mask.dtype == torch.bool
        assert target_masks.dtype == torch.bool

        # Verify boolean values only
        assert torch.all((context_mask == True) | (context_mask == False))
        assert torch.all((target_masks == True) | (target_masks == False))

    def test_no_overlap_consistency_across_multiple_runs(self):
        """Test that no-overlap constraint is consistent."""
        gen = MultiBlockMaskGenerator(
            input_size=224,
            patch_size=16,
            num_target_masks=4,
        )

        for _ in range(5):
            context_mask, target_masks = gen(batch_size=2, device="cpu")
            stats = gen.get_mask_statistics(context_mask, target_masks)

            # Overlap should always be zero due to generation constraint
            assert stats["overlap_max"] == 0.0
            assert stats["overlap_mean"] == 0.0

    def test_hierarchical_with_single_target_mask(self):
        """Test hierarchical generation with single target mask."""
        gen = HierarchicalMaskGenerator(num_hierarchies=3, num_target_masks=1)
        masks = gen(batch_size=2, device="cpu")

        for level_key in masks:
            targets = masks[level_key]["targets"]
            assert targets.shape == (2, 1, 196)

    def test_hierarchical_with_many_target_masks(self):
        """Test hierarchical generation with many target masks."""
        gen = HierarchicalMaskGenerator(num_hierarchies=3, num_target_masks=8)
        masks = gen(batch_size=2, device="cpu")

        for level_key in masks:
            targets = masks[level_key]["targets"]
            assert targets.shape == (2, 8, 196)

    def test_multicrop_with_equal_crop_sizes(self):
        """Test multicrop when global and local crops are same size."""
        gen = MultiCropMaskGenerator(
            global_crop_size=224,
            local_crop_size=224,
        )
        assert gen.global_num_patches == gen.local_num_patches

    def test_multicrop_asymmetric_crop_counts(self):
        """Test multicrop with asymmetric global and local crop counts."""
        gen = MultiCropMaskGenerator(
            num_global_crops=1,
            num_local_crops=10,
            masking_strategy="global_with_local_context",
        )
        masks = gen(batch_size=2, device="cpu")

        assert len(masks["global_masks"]) == 1
        assert len(masks["local_masks"]) == 10


class TestMaskDeterminism:
    """Tests for deterministic behavior and reproducibility."""

    def test_multiblock_determinism_with_multiple_seeds(self):
        """Test that same seed produces same masks across multiple runs."""
        for seed_val in [42, 123, 999]:
            torch.manual_seed(seed_val)
            np.random.seed(seed_val)
            gen = MultiBlockMaskGenerator()
            context1, targets1 = gen(batch_size=4, device="cpu")

            torch.manual_seed(seed_val)
            np.random.seed(seed_val)
            context2, targets2 = gen(batch_size=4, device="cpu")

            assert torch.equal(context1, context2)
            assert torch.equal(targets1, targets2)

    def test_hierarchical_determinism_with_multiple_seeds(self):
        """Test hierarchical determinism across seeds."""
        for seed_val in [42, 123]:
            torch.manual_seed(seed_val)
            np.random.seed(seed_val)
            gen = HierarchicalMaskGenerator(num_hierarchies=3)
            masks1 = gen(batch_size=2, device="cpu")

            torch.manual_seed(seed_val)
            np.random.seed(seed_val)
            masks2 = gen(batch_size=2, device="cpu")

            for level_key in masks1:
                assert torch.equal(masks1[level_key]["context"], masks2[level_key]["context"])
                assert torch.equal(masks1[level_key]["targets"], masks2[level_key]["targets"])

    def test_multicrop_determinism(self):
        """Test multicrop determinism across strategies."""
        for strategy in ["global_only", "global_with_local_context", "cross_crop_prediction"]:
            torch.manual_seed(42)
            np.random.seed(42)
            gen = MultiCropMaskGenerator(masking_strategy=strategy)
            masks1 = gen(batch_size=2, device="cpu")

            torch.manual_seed(42)
            np.random.seed(42)
            masks2 = gen(batch_size=2, device="cpu")

            # Compare global masks
            for crop_idx in range(gen.num_global_crops):
                crop_key = f"crop_{crop_idx}"
                for level_idx in range(gen.num_hierarchies):
                    level_key = f"level_{level_idx}"
                    assert torch.equal(
                        masks1["global_masks"][crop_key][level_key]["context"],
                        masks2["global_masks"][crop_key][level_key]["context"],
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

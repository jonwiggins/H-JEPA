"""
Tests for LeWorldModel-derived features added to H-JEPA.

Coverage:
    - SIGReg vectorization correctness vs the legacy per-slice loop.
    - Characteristic-function Epps-Pulley test variant.
    - BatchNorm1dForTokens wrapper for [B, N, D] inputs.
    - HJEPA end-to-end mode (use_target_encoder=False).
    - HJEPA BatchNorm projection head construction.
    - HJEPALoss detach_targets toggle.
    - ActionPredictor forward pass with per-step actions.
    - LeWM full forward pass on synthetic frames + actions.
    - CEMPlanner returns valid action shapes on a toy dynamics setup.
"""

from pathlib import Path

import pytest
import torch

from src.data import (
    EpisodeFileSequentialDataset,
    SyntheticSequentialDataset,
    build_sequential_dataloader,
)
from src.inference import CEMPlanner
from src.losses import (
    EppsPulleyTest,
    HJEPALoss,
    HJEPASIGRegLoss,
    SIGRegLoss,
    create_loss_from_config,
)
from src.models import (
    AdaLNZeroBlock,
    BatchNorm1dForTokens,
    create_action_predictor,
    create_hjepa,
    create_lewm,
)

# ---------------------------------------------------------------------------
# SIGReg refinements
# ---------------------------------------------------------------------------


class TestSIGRegVectorization:
    def test_vectorized_matches_loop(self):
        """Vectorized path must produce the same value as the per-slice loop."""
        torch.manual_seed(0)
        loss_loop = SIGRegLoss(num_slices=8, vectorized=False, fixed_slices=True)
        loss_vec = SIGRegLoss(num_slices=8, vectorized=True, fixed_slices=True)
        # Share the cached random slices so both losses use the same projections.
        loss_vec._fixed_random_slices = None
        loss_loop._fixed_random_slices = None
        z_a = torch.randn(8, 16)
        z_b = torch.randn(8, 16)

        torch.manual_seed(123)
        out_loop = loss_loop(z_a, z_b)
        torch.manual_seed(123)
        out_vec = loss_vec(z_a, z_b)

        assert torch.allclose(
            out_loop["sigreg_loss"], out_vec["sigreg_loss"], atol=1e-5
        ), "vectorized SIGReg should match the legacy loop value"


class TestCharFunctionEppsPulley:
    def test_low_for_gaussian(self):
        torch.manual_seed(0)
        test = EppsPulleyTest(test_method="char_function")
        x = torch.randn(2000)
        stat = test(x)
        assert stat.item() < 1.0, "char-function test should be low for Gaussian"

    def test_higher_for_uniform(self):
        torch.manual_seed(0)
        test = EppsPulleyTest(test_method="char_function")
        gauss = test(torch.randn(2000))
        unif = test(torch.rand(2000) * 2 - 1)
        assert unif > gauss, "uniform should produce a higher statistic than Gaussian"

    def test_batched_shape(self):
        test = EppsPulleyTest(test_method="char_function")
        x = torch.randn(4, 1000)
        stat = test(x)
        assert stat.shape == (4,), "batched call should produce one stat per row"


# ---------------------------------------------------------------------------
# HJEPA additive flags
# ---------------------------------------------------------------------------


class TestBatchNormForTokens:
    def test_2d_passthrough(self):
        bn = BatchNorm1dForTokens(8)
        x = torch.randn(16, 8)
        out = bn(x)
        assert out.shape == x.shape

    def test_3d_input(self):
        bn = BatchNorm1dForTokens(8)
        x = torch.randn(4, 7, 8)
        out = bn(x)
        assert out.shape == x.shape

    def test_invalid_shape(self):
        bn = BatchNorm1dForTokens(8)
        with pytest.raises(ValueError):
            bn(torch.randn(2, 3, 4, 5))


class TestHJEPABatchNormProjection:
    def test_constructs_with_batchnorm(self):
        model = create_hjepa(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            predictor_depth=2,
            predictor_num_heads=3,
            num_hierarchies=2,
            projection_norm="batchnorm",
        )
        # Each projection head is Sequential(Linear, BatchNorm1dForTokens)
        for proj in model.hierarchy_projections:
            assert isinstance(proj[1], BatchNorm1dForTokens)


class TestHJEPAEndToEndMode:
    def test_no_target_encoder_attribute(self):
        model = create_hjepa(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            predictor_depth=2,
            predictor_num_heads=3,
            num_hierarchies=2,
            use_target_encoder=False,
        )
        # When end-to-end, no separate target_encoder is registered as a child module.
        assert not hasattr(
            model, "target_encoder"
        ), "end-to-end mode should not register a target_encoder module"
        assert model.use_target_encoder is False

    def test_update_target_encoder_is_noop(self):
        model = create_hjepa(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            predictor_depth=2,
            predictor_num_heads=3,
            num_hierarchies=2,
            use_target_encoder=False,
        )
        # update_target_encoder must return 1.0 (no update) and not raise.
        ret = model.update_target_encoder(0)
        assert ret == 1.0


class TestHJEPALossDetachToggle:
    def test_detach_default_blocks_gradient(self):
        loss_fn = HJEPALoss(num_hierarchies=1, detach_targets=True)
        pred = torch.randn(2, 4, 8, requires_grad=True)
        target = torch.randn(2, 4, 8, requires_grad=True)
        out = loss_fn([pred], [target])
        out["loss"].backward()
        assert target.grad is None, "with detach_targets=True the target should have no gradient"

    def test_no_detach_propagates_gradient(self):
        loss_fn = HJEPALoss(num_hierarchies=1, detach_targets=False)
        pred = torch.randn(2, 4, 8, requires_grad=True)
        target = torch.randn(2, 4, 8, requires_grad=True)
        out = loss_fn([pred], [target])
        out["loss"].backward()
        assert (
            target.grad is not None
        ), "with detach_targets=False the target should receive gradient"


# ---------------------------------------------------------------------------
# Action predictor and LeWM
# ---------------------------------------------------------------------------


class TestActionPredictor:
    def test_forward_shape(self):
        predictor = create_action_predictor(
            embed_dim=32, action_dim=4, depth=2, num_heads=4, max_seq_len=16
        )
        emb = torch.randn(3, 8, 32)
        act = torch.randn(3, 8, 4)
        out = predictor(emb, act)
        assert out.shape == (3, 8, 32)

    def test_zero_init_modulation_is_unconditional_at_init(self):
        """At init, gating is zero so the block is identity in the action input."""
        predictor = create_action_predictor(
            embed_dim=16, action_dim=4, depth=2, num_heads=4, max_seq_len=8
        )
        predictor.eval()
        emb = torch.randn(2, 4, 16)
        act_a = torch.zeros(2, 4, 4)
        act_b = torch.randn(2, 4, 4)
        with torch.no_grad():
            out_a = predictor(emb, act_a)
            out_b = predictor(emb, act_b)
        # AdaLN-Zero gating starts at 0 so attention/MLP outputs are gated to zero,
        # and the action MLP also starts contributing nothing to scale/shift.
        # Therefore both outputs must be equal at init regardless of action.
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_rejects_mismatched_shapes(self):
        predictor = create_action_predictor(embed_dim=16, action_dim=4, max_seq_len=8)
        with pytest.raises(ValueError):
            predictor(torch.randn(2, 4, 16), torch.randn(2, 5, 4))


class TestLeWM:
    def test_forward_returns_expected_keys(self):
        model = create_lewm(
            encoder_type="vit_tiny_patch16_224",
            img_size=224,
            embed_dim=192,
            action_dim=4,
            predictor_depth=2,
            predictor_num_heads=4,
            predictor_max_seq_len=8,
        )
        frames = torch.randn(1, 4, 3, 224, 224)
        actions = torch.randn(1, 4, 4)
        out = model(frames, actions)
        assert {"embeddings", "predictions", "prediction_inputs", "target_embeddings"} <= out.keys()
        assert out["embeddings"].shape == (1, 4, 192)
        assert out["predictions"].shape == (1, 4, 192)
        assert out["prediction_inputs"].shape == (1, 3, 192)
        assert out["target_embeddings"].shape == (1, 3, 192)

    def test_gradients_flow_to_encoder(self):
        model = create_lewm(
            encoder_type="vit_tiny_patch16_224",
            img_size=224,
            embed_dim=192,
            action_dim=4,
            predictor_depth=2,
            predictor_num_heads=4,
            predictor_max_seq_len=8,
        )
        frames = torch.randn(1, 3, 3, 224, 224)
        actions = torch.randn(1, 3, 4)
        out = model(frames, actions)
        loss = ((out["prediction_inputs"] - out["target_embeddings"]) ** 2).mean()
        loss.backward()
        # At least one encoder parameter must receive a gradient (end-to-end).
        encoder_grads = [
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.frame_encoder.parameters()
        ]
        assert any(encoder_grads), "encoder must receive gradients in end-to-end LeWM"


# ---------------------------------------------------------------------------
# Sequential dataset + CEM planner
# ---------------------------------------------------------------------------


class TestSyntheticSequentialDataset:
    def test_shapes(self):
        ds = SyntheticSequentialDataset(
            num_episodes=4, seq_len=8, image_size=32, channels=3, action_dim=2, seed=0
        )
        sample = ds[0]
        assert sample["frames"].shape == (8, 3, 32, 32)
        assert sample["actions"].shape == (8, 2)

    def test_dataloader(self):
        ds = SyntheticSequentialDataset(
            num_episodes=4, seq_len=8, image_size=32, channels=3, action_dim=2, seed=0
        )
        loader = build_sequential_dataloader(ds, batch_size=2, num_workers=0, shuffle=False)
        batch = next(iter(loader))
        assert batch["frames"].shape == (2, 8, 3, 32, 32)
        assert batch["actions"].shape == (2, 8, 2)


class TestCEMPlanner:
    def test_plan_returns_action_shape(self):
        torch.manual_seed(0)
        # Build a tiny LeWM and use it as encoder/predictor for the planner.
        model = create_lewm(
            encoder_type="vit_tiny_patch16_224",
            img_size=224,
            embed_dim=192,
            action_dim=2,
            predictor_depth=2,
            predictor_num_heads=4,
            predictor_max_seq_len=8,
        )
        planner = CEMPlanner(
            encoder=model.frame_encoder,
            predictor=model.action_predictor,
            action_dim=2,
            horizon=4,
            num_samples=8,
            num_elites=2,
            num_iterations=2,
        )
        obs = torch.randn(3, 224, 224)
        goal = torch.randn(3, 224, 224)
        action = planner.plan(obs, goal)
        assert action.shape == (
            2,
        ), f"first action should have shape [action_dim], got {action.shape}"

    def test_plan_returns_full_sequence(self):
        torch.manual_seed(0)
        model = create_lewm(
            encoder_type="vit_tiny_patch16_224",
            img_size=224,
            embed_dim=192,
            action_dim=2,
            predictor_depth=2,
            predictor_num_heads=4,
            predictor_max_seq_len=8,
        )
        planner = CEMPlanner(
            encoder=model.frame_encoder,
            predictor=model.action_predictor,
            action_dim=2,
            horizon=4,
            num_samples=8,
            num_elites=2,
            num_iterations=2,
        )
        obs = torch.randn(3, 224, 224)
        goal = torch.randn(3, 224, 224)
        actions = planner.plan(obs, goal, return_full_sequence=True)
        assert actions.shape == (4, 2)

    def test_constructor_rejects_too_many_elites(self):
        model = create_lewm(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            action_dim=2,
            predictor_depth=2,
            predictor_num_heads=4,
            predictor_max_seq_len=8,
        )
        with pytest.raises(ValueError, match="num_elites"):
            CEMPlanner(
                encoder=model.frame_encoder,
                predictor=model.action_predictor,
                action_dim=2,
                horizon=4,
                num_samples=4,
                num_elites=8,  # > num_samples
            )

    def test_planner_restores_train_mode(self):
        """The planner must leave encoder/predictor in their original mode."""
        model = create_lewm(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            action_dim=2,
            predictor_depth=2,
            predictor_num_heads=4,
            predictor_max_seq_len=8,
        )
        planner = CEMPlanner(
            encoder=model.frame_encoder,
            predictor=model.action_predictor,
            action_dim=2,
            horizon=4,
            num_samples=8,
            num_elites=2,
            num_iterations=2,
        )
        model.train()
        assert model.frame_encoder.training and model.action_predictor.training
        planner.plan(torch.randn(3, 224, 224), torch.randn(3, 224, 224))
        assert model.frame_encoder.training, "encoder train mode must be restored"
        assert model.action_predictor.training, "predictor train mode must be restored"

    def test_custom_cost_function(self):
        """A user-supplied cost_fn must be used in place of the default."""
        model = create_lewm(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            action_dim=2,
            predictor_depth=2,
            predictor_num_heads=4,
            predictor_max_seq_len=8,
        )
        called: list[bool] = []

        def custom_cost(traj: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
            called.append(True)
            return traj.abs().sum(dim=(-2, -1))  # arbitrary scoring

        planner = CEMPlanner(
            encoder=model.frame_encoder,
            predictor=model.action_predictor,
            action_dim=2,
            horizon=4,
            num_samples=8,
            num_elites=2,
            num_iterations=2,
            cost_fn=custom_cost,
        )
        planner.plan(torch.randn(3, 224, 224), torch.randn(3, 224, 224))
        assert called, "custom cost_fn must be invoked during planning"

    def test_per_dim_action_bounds(self):
        """Tensor-valued action_low/action_high should broadcast per-dim."""
        model = create_lewm(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            action_dim=3,
            predictor_depth=2,
            predictor_num_heads=4,
            predictor_max_seq_len=8,
        )
        low = torch.tensor([-1.0, -2.0, -3.0])
        high = torch.tensor([1.0, 2.0, 3.0])
        planner = CEMPlanner(
            encoder=model.frame_encoder,
            predictor=model.action_predictor,
            action_dim=3,
            horizon=2,
            num_samples=8,
            num_elites=2,
            num_iterations=1,
            action_low=low,
            action_high=high,
        )
        action = planner.plan(torch.randn(3, 224, 224), torch.randn(3, 224, 224))
        assert (action >= low).all() and (action <= high).all()


# ---------------------------------------------------------------------------
# HJEPASIGRegLoss wrapper
# ---------------------------------------------------------------------------


class TestHJEPASIGRegLoss:
    def test_forward_signature_matches_trainer(self):
        """The wrapper must accept (predictions, targets, masks, context_features)."""
        jepa = HJEPALoss(num_hierarchies=2, normalize_embeddings=False)
        sigreg = SIGRegLoss(num_slices=16, sigreg_weight=1.0, invariance_weight=0.0)
        loss_fn = HJEPASIGRegLoss(jepa, sigreg, sigreg_weight=0.5)

        preds = [torch.randn(4, 8, 16), torch.randn(4, 8, 16)]
        targets = [torch.randn(4, 8, 16), torch.randn(4, 8, 16)]
        ctx = torch.randn(4, 8, 16)

        out = loss_fn(predictions=preds, targets=targets, context_features=ctx)
        assert "loss" in out
        assert "jepa_loss" in out
        assert "sigreg_loss" in out

    def test_no_context_features_returns_zero_sigreg(self):
        jepa = HJEPALoss(num_hierarchies=1, normalize_embeddings=False)
        sigreg = SIGRegLoss(num_slices=16, sigreg_weight=1.0, invariance_weight=0.0)
        loss_fn = HJEPASIGRegLoss(jepa, sigreg, sigreg_weight=0.5)

        out = loss_fn(
            predictions=[torch.randn(2, 4, 16)],
            targets=[torch.randn(2, 4, 16)],
            context_features=None,
        )
        assert out["sigreg_loss"].item() == 0.0
        # When SIGReg is zero, total loss must equal jepa_loss exactly.
        assert torch.allclose(out["loss"], out["jepa_loss"])

    def test_gradients_flow_through_sigreg_path(self):
        jepa = HJEPALoss(num_hierarchies=1, normalize_embeddings=False)
        sigreg = SIGRegLoss(num_slices=16, sigreg_weight=1.0, invariance_weight=0.0)
        loss_fn = HJEPASIGRegLoss(jepa, sigreg, sigreg_weight=1.0)
        ctx = torch.randn(4, 8, 16, requires_grad=True)
        out = loss_fn(
            predictions=[torch.randn(4, 8, 16)],
            targets=[torch.randn(4, 8, 16)],
            context_features=ctx,
        )
        out["loss"].backward()
        assert ctx.grad is not None and ctx.grad.abs().sum() > 0


class TestLossFactory:
    def test_hjepa_sigreg_type_returns_wrapper(self):
        config = {
            "loss": {"type": "hjepa_sigreg", "sigreg_num_slices": 32},
            "model": {"num_hierarchies": 2},
        }
        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, HJEPASIGRegLoss)

    def test_sigreg_combined_alias_returns_wrapper(self):
        config = {
            "loss": {"type": "sigreg_combined", "sigreg_num_slices": 32},
            "model": {"num_hierarchies": 1},
        }
        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, HJEPASIGRegLoss)

    def test_bare_sigreg_type_unchanged(self):
        """The original 'sigreg' type must still return a bare SIGRegLoss."""
        config = {"loss": {"type": "sigreg"}, "model": {"num_hierarchies": 1}}
        loss_fn = create_loss_from_config(config)
        assert isinstance(loss_fn, SIGRegLoss)


# ---------------------------------------------------------------------------
# AdaLN-Zero block direct
# ---------------------------------------------------------------------------


class TestAdaLNZeroBlock:
    def test_sequence_level_action(self):
        block = AdaLNZeroBlock(embed_dim=16, num_heads=4, action_embed_dim=16)
        x = torch.randn(2, 5, 16)
        action = torch.randn(2, 16)  # one vector per batch element
        out = block(x, action)
        assert out.shape == x.shape

    def test_per_token_action(self):
        block = AdaLNZeroBlock(embed_dim=16, num_heads=4, action_embed_dim=16)
        x = torch.randn(2, 5, 16)
        action = torch.randn(2, 5, 16)  # one vector per (batch, token)
        out = block(x, action)
        assert out.shape == x.shape

    def test_modulation_zero_init_is_identity_at_init(self):
        torch.manual_seed(0)
        block = AdaLNZeroBlock(embed_dim=16, num_heads=4, action_embed_dim=16)
        block.eval()
        x = torch.randn(3, 4, 16)
        out = block(x, torch.randn(3, 16))
        # With zero-init modulation, the block reduces to x + 0*attn + 0*mlp = x.
        assert torch.allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# ActionPredictor edge cases
# ---------------------------------------------------------------------------


class TestActionPredictorEdgeCases:
    def test_rejects_2d_embeddings(self):
        predictor = create_action_predictor(embed_dim=16, action_dim=4)
        with pytest.raises(ValueError, match="embeddings"):
            predictor(torch.randn(4, 16), torch.randn(4, 1, 4))

    def test_rejects_2d_actions(self):
        predictor = create_action_predictor(embed_dim=16, action_dim=4)
        with pytest.raises(ValueError, match="actions"):
            predictor(torch.randn(2, 4, 16), torch.randn(2, 4))

    def test_rejects_sequence_longer_than_max(self):
        predictor = create_action_predictor(embed_dim=16, action_dim=4, max_seq_len=4)
        with pytest.raises(ValueError, match="max_seq_len"):
            predictor(torch.randn(2, 8, 16), torch.randn(2, 8, 4))


# ---------------------------------------------------------------------------
# SIGRegLoss extras
# ---------------------------------------------------------------------------


class TestSIGRegLossExtras:
    def test_quadrature_points_registered_as_buffer(self):
        loss = SIGRegLoss(test_method="char_function")
        # Buffer means it moves with .to(device) and is part of state_dict.
        assert "univariate_test.quadrature_points" in dict(loss.named_buffers())

    def test_char_function_lambda_changes_value(self):
        torch.manual_seed(0)
        x = torch.randn(500)
        small = EppsPulleyTest(test_method="char_function", char_function_lambda=0.5)
        large = EppsPulleyTest(test_method="char_function", char_function_lambda=2.0)
        # Different λ should produce different test statistics.
        assert not torch.allclose(small(x), large(x))


# ---------------------------------------------------------------------------
# HJEPA combined flag tests
# ---------------------------------------------------------------------------


class TestHJEPACombinedFlags:
    def test_no_target_encoder_with_batchnorm_projection(self):
        """End-to-end + batchnorm together — both flags wire correctly."""
        model = create_hjepa(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            predictor_depth=2,
            predictor_num_heads=3,
            num_hierarchies=2,
            use_target_encoder=False,
            projection_norm="batchnorm",
        )
        assert model.use_target_encoder is False
        assert isinstance(model.hierarchy_projections[0][1], BatchNorm1dForTokens)

    def test_layernorm_projection_default_preserved(self):
        """Default projection_norm must remain LayerNorm for backward compat."""
        model = create_hjepa(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            predictor_depth=2,
            predictor_num_heads=3,
            num_hierarchies=2,
        )
        # Default projection should still be LayerNorm.
        assert isinstance(model.hierarchy_projections[0][1], torch.nn.LayerNorm)

    def test_projection_norm_none(self):
        model = create_hjepa(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            predictor_depth=2,
            predictor_num_heads=3,
            num_hierarchies=2,
            projection_norm="none",
        )
        assert isinstance(model.hierarchy_projections[0][1], torch.nn.Identity)

    def test_invalid_projection_norm_raises(self):
        with pytest.raises(ValueError, match="projection_norm"):
            create_hjepa(
                encoder_type="vit_tiny_patch16_224",
                embed_dim=192,
                predictor_depth=2,
                predictor_num_heads=3,
                num_hierarchies=2,
                projection_norm="invalid",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# EpisodeFileSequentialDataset
# ---------------------------------------------------------------------------


class TestEpisodeFileSequentialDataset:
    def test_loads_episode_files(self, tmp_path: Path):
        """Round-trip: write fake episodes, load via dataset, get expected shapes."""
        # Two episodes of length 10 each.
        for i in range(2):
            torch.save(
                {
                    "frames": torch.randn(10, 3, 32, 32),
                    "actions": torch.randn(10, 4),
                },
                tmp_path / f"ep_{i}.pt",
            )

        ds = EpisodeFileSequentialDataset(
            root_dir=tmp_path,
            seq_len=4,
            random_window=False,
        )
        # 2 episodes × (10 - 4 + 1) = 14 windows total in deterministic mode.
        assert len(ds) == 14

        sample = ds[0]
        assert sample["frames"].shape == (4, 3, 32, 32)
        assert sample["actions"].shape == (4, 4)

    def test_random_window_one_per_episode(self, tmp_path: Path):
        for i in range(3):
            torch.save(
                {
                    "frames": torch.randn(8, 3, 16, 16),
                    "actions": torch.randn(8, 2),
                },
                tmp_path / f"ep_{i}.pt",
            )
        ds = EpisodeFileSequentialDataset(
            root_dir=tmp_path,
            seq_len=4,
            random_window=True,
        )
        # In random mode each episode contributes exactly one virtual sample.
        assert len(ds) == 3

    def test_skips_too_short_episodes(self, tmp_path: Path):
        # Episode of length 2 is too short for seq_len=8 — must be skipped.
        torch.save(
            {"frames": torch.randn(2, 3, 16, 16), "actions": torch.randn(2, 2)},
            tmp_path / "short.pt",
        )
        torch.save(
            {"frames": torch.randn(10, 3, 16, 16), "actions": torch.randn(10, 2)},
            tmp_path / "long.pt",
        )
        ds = EpisodeFileSequentialDataset(
            root_dir=tmp_path,
            seq_len=8,
            random_window=True,
        )
        assert len(ds) == 1, "too-short episode must be excluded"

    def test_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            EpisodeFileSequentialDataset(
                root_dir=tmp_path / "does_not_exist",
                seq_len=4,
            )

    def test_empty_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No .pt files"):
            EpisodeFileSequentialDataset(root_dir=tmp_path, seq_len=4)


# ---------------------------------------------------------------------------
# LeWM additional cases
# ---------------------------------------------------------------------------


class TestLeWMValidation:
    def test_rejects_4d_frames(self):
        model = create_lewm(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            action_dim=4,
            predictor_depth=2,
            predictor_num_heads=4,
            predictor_max_seq_len=8,
        )
        with pytest.raises(ValueError, match="frames"):
            model.encode_sequence(torch.randn(2, 3, 224, 224))  # missing time dim

    def test_rejects_action_time_mismatch(self):
        model = create_lewm(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            action_dim=4,
            predictor_depth=2,
            predictor_num_heads=4,
            predictor_max_seq_len=8,
        )
        frames = torch.randn(1, 4, 3, 224, 224)
        actions = torch.randn(1, 5, 4)  # wrong T
        with pytest.raises(ValueError, match="dims must match"):
            model(frames, actions)

    def test_encode_sequence_shape(self):
        model = create_lewm(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            action_dim=4,
            predictor_depth=2,
            predictor_num_heads=4,
            predictor_max_seq_len=8,
        )
        frames = torch.randn(2, 4, 3, 224, 224)
        embs = model.encode_sequence(frames)
        assert embs.shape == (2, 4, 192)

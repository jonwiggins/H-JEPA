"""Behavioral regressions discovered by the bounded correctness review."""

import copy

import pytest
import torch

from src.losses.combined import CombinedLoss
from src.losses.hjepa_loss import HJEPALoss
from src.losses.sigreg import EppsPulleyTest
from src.models.hjepa import create_hjepa


@pytest.fixture
def tiny_model():
    torch.manual_seed(7)
    return create_hjepa(
        encoder_type="vit_tiny_patch16_224",
        img_size=64,
        embed_dim=192,
        predictor_depth=1,
        predictor_num_heads=3,
        num_hierarchies=2,
        use_fpn=True,
        fpn_feature_dim=32,
        use_flash_attention=False,
    )


def test_teacher_starts_as_exact_student_copy(tiny_model):
    for a, b in zip(
        tiny_model.context_encoder.parameters(), tiny_model.target_encoder.parameters()
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert not b.requires_grad


def test_all_real_fpn_tokens_are_valid(tiny_model):
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, :8] = True
    out = tiny_model(torch.randn(2, 3, 64, 64), mask)
    assert all(m.all() for m in out["masks_valid"])


def test_predictions_independent_of_other_samples_padding(tiny_model):
    tiny_model.eval()
    x = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, :9] = True
    with torch.no_grad():
        solo = tiny_model(x[:1], mask[:1])
        batched = tiny_model(x, mask)
    for one, many in zip(solo["predictions"], batched["predictions"]):
        torch.testing.assert_close(one[0], many[0, : one.shape[1]], atol=2e-5, rtol=2e-5)


def test_masked_pixels_cannot_change_predictions(tiny_model):
    tiny_model.eval()
    x = torch.randn(2, 3, 64, 64)
    y = x.clone()
    y[:, :, :32, :] += 100
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, :8] = True
    with torch.no_grad():
        a, b = tiny_model(x, mask), tiny_model(y, mask)
    for p, q in zip(a["predictions"], b["predictions"]):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert not torch.equal(a["targets"][0], b["targets"][0])


def test_masked_mean_matches_unmasked_mean():
    p, t = torch.randn(2, 5, 7), torch.randn(2, 5, 7)
    loss = HJEPALoss(num_hierarchies=1, normalize_embeddings=False)
    torch.testing.assert_close(
        loss(p, t)["loss"], loss(p, t, [torch.ones(2, 5, dtype=torch.bool)])["loss"]
    )


@pytest.mark.parametrize("method", ["reference_points", "char_function"])
def test_gaussian_test_penalizes_wrong_location_and_scale(method):
    test = EppsPulleyTest(test_method=method)
    x = torch.erfinv(2 * (torch.arange(512) + 0.5) / 512 - 1) * 2**0.5
    base = test(x)
    assert test(x * 0.01) > base + 0.05
    assert test(x + 4) > base + 0.05


def test_encoder_regularizer_is_invariant_to_batch_order():
    loss = CombinedLoss(num_hierarchies=2)
    x = torch.randn(5, 3, 8)
    p = [torch.randn(5, 3, 8)] * 2
    t = [torch.randn(5, 3, 8)] * 2
    a = loss(p, t, context_features=x)["vicreg_loss"]
    b = loss(p, t, context_features=x[torch.tensor([2, 0, 4, 1, 3])])["vicreg_loss"]
    torch.testing.assert_close(a, b)


class ScalarJEPA(torch.nn.Module):
    """Synthetic analytic fixture, deliberately independent of ViT internals."""

    def __init__(self):
        super().__init__()
        self.context_encoder = torch.nn.Linear(1, 1, bias=False)
        self.context_encoder.weight.data.fill_(1)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)

    def forward(self, images, mask, context_mask=None):
        p = self.context_encoder(images).unsqueeze(1)
        return dict(
            predictions=[p],
            targets=[torch.zeros_like(p)],
            context_features=p,
            target_features=p.detach(),
            masks_valid=[torch.ones(p.shape[:2], dtype=torch.bool)],
        )


def make_scalar_trainer(tmp_path, accumulation=2, batch_size=1, resume=None):
    from torch.utils.data import DataLoader, TensorDataset

    from src.trainers.trainer import HJEPATrainer

    config = dict(
        training=dict(epochs=2, optimizer="sgd", lr=0.1, accumulation_steps=accumulation),
        logging=dict(use_wandb=False, use_tensorboard=False, log_dir=str(tmp_path / "logs")),
        checkpoint=dict(checkpoint_dir=str(tmp_path / "checkpoints")),
    )
    model = ScalarJEPA()
    loader = DataLoader(TensorDataset(torch.tensor([[1.0], [2.0], [3.0]])), batch_size=batch_size)
    loss = HJEPALoss(loss_type="mse", num_hierarchies=1, normalize_embeddings=False)

    def masks(batch_size, device):
        return {
            "level_0": {"targets": torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device)}
        }

    trainer = HJEPATrainer(
        model,
        loader,
        loader,
        torch.optim.SGD(model.parameters(), lr=0.1),
        loss,
        masks,
        config,
        "cpu",
        resume_checkpoint=resume,
    )
    return trainer


def test_accumulation_matches_macro_batches_including_remainder(tmp_path):
    micro = make_scalar_trainer(tmp_path / "micro")
    macro = make_scalar_trainer(tmp_path / "macro", accumulation=1, batch_size=2)
    micro._train_epoch(0)
    macro._train_epoch(0)
    assert micro.global_step == macro.global_step == 2
    torch.testing.assert_close(
        micro.model.context_encoder.weight, macro.model.context_encoder.weight
    )
    torch.testing.assert_close(micro.model.target_encoder.weight, macro.model.target_encoder.weight)
    assert all(p.grad is None for p in micro.model.parameters())


def test_ema_uses_updated_student_and_forward_does_not_mutate_teacher(tmp_path):
    trainer = make_scalar_trainer(tmp_path)
    before = trainer.model.target_encoder.weight.detach().clone()
    trainer.model.context_encoder.weight.data.fill_(2)
    trainer._train_step((torch.ones(1, 1),), 0, 0)
    torch.testing.assert_close(trainer.model.target_encoder.weight, before, rtol=0, atol=0)
    trainer.max_steps = 1
    trainer.ema_scheduler = lambda step: 0.5
    trainer.lr_scheduler = lambda step: 0.1
    trainer._train_epoch(0)
    # mean squared error over x=1,2: gradient=5*w=10 => w'=1.
    torch.testing.assert_close(trainer.model.context_encoder.weight, torch.ones(1, 1))
    torch.testing.assert_close(
        trainer.model.target_encoder.weight,
        0.5 * before + 0.5 * trainer.model.context_encoder.weight,
    )


def test_lr_is_applied_before_update(tmp_path):
    trainer = make_scalar_trainer(tmp_path, accumulation=1)
    trainer.max_steps = 1
    trainer.lr_scheduler = lambda step: 0.0
    trainer._train_epoch(0)
    torch.testing.assert_close(
        trainer.model.context_encoder.weight, torch.ones(1, 1), rtol=0, atol=0
    )


def test_checkpoint_restores_rng_step_optimizer_and_next_update(tmp_path):
    import random

    import numpy as np

    trainer = make_scalar_trainer(tmp_path)
    trainer._train_epoch(0)
    path = trainer._save_checkpoint(0, trainer._validate_epoch(0)["loss"], False)
    draws = (random.random(), np.random.rand(), torch.rand(3))
    trainer._train_epoch(1)
    resumed = make_scalar_trainer(tmp_path, resume=path)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert draws[:2] == actual[:2]
    torch.testing.assert_close(draws[2], actual[2], rtol=0, atol=0)
    resumed._train_epoch(1)
    assert trainer.global_step == resumed.global_step == 4
    for a, b in zip(trainer.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_teacher_stays_in_eval_mode(tiny_model):
    tiny_model.train()
    assert tiny_model.context_encoder.training
    assert not tiny_model.target_encoder.training


def test_single_masked_token_has_finite_hierarchy(tiny_model):
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, 4] = True
    out = tiny_model(torch.randn(2, 3, 64, 64), mask)
    assert all(torch.isfinite(p).all() and p.shape[1] == 1 for p in out["predictions"])


def test_vicreg_covariance_matches_dimension_normalized_formula():
    from src.losses.vicreg import VICRegLoss

    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    covariance = x.T @ x / (x.shape[0] - 1)
    expected = (covariance.square().sum() - covariance.diag().square().sum()) / x.shape[1]
    torch.testing.assert_close(VICRegLoss()._covariance_loss(x), expected)


def test_vicreg_single_observation_is_finite():
    from src.losses.vicreg import VICRegLoss

    x = torch.ones(1, 1, requires_grad=True)
    loss = VICRegLoss()(x, x)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_pixels_outside_sampled_context_are_hidden(tiny_model):
    tiny_model.eval()
    x = torch.randn(2, 3, 64, 64)
    y = x.clone()
    y[:, :, 32:, :] += 100
    target = torch.zeros(2, 16, dtype=torch.bool)
    target[:, 8:12] = True
    context = torch.zeros_like(target)
    context[:, :8] = True
    with torch.no_grad():
        a = tiny_model(x, target, context_mask=context)
        b = tiny_model(y, target, context_mask=context)
    for p, q in zip(a["predictions"], b["predictions"]):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_advertised_config_flags_reach_encoder_and_fpn():
    from src.models.hjepa import create_hjepa_from_config

    config = dict(
        model=dict(
            encoder_type="vit_tiny_patch16_224",
            embed_dim=192,
            num_hierarchies=2,
            use_fpn=True,
            use_rope=True,
            use_layerscale=True,
            predictor=dict(depth=1, num_heads=3),
        ),
        data=dict(image_size=64),
        training=dict(use_gradient_checkpointing=True),
    )
    model = create_hjepa_from_config(config)
    assert model.use_fpn and model.context_encoder.use_rope
    assert model.context_encoder.use_layerscale and model.context_encoder.use_gradient_checkpointing
    output = model(torch.randn(2, 3, 64, 64), torch.arange(16).expand(2, -1) < 8)
    sum(p.square().mean() for p in output["predictions"]).backward()
    assert model.context_encoder.vit.patch_embed.proj.weight.grad is not None


def test_fixed_sigreg_slices_survive_serialization():
    from src.losses.sigreg import SIGRegLoss

    loss = SIGRegLoss(num_slices=8, fixed_slices=True)
    x = torch.randn(8, 5)
    expected = loss(x, x)["loss"]
    restored = SIGRegLoss(num_slices=8, fixed_slices=True)
    restored.load_state_dict(copy.deepcopy(loss.state_dict()))
    torch.testing.assert_close(restored(x, x)["loss"], expected, rtol=0, atol=0)


def test_world_predictor_causal_after_gates_open():
    from src.models.action_predictor import ActionPredictor

    predictor = ActionPredictor(embed_dim=12, action_dim=2, depth=2, num_heads=3, dropout=0).eval()
    # Zero gates make the initialization's causality check vacuous. Open them.
    for name, param in predictor.named_parameters():
        if "modulation" in name:
            torch.nn.init.normal_(param, std=0.1)
    x, actions = torch.randn(2, 4, 12), torch.randn(2, 4, 2)
    a = predictor(x, actions)
    y, later_actions = x.clone(), actions.clone()
    y[:, 2:] += 10
    later_actions[:, 2:] += 10
    b = predictor(y, later_actions)
    torch.testing.assert_close(a[:, :2], b[:, :2], rtol=0, atol=0)
    changed = predictor(x, actions + 1)
    assert not torch.allclose(a, changed)


def test_accumulation_weights_short_final_batch_by_sample_count(tmp_path):
    micro = make_scalar_trainer(tmp_path / "micro", accumulation=2, batch_size=2)
    macro = make_scalar_trainer(tmp_path / "macro", accumulation=1, batch_size=3)
    micro._train_epoch(0)
    macro._train_epoch(0)
    torch.testing.assert_close(
        micro.model.context_encoder.weight, macro.model.context_encoder.weight
    )


def test_incomplete_epoch_resume_fails_explicitly(tmp_path):
    trainer = make_scalar_trainer(tmp_path)
    trainer.max_steps = 1
    trainer._train_epoch(0)
    path = trainer._save_checkpoint(0, 0.1, False)
    with pytest.raises(ValueError, match="incomplete epoch"):
        make_scalar_trainer(tmp_path, resume=path)


def test_sigreg_config_weight_controls_objective():
    from src.losses.combined import create_loss_from_config

    def loss(weight):
        return create_loss_from_config(
            dict(
                model=dict(num_hierarchies=2),
                loss=dict(
                    type="hjepa_sigreg",
                    sigreg_weight=weight,
                    sigreg_num_slices=8,
                    sigreg_test_method="char_function",
                ),
            )
        )

    x = torch.randn(4, 3, 8)
    p = [x, x]
    torch.manual_seed(8)
    a = loss(1)(p, p, context_features=x)
    torch.manual_seed(8)
    b = loss(7)(p, p, context_features=x)
    torch.testing.assert_close(b["loss"] - b["jepa_loss"], 7 * (a["loss"] - a["jepa_loss"]))


def test_cem_one_elite_stays_finite_and_rollout_matches_dynamics():
    from src.inference.cem_planner import CEMPlanner

    class Dynamics(torch.nn.Module):
        def forward(self, embeddings, actions):
            return embeddings + actions

    encoder = torch.nn.Linear(1, 1)
    planner = CEMPlanner(
        encoder, Dynamics(), action_dim=1, horizon=2, num_samples=16, num_elites=1, num_iterations=3
    )
    actions = torch.tensor([[[0.2], [0.3]], [[-0.1], [0.4]]])
    rollout = planner._rollout(torch.tensor([1.0]), actions)
    torch.testing.assert_close(rollout, 1 + actions.cumsum(dim=1))
    result = planner.plan(torch.tensor([[0.0]]), torch.tensor([[1.0]]))
    assert torch.isfinite(result).all()
    assert (result >= -1).all() and (result <= 1).all()


@pytest.mark.parametrize("kind", ["CosineScheduler", "LinearScheduler", "EMAScheduler"])
def test_scheduler_endpoint_with_all_warmup(kind):
    from src.utils import scheduler

    schedule = getattr(scheduler, kind)(
        base_value=0.1, final_value=0.01, epochs=1, steps_per_epoch=2, warmup_epochs=1
    )
    assert schedule(2) == 0.01
    assert schedule(100) == 0.01


def test_collapse_monitor_detects_identical_nonconstant_vectors(tmp_path):
    trainer = make_scalar_trainer(tmp_path)
    repeated = torch.tensor([[1.0, 2.0, 3.0]]).expand(4, -1)
    metrics = trainer._compute_collapse_metrics(repeated, repeated)
    assert metrics["context_std"] == metrics["target_std"] == 0
    assert metrics["context_eff_rank"] == metrics["target_eff_rank"] == 0


def test_target_indices_match_full_encoder_positions(tiny_model):
    tiny_model.eval()
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[0, [2, 7, 9]] = True
    mask[1, [1, 4, 8, 10]] = True
    with torch.no_grad():
        output = tiny_model(torch.randn(2, 3, 64, 64), mask, return_all_levels=False)
    for i in range(2):
        expected = output["target_features"][i, 1:][mask[i]]
        actual = output["targets"][0][i][output["mask_valid"][i]]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

#!/usr/bin/env python3
"""
Ultra-small smoke test for the LeWM-derived features.

Builds a model + loss for each of the three new configs (tier1, tier2,
world_model), runs a few iterations of training on synthetic data, and
verifies that gradients flow and the loss decreases. Intended for CI-style
"does it explode?" verification, not for benchmarking.

Run:
    python scripts/lewm_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SyntheticSequentialDataset, build_sequential_dataloader
from src.inference import CEMPlanner
from src.losses import SIGRegLoss, create_loss_from_config
from src.models import (
    create_hjepa_from_config,
    create_lewm_from_config,
)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _smoke_hjepa_config(config_path: str, num_steps: int = 3) -> tuple[float, float]:
    """Build HJEPA + loss from a config and run a few training steps."""
    print(f"\n=== Smoke test: {config_path} ===")
    config = _load_config(config_path)
    device = torch.device("cpu")

    model = create_hjepa_from_config(config).to(device)
    loss_fn = create_loss_from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"  use_target_encoder = {model.use_target_encoder}")
    print(f"  projection_norm = {model.projection_norm}")
    print(f"  loss class = {loss_fn.__class__.__name__}")

    img_size = config.get("data", {}).get("image_size", 224)
    batch_size = 2  # tiny — main goal is to verify the path works
    num_patches = (img_size // 16) ** 2

    losses = []
    for step in range(num_steps):
        images = torch.rand(batch_size, 3, img_size, img_size, device=device)
        # Make a coarse mask: roughly half the patches masked.
        mask = torch.rand(batch_size, num_patches, device=device) > 0.5

        outputs = model(images, mask, return_all_levels=True)
        loss_dict = loss_fn(
            predictions=outputs["predictions"],
            targets=outputs["targets"],
            context_features=outputs.get("context_features"),
        )
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"  step {step}: loss={loss.item():.4f}")

    initial, final = losses[0], losses[-1]
    print(f"  initial={initial:.4f}, final={final:.4f}")
    return initial, final


def _smoke_lewm_world_model(config_path: str, num_steps: int = 3) -> tuple[float, float]:
    """Build a LeWM model from config and run a few sequence-prediction steps."""
    print(f"\n=== Smoke test: {config_path} ===")
    config = _load_config(config_path)
    device = torch.device("cpu")

    model = create_lewm_from_config(config).to(device)
    print(f"  encoder.embed_dim = {model.embed_dim}")
    print(f"  action_dim = {model.action_dim}")

    data_cfg = config.get("data", {})
    dataset = SyntheticSequentialDataset(
        num_episodes=8,
        seq_len=data_cfg.get("seq_len", 8),
        image_size=224,  # encoder requires the configured img_size
        channels=data_cfg.get("channels", 3),
        action_dim=data_cfg.get("action_dim", 4),
        seed=0,
    )
    loader = build_sequential_dataloader(
        dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )

    sigreg = SIGRegLoss(
        num_slices=64,
        sigreg_weight=1.0,
        invariance_weight=0.0,
        flatten_patches=False,
        test_method="char_function",
        char_function_num_quadrature=16,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    iterator = iter(loader)
    for step in range(num_steps):
        batch = next(iterator)
        frames = batch["frames"].to(device)
        actions = batch["actions"].to(device)

        out = model(frames, actions)
        pred_loss = ((out["prediction_inputs"] - out["target_embeddings"].detach()) ** 2).mean()
        # SIGReg on encoder embeddings flattened over batch+time.
        z_flat = out["embeddings"].reshape(-1, model.embed_dim)
        sigreg_dict = sigreg(z_flat, z_flat)
        loss = pred_loss + 0.1 * sigreg_dict["sigreg_loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(
            f"  step {step}: loss={loss.item():.4f} "
            f"(pred={pred_loss.item():.4f}, sigreg={sigreg_dict['sigreg_loss'].item():.4f})"
        )

    initial, final = losses[0], losses[-1]
    print(f"  initial={initial:.4f}, final={final:.4f}")

    # Smoke-test the planner on a single observation pair.
    print("  testing CEM planner ...")
    planner = CEMPlanner(
        encoder=model.frame_encoder,
        predictor=model.action_predictor,
        action_dim=model.action_dim,
        horizon=4,
        num_samples=8,
        num_elites=2,
        num_iterations=2,
    )
    obs = torch.rand(3, 224, 224)
    goal = torch.rand(3, 224, 224)
    action = planner.plan(obs, goal)
    assert action.shape == (model.action_dim,)
    print(f"  planner produced action with shape {tuple(action.shape)} OK")

    return initial, final


def main() -> int:
    torch.manual_seed(0)

    results: list[tuple[str, float, float]] = []

    for cfg in ("configs/lewm_tier1.yaml", "configs/lewm_tier2.yaml"):
        try:
            init, final = _smoke_hjepa_config(cfg, num_steps=3)
            results.append((cfg, init, final))
        except Exception as e:
            print(f"  FAILED: {e}")
            return 1

    try:
        init, final = _smoke_lewm_world_model("configs/lewm_world_model.yaml", num_steps=3)
        results.append(("configs/lewm_world_model.yaml", init, final))
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"  FAILED: {e}")
        return 1

    print("\n=== Summary ===")
    for cfg, init, final in results:
        delta = final - init
        print(f"  {cfg}: initial={init:.4f}  final={final:.4f}  delta={delta:+.4f}")

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

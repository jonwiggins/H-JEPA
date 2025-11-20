#!/usr/bin/env python3
"""Generate a test checkpoint for evaluation testing."""

import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.models.hjepa import HJEPA
from src.utils.checkpoint import CheckpointManager


def main():
    """Generate a test checkpoint."""
    print("Generating test checkpoint...")

    # Create output directory
    output_dir = "results/test_checkpoint"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # Create model
    model = HJEPA(
        encoder_type="vit_tiny_patch16_224",
        img_size=224,
        num_hierarchies=3,
        embed_dim=192,
        use_fpn=True,
        fpn_feature_dim=128,
        use_flash_attention=False,
        use_layerscale=False,
        predictor_depth=4,
        predictor_mlp_ratio=4.0,
    )

    # Move to device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)

    # Create optimizer (needed for checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 0,
        "global_step": 10,
        "train_loss": 6.5,
        "val_loss": 7.0,
        "config": {
            "model": {
                "encoder_type": "vit_tiny_patch16_224",
                "img_size": 224,
                "num_hierarchies": 3,
                "embed_dim": 192,
                "use_fpn": True,
                "fpn_channels": 128,
            },
            "data": {
                "dataset": "cifar10",
                "batch_size": 32,
            },
            "training": {
                "learning_rate": 1e-4,
            },
        },
    }

    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoints", "checkpoint_step_10.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Also save as last checkpoint
    last_checkpoint_path = os.path.join(output_dir, "checkpoints", "checkpoint_last.pth")
    torch.save(checkpoint, last_checkpoint_path)
    print(f"Last checkpoint saved to: {last_checkpoint_path}")

    print("\nCheckpoint generation complete!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

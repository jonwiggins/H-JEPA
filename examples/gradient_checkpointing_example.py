#!/usr/bin/env python3
"""
Example: Using Gradient Checkpointing with H-JEPA

This script demonstrates how to use gradient checkpointing for memory-efficient training.
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hjepa import create_hjepa, create_hjepa_from_config


def print_memory_stats(stage: str):
    """Print current GPU memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage:30s} - Allocated: {allocated:6.2f} GB, Reserved: {reserved:6.2f} GB")


def compare_memory_usage():
    """Compare memory usage with and without gradient checkpointing."""
    print("\n" + "="*80)
    print("Gradient Checkpointing Memory Comparison")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if not torch.cuda.is_available():
        print("CUDA not available. Memory comparison requires GPU.")
        return

    # Configuration
    batch_size = 32
    img_size = 224

    # Create dummy batch
    images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    num_patches = (img_size // 16) ** 2  # For patch size 16
    mask = torch.zeros(batch_size, num_patches).to(device)
    mask[:, :num_patches//2] = 1  # Mask first half of patches

    print(f"\nBatch size: {batch_size}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Number of patches: {num_patches}")

    # Test WITHOUT gradient checkpointing
    print("\n" + "-"*80)
    print("WITHOUT Gradient Checkpointing")
    print("-"*80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_without = create_hjepa(
        encoder_type="vit_base_patch16_224",
        img_size=img_size,
        embed_dim=768,
        predictor_depth=6,
        use_gradient_checkpointing=False,  # Disabled
    ).to(device)

    model_without.train()
    print_memory_stats("After model creation")

    # Forward pass
    outputs = model_without(images, mask)
    print_memory_stats("After forward pass")

    # Compute loss (simplified)
    predictions = outputs['predictions']
    targets = outputs['targets']
    loss = sum([
        nn.functional.mse_loss(pred, tgt)
        for pred, tgt in zip(predictions, targets)
    ])

    # Backward pass
    loss.backward()
    print_memory_stats("After backward pass")

    peak_memory_without = torch.cuda.max_memory_allocated() / 1024**3

    # Clean up
    del model_without, outputs, predictions, targets, loss
    torch.cuda.empty_cache()

    # Test WITH gradient checkpointing
    print("\n" + "-"*80)
    print("WITH Gradient Checkpointing")
    print("-"*80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_with = create_hjepa(
        encoder_type="vit_base_patch16_224",
        img_size=img_size,
        embed_dim=768,
        predictor_depth=6,
        use_gradient_checkpointing=True,  # Enabled
    ).to(device)

    model_with.train()
    print_memory_stats("After model creation")

    # Forward pass
    outputs = model_with(images, mask)
    print_memory_stats("After forward pass")

    # Compute loss (simplified)
    predictions = outputs['predictions']
    targets = outputs['targets']
    loss = sum([
        nn.functional.mse_loss(pred, tgt)
        for pred, tgt in zip(predictions, targets)
    ])

    # Backward pass
    loss.backward()
    print_memory_stats("After backward pass")

    peak_memory_with = torch.cuda.max_memory_allocated() / 1024**3

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Peak memory WITHOUT checkpointing: {peak_memory_without:.2f} GB")
    print(f"Peak memory WITH checkpointing:    {peak_memory_with:.2f} GB")
    savings = (1 - peak_memory_with / peak_memory_without) * 100
    print(f"Memory savings:                     {savings:.1f}%")
    print("="*80)


def example_config_usage():
    """Example of using gradient checkpointing via configuration."""
    print("\n" + "="*80)
    print("Example: Using Gradient Checkpointing via Configuration")
    print("="*80)

    # Example configuration
    config = {
        'model': {
            'encoder_type': 'vit_base_patch16_224',
            'embed_dim': 768,
            'num_hierarchies': 3,
            'predictor': {
                'depth': 6,
                'num_heads': 12,
                'mlp_ratio': 4.0,
            },
            'ema': {
                'momentum': 0.996,
                'momentum_end': 1.0,
                'momentum_warmup_epochs': 30,
            },
        },
        'training': {
            'use_gradient_checkpointing': True,  # Enable here
            'drop_path_rate': 0.1,
        },
        'data': {
            'image_size': 224,
        },
    }

    # Create model from config
    model = create_hjepa_from_config(config)

    print("\nModel created with configuration:")
    print(f"  use_gradient_checkpointing: {model.use_gradient_checkpointing}")
    print(f"  context_encoder.use_gradient_checkpointing: {model.context_encoder.use_gradient_checkpointing}")
    print(f"  predictor.use_gradient_checkpointing: {model.predictor.use_gradient_checkpointing}")

    # Save example config
    config_path = Path(__file__).parent / "gradient_checkpointing_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\nExample configuration saved to: {config_path}")


def example_programmatic_usage():
    """Example of using gradient checkpointing programmatically."""
    print("\n" + "="*80)
    print("Example: Using Gradient Checkpointing Programmatically")
    print("="*80)

    # Create model with checkpointing enabled
    model = create_hjepa(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        embed_dim=768,
        predictor_depth=6,
        predictor_num_heads=12,
        predictor_mlp_ratio=4.0,
        num_hierarchies=3,
        use_gradient_checkpointing=True,  # Enable checkpointing
    )

    print("\nModel created with gradient checkpointing enabled:")
    print(f"  model.use_gradient_checkpointing: {model.use_gradient_checkpointing}")
    print(f"  context_encoder.use_gradient_checkpointing: {model.context_encoder.use_gradient_checkpointing}")
    print(f"  predictor.use_gradient_checkpointing: {model.predictor.use_gradient_checkpointing}")

    # Can also toggle dynamically
    print("\nToggling checkpointing dynamically:")
    model.use_gradient_checkpointing = False
    model.context_encoder.use_gradient_checkpointing = False
    model.predictor.use_gradient_checkpointing = False
    print(f"  Checkpointing disabled: {model.use_gradient_checkpointing}")

    model.use_gradient_checkpointing = True
    model.context_encoder.use_gradient_checkpointing = True
    model.predictor.use_gradient_checkpointing = True
    print(f"  Checkpointing re-enabled: {model.use_gradient_checkpointing}")


def example_training_mode():
    """Demonstrate that checkpointing only applies during training."""
    print("\n" + "="*80)
    print("Example: Checkpointing Only Active During Training")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_hjepa(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        use_gradient_checkpointing=True,
    ).to(device)

    # Create dummy input
    images = torch.randn(4, 3, 224, 224).to(device)
    mask = torch.zeros(4, 196).to(device)
    mask[:, :98] = 1

    # Training mode
    model.train()
    print("\nTraining mode (checkpointing ACTIVE):")
    print(f"  model.training: {model.training}")
    print(f"  Checkpointing will be applied during backward pass")

    # Evaluation mode
    model.eval()
    print("\nEvaluation mode (checkpointing INACTIVE):")
    print(f"  model.training: {model.training}")
    print(f"  Standard forward pass for maximum speed")

    with torch.no_grad():
        outputs = model(images, mask)
        print(f"  Forward pass completed successfully")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("H-JEPA Gradient Checkpointing Examples")
    print("="*80)

    # Example 1: Programmatic usage
    example_programmatic_usage()

    # Example 2: Config usage
    example_config_usage()

    # Example 3: Training mode behavior
    example_training_mode()

    # Example 4: Memory comparison (requires CUDA)
    if torch.cuda.is_available():
        compare_memory_usage()
    else:
        print("\n" + "="*80)
        print("Memory Comparison")
        print("="*80)
        print("\nCUDA not available. Skipping memory comparison.")
        print("Run on a GPU to see memory savings.")

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
    print("\nFor more information, see:")
    print("  - docs/gradient_checkpointing.md")
    print("  - docs/gradient_checkpointing_implementation_report.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

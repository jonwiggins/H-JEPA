"""
Example usage of H-JEPA loss functions.

This script demonstrates how to use the various loss functions implemented
in src/losses/ for training H-JEPA models.
"""

import torch
import torch.nn as nn
from src.losses import (
    HJEPALoss,
    VICRegLoss,
    CombinedLoss,
    HierarchicalCombinedLoss,
    create_loss_from_config,
)


def example_1_basic_hjepa_loss():
    """Example 1: Basic H-JEPA Loss"""
    print("=" * 60)
    print("Example 1: Basic H-JEPA Loss")
    print("=" * 60)

    # Create loss function
    loss_fn = HJEPALoss(
        loss_type='smoothl1',
        hierarchy_weights=[1.0, 0.5, 0.25],
        num_hierarchies=3,
        normalize_embeddings=True,
    )

    # Simulate predictions and targets for 3 hierarchy levels
    # Each level has [batch_size, num_patches, embedding_dim]
    batch_size, num_patches, embed_dim = 32, 196, 768

    predictions = [
        torch.randn(batch_size, num_patches, embed_dim)
        for _ in range(3)
    ]
    targets = [
        torch.randn(batch_size, num_patches, embed_dim)
        for _ in range(3)
    ]

    # Compute loss
    loss_dict = loss_fn(predictions, targets)

    # Print results
    print(f"Total Loss: {loss_dict['loss'].item():.6f}")
    print(f"Loss at level 0: {loss_dict['loss_h0'].item():.6f}")
    print(f"Loss at level 1: {loss_dict['loss_h1'].item():.6f}")
    print(f"Loss at level 2: {loss_dict['loss_h2'].item():.6f}")
    print(f"Unweighted Loss: {loss_dict['loss_unweighted'].item():.6f}")
    print()


def example_2_vicreg_loss():
    """Example 2: VICReg Loss for Collapse Prevention"""
    print("=" * 60)
    print("Example 2: VICReg Loss")
    print("=" * 60)

    # Create VICReg loss
    loss_fn = VICRegLoss(
        invariance_weight=25.0,
        variance_weight=25.0,
        covariance_weight=1.0,
        variance_threshold=1.0,
    )

    # Two views of the same data (e.g., different augmentations)
    batch_size, num_patches, embed_dim = 32, 196, 768

    view_a = torch.randn(batch_size, num_patches, embed_dim)
    view_b = torch.randn(batch_size, num_patches, embed_dim)

    # Compute loss
    loss_dict = loss_fn(view_a, view_b)

    # Print results
    print(f"Total Loss: {loss_dict['loss'].item():.6f}")
    print(f"Invariance Loss: {loss_dict['invariance_loss'].item():.6f}")
    print(f"Variance Loss: {loss_dict['variance_loss'].item():.6f}")
    print(f"Covariance Loss: {loss_dict['covariance_loss'].item():.6f}")
    print()


def example_3_combined_loss():
    """Example 3: Combined H-JEPA + VICReg Loss"""
    print("=" * 60)
    print("Example 3: Combined H-JEPA + VICReg Loss")
    print("=" * 60)

    # Create combined loss
    loss_fn = CombinedLoss(
        jepa_loss_type='smoothl1',
        jepa_hierarchy_weights=[1.0, 0.5, 0.25],
        num_hierarchies=3,
        vicreg_weight=0.1,  # Scale VICReg relative to JEPA
        vicreg_invariance_weight=25.0,
        vicreg_variance_weight=25.0,
        vicreg_covariance_weight=1.0,
        apply_vicreg_per_level=True,
    )

    # Simulate hierarchical predictions and targets
    batch_size, num_patches, embed_dim = 32, 196, 768

    predictions = [
        torch.randn(batch_size, num_patches, embed_dim)
        for _ in range(3)
    ]
    targets = [
        torch.randn(batch_size, num_patches, embed_dim)
        for _ in range(3)
    ]

    # Compute loss
    loss_dict = loss_fn(predictions, targets)

    # Print results
    print(f"Total Loss: {loss_dict['loss'].item():.6f}")
    print(f"JEPA Loss: {loss_dict['jepa_loss'].item():.6f}")
    print(f"VICReg Loss: {loss_dict['vicreg_loss'].item():.6f}")
    print()
    print("Per-level breakdown:")
    for i in range(3):
        jepa = loss_dict[f'loss_h{i}'].item()
        vicreg = loss_dict.get(f'vicreg_h{i}', torch.tensor(0.0)).item()
        print(f"  Level {i}: JEPA={jepa:.6f}, VICReg={vicreg:.6f}")
    print()

    # Get formatted summary
    summary = loss_fn.get_loss_summary(loss_dict)
    print(summary)
    print()


def example_4_hierarchical_combined():
    """Example 4: Hierarchical Combined Loss with Level-Specific Configs"""
    print("=" * 60)
    print("Example 4: Hierarchical Combined Loss")
    print("=" * 60)

    # Different VICReg configurations for each hierarchy level
    vicreg_configs = [
        {
            'invariance_weight': 25.0,
            'variance_weight': 25.0,
            'covariance_weight': 1.0,
        },
        {
            'invariance_weight': 15.0,
            'variance_weight': 15.0,
            'covariance_weight': 0.5,
        },
        {
            'invariance_weight': 10.0,
            'variance_weight': 10.0,
            'covariance_weight': 0.25,
        },
    ]

    loss_fn = HierarchicalCombinedLoss(
        jepa_loss_type='smoothl1',
        jepa_hierarchy_weights=[1.0, 0.5, 0.25],
        num_hierarchies=3,
        vicreg_weight=[0.1, 0.05, 0.025],  # Different weights per level
        vicreg_configs=vicreg_configs,
    )

    # Simulate data
    batch_size, num_patches, embed_dim = 32, 196, 768
    predictions = [
        torch.randn(batch_size, num_patches, embed_dim)
        for _ in range(3)
    ]
    targets = [
        torch.randn(batch_size, num_patches, embed_dim)
        for _ in range(3)
    ]

    # Compute loss
    loss_dict = loss_fn(predictions, targets)

    # Print results
    print(f"Total Loss: {loss_dict['loss'].item():.6f}")
    print(f"JEPA Loss: {loss_dict['jepa_loss'].item():.6f}")
    print(f"VICReg Loss: {loss_dict['vicreg_loss'].item():.6f}")
    print()


def example_5_with_masking():
    """Example 5: H-JEPA Loss with Patch Masking"""
    print("=" * 60)
    print("Example 5: H-JEPA Loss with Patch Masking")
    print("=" * 60)

    loss_fn = HJEPALoss(
        loss_type='smoothl1',
        hierarchy_weights=[1.0, 0.5, 0.25],
        num_hierarchies=3,
    )

    # Simulate data with masks
    batch_size, num_patches, embed_dim = 32, 196, 768

    predictions = [
        torch.randn(batch_size, num_patches, embed_dim)
        for _ in range(3)
    ]
    targets = [
        torch.randn(batch_size, num_patches, embed_dim)
        for _ in range(3)
    ]

    # Create random binary masks (1 = include in loss, 0 = exclude)
    masks = [
        torch.randint(0, 2, (batch_size, num_patches), dtype=torch.float32)
        for _ in range(3)
    ]

    # Compute loss with masking
    loss_dict = loss_fn(predictions, targets, masks)

    print(f"Total Loss (with masking): {loss_dict['loss'].item():.6f}")
    print(f"Loss at level 0: {loss_dict['loss_h0'].item():.6f}")
    print()


def example_6_from_config():
    """Example 6: Creating Loss from Configuration Dict"""
    print("=" * 60)
    print("Example 6: Creating Loss from Config")
    print("=" * 60)

    # Configuration dictionary (similar to YAML config)
    config = {
        'type': 'combined',
        'jepa_loss_type': 'smoothl1',
        'hierarchy_weights': [1.0, 0.5, 0.25],
        'num_hierarchies': 3,
        'normalize_embeddings': True,
        'vicreg_weight': 0.1,
        'vicreg_invariance_weight': 25.0,
        'vicreg_variance_weight': 25.0,
        'vicreg_covariance_weight': 1.0,
    }

    # Create loss from config
    loss_fn = create_loss_from_config(config)

    print(f"Created loss: {loss_fn.__class__.__name__}")
    print(f"Configuration: {loss_fn}")
    print()


def example_7_training_loop():
    """Example 7: Integration in Training Loop"""
    print("=" * 60)
    print("Example 7: Training Loop Integration")
    print("=" * 60)

    # Setup
    loss_fn = CombinedLoss(
        jepa_loss_type='smoothl1',
        jepa_hierarchy_weights=[1.0, 0.5, 0.25],
        num_hierarchies=3,
        vicreg_weight=0.1,
    )

    # Dummy model (in practice, this would be your H-JEPA model)
    class DummyModel(nn.Module):
        def forward(self, x):
            # Returns predictions and targets for 3 levels
            batch_size = x.shape[0]
            predictions = [torch.randn(batch_size, 196, 768) for _ in range(3)]
            targets = [torch.randn(batch_size, 196, 768) for _ in range(3)]
            return predictions, targets

    model = DummyModel()
    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=1e-4)

    # Simulate training step
    print("Simulating training step...")

    batch = torch.randn(32, 3, 224, 224)  # Dummy batch

    # Forward pass
    predictions, targets = model(batch)

    # Compute loss
    loss_dict = loss_fn(predictions, targets)
    total_loss = loss_dict['loss']

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"Step completed. Loss: {total_loss.item():.6f}")
    print(f"  JEPA component: {loss_dict['jepa_loss'].item():.6f}")
    print(f"  VICReg component: {loss_dict['vicreg_loss'].item():.6f}")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("H-JEPA Loss Functions - Usage Examples")
    print("=" * 60 + "\n")

    # Run all examples
    example_1_basic_hjepa_loss()
    example_2_vicreg_loss()
    example_3_combined_loss()
    example_4_hierarchical_combined()
    example_5_with_masking()
    example_6_from_config()
    example_7_training_loop()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

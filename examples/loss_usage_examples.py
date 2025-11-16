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
    SIGRegLoss,
    HybridVICRegSIGRegLoss,
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


def example_7_sigreg_loss():
    """Example 7: SIGReg Loss (Improved Stability)"""
    print("=" * 60)
    print("Example 7: SIGReg Loss - Improved Training Stability")
    print("=" * 60)

    # Create SIGReg loss
    loss_fn = SIGRegLoss(
        num_slices=1024,              # Number of random projections
        num_test_points=17,            # Reference Gaussian points
        invariance_weight=25.0,        # MSE weight
        sigreg_weight=25.0,            # SIGReg regularization weight
        flatten_patches=True,
        fixed_slices=False,
    )

    # Two views of the same data
    batch_size, num_patches, embed_dim = 32, 196, 768

    view_a = torch.randn(batch_size, num_patches, embed_dim)
    view_b = torch.randn(batch_size, num_patches, embed_dim)

    # Compute loss
    loss_dict = loss_fn(view_a, view_b)

    # Print results
    print(f"Total Loss: {loss_dict['loss'].item():.6f}")
    print(f"Invariance Loss: {loss_dict['invariance_loss'].item():.6f}")
    print(f"SIGReg Loss: {loss_dict['sigreg_loss'].item():.6f}")
    print(f"  SIGReg A: {loss_dict['sigreg_loss_a'].item():.6f}")
    print(f"  SIGReg B: {loss_dict['sigreg_loss_b'].item():.6f}")
    print()
    print("SIGReg Benefits:")
    print("  - O(K) complexity vs O(K²) for VICReg covariance")
    print("  - Single hyperparameter (num_slices)")
    print("  - Better training stability (from LeJEPA paper)")
    print("  - Theoretically grounded in optimal Gaussian distribution")
    print()


def example_8_hybrid_vicreg_sigreg():
    """Example 8: Hybrid VICReg + SIGReg Loss"""
    print("=" * 60)
    print("Example 8: Hybrid VICReg + SIGReg Loss")
    print("=" * 60)

    # Create hybrid loss for gradual transition or ablation studies
    loss_fn = HybridVICRegSIGRegLoss(
        vicreg_weight=1.0,      # Start with VICReg
        sigreg_weight=0.0,      # Gradually increase SIGReg
        invariance_weight=25.0,
        variance_weight=25.0,
        covariance_weight=1.0,
        num_slices=1024,
        num_test_points=17,
    )

    # Two views
    batch_size, num_patches, embed_dim = 32, 196, 768
    view_a = torch.randn(batch_size, num_patches, embed_dim)
    view_b = torch.randn(batch_size, num_patches, embed_dim)

    # Compute loss
    loss_dict = loss_fn(view_a, view_b)

    # Print results
    print(f"Total Loss: {loss_dict['loss'].item():.6f}")
    print(f"VICReg Component: {loss_dict['vicreg_loss'].item():.6f}")
    print(f"SIGReg Component: {loss_dict['sigreg_loss'].item():.6f}")
    print()
    print("Hybrid Loss Use Cases:")
    print("  - Gradual transition from VICReg to SIGReg during training")
    print("  - Ablation studies comparing VICReg vs SIGReg")
    print("  - Combining strengths of both approaches")
    print()
    print("Adjust weights during training:")
    print("  loss_fn.vicreg_weight = 0.5  # Decrease VICReg")
    print("  loss_fn.sigreg_weight = 0.5  # Increase SIGReg")
    print()


def example_9_sigreg_from_config():
    """Example 9: SIGReg from Configuration"""
    print("=" * 60)
    print("Example 9: Creating SIGReg from Config")
    print("=" * 60)

    # Configuration dictionary for SIGReg
    config = {
        'type': 'sigreg',
        'sigreg_num_slices': 1024,
        'sigreg_num_test_points': 17,
        'sigreg_invariance_weight': 25.0,
        'sigreg_weight': 25.0,
        'sigreg_fixed_slices': False,
        'flatten_patches': True,
        'eps': 1e-6,
    }

    # Create loss from config
    loss_fn = create_loss_from_config(config)

    print(f"Created loss: {loss_fn.__class__.__name__}")
    print(f"Configuration: {loss_fn}")
    print()

    # Simulate usage
    batch_size, num_patches, embed_dim = 16, 196, 768
    view_a = torch.randn(batch_size, num_patches, embed_dim)
    view_b = torch.randn(batch_size, num_patches, embed_dim)

    loss_dict = loss_fn(view_a, view_b)
    print(f"Total Loss: {loss_dict['loss'].item():.6f}")
    print()


def example_10_training_loop():
    """Example 10: Integration in Training Loop"""
    print("=" * 60)
    print("Example 10: Training Loop Integration with SIGReg")
    print("=" * 60)

    # Setup with SIGReg for better stability
    loss_fn = SIGRegLoss(
        num_slices=1024,
        invariance_weight=25.0,
        sigreg_weight=25.0,
    )

    # Dummy model (in practice, this would be your H-JEPA model)
    class DummyModel(nn.Module):
        def forward(self, x):
            # Returns two views of representations
            batch_size = x.shape[0]
            view_a = torch.randn(batch_size, 196, 768)
            view_b = torch.randn(batch_size, 196, 768)
            return view_a, view_b

    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Simulate training step
    print("Simulating training step...")

    batch = torch.randn(32, 3, 224, 224)  # Dummy batch

    # Forward pass
    view_a, view_b = model(batch)

    # Compute loss
    loss_dict = loss_fn(view_a, view_b)
    total_loss = loss_dict['loss']

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"Step completed. Loss: {total_loss.item():.6f}")
    print(f"  Invariance component: {loss_dict['invariance_loss'].item():.6f}")
    print(f"  SIGReg component: {loss_dict['sigreg_loss'].item():.6f}")
    print()


def example_11_comparison():
    """Example 11: VICReg vs SIGReg Comparison"""
    print("=" * 60)
    print("Example 11: VICReg vs SIGReg Comparison")
    print("=" * 60)

    # Same input for both losses
    batch_size, num_patches, embed_dim = 32, 196, 768
    view_a = torch.randn(batch_size, num_patches, embed_dim)
    view_b = torch.randn(batch_size, num_patches, embed_dim)

    # VICReg Loss
    vicreg_loss_fn = VICRegLoss(
        invariance_weight=25.0,
        variance_weight=25.0,
        covariance_weight=1.0,
    )

    # SIGReg Loss
    sigreg_loss_fn = SIGRegLoss(
        num_slices=1024,
        invariance_weight=25.0,
        sigreg_weight=25.0,
    )

    # Compute both losses
    import time

    start = time.time()
    vicreg_dict = vicreg_loss_fn(view_a, view_b)
    vicreg_time = time.time() - start

    start = time.time()
    sigreg_dict = sigreg_loss_fn(view_a, view_b)
    sigreg_time = time.time() - start

    # Print comparison
    print("VICReg Results:")
    print(f"  Total Loss: {vicreg_dict['loss'].item():.6f}")
    print(f"  Invariance: {vicreg_dict['invariance_loss'].item():.6f}")
    print(f"  Variance: {vicreg_dict['variance_loss'].item():.6f}")
    print(f"  Covariance: {vicreg_dict['covariance_loss'].item():.6f}")
    print(f"  Time: {vicreg_time*1000:.2f}ms")
    print()

    print("SIGReg Results:")
    print(f"  Total Loss: {sigreg_dict['loss'].item():.6f}")
    print(f"  Invariance: {sigreg_dict['invariance_loss'].item():.6f}")
    print(f"  SIGReg Reg: {sigreg_dict['sigreg_loss'].item():.6f}")
    print(f"  Time: {sigreg_time*1000:.2f}ms")
    print()

    print("Comparison Summary:")
    print(f"  Complexity: VICReg O(K²) vs SIGReg O(K)")
    print(f"  Hyperparameters: VICReg 3 weights vs SIGReg 1 weight")
    print(f"  Stability: SIGReg superior (from LeJEPA paper)")
    print(f"  Time Ratio: {sigreg_time/vicreg_time:.2f}x")
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
    example_7_sigreg_loss()
    example_8_hybrid_vicreg_sigreg()
    example_9_sigreg_from_config()
    example_10_training_loop()
    example_11_comparison()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

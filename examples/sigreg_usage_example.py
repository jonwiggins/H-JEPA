"""
SIGReg Usage Examples - Sign-based Regularization

This script demonstrates how to use SIGReg (Sketched Isotropic Gaussian
Regularization) from the LeJEPA paper as an improved alternative to VICReg
for H-JEPA training.

SIGReg provides:
- Better training stability
- O(K) complexity vs O(K²) for VICReg
- Single hyperparameter (num_slices) vs 3 weights
- Theoretically grounded in optimal Gaussian distribution

Reference: LeJEPA paper (https://arxiv.org/abs/2511.08544)
"""

import torch
import torch.nn as nn
from src.losses import (
    SIGRegLoss,
    HybridVICRegSIGRegLoss,
    EppsPulleyTest,
    VICRegLoss,
    create_loss_from_config,
)


def example_1_basic_sigreg():
    """Example 1: Basic SIGReg Loss Usage"""
    print("=" * 70)
    print("Example 1: Basic SIGReg Loss Usage")
    print("=" * 70)

    # Create SIGReg loss with default parameters
    loss_fn = SIGRegLoss(
        num_slices=1024,              # Number of random projections
        num_test_points=17,            # Reference Gaussian points
        invariance_weight=25.0,        # MSE weight
        sigreg_weight=25.0,            # SIGReg regularization weight
    )

    # Simulate two views of the same data (different augmentations)
    batch_size, num_patches, embed_dim = 32, 196, 768
    view_a = torch.randn(batch_size, num_patches, embed_dim)
    view_b = torch.randn(batch_size, num_patches, embed_dim)

    # Compute loss
    loss_dict = loss_fn(view_a, view_b)

    # Print results
    print(f"\nTotal Loss: {loss_dict['loss'].item():.6f}")
    print(f"  Invariance Loss: {loss_dict['invariance_loss'].item():.6f}")
    print(f"  SIGReg Loss: {loss_dict['sigreg_loss'].item():.6f}")
    print(f"    - SIGReg View A: {loss_dict['sigreg_loss_a'].item():.6f}")
    print(f"    - SIGReg View B: {loss_dict['sigreg_loss_b'].item():.6f}")
    print()


def example_2_epps_pulley_test():
    """Example 2: Understanding the Epps-Pulley Test"""
    print("=" * 70)
    print("Example 2: Epps-Pulley Test - Statistical Testing")
    print("=" * 70)

    # Create Epps-Pulley test
    test = EppsPulleyTest(num_points=17)

    print("\nTesting different distributions:")
    print("-" * 50)

    # Test 1: Standard Gaussian (should be close to 0)
    gaussian = torch.randn(1000)
    stat_gaussian = test(gaussian).item()
    print(f"Standard Gaussian N(0,1): {stat_gaussian:.6f} (should be ~0)")

    # Test 2: Non-standard Gaussian (shifted mean)
    shifted = torch.randn(1000) + 5.0
    stat_shifted = test(shifted).item()
    print(f"Shifted Gaussian N(5,1): {stat_shifted:.6f} (higher)")

    # Test 3: Uniform distribution
    uniform = torch.rand(1000) * 2 - 1  # Uniform[-1, 1]
    stat_uniform = test(uniform).item()
    print(f"Uniform[-1,1]: {stat_uniform:.6f} (much higher)")

    # Test 4: Constant (complete collapse)
    constant = torch.ones(1000)
    stat_constant = test(constant).item()
    print(f"Constant (collapsed): {stat_constant:.6f} (very high)")

    print("\nInterpretation:")
    print("  - Lower values = closer to standard Gaussian")
    print("  - Higher values = further from ideal distribution")
    print("  - SIGReg minimizes this statistic during training")
    print()


def example_3_hyperparameter_tuning():
    """Example 3: Hyperparameter Tuning Guide"""
    print("=" * 70)
    print("Example 3: Hyperparameter Tuning")
    print("=" * 70)

    print("\nRecommended settings from LeJEPA paper:")
    print("-" * 50)
    print("  num_slices:           1024    (balance of accuracy/speed)")
    print("  num_test_points:      17      (sufficient for testing)")
    print("  invariance_weight:    25.0    (same as VICReg)")
    print("  sigreg_weight:        25.0    (equal to invariance)")
    print("  fixed_slices:         False   (random reduces variance)")
    print()

    # Demonstrate different configurations
    configs = [
        ("Small/Fast", {"num_slices": 512, "sigreg_weight": 25.0}),
        ("Standard", {"num_slices": 1024, "sigreg_weight": 25.0}),
        ("Large/Thorough", {"num_slices": 2048, "sigreg_weight": 30.0}),
    ]

    batch_size, num_patches, embed_dim = 16, 196, 768
    view_a = torch.randn(batch_size, num_patches, embed_dim)
    view_b = torch.randn(batch_size, num_patches, embed_dim)

    print("Performance comparison:")
    print("-" * 50)

    import time

    for name, params in configs:
        loss_fn = SIGRegLoss(**params, invariance_weight=25.0)

        # Time the computation
        start = time.time()
        loss_dict = loss_fn(view_a, view_b)
        elapsed = time.time() - start

        print(f"{name:15s}: Loss={loss_dict['loss'].item():.4f}, "
              f"Time={elapsed*1000:.2f}ms, Slices={params['num_slices']}")

    print()


def example_4_fixed_vs_random_slices():
    """Example 4: Fixed vs Random Slices"""
    print("=" * 70)
    print("Example 4: Fixed vs Random Slices")
    print("=" * 70)

    # Same input
    batch_size, num_patches, embed_dim = 32, 196, 768
    view_a = torch.randn(batch_size, num_patches, embed_dim)
    view_b = torch.randn(batch_size, num_patches, embed_dim)

    # Fixed slices (reproducible)
    loss_fn_fixed = SIGRegLoss(
        num_slices=1024,
        fixed_slices=True,
        sigreg_weight=25.0,
    )

    # Random slices (default)
    loss_fn_random = SIGRegLoss(
        num_slices=1024,
        fixed_slices=False,
        sigreg_weight=25.0,
    )

    print("\nFixed slices (3 runs):")
    for i in range(3):
        loss_dict = loss_fn_fixed(view_a, view_b)
        print(f"  Run {i+1}: {loss_dict['sigreg_loss'].item():.6f}")

    print("\nRandom slices (3 runs):")
    for i in range(3):
        loss_dict = loss_fn_random(view_a, view_b)
        print(f"  Run {i+1}: {loss_dict['sigreg_loss'].item():.6f}")

    print("\nTrade-offs:")
    print("  Fixed slices:")
    print("    + Reproducible results")
    print("    + Slightly faster (cached)")
    print("    - May have higher variance in gradients")
    print("  Random slices:")
    print("    + Better statistical properties")
    print("    + Lower gradient variance")
    print("    - Non-deterministic")
    print()


def example_5_vicreg_vs_sigreg():
    """Example 5: VICReg vs SIGReg Comparison"""
    print("=" * 70)
    print("Example 5: VICReg vs SIGReg Comparison")
    print("=" * 70)

    # Same input
    batch_size, num_patches, embed_dim = 32, 196, 768
    view_a = torch.randn(batch_size, num_patches, embed_dim)
    view_b = torch.randn(batch_size, num_patches, embed_dim)

    # VICReg
    vicreg = VICRegLoss(
        invariance_weight=25.0,
        variance_weight=25.0,
        covariance_weight=1.0,
    )

    # SIGReg
    sigreg = SIGRegLoss(
        num_slices=1024,
        invariance_weight=25.0,
        sigreg_weight=25.0,
    )

    # Time both
    import time

    # VICReg
    start = time.time()
    vicreg_dict = vicreg(view_a, view_b)
    vicreg_time = time.time() - start

    # SIGReg
    start = time.time()
    sigreg_dict = sigreg(view_a, view_b)
    sigreg_time = time.time() - start

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'VICReg':<20} {'SIGReg':<20}")
    print("=" * 70)
    print(f"{'Total Loss':<25} {vicreg_dict['loss'].item():<20.6f} "
          f"{sigreg_dict['loss'].item():<20.6f}")
    print(f"{'Invariance Loss':<25} {vicreg_dict['invariance_loss'].item():<20.6f} "
          f"{sigreg_dict['invariance_loss'].item():<20.6f}")
    print(f"{'Variance Loss':<25} {vicreg_dict['variance_loss'].item():<20.6f} "
          f"{'N/A':<20}")
    print(f"{'Covariance Loss':<25} {vicreg_dict['covariance_loss'].item():<20.6f} "
          f"{'N/A':<20}")
    print(f"{'SIGReg Loss':<25} {'N/A':<20} "
          f"{sigreg_dict['sigreg_loss'].item():<20.6f}")
    print(f"{'Computation Time':<25} {vicreg_time*1000:<20.2f} "
          f"{sigreg_time*1000:<20.2f}")
    print("=" * 70)

    print("\nKey Differences:")
    print(f"  Complexity:        VICReg O(K²)  vs  SIGReg O(K)")
    print(f"  Hyperparameters:   VICReg 3      vs  SIGReg 1")
    print(f"  Time Ratio:        {sigreg_time/vicreg_time:.2f}x")
    print(f"  Memory Usage:      VICReg Higher vs  SIGReg Lower")
    print(f"  Training Stability: VICReg Good   vs  SIGReg Superior")
    print()


def example_6_hybrid_transition():
    """Example 6: Hybrid VICReg → SIGReg Transition"""
    print("=" * 70)
    print("Example 6: Hybrid Transition from VICReg to SIGReg")
    print("=" * 70)

    # Create hybrid loss
    loss_fn = HybridVICRegSIGRegLoss(
        vicreg_weight=1.0,      # Start with full VICReg
        sigreg_weight=0.0,      # Start with no SIGReg
        invariance_weight=25.0,
        num_slices=1024,
    )

    # Simulate data
    batch_size, num_patches, embed_dim = 32, 196, 768
    view_a = torch.randn(batch_size, num_patches, embed_dim)
    view_b = torch.randn(batch_size, num_patches, embed_dim)

    print("\nSimulating training progression:")
    print("-" * 70)

    # Simulate different training phases
    phases = [
        (0, 1.0, 0.0, "Early: 100% VICReg"),
        (100, 0.75, 0.25, "Early-Mid: 75% VICReg, 25% SIGReg"),
        (200, 0.5, 0.5, "Mid: 50% VICReg, 50% SIGReg"),
        (300, 0.25, 0.75, "Mid-Late: 25% VICReg, 75% SIGReg"),
        (400, 0.0, 1.0, "Late: 100% SIGReg"),
    ]

    for epoch, vic_w, sig_w, desc in phases:
        loss_fn.vicreg_weight = vic_w
        loss_fn.sigreg_weight = sig_w

        loss_dict = loss_fn(view_a, view_b)

        print(f"Epoch {epoch:3d} ({desc:30s}): "
              f"Total={loss_dict['loss'].item():.4f}, "
              f"VIC={loss_dict['vicreg_loss'].item():.4f}, "
              f"SIG={loss_dict['sigreg_loss'].item():.4f}")

    print("\nUse Cases for Hybrid Loss:")
    print("  1. Gradual transition during training")
    print("  2. Ablation studies comparing both methods")
    print("  3. Combining strengths of both approaches")
    print()


def example_7_config_based():
    """Example 7: Configuration-Based Usage"""
    print("=" * 70)
    print("Example 7: Configuration-Based Usage")
    print("=" * 70)

    # Define different configurations
    configs = {
        "Small Model": {
            'type': 'sigreg',
            'sigreg_num_slices': 512,
            'sigreg_weight': 25.0,
            'sigreg_invariance_weight': 25.0,
        },
        "Standard Model": {
            'type': 'sigreg',
            'sigreg_num_slices': 1024,
            'sigreg_weight': 25.0,
            'sigreg_invariance_weight': 25.0,
        },
        "Large Model": {
            'type': 'sigreg',
            'sigreg_num_slices': 2048,
            'sigreg_weight': 30.0,
            'sigreg_invariance_weight': 25.0,
            'sigreg_fixed_slices': True,
        },
        "Hybrid": {
            'type': 'hybrid_vicreg_sigreg',  # Not yet in factory, but shown for reference
            'vicreg_weight': 0.5,
            'sigreg_weight': 0.5,
        },
    }

    print("\nCreating losses from configs:")
    print("-" * 70)

    for name, config in configs.items():
        if config['type'] == 'hybrid_vicreg_sigreg':
            print(f"{name:15s}: Hybrid not in factory (use HybridVICRegSIGRegLoss directly)")
            continue

        loss_fn = create_loss_from_config(config)
        print(f"{name:15s}: {loss_fn.__class__.__name__}")
        print(f"                 Config: {loss_fn}")

    print()


def example_8_training_integration():
    """Example 8: Integration in Training Loop"""
    print("=" * 70)
    print("Example 8: Training Loop Integration")
    print("=" * 70)

    # Setup
    loss_fn = SIGRegLoss(
        num_slices=1024,
        invariance_weight=25.0,
        sigreg_weight=25.0,
    )

    # Dummy encoder (in practice, this is your H-JEPA encoder)
    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(768, 768)

        def forward(self, x):
            # x: [B, N, D]
            return self.proj(x)

    encoder = SimpleEncoder()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4)

    print("\nSimulating training steps:")
    print("-" * 70)

    # Simulate a few training steps
    for step in range(5):
        # Create two augmented views
        batch_size, num_patches, embed_dim = 32, 196, 768
        view_a = torch.randn(batch_size, num_patches, embed_dim)
        view_b = torch.randn(batch_size, num_patches, embed_dim)

        # Encode
        z_a = encoder(view_a)
        z_b = encoder(view_b)

        # Compute loss
        loss_dict = loss_fn(z_a, z_b)
        total_loss = loss_dict['loss']

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log
        print(f"Step {step:2d}: Loss={total_loss.item():.4f}, "
              f"Inv={loss_dict['invariance_loss'].item():.4f}, "
              f"SIG={loss_dict['sigreg_loss'].item():.4f}")

    print("\nTraining tips:")
    print("  - Monitor both invariance and SIGReg terms")
    print("  - SIGReg should stay relatively stable and low")
    print("  - Invariance should decrease as views align")
    print("  - Keep ratio around 1:1 for balanced training")
    print()


def example_9_memory_efficiency():
    """Example 9: Memory Efficiency Comparison"""
    print("=" * 70)
    print("Example 9: Memory Efficiency")
    print("=" * 70)

    print("\nMemory usage comparison (approximate):")
    print("-" * 70)

    # Different embedding dimensions
    dims = [384, 768, 1024, 2048]

    print(f"{'Embed Dim':<12} {'VICReg Cov':<15} {'SIGReg Slices':<15} {'Ratio':<10}")
    print("-" * 70)

    for dim in dims:
        # VICReg stores D x D covariance matrix
        vicreg_mem = dim * dim * 4 / 1024  # KB (float32)

        # SIGReg stores M x D random slices
        sigreg_mem = 1024 * dim * 4 / 1024  # KB (M=1024)

        ratio = vicreg_mem / sigreg_mem

        print(f"{dim:<12} {vicreg_mem:<15.1f} {sigreg_mem:<15.1f} {ratio:<10.2f}x")

    print()
    print("Key observations:")
    print("  - VICReg memory grows quadratically O(D²)")
    print("  - SIGReg memory grows linearly O(M×D)")
    print("  - For large models (D > 1024), SIGReg much more efficient")
    print()


def example_10_troubleshooting():
    """Example 10: Common Issues and Solutions"""
    print("=" * 70)
    print("Example 10: Troubleshooting Guide")
    print("=" * 70)

    print("\nCommon Issues and Solutions:")
    print("=" * 70)

    issues = [
        ("SIGReg loss increasing",
         "Embeddings diverging from Gaussian",
         ["Increase sigreg_weight", "Check normalization", "Reduce learning rate"]),

        ("Out of memory",
         "Too many slices or large batch",
         ["Reduce num_slices to 512", "Use fixed_slices=True", "Reduce batch size"]),

        ("Training unstable",
         "Imbalanced loss terms",
         ["Balance inv/sig weights 1:1", "Use gradient clipping", "Check for NaN"]),

        ("Slower than VICReg",
         "Large batch, small embedding dim",
         ["Reduce num_slices", "Use fixed_slices=True", "Consider VICReg for D<512"]),
    ]

    for issue, cause, solutions in issues:
        print(f"\nIssue: {issue}")
        print(f"Cause: {cause}")
        print("Solutions:")
        for sol in solutions:
            print(f"  - {sol}")

    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SIGReg Usage Examples - Sign-based Regularization")
    print("Improved training stability from LeJEPA paper")
    print("=" * 70 + "\n")

    # Run all examples
    example_1_basic_sigreg()
    example_2_epps_pulley_test()
    example_3_hyperparameter_tuning()
    example_4_fixed_vs_random_slices()
    example_5_vicreg_vs_sigreg()
    example_6_hybrid_transition()
    example_7_config_based()
    example_8_training_integration()
    example_9_memory_efficiency()
    example_10_troubleshooting()

    print("=" * 70)
    print("All SIGReg examples completed successfully!")
    print("=" * 70)
    print("\nFor more information:")
    print("  - Documentation: docs/SIGREG_IMPLEMENTATION.md")
    print("  - Configuration: configs/sigreg_example.yaml")
    print("  - Implementation: src/losses/sigreg.py")
    print("  - Paper: https://arxiv.org/abs/2511.08544")
    print("=" * 70)

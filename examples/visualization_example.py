#!/usr/bin/env python3
"""
Simple example demonstrating H-JEPA visualization capabilities.

This script shows basic usage of all visualization modules without requiring
a trained model or real images.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.hjepa import create_hjepa
from src.visualization import *


def example_masking_visualization():
    """Example: Visualize masking strategies."""
    print("\n" + "="*80)
    print("Example 1: Masking Visualization")
    print("="*80)

    # Generate random masks
    num_patches = 196  # 14x14 grid

    # Single mask visualization
    mask = torch.zeros(num_patches)
    masked_indices = torch.randperm(num_patches)[:int(num_patches * 0.5)]
    mask[masked_indices] = 1

    fig = visualize_masking_strategy(
        mask,
        title="Example Masking Pattern"
    )
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_mask.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_mask.png")

    # Multi-block masking
    fig = visualize_multi_block_masking(
        num_samples=6,
        grid_size=14,
        num_blocks=4
    )
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_multiblock.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_multiblock.png")

    # Compare strategies
    masks = []
    labels = []
    for ratio in [0.3, 0.5, 0.7]:
        m = torch.zeros(num_patches)
        m[torch.randperm(num_patches)[:int(num_patches * ratio)]] = 1
        masks.append(m)
        labels.append(f'{ratio:.0%} masked')

    fig = compare_masking_strategies(masks, labels)
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_comparison.png")


def example_attention_visualization():
    """Example: Visualize attention patterns with a simple model."""
    print("\n" + "="*80)
    print("Example 2: Attention Visualization")
    print("="*80)

    # Create a simple model (randomly initialized)
    model = create_hjepa(
        encoder_type='vit_base_patch16_224',
        img_size=224,
        embed_dim=768,
        num_hierarchies=3,
        pretrained=False
    )
    model.eval()

    # Create dummy image
    image = torch.randn(1, 3, 224, 224)

    print("Extracting attention maps...")
    from src.visualization.attention_viz import extract_attention_maps

    # Extract attention from a few layers
    attention_maps = extract_attention_maps(model, image, layer_indices=[0, 5, 11])

    # Note: For randomly initialized model, attention won't be meaningful
    # but this demonstrates the API
    print(f"Extracted attention from {len(attention_maps)} layers")
    for layer, attn in attention_maps.items():
        print(f"  {layer}: {attn.shape}")

    print("\nNote: Use a trained model for meaningful attention visualizations")


def example_feature_space_visualization():
    """Example: Visualize feature space."""
    print("\n" + "="*80)
    print("Example 3: Feature Space Visualization")
    print("="*80)

    # Generate random features (simulating embeddings)
    num_samples = 500
    embed_dim = 768

    # Create clustered features for demonstration
    features = []
    labels = []

    for cluster in range(5):
        # Each cluster has different mean
        cluster_mean = torch.randn(embed_dim) * 2
        cluster_features = torch.randn(num_samples // 5, embed_dim) * 0.5 + cluster_mean
        features.append(cluster_features)
        labels.extend([cluster] * (num_samples // 5))

    features = torch.cat(features, dim=0)
    labels = np.array(labels)

    print("Visualizing feature space with t-SNE...")
    fig = visualize_feature_space(
        features,
        labels=labels,
        method='tsne',
        perplexity=30,
        random_state=42
    )
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_tsne.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_tsne.png")

    print("Visualizing feature space with PCA...")
    fig = visualize_feature_space(
        features,
        labels=labels,
        method='pca'
    )
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_pca.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_pca.png")


def example_embedding_analysis():
    """Example: Analyze embedding distribution and check for collapse."""
    print("\n" + "="*80)
    print("Example 4: Embedding Analysis")
    print("="*80)

    # Generate random embeddings
    num_samples = 1000
    embed_dim = 768

    # Case 1: Healthy embeddings
    print("\nCase 1: Healthy embeddings")
    healthy_features = torch.randn(num_samples, embed_dim) * 0.5

    fig = visualize_embedding_distribution(healthy_features)
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_healthy_embeddings.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_healthy_embeddings.png")

    fig = plot_collapse_metrics(healthy_features)
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_healthy_collapse.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_healthy_collapse.png")

    # Case 2: Collapsed embeddings (for demonstration)
    print("\nCase 2: Collapsed embeddings (for demonstration)")
    collapsed_features = torch.randn(num_samples, embed_dim) * 0.01  # Very small variance

    fig = visualize_embedding_distribution(collapsed_features)
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_collapsed_embeddings.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_collapsed_embeddings.png")

    fig = plot_collapse_metrics(collapsed_features)
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_collapsed_collapse.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_collapsed_collapse.png")


def example_training_visualization():
    """Example: Visualize training metrics."""
    print("\n" + "="*80)
    print("Example 5: Training Visualization")
    print("="*80)

    # Generate synthetic training logs
    num_epochs = 100

    # Simulate decreasing loss with noise
    train_loss = [10.0 * np.exp(-0.05 * i) + np.random.randn() * 0.1 for i in range(num_epochs)]
    val_loss = [10.5 * np.exp(-0.045 * i) + np.random.randn() * 0.15 for i in range(num_epochs)]
    learning_rate = [0.001 * (0.95 ** (i // 10)) for i in range(num_epochs)]

    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': learning_rate,
    }

    fig = plot_training_curves(metrics)
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_training_curves.png")

    # Hierarchical losses
    hierarchical_losses = {
        0: [5.0 * np.exp(-0.05 * i) + np.random.randn() * 0.1 for i in range(num_epochs)],
        1: [4.0 * np.exp(-0.05 * i) + np.random.randn() * 0.1 for i in range(num_epochs)],
        2: [3.0 * np.exp(-0.05 * i) + np.random.randn() * 0.1 for i in range(num_epochs)],
    }

    fig = plot_hierarchical_losses(hierarchical_losses)
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_hierarchical_losses.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_hierarchical_losses.png")

    # EMA momentum schedule
    ema_momentum = [0.996 + (1.0 - 0.996) * min(i / 1000, 1.0) for i in range(num_epochs * 100)]

    fig = plot_ema_momentum(ema_momentum)
    plt.savefig('/home/user/H-JEPA/results/visualizations/example_ema_momentum.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: example_ema_momentum.png")


def main():
    """Run all visualization examples."""
    print("="*80)
    print("H-JEPA Visualization Examples")
    print("="*80)
    print("\nThis script demonstrates all visualization capabilities.")
    print("Outputs will be saved to: results/visualizations/")

    # Create output directory
    output_dir = Path('/home/user/H-JEPA/results/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run examples
    try:
        example_masking_visualization()
    except Exception as e:
        print(f"Error in masking visualization: {e}")

    try:
        example_attention_visualization()
    except Exception as e:
        print(f"Error in attention visualization: {e}")

    try:
        example_feature_space_visualization()
    except Exception as e:
        print(f"Error in feature space visualization: {e}")

    try:
        example_embedding_analysis()
    except Exception as e:
        print(f"Error in embedding analysis: {e}")

    try:
        example_training_visualization()
    except Exception as e:
        print(f"Error in training visualization: {e}")

    print("\n" + "="*80)
    print("Examples Complete!")
    print("="*80)
    print(f"\nGenerated visualizations in: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("example_*.png")):
        print(f"  - {file.name}")

    print("\nNext steps:")
    print("  1. Train a model: python scripts/train.py")
    print("  2. Visualize results: python scripts/visualize.py --visualize-all")
    print("  3. Explore interactively: jupyter notebook notebooks/demo.ipynb")


if __name__ == "__main__":
    main()

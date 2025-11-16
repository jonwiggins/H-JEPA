#!/usr/bin/env python3.11
"""
Feature Visualization for H-JEPA

This script visualizes what patterns different feature channels detect
across the hierarchical representations. It shows:

1. Feature activation maps for individual channels
2. Top activating patches for each channel
3. Feature statistics and distributions
4. Comparison across hierarchy levels
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, List, Tuple
import argparse

from src.models.hjepa import create_hjepa
from src.data.datasets import get_dataset


def load_model(checkpoint_path: str, device: str) -> Tuple:
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', {})
    model_state = checkpoint.get('model_state_dict', checkpoint.get('target_encoder', {}))

    model = create_hjepa(
        encoder_type=config.get('model', {}).get('encoder_type', 'vit_base_patch16_224'),
        img_size=config.get('data', {}).get('image_size', 224),
        num_hierarchies=config.get('model', {}).get('num_hierarchies', 3),
        predictor_depth=config.get('model', {}).get('predictor', {}).get('depth', 6),
        predictor_heads=config.get('model', {}).get('predictor', {}).get('num_heads', 6),
        use_rope=config.get('model', {}).get('use_rope', True),
        use_flash_attention=config.get('model', {}).get('use_flash_attention', True),
    )

    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")

    return model, config


def visualize_feature_maps(
    model,
    image: torch.Tensor,
    hierarchy_idx: int,
    num_channels: int = 16,
    device: str = "mps",
    save_path: Optional[str] = None
):
    """Visualize feature activation maps for top channels"""
    print(f"\n=== Feature Activation Maps (Hierarchy {hierarchy_idx+1}) ===")

    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        embeddings = model.target_encoder.forward_hierarchical(image_input)
        features = embeddings[hierarchy_idx]  # (1, C, H, W)

    B, C, H, W = features.shape

    # Compute channel-wise activation magnitudes
    channel_magnitudes = features[0].abs().mean(dim=(1, 2))  # (C,)
    top_channels = channel_magnitudes.topk(num_channels).indices.cpu()

    # Visualize
    ncols = 4
    nrows = (num_channels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Top {num_channels} Feature Activation Maps - Hierarchy {hierarchy_idx+1} ({H}x{W})', fontsize=14)

    for idx, channel_idx in enumerate(top_channels):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        # Get feature map for this channel
        feat_map = features[0, channel_idx].cpu().numpy()

        # Visualize
        im = ax.imshow(feat_map, cmap='viridis', interpolation='bilinear')
        ax.set_title(f'Channel {channel_idx}\nMag: {channel_magnitudes[channel_idx]:.3f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide empty subplots
    for idx in range(num_channels, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved feature maps to {save_path}")
    else:
        plt.show()


def visualize_top_activating_patches(
    model,
    dataset,
    hierarchy_idx: int,
    channel_idx: int,
    num_samples: int = 100,
    top_k: int = 9,
    device: str = "mps",
    save_path: Optional[str] = None
):
    """Find and visualize patches that maximally activate a specific channel"""
    print(f"\n=== Top Activating Patches (Hierarchy {hierarchy_idx+1}, Channel {channel_idx}) ===")

    activations = []
    images_list = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            if i % 20 == 0:
                print(f"Processing {i}/{num_samples}...")

            image, _ = dataset[i]
            images_list.append(image)

            image_input = image.unsqueeze(0).to(device)

            # Get features
            embeddings = model.target_encoder.forward_hierarchical(image_input)
            features = embeddings[hierarchy_idx]  # (1, C, H, W)

            # Get activation for this channel
            channel_activation = features[0, channel_idx]  # (H, W)

            # Store max activation and location
            max_activation = channel_activation.max()
            max_loc = channel_activation.argmax()
            h_idx = max_loc // channel_activation.shape[1]
            w_idx = max_loc % channel_activation.shape[1]

            activations.append({
                'image_idx': i,
                'activation': max_activation.item(),
                'location': (h_idx.item(), w_idx.item()),
            })

    # Sort by activation
    activations.sort(key=lambda x: x['activation'], reverse=True)
    top_activations = activations[:top_k]

    # Visualize top activating images and patches
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'Top {top_k} Activating Patches - Hierarchy {hierarchy_idx+1}, Channel {channel_idx}', fontsize=14)

    for idx, act_info in enumerate(top_activations):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        # Get image
        image = images_list[act_info['image_idx']]

        # Denormalize for visualization
        image_vis = image.clone()
        image_vis = image_vis * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_vis = image_vis + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image_vis = image_vis.clamp(0, 1)

        # Display image
        ax.imshow(image_vis.permute(1, 2, 0).numpy())

        # Mark the patch location
        h, w = act_info['location']
        _, _, img_h, img_w = image.shape[0], image.shape[0], 224, 224
        H, W = embeddings[hierarchy_idx].shape[2:]

        # Calculate patch size in image space
        patch_h = img_h / H
        patch_w = img_w / W

        # Draw rectangle around activating patch
        rect = plt.Rectangle(
            (w * patch_w, h * patch_h),
            patch_w, patch_h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        ax.set_title(f'Activation: {act_info["activation"]:.3f}')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved top activating patches to {save_path}")
    else:
        plt.show()


def visualize_feature_statistics(
    model,
    dataset,
    hierarchy_idx: int,
    num_samples: int = 200,
    device: str = "mps",
    save_path: Optional[str] = None
):
    """Visualize feature statistics across the dataset"""
    print(f"\n=== Feature Statistics (Hierarchy {hierarchy_idx+1}) ===")

    all_features = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            if i % 50 == 0:
                print(f"Processing {i}/{num_samples}...")

            image, _ = dataset[i]
            image_input = image.unsqueeze(0).to(device)

            embeddings = model.target_encoder.forward_hierarchical(image_input)
            features = embeddings[hierarchy_idx]  # (1, C, H, W)

            # Global average pooling
            pooled = F.adaptive_avg_pool2d(features, 1).flatten()
            all_features.append(pooled.cpu())

    # Stack features
    all_features = torch.stack(all_features)  # (num_samples, C)

    # Compute statistics
    mean_activation = all_features.mean(dim=0)
    std_activation = all_features.std(dim=0)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Feature Statistics - Hierarchy {hierarchy_idx+1}', fontsize=14)

    # 1. Mean activation per channel
    ax = axes[0, 0]
    ax.bar(range(len(mean_activation)), mean_activation.numpy())
    ax.set_title('Mean Activation per Channel')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Mean Activation')
    ax.grid(alpha=0.3)

    # 2. Std activation per channel
    ax = axes[0, 1]
    ax.bar(range(len(std_activation)), std_activation.numpy())
    ax.set_title('Std Activation per Channel')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Std Activation')
    ax.grid(alpha=0.3)

    # 3. Feature correlation matrix
    ax = axes[1, 0]
    # Sample channels if too many
    if all_features.shape[1] > 64:
        sample_indices = torch.linspace(0, all_features.shape[1]-1, 64).long()
        features_sampled = all_features[:, sample_indices]
    else:
        features_sampled = all_features

    corr_matrix = torch.corrcoef(features_sampled.T)
    sns.heatmap(corr_matrix.numpy(), ax=ax, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
    ax.set_title('Channel Correlation Matrix')

    # 4. Feature activation distribution
    ax = axes[1, 1]
    ax.hist(all_features.flatten().numpy(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_title('Feature Activation Distribution')
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Frequency')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved feature statistics to {save_path}")
    else:
        plt.show()

    print(f"\nStatistics Summary:")
    print(f"  Num channels: {all_features.shape[1]}")
    print(f"  Mean activation: {mean_activation.mean():.4f} ± {mean_activation.std():.4f}")
    print(f"  Mean std: {std_activation.mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize H-JEPA features")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "stl10"])
    parser.add_argument("--output-dir", type=str, default="results/feature_viz", help="Output directory")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index")
    parser.add_argument("--hierarchy", type=int, nargs="+", default=[0, 1, 2], help="Hierarchy levels to visualize")
    parser.add_argument("--num-samples", type=int, default=200, help="Samples for statistics")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("H-JEPA Feature Visualization")
    print("=" * 80)

    # Load model
    model, config = load_model(args.checkpoint, args.device)

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = get_dataset(args.dataset, root='./data', train=False, transform=transform)
    sample_image, _ = dataset[args.sample_idx]

    # Run visualizations for each hierarchy
    for h_idx in args.hierarchy:
        print(f"\n{'='*80}")
        print(f"Processing Hierarchy {h_idx+1}")
        print(f"{'='*80}")

        # 1. Feature activation maps
        visualize_feature_maps(
            model, sample_image, h_idx, num_channels=16, device=args.device,
            save_path=output_dir / f"feature_maps_h{h_idx+1}.png"
        )

        # 2. Top activating patches (for first few channels)
        for channel in [0, 4, 8]:
            visualize_top_activating_patches(
                model, dataset, h_idx, channel, num_samples=100, top_k=9, device=args.device,
                save_path=output_dir / f"top_patches_h{h_idx+1}_c{channel}.png"
            )

        # 3. Feature statistics
        visualize_feature_statistics(
            model, dataset, h_idx, num_samples=args.num_samples, device=args.device,
            save_path=output_dir / f"feature_stats_h{h_idx+1}.png"
        )

    print("\n" + "=" * 80)
    print(f"✓ Feature visualization complete! Check {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

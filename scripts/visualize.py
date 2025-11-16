#!/usr/bin/env python3
"""
Comprehensive visualization script for H-JEPA.

This script provides a complete visualization suite including:
- Attention maps and patterns
- Multi-block masking strategies
- Predictions and feature spaces
- Training metrics and analysis

Usage:
    # Visualize masking strategies
    python scripts/visualize.py --visualize-masks --num-samples 6

    # Visualize model predictions
    python scripts/visualize.py --checkpoint results/checkpoints/best_model.pth \
                                --image path/to/image.jpg --visualize-predictions

    # Visualize attention maps
    python scripts/visualize.py --checkpoint results/checkpoints/best_model.pth \
                                --image path/to/image.jpg --visualize-attention

    # Visualize all
    python scripts/visualize.py --checkpoint results/checkpoints/best_model.pth \
                                --image path/to/image.jpg --visualize-all

    # Visualize training logs
    python scripts/visualize.py --visualize-training --log-dir results/logs
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image

from src.models.hjepa import create_hjepa_from_config
from src.visualization import (
    compare_masking_strategies,
    load_training_logs,
    plot_collapse_metrics,
    plot_ema_momentum,
    plot_hierarchical_losses,
    plot_masking_statistics,
    plot_training_curves,
    visualize_attention_maps,
    visualize_attention_rollout,
    visualize_embedding_distribution,
    visualize_feature_space,
    visualize_gradient_flow,
    visualize_hierarchical_attention,
    visualize_hierarchical_predictions,
    visualize_masked_image,
    visualize_masking_strategy,
    visualize_multi_block_masking,
    visualize_multihead_attention,
    visualize_nearest_neighbors,
    visualize_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive H-JEPA visualization")

    # Model and data
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument("--image", type=str, help="Path to input image")

    # Visualization options
    parser.add_argument("--visualize-all", action="store_true", help="Generate all visualizations")
    parser.add_argument("--visualize-masks", action="store_true", help="Visualize masking strategy")
    parser.add_argument(
        "--visualize-predictions", action="store_true", help="Visualize predictions"
    )
    parser.add_argument(
        "--visualize-attention", action="store_true", help="Visualize attention maps"
    )
    parser.add_argument("--visualize-training", action="store_true", help="Visualize training logs")
    parser.add_argument("--visualize-features", action="store_true", help="Visualize feature space")

    # Additional options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--num-samples", type=int, default=6, help="Number of samples for batch visualizations"
    )
    parser.add_argument(
        "--log-dir", type=str, default="results/logs", help="Directory containing training logs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, config, device):
    """Load H-JEPA model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Create model
    model = create_hjepa_from_config(config)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully!")

    return model


def load_image(image_path, img_size=224):
    """Load and preprocess image."""
    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    original_image = np.array(image_pil)

    # Transform for model
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_tensor = transform(image_pil).unsqueeze(0)  # [1, C, H, W]

    # Resize original for display
    original_resized = np.array(image_pil.resize((img_size, img_size)))

    return image_tensor, original_resized


def generate_random_mask(num_patches, mask_ratio=0.5):
    """Generate random mask for visualization."""
    num_masked = int(num_patches * mask_ratio)
    mask = torch.zeros(1, num_patches)
    masked_indices = torch.randperm(num_patches)[:num_masked]
    mask[0, masked_indices] = 1
    return mask


def visualize_all_attention(model, image, original_image, output_dir, device):
    """Generate all attention visualizations."""
    print("\n" + "=" * 80)
    print("Generating Attention Visualizations")
    print("=" * 80)

    image = image.to(device)

    # Extract attention maps
    from src.visualization.attention_viz import extract_attention_maps

    num_layers = len(model.context_encoder.vit.blocks)
    layer_indices = [0, num_layers // 2, num_layers - 1]

    print("Extracting attention maps...")
    attention_maps = extract_attention_maps(model, image, layer_indices)

    # 1. Multi-head attention for last layer
    print("Visualizing multi-head attention...")
    fig = visualize_multihead_attention(
        model, image, layer_idx=-1, save_path=os.path.join(output_dir, "attention_multihead.png")
    )
    plt.close(fig)

    # 2. Attention rollout
    print("Visualizing attention rollout...")
    fig = visualize_attention_rollout(
        model,
        image,
        original_image=original_image,
        save_path=os.path.join(output_dir, "attention_rollout.png"),
    )
    plt.close(fig)

    # 3. Hierarchical attention
    print("Visualizing hierarchical attention...")
    fig = visualize_hierarchical_attention(
        model,
        image,
        original_image=original_image,
        save_path=os.path.join(output_dir, "attention_hierarchical.png"),
    )
    plt.close(fig)

    print(f"Attention visualizations saved to {output_dir}")


def visualize_all_masking(output_dir, num_samples=6):
    """Generate all masking visualizations."""
    print("\n" + "=" * 80)
    print("Generating Masking Visualizations")
    print("=" * 80)

    # 1. Multi-block masking examples
    print("Generating multi-block masking examples...")
    fig = visualize_multi_block_masking(
        num_samples=num_samples, save_path=os.path.join(output_dir, "masking_multi_block.png")
    )
    plt.close(fig)

    # 2. Generate masks for comparison
    print("Comparing masking strategies...")
    grid_size = 14
    num_patches = grid_size**2

    masks = []
    labels = []

    # Random masking
    for ratio in [0.3, 0.5, 0.7]:
        mask = generate_random_mask(num_patches, mask_ratio=ratio)
        masks.append(mask[0])
        labels.append(f"Random {ratio:.0%}")

    # Block masking (simple blocks)
    for num_blocks in [2, 4, 6]:
        mask = torch.zeros(num_patches)
        block_size = grid_size // num_blocks
        for i in range(num_blocks):
            for j in range(num_blocks):
                if (i + j) % 2 == 0:
                    start_i = i * block_size
                    start_j = j * block_size
                    for ii in range(block_size):
                        for jj in range(block_size):
                            if start_i + ii < grid_size and start_j + jj < grid_size:
                                idx = (start_i + ii) * grid_size + (start_j + jj)
                                mask[idx] = 1
        masks.append(mask)
        labels.append(f"{num_blocks}x{num_blocks} Blocks")

    fig = compare_masking_strategies(
        masks, labels, save_path=os.path.join(output_dir, "masking_comparison.png")
    )
    plt.close(fig)

    # 3. Masking statistics
    print("Generating masking statistics...")
    many_masks = [generate_random_mask(num_patches, mask_ratio=0.5)[0] for _ in range(100)]
    fig = plot_masking_statistics(
        many_masks, save_path=os.path.join(output_dir, "masking_statistics.png")
    )
    plt.close(fig)

    print(f"Masking visualizations saved to {output_dir}")


def visualize_all_predictions(model, image, original_image, output_dir, device):
    """Generate all prediction visualizations."""
    print("\n" + "=" * 80)
    print("Generating Prediction Visualizations")
    print("=" * 80)

    image = image.to(device)
    num_patches = model.get_num_patches()

    # Generate mask
    mask = generate_random_mask(num_patches, mask_ratio=0.5).to(device)

    # 1. Basic predictions
    print("Visualizing predictions...")
    fig = visualize_predictions(
        model,
        image,
        mask,
        original_image=original_image,
        save_path=os.path.join(output_dir, "predictions_basic.png"),
    )
    plt.close(fig)

    # 2. Hierarchical predictions
    print("Visualizing hierarchical predictions...")
    fig = visualize_hierarchical_predictions(
        model, image, mask, save_path=os.path.join(output_dir, "predictions_hierarchical.png")
    )
    plt.close(fig)

    # 3. Feature space visualization
    print("Visualizing feature space...")
    with torch.no_grad():
        features = model.extract_features(image, level=0)
        features_flat = features.view(-1, features.shape[-1])

    fig = visualize_feature_space(
        features_flat, method="tsne", save_path=os.path.join(output_dir, "feature_space_tsne.png")
    )
    plt.close(fig)

    # 4. Embedding distribution
    print("Visualizing embedding distribution...")
    fig = visualize_embedding_distribution(
        features_flat, save_path=os.path.join(output_dir, "embedding_distribution.png")
    )
    plt.close(fig)

    print(f"Prediction visualizations saved to {output_dir}")


def visualize_all_training(log_dir, output_dir):
    """Generate all training visualizations."""
    print("\n" + "=" * 80)
    print("Generating Training Visualizations")
    print("=" * 80)

    # Load training logs
    print(f"Loading training logs from {log_dir}...")
    metrics = load_training_logs(log_dir)

    if not metrics:
        print("No training logs found!")
        return

    # 1. Training curves
    if any("loss" in k.lower() for k in metrics.keys()):
        print("Plotting training curves...")
        fig = plot_training_curves(
            metrics, save_path=os.path.join(output_dir, "training_curves.png")
        )
        plt.close(fig)

    # 2. EMA momentum schedule (if available)
    if "ema_momentum" in metrics:
        print("Plotting EMA momentum schedule...")
        fig = plot_ema_momentum(
            metrics["ema_momentum"], save_path=os.path.join(output_dir, "ema_momentum.png")
        )
        plt.close(fig)

    print(f"Training visualizations saved to {output_dir}")


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Determine what to visualize
    visualize_masks = args.visualize_masks or args.visualize_all
    visualize_predictions = args.visualize_predictions or args.visualize_all
    visualize_attention = args.visualize_attention or args.visualize_all
    visualize_training = args.visualize_training or args.visualize_all
    visualize_features = args.visualize_features or args.visualize_all

    # Visualize masking (doesn't need model)
    if visualize_masks:
        visualize_all_masking(args.output_dir, num_samples=args.num_samples)

    # Visualize training logs (doesn't need model or image)
    if visualize_training:
        visualize_all_training(args.log_dir, args.output_dir)

    # Load model if needed
    model = None
    if visualize_predictions or visualize_attention or visualize_features:
        if not args.checkpoint:
            print("\nError: --checkpoint required for model-based visualizations")
            print("Available visualizations without model:")
            print("  --visualize-masks")
            print("  --visualize-training")
            return

        try:
            model = load_model(args.checkpoint, config, args.device)
        except Exception as e:
            print(f"\nError loading model: {e}")
            print("Continuing with non-model visualizations...")

    # Load image if needed
    image = None
    original_image = None
    if (visualize_predictions or visualize_attention) and model is not None:
        if not args.image:
            print("\nError: --image required for image-based visualizations")
            return

        try:
            img_size = config.get("data", {}).get("image_size", 224)
            image, original_image = load_image(args.image, img_size=img_size)
            print(f"Loaded image from {args.image}")
        except Exception as e:
            print(f"\nError loading image: {e}")
            return

    # Generate visualizations
    if visualize_attention and model is not None and image is not None:
        visualize_all_attention(model, image, original_image, args.output_dir, args.device)

    if visualize_predictions and model is not None and image is not None:
        visualize_all_predictions(model, image, original_image, args.output_dir, args.device)

    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("\nGenerated files:")
    for file in sorted(Path(args.output_dir).glob("*.png")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()

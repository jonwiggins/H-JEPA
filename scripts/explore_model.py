#!/usr/bin/env python3.11
"""
Interactive H-JEPA Model Explorer

This script loads a trained H-JEPA checkpoint and provides various
visualization and analysis capabilities to understand what the model learned.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from src.data.datasets import get_dataset
from src.models.hjepa import create_hjepa


def load_model(checkpoint_path: str, device: str = "mps") -> Tuple:
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config and model state
    config = checkpoint.get("config", {})
    model_state = checkpoint.get("model_state_dict", checkpoint.get("target_encoder", {}))

    # Create model
    print("Creating H-JEPA model...")
    model = create_hjepa(
        encoder_type=config.get("model", {}).get("encoder_type", "vit_base_patch16_224"),
        img_size=config.get("data", {}).get("image_size", 224),
        num_hierarchies=config.get("model", {}).get("num_hierarchies", 3),
        predictor_depth=config.get("model", {}).get("predictor", {}).get("depth", 6),
        predictor_heads=config.get("model", {}).get("predictor", {}).get("num_heads", 6),
        use_rope=config.get("model", {}).get("use_rope", True),
        use_flash_attention=config.get("model", {}).get("use_flash_attention", True),
    )

    # Load weights
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")

    return model, config


def visualize_attention_maps(
    model, image: torch.Tensor, device: str = "mps", save_path: Optional[str] = None
):
    """Visualize attention maps from the encoder"""
    print("\n=== Attention Map Visualization ===")

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Get encoder outputs with attention
        context_encoder = model.context_encoder

        # Forward pass through patch embedding
        x = context_encoder.vit.patch_embed(image)

        # Add position embeddings (if not using RoPE)
        if hasattr(context_encoder.vit, "pos_embed"):
            x = x + context_encoder.vit.pos_embed

        # Store attention maps
        attention_maps = []

        # Forward through each block and extract attention
        for i, block in enumerate(context_encoder.vit.blocks):
            # Get attention weights before softmax
            B, N, C = x.shape
            qkv = block.attn.qkv(block.norm1(x))
            qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Compute attention weights
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.cpu().numpy())

            # Continue forward pass
            x = block(x)

    # Visualize attention from different layers and heads
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Multi-Head Attention Maps Across Layers", fontsize=16)

    layers_to_viz = [0, len(attention_maps) // 2, -1]  # First, middle, last layer

    for row, layer_idx in enumerate(layers_to_viz):
        attn_layer = attention_maps[layer_idx]  # Shape: (1, num_heads, N, N)

        for col in range(4):
            if col < attn_layer.shape[1]:  # num_heads
                # Visualize attention from CLS token to patches
                attn_weights = attn_layer[0, col, 0, 1:]  # Attention from CLS to patches

                # Reshape to spatial grid (14x14 for 224x224 images with patch_size=16)
                grid_size = int(np.sqrt(len(attn_weights)))
                attn_grid = attn_weights.reshape(grid_size, grid_size)

                ax = axes[row, col]
                im = ax.imshow(attn_grid, cmap="viridis", interpolation="nearest")
                ax.set_title(f"Layer {layer_idx+1}, Head {col+1}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved attention visualization to {save_path}")
    else:
        plt.show()

    return attention_maps


def visualize_hierarchical_representations(
    model, image: torch.Tensor, device: str = "mps", save_path: Optional[str] = None
):
    """Visualize hierarchical representations learned by the model"""
    print("\n=== Hierarchical Representation Visualization ===")

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Get hierarchical embeddings
        embeddings = model.context_encoder.forward_hierarchical(image)

    # Visualize embeddings from each hierarchy
    fig, axes = plt.subplots(1, len(embeddings), figsize=(6 * len(embeddings), 5))
    if len(embeddings) == 1:
        axes = [axes]

    fig.suptitle("Hierarchical Embeddings (PCA Projection)", fontsize=16)

    for i, emb in enumerate(embeddings):
        # Flatten spatial dimensions
        B, C, H, W = emb.shape
        emb_flat = emb.view(B, C, -1).squeeze(0)  # (C, H*W)

        # Simple PCA-like projection to 3D for visualization
        U, S, V = torch.svd(emb_flat)
        proj_3d = (U[:, :3].T @ emb_flat).cpu().numpy()  # (3, H*W)

        # Reshape back to spatial grid
        proj_grid = proj_3d.reshape(3, H, W).transpose(1, 2, 0)

        # Normalize to [0, 1] for visualization
        proj_grid = (proj_grid - proj_grid.min()) / (proj_grid.max() - proj_grid.min() + 1e-8)

        ax = axes[i]
        ax.imshow(proj_grid)
        ax.set_title(f"Hierarchy {i+1}\n{H}x{W} spatial resolution")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved hierarchical representations to {save_path}")
    else:
        plt.show()

    return embeddings


def masked_prediction_demo(
    model,
    image: torch.Tensor,
    device: str = "mps",
    mask_ratio: float = 0.5,
    save_path: Optional[str] = None,
):
    """Demonstrate masked region prediction"""
    print("\n=== Masked Prediction Demonstration ===")
    print(f"Masking {mask_ratio*100:.0f}% of the image...")

    image = image.unsqueeze(0).to(device)
    B, C, H, W = image.shape

    # Create a random mask
    mask_h, mask_w = H // 16, W // 16  # Patch grid size
    mask = torch.rand(mask_h, mask_w, device=device) > mask_ratio
    mask = mask.unsqueeze(0).unsqueeze(0).float()
    mask = F.interpolate(mask, size=(H, W), mode="nearest")

    with torch.no_grad():
        # Get context embeddings (from visible patches)
        masked_image = image * mask
        context_embeddings = model.context_encoder.forward_hierarchical(masked_image)

        # Get target embeddings (from full image)
        target_embeddings = model.target_encoder.forward_hierarchical(image)

        # Predict masked regions
        predictions = []
        for ctx, tgt in zip(context_embeddings, target_embeddings):
            # Simple visualization: show prediction error
            pred_error = F.mse_loss(ctx, tgt, reduction="none").mean(dim=1, keepdim=True)
            predictions.append(pred_error)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Masked Prediction (Mask Ratio: {mask_ratio})", fontsize=16)

    # Row 1: Original, Masked, Mask
    axes[0, 0].imshow(image[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(masked_image[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title("Masked Image (Context)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask[0, 0].cpu().numpy(), cmap="gray")
    axes[0, 2].set_title("Mask (White=Visible)")
    axes[0, 2].axis("off")

    # Row 2: Prediction errors at different hierarchies
    for i, pred_err in enumerate(predictions):
        if i < 3:
            err_map = pred_err[0, 0].cpu().numpy()
            im = axes[1, i].imshow(err_map, cmap="hot", interpolation="bilinear")
            axes[1, i].set_title(f"Prediction Error\nHierarchy {i+1}")
            axes[1, i].axis("off")
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved masked prediction demo to {save_path}")
    else:
        plt.show()


def embedding_similarity_analysis(
    model, dataset, num_samples: int = 100, device: str = "mps", save_path: Optional[str] = None
):
    """Analyze embedding similarities across the dataset"""
    print(f"\n=== Embedding Similarity Analysis ({num_samples} samples) ===")

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            if i % 20 == 0:
                print(f"Processing {i}/{num_samples}...")

            image, label = dataset[i]
            image = image.unsqueeze(0).to(device)

            # Get top-level embedding
            emb = model.context_encoder.forward_hierarchical(image)[-1]  # Highest hierarchy
            emb_pooled = F.adaptive_avg_pool2d(emb, 1).flatten()

            embeddings_list.append(emb_pooled.cpu())
            labels_list.append(label)

    # Stack embeddings
    embeddings = torch.stack(embeddings_list)  # (num_samples, embed_dim)
    labels = torch.tensor(labels_list)

    # Compute similarity matrix
    print("Computing similarity matrix...")
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    similarity = embeddings_norm @ embeddings_norm.T

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Similarity heatmap
    sns.heatmap(similarity.numpy(), cmap="viridis", ax=axes[0], square=True)
    axes[0].set_title("Embedding Similarity Matrix")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Sample Index")

    # t-SNE visualization (simplified - use mean instead of t-SNE for speed)
    # Project to 2D using PCA
    U, S, V = torch.svd(embeddings)
    proj_2d = (embeddings @ V[:, :2]).numpy()

    scatter = axes[1].scatter(proj_2d[:, 0], proj_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    axes[1].set_title("Embedding Space (PCA projection)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    plt.colorbar(scatter, ax=axes[1], label="Class")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved embedding analysis to {save_path}")
    else:
        plt.show()

    print(f"\nEmbedding Statistics:")
    print(f"  Dimension: {embeddings.shape[1]}")
    print(
        f"  Mean similarity (same class): {similarity[labels.unsqueeze(0) == labels.unsqueeze(1)].mean():.4f}"
    )
    print(
        f"  Mean similarity (diff class): {similarity[labels.unsqueeze(0) != labels.unsqueeze(1)].mean():.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Explore trained H-JEPA model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/exploration",
        help="Output directory for visualizations",
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index from test set")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("H-JEPA Model Explorer")
    print("=" * 80)

    # Load model
    model, config = load_model(args.checkpoint, args.device)

    # Load dataset
    print("\nLoading dataset...")
    dataset = get_dataset(
        dataset_name=config.get("data", {}).get("dataset", "cifar10"),
        root="./data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    print(f"✓ Loaded {len(dataset)} test samples")

    # Get a sample image
    sample_image, sample_label = dataset[args.sample_idx]
    print(f"\nUsing sample {args.sample_idx} (class: {sample_label})")

    # Run all visualizations
    print("\n" + "=" * 80)
    print("Running Visualizations...")
    print("=" * 80)

    # 1. Attention maps
    visualize_attention_maps(
        model, sample_image, args.device, save_path=output_dir / "attention_maps.png"
    )

    # 2. Hierarchical representations
    visualize_hierarchical_representations(
        model, sample_image, args.device, save_path=output_dir / "hierarchical_representations.png"
    )

    # 3. Masked prediction demo
    masked_prediction_demo(
        model,
        sample_image,
        args.device,
        mask_ratio=0.5,
        save_path=output_dir / "masked_prediction.png",
    )

    # 4. Embedding similarity analysis
    embedding_similarity_analysis(
        model,
        dataset,
        num_samples=100,
        device=args.device,
        save_path=output_dir / "embedding_similarity.png",
    )

    print("\n" + "=" * 80)
    print(f"✓ All visualizations complete! Check {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

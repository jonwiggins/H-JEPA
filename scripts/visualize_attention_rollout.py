#!/usr/bin/env python3.11
"""
Attention Rollout Visualization for H-JEPA

This script implements attention rollout - a technique that aggregates
attention maps across all layers to show what parts of the image the
model is attending to for its final representation.

Reference: "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
        use_flash_attention=config.get('model', {}).get('use_flash_attention', False),  # Need explicit attention for visualization
    )

    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")

    return model, config


def extract_attention_maps(model, image: torch.Tensor, device: str) -> List[torch.Tensor]:
    """Extract attention maps from all layers"""

    image = image.unsqueeze(0).to(device)
    encoder = model.target_encoder

    # Forward pass through patch embedding
    x = encoder.vit.patch_embed(image)

    # Add position embeddings if not using RoPE
    if hasattr(encoder.vit, 'pos_embed') and encoder.vit.pos_embed is not None:
        x = x + encoder.vit.pos_embed

    attention_maps = []

    # Extract attention from each block
    with torch.no_grad():
        for block in encoder.vit.blocks:
            B, N, C = x.shape

            # Get QKV
            attn_module = block.attn
            if hasattr(attn_module, 'attn'):  # If wrapped
                attn_module = attn_module.attn

            qkv = attn_module.qkv(block.norm1(x))
            qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Compute attention weights
            attn = (q @ k.transpose(-2, -1)) * attn_module.scale
            attn = attn.softmax(dim=-1)

            # Average over heads
            attn_avg = attn.mean(dim=1)  # (B, N, N)
            attention_maps.append(attn_avg[0])  # (N, N)

            # Continue forward pass
            x = block(x)

    return attention_maps


def compute_attention_rollout(
    attention_maps: List[torch.Tensor],
    discard_ratio: float = 0.1
) -> torch.Tensor:
    """
    Compute attention rollout by recursively multiplying attention matrices.

    Args:
        attention_maps: List of attention matrices (N, N) from each layer
        discard_ratio: Ratio of lowest attention values to discard (helps reduce noise)

    Returns:
        rollout: Final attention map (N,) showing attention to each patch
    """

    # Start with identity matrix
    num_tokens = attention_maps[0].shape[0]
    rollout = torch.eye(num_tokens, device=attention_maps[0].device)

    # Recursively multiply attention matrices
    for attn in attention_maps:
        # Optionally discard low attention values
        if discard_ratio > 0:
            flat = attn.view(-1)
            threshold_index = int(flat.shape[0] * discard_ratio)
            threshold = torch.sort(flat)[0][threshold_index]
            attn = attn.clone()
            attn[attn < threshold] = 0

        # Normalize rows to sum to 1
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # Multiply
        rollout = torch.matmul(attn, rollout)

    # Get attention from CLS token (index 0) to all patches
    rollout_cls = rollout[0, 1:]  # Exclude CLS token itself

    return rollout_cls


def visualize_attention_rollout(
    image: torch.Tensor,
    rollout: torch.Tensor,
    save_path: Optional[str] = None
):
    """Visualize attention rollout overlaid on the image"""

    # Denormalize image
    image_vis = image.clone()
    image_vis = image_vis * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_vis = image_vis + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image_vis = image_vis.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    # Reshape rollout to spatial grid
    num_patches = int(np.sqrt(rollout.shape[0]))
    rollout_grid = rollout.reshape(num_patches, num_patches).cpu().numpy()

    # Resize attention map to match image size
    rollout_resized = np.array(Image.fromarray(rollout_grid).resize(
        (224, 224), Image.BILINEAR
    ))

    # Normalize
    rollout_resized = (rollout_resized - rollout_resized.min()) / (
        rollout_resized.max() - rollout_resized.min() + 1e-8
    )

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image_vis)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Attention map
    axes[1].imshow(rollout_resized, cmap='jet')
    axes[1].set_title('Attention Rollout')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image_vis)
    axes[2].imshow(rollout_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved attention rollout to {save_path}")
    else:
        plt.show()


def compare_layers_attention(
    image: torch.Tensor,
    attention_maps: List[torch.Tensor],
    layers_to_show: List[int],
    save_path: Optional[str] = None
):
    """Compare attention patterns at different layers"""

    # Denormalize image
    image_vis = image.clone()
    image_vis = image_vis * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_vis = image_vis + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image_vis = image_vis.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    num_layers = len(layers_to_show)
    fig, axes = plt.subplots(2, num_layers + 1, figsize=(4 * (num_layers + 1), 8))

    # Show original image
    axes[0, 0].imshow(image_vis)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    # Show attention from each layer
    for idx, layer in enumerate(layers_to_show):
        attn = attention_maps[layer]

        # Get CLS token attention to patches
        attn_cls = attn[0, 1:]  # (N-1,)

        # Reshape to spatial grid
        num_patches = int(np.sqrt(attn_cls.shape[0]))
        attn_grid = attn_cls.reshape(num_patches, num_patches).cpu().numpy()

        # Resize to image size
        attn_resized = np.array(Image.fromarray(attn_grid).resize(
            (224, 224), Image.BILINEAR
        ))

        # Normalize
        attn_resized = (attn_resized - attn_resized.min()) / (
            attn_resized.max() - attn_resized.min() + 1e-8
        )

        # Show attention map
        axes[0, idx + 1].imshow(attn_resized, cmap='jet')
        axes[0, idx + 1].set_title(f'Layer {layer + 1}')
        axes[0, idx + 1].axis('off')

        # Show overlay
        axes[1, idx + 1].imshow(image_vis)
        axes[1, idx + 1].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[1, idx + 1].set_title(f'Layer {layer + 1} Overlay')
        axes[1, idx + 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved layer comparison to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize attention rollout for H-JEPA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "stl10"])
    parser.add_argument("--output-dir", type=str, default="results/attention_rollout", help="Output directory")
    parser.add_argument("--sample-idx", type=int, nargs="+", default=[0, 10, 20], help="Sample indices")
    parser.add_argument("--discard-ratio", type=float, default=0.1, help="Ratio of low attentions to discard")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("H-JEPA Attention Rollout Visualization")
    print("=" * 80)
    print("NOTE: Flash Attention is disabled for this visualization to extract attention weights")
    print()

    # Load model
    model, config = load_model(args.checkpoint, args.device)

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = get_dataset(args.dataset, root='./data', train=False, transform=transform)

    # Process each sample
    for sample_idx in args.sample_idx:
        print(f"\n{'='*80}")
        print(f"Processing sample {sample_idx}")
        print(f"{'='*80}")

        sample_image, sample_label = dataset[sample_idx]

        # Extract attention maps
        print("Extracting attention maps...")
        attention_maps = extract_attention_maps(model, sample_image, args.device)
        print(f"✓ Extracted attention from {len(attention_maps)} layers")

        # Compute attention rollout
        print("Computing attention rollout...")
        rollout = compute_attention_rollout(attention_maps, discard_ratio=args.discard_ratio)
        print(f"✓ Rollout computed")

        # Visualize rollout
        visualize_attention_rollout(
            sample_image,
            rollout,
            save_path=output_dir / f"rollout_sample_{sample_idx}.png"
        )

        # Compare layers
        num_layers = len(attention_maps)
        layers_to_compare = [0, num_layers // 2, num_layers - 1]
        compare_layers_attention(
            sample_image,
            attention_maps,
            layers_to_compare,
            save_path=output_dir / f"layers_sample_{sample_idx}.png"
        )

    print("\n" + "=" * 80)
    print(f"✓ Attention rollout visualization complete! Check {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

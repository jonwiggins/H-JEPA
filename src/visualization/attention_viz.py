"""
Attention visualization utilities for H-JEPA.

Provides functions to visualize attention maps, multi-head patterns,
and hierarchical attention across different levels.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.figure as mfigure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

try:
    from einops import rearrange
except ImportError:

    def rearrange(  # type: ignore[misc]
        tensor: Union[torch.Tensor, List[torch.Tensor]], pattern: str, **axes_lengths: Any
    ) -> torch.Tensor:
        """Fallback rearrange for basic patterns."""
        if not isinstance(tensor, torch.Tensor):
            raise NotImplementedError("Fallback rearrange only supports single tensors")
        if pattern == "b n d -> b d n":
            return tensor.transpose(1, 2)
        elif pattern == "b d n -> b n d":
            return tensor.transpose(1, 2)
        else:
            raise NotImplementedError(f"Pattern {pattern} not supported in fallback")


try:
    import seaborn as sns
except ImportError:
    sns = None


def extract_attention_maps(
    model: nn.Module,
    images: torch.Tensor,
    layer_indices: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract attention maps from Vision Transformer encoder.

    Args:
        model: H-JEPA model
        images: Input images [B, C, H, W]
        layer_indices: Which layers to extract (None = all layers)

    Returns:
        Dictionary containing attention maps per layer
    """
    attention_maps: Dict[str, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def get_attention_hook(name: str) -> Callable[[nn.Module, Any, Any], None]:
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # For timm ViT, attention weights are in attn_drop
            if hasattr(module, "attn"):
                attn_module: Any = module.attn
                attention_maps[name] = attn_module.detach().cpu()

        return hook

    # Register hooks for transformer blocks
    encoder: Any = model.context_encoder.vit  # type: ignore
    blocks: Any = encoder.blocks

    if layer_indices is None:
        layer_indices = list(range(len(blocks)))

    for idx in layer_indices:
        hook = blocks[idx].attn.register_forward_hook(get_attention_hook(f"layer_{idx}"))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = model.context_encoder(images)  # type: ignore[operator]

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return attention_maps


def visualize_attention_maps(
    attention_maps: Dict[str, torch.Tensor],
    image: Optional[npt.NDArray[np.float64]] = None,
    layer_indices: Optional[List[int]] = None,
    head_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> mfigure.Figure:
    """
    Visualize attention maps from multiple layers and heads.

    Args:
        attention_maps: Dictionary of attention tensors per layer
        image: Optional original image to overlay [H, W, 3]
        layer_indices: Which layers to visualize
        head_indices: Which heads to visualize
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if layer_indices is None:
        layer_indices = list(range(len(attention_maps)))

    num_layers = len(layer_indices)

    # Get first attention map to determine number of heads
    first_key = list(attention_maps.keys())[0]
    num_heads = attention_maps[first_key].shape[1]

    if head_indices is None:
        head_indices = [0, num_heads // 4, num_heads // 2, num_heads - 1]

    num_heads_viz = len(head_indices)

    fig, axes = plt.subplots(num_layers, num_heads_viz, figsize=figsize, squeeze=False)

    for i, layer_idx in enumerate(layer_indices):
        attn = attention_maps[f"layer_{layer_idx}"][0]  # First image in batch

        for j, head_idx in enumerate(head_indices):
            ax = axes[i, j]

            # Get attention from CLS token to all patches
            attn_head = attn[head_idx, 0, 1:]  # [num_patches]

            # Reshape to 2D grid
            grid_size = int(np.sqrt(len(attn_head)))
            attn_2d = attn_head.reshape(grid_size, grid_size).cpu().numpy()

            # Plot
            im = ax.imshow(attn_2d, cmap="hot", interpolation="bilinear")

            if i == 0:
                ax.set_title(f"Head {head_idx}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=10)

            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Attention Maps: CLS Token to Patches", fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_multihead_attention(
    model: nn.Module,
    image: torch.Tensor,
    layer_idx: int = -1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
) -> mfigure.Figure:
    """
    Visualize all attention heads from a specific layer.

    Args:
        model: H-JEPA model
        image: Input image [1, C, H, W]
        layer_idx: Which layer to visualize (-1 = last layer)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Extract attention maps
    attention_maps = extract_attention_maps(model, image, [layer_idx])

    # Get attention for specified layer
    if layer_idx >= 0:
        attn_key = f"layer_{layer_idx}"
    else:
        encoder_vit: Any = model.context_encoder.vit  # type: ignore
        num_blocks: int = len(encoder_vit.blocks)
        attn_key = f"layer_{num_blocks + layer_idx}"
    attn = attention_maps[attn_key][0]  # [num_heads, seq_len, seq_len]

    num_heads = attn.shape[0]
    num_cols = 4
    num_rows = (num_heads + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    for head_idx in range(num_heads):
        row = head_idx // num_cols
        col = head_idx % num_cols
        ax = axes[row, col]

        # Get attention from CLS token to all patches
        attn_head = attn[head_idx, 0, 1:]  # [num_patches]

        # Reshape to 2D grid
        grid_size = int(np.sqrt(len(attn_head)))
        attn_2d = attn_head.reshape(grid_size, grid_size).cpu().numpy()

        # Plot
        im = ax.imshow(attn_2d, cmap="viridis", interpolation="bilinear")
        ax.set_title(f"Head {head_idx}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide extra subplots
    for idx in range(num_heads, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis("off")

    plt.suptitle(f"Multi-Head Attention - Layer {layer_idx}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_attention_rollout(
    model: nn.Module,
    image: torch.Tensor,
    original_image: Optional[npt.NDArray[np.float64]] = None,
    start_layer: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> mfigure.Figure:
    """
    Compute and visualize attention rollout (accumulated attention across layers).

    Args:
        model: H-JEPA model
        image: Input image [1, C, H, W]
        original_image: Original image for overlay [H, W, 3]
        start_layer: Layer to start rollout from
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Extract all attention maps
    encoder_vit: Any = model.context_encoder.vit  # type: ignore
    num_layers: int = len(encoder_vit.blocks)
    layer_indices = list(range(start_layer, num_layers))
    attention_maps = extract_attention_maps(model, image, layer_indices)

    # Compute attention rollout
    rollout: Optional[torch.Tensor] = None
    for layer_idx in layer_indices:
        attn = attention_maps[f"layer_{layer_idx}"][0]  # [num_heads, seq_len, seq_len]

        # Average across heads
        attn_avg = attn.mean(dim=0)  # [seq_len, seq_len]

        # Add residual connection (identity matrix)
        I = torch.eye(attn_avg.shape[0])
        attn_avg = 0.5 * attn_avg + 0.5 * I

        # Normalize
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)

        # Accumulate
        if rollout is None:
            rollout = attn_avg
        else:
            rollout = torch.matmul(attn_avg, rollout)

    # Get attention from CLS token
    assert rollout is not None, "Rollout should be computed"
    rollout_cls = rollout[0, 1:].cpu().numpy()  # [num_patches]

    # Reshape to 2D
    grid_size = int(np.sqrt(len(rollout_cls)))
    rollout_2d = rollout_cls.reshape(grid_size, grid_size)

    # Visualize
    fig, axes = plt.subplots(1, 3 if original_image is not None else 2, figsize=figsize)

    # Plot 1: Attention rollout
    im0 = axes[0].imshow(rollout_2d, cmap="hot", interpolation="bilinear")
    axes[0].set_title("Attention Rollout")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot 2: Upsampled rollout
    from scipy.ndimage import zoom

    if image.shape[-1] != rollout_2d.shape[0]:
        scale_factor = image.shape[-1] // rollout_2d.shape[0]
        rollout_upsampled = zoom(rollout_2d, scale_factor, order=1)
    else:
        rollout_upsampled = rollout_2d

    im1 = axes[1].imshow(rollout_upsampled, cmap="hot", interpolation="bilinear")
    axes[1].set_title("Upsampled Attention")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot 3: Overlay on original image (if provided)
    if original_image is not None:
        axes[2].imshow(original_image)
        axes[2].imshow(rollout_upsampled, cmap="hot", alpha=0.5, interpolation="bilinear")
        axes[2].set_title("Attention Overlay")
        axes[2].axis("off")

    plt.suptitle(f"Attention Rollout (Layers {start_layer}-{num_layers-1})", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_hierarchical_attention(
    model: nn.Module,
    image: torch.Tensor,
    original_image: Optional[npt.NDArray[np.float64]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 5),
) -> mfigure.Figure:
    """
    Visualize attention patterns at different hierarchical levels.

    Args:
        model: H-JEPA model
        image: Input image [1, C, H, W]
        original_image: Original image for reference [H, W, 3]
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    encoder_vit: Any = model.context_encoder.vit  # type: ignore
    num_layers: int = len(encoder_vit.blocks)

    # Select layers at different depths (early, middle, late)
    if num_layers >= 12:
        layer_indices = [2, num_layers // 2, num_layers - 2]
        layer_names = ["Early (Local)", "Middle (Mixed)", "Late (Global)"]
    else:
        layer_indices = [0, num_layers // 2, num_layers - 1]
        layer_names = ["Early", "Middle", "Late"]

    # Extract attention maps
    attention_maps = extract_attention_maps(model, image, layer_indices)

    num_levels = len(layer_indices)
    fig, axes = plt.subplots(2, num_levels, figsize=figsize)

    for i, (layer_idx, layer_name) in enumerate(zip(layer_indices, layer_names)):
        attn = attention_maps[f"layer_{layer_idx}"][0]  # [num_heads, seq_len, seq_len]

        # Average across heads
        attn_avg = attn.mean(dim=0)  # [seq_len, seq_len]

        # Get attention from CLS token
        attn_cls = attn_avg[0, 1:].cpu().numpy()  # [num_patches]

        # Reshape to 2D
        grid_size = int(np.sqrt(len(attn_cls)))
        attn_2d = attn_cls.reshape(grid_size, grid_size)

        # Plot attention map
        im0 = axes[0, i].imshow(attn_2d, cmap="hot", interpolation="bilinear")
        axes[0, i].set_title(f"{layer_name}\nLayer {layer_idx}", fontsize=10)
        axes[0, i].axis("off")
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Plot overlay if original image provided
        if original_image is not None:
            from scipy.ndimage import zoom

            scale_factor = original_image.shape[0] // attn_2d.shape[0]
            attn_upsampled = zoom(attn_2d, scale_factor, order=1)

            axes[1, i].imshow(original_image)
            axes[1, i].imshow(attn_upsampled, cmap="hot", alpha=0.5, interpolation="bilinear")
            axes[1, i].set_title("Overlay", fontsize=10)
            axes[1, i].axis("off")
        else:
            # Show attention distribution
            axes[1, i].hist(attn_cls, bins=50, alpha=0.7, edgecolor="black")
            axes[1, i].set_title("Attention Distribution", fontsize=10)
            axes[1, i].set_xlabel("Attention Weight")
            axes[1, i].set_ylabel("Frequency")

    plt.suptitle("Hierarchical Attention Patterns", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_patch_to_patch_attention(
    model: nn.Module,
    image: torch.Tensor,
    patch_idx: int,
    layer_idx: int = -1,
    original_image: Optional[npt.NDArray[np.float64]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> mfigure.Figure:
    """
    Visualize attention from a specific patch to all other patches.

    Args:
        model: H-JEPA model
        image: Input image [1, C, H, W]
        patch_idx: Index of source patch
        layer_idx: Which layer to visualize
        original_image: Original image for overlay [H, W, 3]
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Extract attention maps
    if layer_idx < 0:
        encoder_vit: Any = model.context_encoder.vit  # type: ignore
        layer_idx = len(encoder_vit.blocks) + layer_idx

    attention_maps = extract_attention_maps(model, image, [layer_idx])
    attn = attention_maps[f"layer_{layer_idx}"][0]  # [num_heads, seq_len, seq_len]

    # Average across heads
    attn_avg = attn.mean(dim=0)  # [seq_len, seq_len]

    # Get attention from specified patch (add 1 for CLS token)
    patch_attn = attn_avg[patch_idx + 1, 1:].cpu().numpy()  # [num_patches]

    # Reshape to 2D
    grid_size = int(np.sqrt(len(patch_attn)))
    attn_2d = patch_attn.reshape(grid_size, grid_size)

    # Calculate patch position
    patch_row = patch_idx // grid_size
    patch_col = patch_idx % grid_size

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot attention map
    im0 = axes[0].imshow(attn_2d, cmap="hot", interpolation="bilinear")

    # Mark source patch
    rect = patches.Rectangle(
        (patch_col - 0.5, patch_row - 0.5), 1, 1, linewidth=2, edgecolor="cyan", facecolor="none"
    )
    axes[0].add_patch(rect)

    axes[0].set_title(f"Attention from Patch {patch_idx}\n(Layer {layer_idx})")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot overlay on original image
    if original_image is not None:
        from scipy.ndimage import zoom

        scale_factor = original_image.shape[0] // attn_2d.shape[0]
        attn_upsampled = zoom(attn_2d, scale_factor, order=1)

        axes[1].imshow(original_image)
        axes[1].imshow(attn_upsampled, cmap="hot", alpha=0.5, interpolation="bilinear")

        # Mark source patch on original image
        patch_size = original_image.shape[0] // grid_size
        rect_orig = patches.Rectangle(
            (patch_col * patch_size, patch_row * patch_size),
            patch_size,
            patch_size,
            linewidth=3,
            edgecolor="cyan",
            facecolor="none",
        )
        axes[1].add_patch(rect_orig)

        axes[1].set_title("Overlay on Original Image")
        axes[1].axis("off")
    else:
        axes[1].hist(patch_attn, bins=50, alpha=0.7, edgecolor="black")
        axes[1].set_title("Attention Distribution")
        axes[1].set_xlabel("Attention Weight")
        axes[1].set_ylabel("Frequency")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

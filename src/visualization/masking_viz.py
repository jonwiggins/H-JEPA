"""
Masking visualization utilities for H-JEPA.

Provides functions to visualize multi-block masking strategies,
context/target regions, and compare different masking approaches.
"""

from typing import Any

import matplotlib.animation as animation
import matplotlib.figure as mfigure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch

try:
    from einops import rearrange
except ImportError:

    def rearrange(  # type: ignore[misc, no-redef]
        tensor: torch.Tensor | list[torch.Tensor], pattern: str, **axes_lengths: Any
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


def visualize_masking_strategy(
    mask: torch.Tensor,
    image: npt.NDArray[np.float64] | None = None,
    patch_size: int = 16,
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 5),
    title: str | None = None,
) -> mfigure.Figure:
    """
    Visualize a single masking instance.

    Args:
        mask: Binary mask tensor [N] where 1 = masked
        image: Optional original image [H, W, 3]
        patch_size: Size of each patch
        save_path: Path to save figure
        figsize: Figure size
        title: Optional title for the plot

    Returns:
        Matplotlib figure
    """
    # Convert mask to numpy
    mask_np: npt.NDArray[np.float64] = mask.cpu().numpy()

    # Reshape mask to 2D grid
    grid_size = int(np.sqrt(len(mask_np)))
    mask_2d = mask_np.reshape(grid_size, grid_size)

    # Create figure
    num_plots = 3 if image is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    # Plot 1: Mask visualization
    im0 = axes[0].imshow(mask_2d, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Mask Pattern\n(Red=Masked, Green=Visible)")
    axes[0].axis("off")

    # Add grid
    for i in range(grid_size + 1):
        axes[0].axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
        axes[0].axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)

    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot 2: Statistics
    mask_ratio = mask_np.mean()
    num_masked = int(mask_np.sum())
    num_visible = len(mask_np) - num_masked

    stats_text = "Masking Statistics:\n\n"
    stats_text += f"Total patches: {len(mask_np)}\n"
    stats_text += f"Masked patches: {num_masked}\n"
    stats_text += f"Visible patches: {num_visible}\n"
    stats_text += f"Mask ratio: {mask_ratio:.2%}"

    axes[1].text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    axes[1].axis("off")
    axes[1].set_title("Statistics")

    # Plot 3: Overlay on image (if provided)
    if image is not None:
        # Upsample mask to image size
        mask_upsampled = np.repeat(np.repeat(mask_2d, patch_size, axis=0), patch_size, axis=1)

        # Ensure mask matches image size
        if mask_upsampled.shape[0] > image.shape[0]:
            mask_upsampled = mask_upsampled[: image.shape[0], : image.shape[1]]

        # Create masked image (darken masked regions)
        masked_image = image.copy()
        masked_image[mask_upsampled > 0.5] = masked_image[mask_upsampled > 0.5] * 0.3

        axes[2].imshow(masked_image)
        axes[2].set_title("Masked Image\n(Darkened=Masked)")
        axes[2].axis("off")

    if title:
        plt.suptitle(title, fontsize=14, y=1.02)
    else:
        plt.suptitle("H-JEPA Masking Strategy", fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_masked_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = 16,
    mask_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> mfigure.Figure:
    """
    Visualize original image, mask, and masked image side by side.

    Args:
        image: Input image tensor [C, H, W] or [H, W, C]
        mask: Binary mask [N] where 1 = masked
        patch_size: Size of each patch
        mask_color: RGB color for masked regions
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Convert to numpy and normalize
    image_np: npt.NDArray[np.float64] = image.cpu().numpy()

    if image_np.shape[0] == 3:  # [C, H, W]
        image_np = np.transpose(image_np, (1, 2, 0))

    # Normalize to [0, 1]
    if image_np.max() > 1.0:
        image_np = image_np / 255.0

    mask_np: npt.NDArray[np.float64] = mask.cpu().numpy()

    # Reshape mask to 2D
    grid_size = int(np.sqrt(len(mask_np)))
    mask_2d = mask_np.reshape(grid_size, grid_size)

    # Create masked image
    masked_image = image_np.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            if mask_2d[i, j] > 0.5:
                y_start = i * patch_size
                y_end = min((i + 1) * patch_size, image_np.shape[0])
                x_start = j * patch_size
                x_end = min((j + 1) * patch_size, image_np.shape[1])
                masked_image[y_start:y_end, x_start:x_end] = mask_color

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Mask pattern
    axes[1].imshow(mask_2d, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Mask ({mask_np.mean():.1%} masked)")
    axes[1].axis("off")

    # Masked image
    axes[2].imshow(masked_image)
    axes[2].set_title("Masked Image")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_context_target_regions(
    mask: torch.Tensor,
    target_blocks: list[tuple[int, int, int, int]] | None = None,
    image: npt.NDArray[np.float64] | None = None,
    patch_size: int = 16,
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 6),
) -> mfigure.Figure:
    """
    Visualize context (visible) and target (masked) regions separately.

    Args:
        mask: Binary mask [N] where 1 = masked
        target_blocks: List of target blocks as (row, col, height, width)
        image: Optional original image [H, W, 3]
        patch_size: Size of each patch
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    mask_np: npt.NDArray[np.float64] = mask.cpu().numpy()

    grid_size = int(np.sqrt(len(mask_np)))
    mask_2d = mask_np.reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Full mask with target blocks highlighted
    im0 = axes[0].imshow(
        mask_2d, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest", alpha=0.7
    )
    axes[0].set_title("Masking Strategy\n(Red=Masked, Green=Context)")

    # Draw target blocks if provided
    if target_blocks is not None:
        for block in target_blocks:
            row, col, height, width = block
            rect = patches.Rectangle(
                (col - 0.5, row - 0.5),
                width,
                height,
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
                linestyle="--",
            )
            axes[0].add_patch(rect)

    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot 2: Context regions only
    context_mask = 1 - mask_2d
    axes[1].imshow(context_mask, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Context Regions\n({(1-mask_np.mean()):.1%} visible)")
    axes[1].axis("off")

    # Plot 3: Target regions only
    axes[2].imshow(mask_2d, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"Target Regions\n({mask_np.mean():.1%} masked)")
    axes[2].axis("off")

    plt.suptitle("Context and Target Regions", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compare_masking_strategies(
    masks: list[torch.Tensor],
    labels: list[str],
    image: npt.NDArray[np.float64] | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (16, 10),
) -> mfigure.Figure:
    """
    Compare different masking strategies side by side.

    Args:
        masks: List of mask tensors [N]
        labels: Labels for each masking strategy
        image: Optional original image [H, W, 3]
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    num_masks = len(masks)
    num_cols = min(4, num_masks)
    num_rows = (num_masks + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    for idx, (mask, label) in enumerate(zip(masks, labels)):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]

        # Convert mask to numpy
        mask_np: npt.NDArray[np.float64] = mask.cpu().numpy()

        # Reshape to 2D
        grid_size = int(np.sqrt(len(mask_np)))
        mask_2d = mask_np.reshape(grid_size, grid_size)

        # Plot mask
        ax.imshow(mask_2d, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")

        mask_ratio = mask_np.mean()
        ax.set_title(f"{label}\n({mask_ratio:.1%} masked)", fontsize=10)
        ax.axis("off")

        # Add grid
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.2)
            ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.2)

    # Hide extra subplots
    for idx in range(num_masks, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis("off")

    plt.suptitle("Comparison of Masking Strategies", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def animate_masking_process(
    masks: list[torch.Tensor],
    image: npt.NDArray[np.float64] | None = None,
    patch_size: int = 16,
    save_path: str | None = None,
    interval: int = 500,
    figsize: tuple[int, int] = (10, 5),
) -> animation.FuncAnimation:
    """
    Create an animation showing the masking process over time.

    Args:
        masks: List of mask tensors representing different time steps
        image: Optional original image [H, W, 3]
        patch_size: Size of each patch
        save_path: Path to save animation (as .gif or .mp4)
        interval: Interval between frames in milliseconds
        figsize: Figure size

    Returns:
        Animation object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Initialize plots
    grid_size = int(np.sqrt(len(masks[0])))
    mask_2d = masks[0].cpu().numpy().reshape(grid_size, grid_size)

    im0 = axes[0].imshow(mask_2d, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Mask Evolution")
    axes[0].axis("off")

    if image is not None:
        masked_img = image.copy()
        im1 = axes[1].imshow(masked_img)
        axes[1].set_title("Masked Image")
        axes[1].axis("off")

    def update(frame: int) -> list[Any]:
        mask = masks[frame].cpu().numpy()
        mask_2d = mask.reshape(grid_size, grid_size)

        im0.set_array(mask_2d)
        axes[0].set_title(f"Mask Evolution (Frame {frame + 1}/{len(masks)})")

        if image is not None:
            masked_img = image.copy()
            mask_upsampled = np.repeat(np.repeat(mask_2d, patch_size, axis=0), patch_size, axis=1)

            if mask_upsampled.shape[0] > image.shape[0]:
                mask_upsampled = mask_upsampled[: image.shape[0], : image.shape[1]]

            masked_img[mask_upsampled > 0.5] = masked_img[mask_upsampled > 0.5] * 0.3
            im1.set_array(masked_img)

        return [im0] if image is None else [im0, im1]

    anim = animation.FuncAnimation(
        fig, update, frames=len(masks), interval=interval, blit=True, repeat=True
    )

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval)

    return anim


def visualize_multi_block_masking(
    num_samples: int = 6,
    grid_size: int = 14,
    num_blocks: int = 4,
    block_aspect_ratio: tuple[float, float] = (0.75, 1.5),
    mask_scale: tuple[float, float] = (0.15, 0.2),
    save_path: str | None = None,
    figsize: tuple[int, int] = (15, 10),
) -> mfigure.Figure:
    """
    Visualize multiple samples of multi-block masking strategy.

    Args:
        num_samples: Number of masking samples to generate
        grid_size: Size of patch grid
        num_blocks: Number of target blocks to mask
        block_aspect_ratio: Range of aspect ratios for blocks
        mask_scale: Range of mask scales (relative to image)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    num_cols = 3
    num_rows = (num_samples + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    for idx in range(num_samples):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]

        # Generate random multi-block mask
        mask = np.zeros((grid_size, grid_size))

        for _ in range(num_blocks):
            # Random block size
            scale = np.random.uniform(*mask_scale)
            aspect = np.random.uniform(*block_aspect_ratio)

            block_h = int(grid_size * np.sqrt(scale * aspect))
            block_w = int(grid_size * np.sqrt(scale / aspect))

            block_h = max(1, min(block_h, grid_size - 1))
            block_w = max(1, min(block_w, grid_size - 1))

            # Random position
            row_start = np.random.randint(0, grid_size - block_h + 1)
            col_start = np.random.randint(0, grid_size - block_w + 1)

            # Apply mask
            mask[row_start : row_start + block_h, col_start : col_start + block_w] = 1

        # Plot
        ax.imshow(mask, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")

        mask_ratio = mask.mean()
        ax.set_title(f"Sample {idx + 1}\n({mask_ratio:.1%} masked)", fontsize=9)
        ax.axis("off")

        # Add grid
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.3)
            ax.axvline(i - 0.5, color="gray", linewidth=0.3, alpha=0.3)

    # Hide extra subplots
    for idx in range(num_samples, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis("off")

    plt.suptitle("Multi-Block Masking Strategy - Random Samples", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_masking_statistics(
    masks: list[torch.Tensor],
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> mfigure.Figure:
    """
    Plot statistics about masking patterns.

    Args:
        masks: List of mask tensors
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Compute statistics
    mask_ratios = [m.float().mean().item() for m in masks]

    # Compute spatial distribution
    grid_size = int(np.sqrt(len(masks[0])))
    spatial_avg = torch.stack([m.view(grid_size, grid_size) for m in masks]).float().mean(dim=0)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Mask ratio distribution
    axes[0].hist(mask_ratios, bins=30, alpha=0.7, edgecolor="black")
    axes[0].axvline(
        np.mean(mask_ratios), color="red", linestyle="--", label=f"Mean: {np.mean(mask_ratios):.2%}"
    )
    axes[0].set_xlabel("Mask Ratio")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Mask Ratio Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Spatial distribution
    im1 = axes[1].imshow(spatial_avg.cpu().numpy(), cmap="hot", interpolation="bilinear")
    axes[1].set_title("Average Spatial Distribution\n(Hotter = More Often Masked)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot 3: Summary statistics
    stats_text = "Masking Statistics:\n\n"
    stats_text += f"Samples: {len(masks)}\n"
    stats_text += f"Mean ratio: {np.mean(mask_ratios):.2%}\n"
    stats_text += f"Std ratio: {np.std(mask_ratios):.2%}\n"
    stats_text += f"Min ratio: {np.min(mask_ratios):.2%}\n"
    stats_text += f"Max ratio: {np.max(mask_ratios):.2%}\n"

    axes[2].text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    axes[2].axis("off")
    axes[2].set_title("Summary Statistics")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

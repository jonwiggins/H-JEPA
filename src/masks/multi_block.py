"""
Multi-block masking strategy for H-JEPA.

This module implements the multi-block masking strategy used in H-JEPA training,
which samples multiple target blocks and a large context block for predictive learning.
"""

import logging

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch

logger = logging.getLogger(__name__)


class MultiBlockMaskGenerator:
    """
    Multi-block mask generator for H-JEPA.

    Generates masks consisting of:
    - Multiple target blocks (typically 4) with scale 15-20% of image
    - One context block with scale 85-100% of image
    - Ensures no overlaps between context and target blocks

    The masks are generated at the patch level, compatible with Vision Transformer
    patch embeddings.

    Args:
        input_size (int or tuple): Input image size (height, width). If int, assumes square image.
        patch_size (int): Size of each patch (e.g., 16 for ViT-Base/16).
        num_target_masks (int): Number of target blocks to sample (default: 4).
        target_scale (tuple): Scale range for target masks as fraction of image (default: (0.15, 0.2)).
        context_scale (tuple): Scale range for context mask as fraction of image (default: (0.85, 1.0)).
        aspect_ratio_range (tuple): Aspect ratio range for masks (default: (0.75, 1.5)).
        max_attempts (int): Maximum attempts to sample non-overlapping masks (default: 10).

    Example:
        >>> mask_gen = MultiBlockMaskGenerator(
        ...     input_size=224,
        ...     patch_size=16,
        ...     num_target_masks=4,
        ...     target_scale=(0.15, 0.2),
        ...     context_scale=(0.85, 1.0),
        ...     aspect_ratio_range=(0.75, 1.5)
        ... )
        >>> context_mask, target_masks = mask_gen(batch_size=8)
        >>> print(context_mask.shape)  # (8, 196)
        >>> print(target_masks.shape)  # (8, 4, 196)
    """

    def __init__(
        self,
        input_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        num_target_masks: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.2),
        context_scale: tuple[float, float] = (0.85, 1.0),
        aspect_ratio_range: tuple[float, float] = (0.75, 1.5),
        max_attempts: int = 10,
    ) -> None:
        # Handle input size
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            self.input_size = input_size

        self.patch_size = patch_size
        self.num_target_masks = num_target_masks
        self.target_scale = target_scale
        self.context_scale = context_scale
        self.aspect_ratio_range = aspect_ratio_range
        self.max_attempts = max_attempts

        # Calculate number of patches
        self.num_patches_h = self.input_size[0] // patch_size
        self.num_patches_w = self.input_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

    def __call__(self, batch_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multi-block masks for a batch.

        Args:
            batch_size (int): Number of masks to generate.
            device (str): Device to create tensors on ('cuda' or 'cpu').

        Returns:
            Tuple of:
                - context_mask: Boolean tensor of shape (batch_size, num_patches)
                  True indicates patches to use as context.
                - target_masks: Boolean tensor of shape (batch_size, num_target_masks, num_patches)
                  True indicates patches to predict.
        """
        context_masks = []
        target_masks_list = []

        for _ in range(batch_size):
            # Generate one set of masks
            context_mask, target_masks = self._generate_single_mask_set()
            context_masks.append(context_mask)
            target_masks_list.append(target_masks)

        # Stack into batch tensors
        context_mask_batch = torch.stack(context_masks).to(device)
        target_mask_batch = torch.stack(target_masks_list).to(device)

        return context_mask_batch, target_mask_batch

    def _generate_single_mask_set(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate one set of context and target masks.

        Returns:
            Tuple of (context_mask, target_masks) as boolean tensors.
        """
        # Sample target blocks first
        target_blocks = []
        occupied_patches = np.zeros((self.num_patches_h, self.num_patches_w), dtype=bool)

        for _ in range(self.num_target_masks):
            block = self._sample_block(
                scale_range=self.target_scale,
                occupied=occupied_patches,
            )
            if block is not None:
                target_blocks.append(block)
                # Mark patches as occupied
                top, left, height, width = block
                occupied_patches[top : top + height, left : left + width] = True

        # Sample context block
        context_block = self._sample_block(
            scale_range=self.context_scale,
            occupied=None,  # Context can be anywhere
        )

        # Convert blocks to patch masks
        target_masks = []
        for block in target_blocks:
            mask = self._block_to_mask(block)
            target_masks.append(mask)

        # Pad if we couldn't sample enough target blocks
        while len(target_masks) < self.num_target_masks:
            # Create a small random mask as fallback
            fallback_block = self._sample_block(
                scale_range=(0.05, 0.1),
                occupied=None,
            )
            target_masks.append(self._block_to_mask(fallback_block))

        context_mask = self._block_to_mask(context_block)

        # Remove overlaps: context should not include target patches
        target_union = torch.zeros(self.num_patches, dtype=torch.bool)
        for target_mask in target_masks:
            target_union = target_union | target_mask

        context_mask = context_mask & (~target_union)

        # Stack target masks
        target_masks_tensor = torch.stack(target_masks)

        return context_mask, target_masks_tensor

    def _sample_block(
        self,
        scale_range: tuple[float, float],
        occupied: npt.NDArray[np.bool_] | None = None,
    ) -> tuple[int, int, int, int]:
        """
        Sample a single block (top, left, height, width) in patch coordinates.

        Args:
            scale_range: (min_scale, max_scale) as fraction of image area.
            occupied: Optional array of occupied patches to avoid.

        Returns:
            Tuple of (top, left, height, width) in patch coordinates.
        """
        for attempt in range(self.max_attempts):
            # Sample scale and aspect ratio
            scale = np.random.uniform(scale_range[0], scale_range[1])
            aspect_ratio = np.random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])

            # Calculate block size in patches
            area = scale * self.num_patches
            height = int(np.round(np.sqrt(area / aspect_ratio)))
            width = int(np.round(height * aspect_ratio))

            # Ensure minimum size of 1 patch and doesn't exceed grid size
            height = max(1, min(height, self.num_patches_h))
            width = max(1, min(width, self.num_patches_w))

            # Sample random position with safety check for edge cases
            # max() ensures the range is always valid (at least [0, 1))
            top = np.random.randint(0, max(1, self.num_patches_h - height + 1))
            left = np.random.randint(0, max(1, self.num_patches_w - width + 1))

            # Check for overlaps if occupied array provided
            if occupied is not None:
                block_area = occupied[top : top + height, left : left + width]
                if not block_area.any():  # No overlap
                    return (top, left, height, width)
            else:
                return (top, left, height, width)

        # If we couldn't find non-overlapping block, return a small random block
        height = max(1, self.num_patches_h // 4)
        width = max(1, self.num_patches_w // 4)
        # Safety check for edge cases
        top = np.random.randint(0, max(1, self.num_patches_h - height + 1))
        left = np.random.randint(0, max(1, self.num_patches_w - width + 1))

        return (top, left, height, width)

    def _block_to_mask(self, block: tuple[int, int, int, int]) -> torch.Tensor:
        """
        Convert a block specification to a boolean mask over patches.

        Args:
            block: (top, left, height, width) in patch coordinates.

        Returns:
            Boolean tensor of shape (num_patches,).
        """
        top, left, height, width = block
        mask_2d = np.zeros((self.num_patches_h, self.num_patches_w), dtype=bool)
        mask_2d[top : top + height, left : left + width] = True

        # Flatten to 1D
        mask_1d = mask_2d.reshape(-1)

        return torch.from_numpy(mask_1d)

    def visualize_masks(
        self,
        context_mask: torch.Tensor,
        target_masks: torch.Tensor,
        sample_idx: int = 0,
        figsize: tuple[int, int] = (12, 4),
        save_path: str | None = None,
    ) -> "plt.Figure":
        """
        Visualize the generated masks.

        Args:
            context_mask: Context mask tensor of shape (batch_size, num_patches).
            target_masks: Target masks tensor of shape (batch_size, num_target_masks, num_patches).
            sample_idx: Which sample from the batch to visualize.
            figsize: Figure size.
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure object.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Get masks for the specified sample
        ctx_mask = context_mask[sample_idx].cpu().numpy()
        tgt_masks = target_masks[sample_idx].cpu().numpy()

        # Reshape to 2D
        ctx_mask_2d = ctx_mask.reshape(self.num_patches_h, self.num_patches_w)

        # Visualize context mask
        axes[0].imshow(ctx_mask_2d, cmap="Blues", vmin=0, vmax=1)
        axes[0].set_title("Context Mask")
        axes[0].axis("off")

        # Visualize all target masks combined
        target_union = np.zeros((self.num_patches_h, self.num_patches_w))
        for i, tgt_mask in enumerate(tgt_masks):
            tgt_mask_2d = tgt_mask.reshape(self.num_patches_h, self.num_patches_w)
            target_union = np.maximum(target_union, tgt_mask_2d * (i + 1))

        axes[1].imshow(target_union, cmap="Set1", vmin=0, vmax=self.num_target_masks)
        axes[1].set_title(f"Target Masks ({self.num_target_masks} blocks)")
        axes[1].axis("off")

        # Visualize combined view
        combined = np.zeros((self.num_patches_h, self.num_patches_w, 3))
        combined[ctx_mask_2d > 0] = [0.2, 0.6, 1.0]  # Blue for context
        for i, tgt_mask in enumerate(tgt_masks):
            tgt_mask_2d = tgt_mask.reshape(self.num_patches_h, self.num_patches_w)
            # Different colors for each target
            color = cm.get_cmap("Set1")(i / self.num_target_masks)[:3]
            combined[tgt_mask_2d > 0] = color

        axes[2].imshow(combined)
        axes[2].set_title("Combined View")
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def get_mask_statistics(
        self,
        context_mask: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute statistics about the generated masks.

        Args:
            context_mask: Context mask tensor of shape (batch_size, num_patches).
            target_masks: Target masks tensor of shape (batch_size, num_target_masks, num_patches).

        Returns:
            Dictionary with mask statistics.
        """
        batch_size = context_mask.shape[0]

        # Context coverage
        context_coverage = context_mask.float().mean(dim=1)  # Per sample

        # Target coverage
        target_coverage = target_masks.float().mean(dim=2)  # Per target per sample

        # Check overlaps
        overlaps = []
        for i in range(batch_size):
            ctx = context_mask[i]
            for j in range(self.num_target_masks):
                tgt = target_masks[i, j]
                overlap = (ctx & tgt).float().sum() / tgt.float().sum()
                overlaps.append(overlap.item())

        stats = {
            "context_coverage_mean": context_coverage.mean().item(),
            "context_coverage_std": context_coverage.std().item(),
            "target_coverage_mean": target_coverage.mean().item(),
            "target_coverage_std": target_coverage.std().item(),
            "overlap_mean": np.mean(overlaps),
            "overlap_max": np.max(overlaps),
        }

        return stats


def demo() -> None:
    """Demonstration of MultiBlockMaskGenerator."""
    logger.info("Multi-Block Mask Generator Demo")
    logger.info("=" * 50)

    # Create mask generator
    mask_gen = MultiBlockMaskGenerator(
        input_size=224,
        patch_size=16,
        num_target_masks=4,
        target_scale=(0.15, 0.2),
        context_scale=(0.85, 1.0),
        aspect_ratio_range=(0.75, 1.5),
    )

    logger.info("Input size: %s", mask_gen.input_size)
    logger.info("Patch size: %s", mask_gen.patch_size)
    logger.info(
        "Number of patches: %d (%dx%d)",
        mask_gen.num_patches,
        mask_gen.num_patches_h,
        mask_gen.num_patches_w,
    )
    logger.info("Number of target masks: %d", mask_gen.num_target_masks)

    # Generate masks
    batch_size = 4
    context_mask, target_masks = mask_gen(batch_size=batch_size, device="cpu")

    logger.info("Generated masks for batch_size=%d", batch_size)
    logger.info("Context mask shape: %s", context_mask.shape)
    logger.info("Target masks shape: %s", target_masks.shape)

    # Compute statistics
    stats = mask_gen.get_mask_statistics(context_mask, target_masks)
    logger.info("Mask Statistics:")
    for key, value in stats.items():
        logger.info("  %s: %.4f", key, value)

    # Visualize
    logger.info("Generating visualization...")
    mask_gen.visualize_masks(context_mask, target_masks, sample_idx=0)
    plt.savefig("/tmp/multi_block_masks_demo.png", dpi=150, bbox_inches="tight")
    logger.info("Visualization saved to /tmp/multi_block_masks_demo.png")
    plt.close()

    logger.info("Demo complete!")


if __name__ == "__main__":
    demo()

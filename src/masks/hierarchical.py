"""
Hierarchical masking strategy for H-JEPA.

This module implements hierarchical masking that generates different masks
for different levels of the hierarchy, enabling multi-scale representation learning.
"""

from typing import Dict, List, Optional, Tuple, Union, cast

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch


class HierarchicalMaskGenerator:
    """
    Hierarchical mask generator for multi-scale learning in H-JEPA.

    Generates different masking patterns for different hierarchy levels:
    - Level 1 (Fine-grained): Small patches, higher masking ratio
    - Level 2 (Medium): Medium-sized blocks
    - Level 3 (Coarse): Large regions, lower masking ratio

    The progressive masking strategy ensures that:
    1. Fine levels have more aggressive masking (learning local details)
    2. Coarse levels have less aggressive masking (learning global structure)
    3. Masks respect the hierarchical structure (coarse regions contain fine patches)

    Args:
        input_size (int or tuple): Input image size (height, width). If int, assumes square image.
        patch_size (int): Size of each patch at the finest level (e.g., 16 for ViT-Base/16).
        num_hierarchies (int): Number of hierarchy levels (default: 3).
        num_target_masks (int): Number of target blocks per level (default: 4).
        scale_progression (str): How scales progress across levels ('geometric' or 'linear').
        base_scale (tuple): Scale range for finest level (default: (0.05, 0.15)).
        aspect_ratio_range (tuple): Aspect ratio range for masks (default: (0.75, 1.5)).

    Example:
        >>> mask_gen = HierarchicalMaskGenerator(
        ...     input_size=224,
        ...     patch_size=16,
        ...     num_hierarchies=3,
        ...     num_target_masks=4,
        ... )
        >>> masks = mask_gen(batch_size=8)
        >>> # masks is a dict with keys 'level_0', 'level_1', 'level_2'
        >>> # Each contains 'context' and 'targets' masks
        >>> print(masks['level_0']['context'].shape)  # (8, 196)
        >>> print(masks['level_0']['targets'].shape)  # (8, 4, 196)
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 224,
        patch_size: int = 16,
        num_hierarchies: int = 3,
        num_target_masks: int = 4,
        scale_progression: str = "geometric",
        base_scale: Tuple[float, float] = (0.05, 0.15),
        aspect_ratio_range: Tuple[float, float] = (0.75, 1.5),
        max_attempts: int = 10,
    ) -> None:
        # Handle input size
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            self.input_size = input_size

        self.patch_size = patch_size
        self.num_hierarchies = num_hierarchies
        self.num_target_masks = num_target_masks
        self.scale_progression = scale_progression
        self.base_scale = base_scale
        self.aspect_ratio_range = aspect_ratio_range
        self.max_attempts = max_attempts

        # Calculate number of patches
        self.num_patches_h = self.input_size[0] // patch_size
        self.num_patches_w = self.input_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Compute scale ranges for each hierarchy level
        self.level_configs = self._compute_level_configs()

    def _compute_level_configs(self) -> List[Dict[str, Union[Tuple[float, float], int]]]:
        """
        Compute masking configurations for each hierarchy level.

        Returns:
            List of dicts with 'target_scale' and 'context_scale' for each level.
        """
        configs: List[Dict[str, Union[Tuple[float, float], int]]] = []

        for level in range(self.num_hierarchies):
            if self.scale_progression == "geometric":
                # Geometric progression: each level doubles the scale
                scale_factor = 2**level
            else:  # linear
                # Linear progression
                scale_factor = 1 + level

            # Target scale increases with level (coarser levels have larger blocks)
            target_min = min(0.95, self.base_scale[0] * scale_factor)
            target_max = min(0.95, self.base_scale[1] * scale_factor)

            # Context scale: starts smaller at fine levels, larger at coarse levels
            # Fine level (0): Use more patches for prediction (smaller context)
            # Coarse level (N-1): Use fewer patches for prediction (larger context)
            context_ratio = 0.6 + 0.3 * (level / max(1, self.num_hierarchies - 1))
            context_min = min(0.95, context_ratio)
            context_max = min(1.0, context_ratio + 0.15)

            config_dict: Dict[str, Union[Tuple[float, float], int]] = {
                "target_scale": (target_min, target_max),
                "context_scale": (context_min, context_max),
                "level": level,
            }
            configs.append(config_dict)

        return configs

    def __call__(self, batch_size: int, device: str = "cpu") -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate hierarchical masks for a batch.

        Args:
            batch_size (int): Number of mask sets to generate.
            device (str): Device to create tensors on ('cuda' or 'cpu').

        Returns:
            Dictionary mapping level names to mask dicts:
            {
                'level_0': {'context': tensor, 'targets': tensor},
                'level_1': {'context': tensor, 'targets': tensor},
                ...
            }
            Each 'context' is shape (batch_size, num_patches)
            Each 'targets' is shape (batch_size, num_target_masks, num_patches)
        """
        hierarchical_masks = {}

        for level_idx, config in enumerate(self.level_configs):
            context_masks = []
            target_masks_list = []

            for _ in range(batch_size):
                # Generate masks for this level
                context_mask, target_masks = self._generate_level_masks(config)
                context_masks.append(context_mask)
                target_masks_list.append(target_masks)

            # Stack into batch tensors
            context_mask_batch = torch.stack(context_masks).to(device)
            target_mask_batch = torch.stack(target_masks_list).to(device)

            hierarchical_masks[f"level_{level_idx}"] = {
                "context": context_mask_batch,
                "targets": target_mask_batch,
            }

        return hierarchical_masks

    def _generate_level_masks(
        self,
        config: Dict[str, Union[Tuple[float, float], int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate masks for a specific hierarchy level.

        Args:
            config: Configuration dict for this level.

        Returns:
            Tuple of (context_mask, target_masks) as boolean tensors.
        """
        target_scale = cast(Tuple[float, float], config["target_scale"])
        context_scale = cast(Tuple[float, float], config["context_scale"])
        level = cast(int, config["level"])

        # Sample target blocks
        target_blocks = []
        occupied_patches = np.zeros((self.num_patches_h, self.num_patches_w), dtype=bool)

        for _ in range(self.num_target_masks):
            block = self._sample_block(
                scale_range=target_scale,
                occupied=occupied_patches,
                level=level,
            )
            if block is not None:
                target_blocks.append(block)
                top, left, height, width = block
                occupied_patches[top : top + height, left : left + width] = True

        # Sample context block
        context_block = self._sample_block(
            scale_range=context_scale,
            occupied=None,
            level=level,
        )

        # Convert blocks to masks
        target_masks = []
        for block in target_blocks:
            mask = self._block_to_mask(block)
            target_masks.append(mask)

        # Pad if needed
        while len(target_masks) < self.num_target_masks:
            fallback_block = self._sample_block(
                scale_range=(0.02, 0.05),
                occupied=None,
                level=level,
            )
            target_masks.append(self._block_to_mask(fallback_block))

        context_mask = self._block_to_mask(context_block)

        # Remove overlaps
        target_union = torch.zeros(self.num_patches, dtype=torch.bool)
        for target_mask in target_masks:
            target_union = target_union | target_mask

        context_mask = context_mask & (~target_union)

        target_masks_tensor = torch.stack(target_masks)

        return context_mask, target_masks_tensor

    def _sample_block(
        self,
        scale_range: Tuple[float, float],
        occupied: Optional[npt.NDArray[np.bool_]] = None,
        level: int = 0,
    ) -> Tuple[int, int, int, int]:
        """
        Sample a single block at a specific hierarchy level.

        Args:
            scale_range: (min_scale, max_scale) as fraction of image area.
            occupied: Optional array of occupied patches to avoid.
            level: Hierarchy level (affects sampling strategy).

        Returns:
            Tuple of (top, left, height, width) in patch coordinates.
        """
        for attempt in range(self.max_attempts):
            # Sample scale
            scale = np.random.uniform(scale_range[0], scale_range[1])

            # Aspect ratio: finer levels prefer more square blocks
            # Coarser levels can have more elongated blocks
            aspect_min, aspect_max = self.aspect_ratio_range
            if level == 0:  # Finest level - prefer squarer blocks
                aspect_min = max(aspect_min, 0.85)
                aspect_max = min(aspect_max, 1.15)

            aspect_ratio = np.random.uniform(aspect_min, aspect_max)

            # Calculate block size
            area = scale * self.num_patches
            height = int(np.round(np.sqrt(area / aspect_ratio)))
            width = int(np.round(height * aspect_ratio))

            # Ensure valid size
            height = max(1, min(height, self.num_patches_h))
            width = max(1, min(width, self.num_patches_w))

            # Sample position
            top = np.random.randint(0, self.num_patches_h - height + 1)
            left = np.random.randint(0, self.num_patches_w - width + 1)

            # Check overlaps
            if occupied is not None:
                block_area = occupied[top : top + height, left : left + width]
                if not block_area.any():
                    return (top, left, height, width)
            else:
                return (top, left, height, width)

        # Fallback
        height = max(1, self.num_patches_h // (level + 2))
        width = max(1, self.num_patches_w // (level + 2))
        top = np.random.randint(0, self.num_patches_h - height + 1)
        left = np.random.randint(0, self.num_patches_w - width + 1)

        return (top, left, height, width)

    def _block_to_mask(self, block: Tuple[int, int, int, int]) -> torch.Tensor:
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
        mask_1d = mask_2d.reshape(-1)
        return torch.from_numpy(mask_1d)

    def visualize_hierarchical_masks(
        self,
        masks: Dict[str, Dict[str, torch.Tensor]],
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Visualize masks across all hierarchy levels.

        Args:
            masks: Dictionary of hierarchical masks from __call__.
            sample_idx: Which sample from the batch to visualize.
            figsize: Figure size.
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure object.
        """
        num_levels = len(masks)
        fig, axes = plt.subplots(2, num_levels, figsize=figsize)

        if num_levels == 1:
            axes = axes.reshape(2, 1)

        for level_idx in range(num_levels):
            level_key = f"level_{level_idx}"
            level_masks = masks[level_key]

            context_mask = level_masks["context"][sample_idx].cpu().numpy()
            target_masks = level_masks["targets"][sample_idx].cpu().numpy()

            # Reshape to 2D
            context_2d = context_mask.reshape(self.num_patches_h, self.num_patches_w)

            # Visualize context
            axes[0, level_idx].imshow(context_2d, cmap="Blues", vmin=0, vmax=1)
            axes[0, level_idx].set_title(f"Level {level_idx} Context")
            axes[0, level_idx].axis("off")

            # Visualize targets
            target_union = np.zeros((self.num_patches_h, self.num_patches_w))
            for i, tgt_mask in enumerate(target_masks):
                tgt_2d = tgt_mask.reshape(self.num_patches_h, self.num_patches_w)
                target_union = np.maximum(target_union, tgt_2d * (i + 1))

            axes[1, level_idx].imshow(target_union, cmap="Set1", vmin=0, vmax=self.num_target_masks)
            axes[1, level_idx].set_title(f"Level {level_idx} Targets")
            axes[1, level_idx].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def visualize_combined_view(
        self,
        masks: Dict[str, Dict[str, torch.Tensor]],
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Visualize all hierarchy levels in a combined view.

        Args:
            masks: Dictionary of hierarchical masks from __call__.
            sample_idx: Which sample from the batch to visualize.
            figsize: Figure size.
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure object.
        """
        num_levels = len(masks)
        fig, axes = plt.subplots(1, num_levels, figsize=figsize)

        if num_levels == 1:
            axes = [axes]

        for level_idx in range(num_levels):
            level_key = f"level_{level_idx}"
            level_masks = masks[level_key]

            context_mask = level_masks["context"][sample_idx].cpu().numpy()
            target_masks = level_masks["targets"][sample_idx].cpu().numpy()

            # Create combined RGB view
            context_2d = context_mask.reshape(self.num_patches_h, self.num_patches_w)
            combined = np.zeros((self.num_patches_h, self.num_patches_w, 3))

            # Blue for context
            combined[context_2d > 0] = [0.2, 0.6, 1.0]

            # Different colors for targets
            for i, tgt_mask in enumerate(target_masks):
                tgt_2d = tgt_mask.reshape(self.num_patches_h, self.num_patches_w)
                color = cm.get_cmap("Set1")(i / self.num_target_masks)[:3]
                combined[tgt_2d > 0] = color

            axes[level_idx].imshow(combined)
            axes[level_idx].set_title(f"Level {level_idx}")
            axes[level_idx].axis("off")

        plt.suptitle("Hierarchical Masking - Combined View", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def get_hierarchical_statistics(
        self,
        masks: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for masks at each hierarchy level.

        Args:
            masks: Dictionary of hierarchical masks from __call__.

        Returns:
            Dictionary mapping level names to statistics dicts.
        """
        stats = {}

        for level_key, level_masks in masks.items():
            context_mask = level_masks["context"]
            target_masks = level_masks["targets"]

            batch_size = context_mask.shape[0]

            # Coverage
            context_coverage = context_mask.float().mean(dim=1)
            target_coverage = target_masks.float().mean(dim=2)

            # Overlaps
            overlaps = []
            for i in range(batch_size):
                ctx = context_mask[i]
                for j in range(self.num_target_masks):
                    tgt = target_masks[i, j]
                    if tgt.float().sum() > 0:
                        overlap = (ctx & tgt).float().sum() / tgt.float().sum()
                        overlaps.append(overlap.item())

            level_stats: Dict[str, float] = {
                "context_coverage_mean": float(context_coverage.mean().item()),
                "context_coverage_std": float(context_coverage.std().item()),
                "target_coverage_mean": float(target_coverage.mean().item()),
                "target_coverage_std": float(target_coverage.std().item()),
                "overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
                "overlap_max": float(np.max(overlaps)) if overlaps else 0.0,
            }

            stats[level_key] = level_stats

        return stats


def demo() -> None:
    """Demonstration of HierarchicalMaskGenerator."""
    print("Hierarchical Mask Generator Demo")
    print("=" * 50)

    # Create mask generator
    mask_gen = HierarchicalMaskGenerator(
        input_size=224,
        patch_size=16,
        num_hierarchies=3,
        num_target_masks=4,
        scale_progression="geometric",
    )

    print(f"Input size: {mask_gen.input_size}")
    print(f"Patch size: {mask_gen.patch_size}")
    print(
        f"Number of patches: {mask_gen.num_patches} ({mask_gen.num_patches_h}x{mask_gen.num_patches_w})"
    )
    print(f"Number of hierarchies: {mask_gen.num_hierarchies}")
    print(f"Number of target masks per level: {mask_gen.num_target_masks}")
    print()

    # Print level configurations
    print("Level Configurations:")
    for config in mask_gen.level_configs:
        print(f"  Level {config['level']}:")
        print(f"    Target scale: {config['target_scale']}")
        print(f"    Context scale: {config['context_scale']}")
    print()

    # Generate masks
    batch_size = 4
    masks = mask_gen(batch_size=batch_size, device="cpu")

    print(f"Generated masks for batch_size={batch_size}")
    for level_key, level_masks in masks.items():
        print(f"  {level_key}:")
        print(f"    Context shape: {level_masks['context'].shape}")
        print(f"    Targets shape: {level_masks['targets'].shape}")
    print()

    # Compute statistics
    stats = mask_gen.get_hierarchical_statistics(masks)
    print("Hierarchical Mask Statistics:")
    for level_key, level_stats in stats.items():
        print(f"  {level_key}:")
        for key, value in level_stats.items():
            print(f"    {key}: {value:.4f}")
    print()

    # Visualize
    print("Generating visualizations...")
    fig1 = mask_gen.visualize_hierarchical_masks(
        masks, sample_idx=0, save_path="/tmp/hierarchical_masks_demo.png"
    )
    plt.close(fig1)
    print("Visualization saved to /tmp/hierarchical_masks_demo.png")

    fig2 = mask_gen.visualize_combined_view(
        masks, sample_idx=0, save_path="/tmp/hierarchical_masks_combined_demo.png"
    )
    plt.close(fig2)
    print("Combined view saved to /tmp/hierarchical_masks_combined_demo.png")

    print("\nDemo complete!")


if __name__ == "__main__":
    demo()

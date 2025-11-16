"""
Multi-crop masking strategy for H-JEPA.

This module implements masking strategies designed for multi-crop training.
The key idea is to:
- Apply standard hierarchical masks to global crops (full resolution)
- Use local crops as additional context or prediction targets
- Enable cross-crop prediction for richer learning

This combines the benefits of multi-crop augmentation with H-JEPA's
hierarchical predictive learning.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional, Dict, Literal
import matplotlib.pyplot as plt

from .hierarchical import HierarchicalMaskGenerator


class MultiCropMaskGenerator:
    """
    Multi-crop mask generator for H-JEPA training.

    This generator creates masks for multi-crop inputs where we have:
    - Global crops: Full resolution views (e.g., 224x224)
    - Local crops: Lower resolution views (e.g., 96x96)

    Masking strategies:
    1. 'global_only': Apply hierarchical masks only to global crops
    2. 'global_with_local_context': Mask global crops, use local crops as context
    3. 'cross_crop_prediction': Enable prediction across crops

    Args:
        global_crop_size: Size of global crops (default: 224)
        local_crop_size: Size of local crops (default: 96)
        num_global_crops: Number of global crops (default: 2)
        num_local_crops: Number of local crops (default: 6)
        patch_size: Size of patches for vision transformer (default: 16)
        num_hierarchies: Number of hierarchical levels (default: 3)
        num_target_masks: Number of target masks per level (default: 4)
        masking_strategy: Strategy for masking ('global_only', 'global_with_local_context',
                          'cross_crop_prediction') (default: 'global_only')
        base_scale: Base scale for finest level masks (default: (0.05, 0.15))
        aspect_ratio_range: Aspect ratio range for masks (default: (0.75, 1.5))

    Example:
        >>> mask_gen = MultiCropMaskGenerator(
        ...     num_global_crops=2,
        ...     num_local_crops=6,
        ...     global_crop_size=224,
        ...     local_crop_size=96,
        ... )
        >>> masks = mask_gen(batch_size=8)
        >>> # Returns masks for global crops with hierarchical structure
    """

    def __init__(
        self,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        num_global_crops: int = 2,
        num_local_crops: int = 6,
        patch_size: int = 16,
        num_hierarchies: int = 3,
        num_target_masks: int = 4,
        masking_strategy: Literal['global_only', 'global_with_local_context', 'cross_crop_prediction'] = 'global_only',
        base_scale: Tuple[float, float] = (0.05, 0.15),
        aspect_ratio_range: Tuple[float, float] = (0.75, 1.5),
        max_attempts: int = 10,
    ):
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.patch_size = patch_size
        self.num_hierarchies = num_hierarchies
        self.num_target_masks = num_target_masks
        self.masking_strategy = masking_strategy

        # Calculate patch dimensions for global crops
        self.global_num_patches_h = global_crop_size // patch_size
        self.global_num_patches_w = global_crop_size // patch_size
        self.global_num_patches = self.global_num_patches_h * self.global_num_patches_w

        # Calculate patch dimensions for local crops
        self.local_num_patches_h = local_crop_size // patch_size
        self.local_num_patches_w = local_crop_size // patch_size
        self.local_num_patches = self.local_num_patches_h * self.local_num_patches_w

        # Create hierarchical mask generator for global crops
        self.global_mask_gen = HierarchicalMaskGenerator(
            input_size=global_crop_size,
            patch_size=patch_size,
            num_hierarchies=num_hierarchies,
            num_target_masks=num_target_masks,
            base_scale=base_scale,
            aspect_ratio_range=aspect_ratio_range,
            max_attempts=max_attempts,
        )

        # Create mask generator for local crops if needed
        if masking_strategy == 'cross_crop_prediction':
            self.local_mask_gen = HierarchicalMaskGenerator(
                input_size=local_crop_size,
                patch_size=patch_size,
                num_hierarchies=num_hierarchies,
                num_target_masks=num_target_masks,
                base_scale=base_scale,
                aspect_ratio_range=aspect_ratio_range,
                max_attempts=max_attempts,
            )
        else:
            self.local_mask_gen = None

    def __call__(
        self,
        batch_size: int,
        device: str = 'cuda'
    ) -> Dict[str, Dict]:
        """
        Generate masks for multi-crop inputs.

        Args:
            batch_size: Number of samples in batch
            device: Device to create tensors on

        Returns:
            Dictionary with mask information:
            {
                'global_masks': {
                    'crop_0': {'level_0': {...}, 'level_1': {...}, ...},
                    'crop_1': {...},
                },
                'local_masks': None or similar structure,
                'strategy': masking strategy name,
            }
        """
        masks = {
            'strategy': self.masking_strategy,
            'num_global_crops': self.num_global_crops,
            'num_local_crops': self.num_local_crops,
        }

        if self.masking_strategy == 'global_only':
            masks.update(self._generate_global_only_masks(batch_size, device))
        elif self.masking_strategy == 'global_with_local_context':
            masks.update(self._generate_global_with_local_masks(batch_size, device))
        elif self.masking_strategy == 'cross_crop_prediction':
            masks.update(self._generate_cross_crop_masks(batch_size, device))
        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")

        return masks

    def _generate_global_only_masks(
        self,
        batch_size: int,
        device: str
    ) -> Dict[str, Dict]:
        """
        Generate masks only for global crops.

        Local crops are not masked and can be used as-is for additional context.
        """
        global_masks = {}

        for crop_idx in range(self.num_global_crops):
            # Generate hierarchical masks for this global crop
            crop_masks = self.global_mask_gen(batch_size=batch_size, device=device)
            global_masks[f'crop_{crop_idx}'] = crop_masks

        return {
            'global_masks': global_masks,
            'local_masks': None,
        }

    def _generate_global_with_local_masks(
        self,
        batch_size: int,
        device: str
    ) -> Dict[str, Dict]:
        """
        Generate masks for global crops, treat local crops as context.

        Local crops provide additional context without explicit masking.
        This encourages the model to use multi-scale information.
        """
        global_masks = {}

        for crop_idx in range(self.num_global_crops):
            crop_masks = self.global_mask_gen(batch_size=batch_size, device=device)
            global_masks[f'crop_{crop_idx}'] = crop_masks

        # Create simple "all-visible" masks for local crops
        local_masks = {}
        for crop_idx in range(self.num_local_crops):
            # All patches visible (no masking)
            local_context = torch.ones(
                batch_size, self.local_num_patches,
                dtype=torch.bool, device=device
            )
            local_masks[f'crop_{crop_idx}'] = {
                'context': local_context,
                'targets': None,  # No prediction targets for local crops
            }

        return {
            'global_masks': global_masks,
            'local_masks': local_masks,
        }

    def _generate_cross_crop_masks(
        self,
        batch_size: int,
        device: str
    ) -> Dict[str, Dict]:
        """
        Generate masks for cross-crop prediction.

        Both global and local crops can have masks, enabling prediction
        from one crop to another (e.g., predict global from local).
        """
        global_masks = {}
        local_masks = {}

        # Mask all global crops
        for crop_idx in range(self.num_global_crops):
            crop_masks = self.global_mask_gen(batch_size=batch_size, device=device)
            global_masks[f'crop_{crop_idx}'] = crop_masks

        # Mask subset of local crops for cross-prediction
        # Strategy: Mask first half, leave second half for context
        num_masked_local = self.num_local_crops // 2

        for crop_idx in range(self.num_local_crops):
            if crop_idx < num_masked_local:
                # Apply masking to this local crop
                crop_masks = self.local_mask_gen(batch_size=batch_size, device=device)
                local_masks[f'crop_{crop_idx}'] = crop_masks
            else:
                # No masking - pure context
                local_context = torch.ones(
                    batch_size, self.local_num_patches,
                    dtype=torch.bool, device=device
                )
                local_masks[f'crop_{crop_idx}'] = {
                    'context': local_context,
                    'targets': None,
                }

        return {
            'global_masks': global_masks,
            'local_masks': local_masks,
        }

    def get_crop_info(self) -> Dict[str, int]:
        """
        Get information about crop dimensions.

        Returns:
            Dictionary with crop and patch information
        """
        return {
            'global_crop_size': self.global_crop_size,
            'local_crop_size': self.local_crop_size,
            'global_num_patches': self.global_num_patches,
            'local_num_patches': self.local_num_patches,
            'num_global_crops': self.num_global_crops,
            'num_local_crops': self.num_local_crops,
            'total_crops': self.num_global_crops + self.num_local_crops,
        }

    def visualize_multicrop_masks(
        self,
        masks: Dict[str, Dict],
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (20, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize masks across all crops.

        Args:
            masks: Mask dictionary from __call__
            sample_idx: Which sample to visualize
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        global_masks = masks['global_masks']
        local_masks = masks.get('local_masks', None)

        num_global = len(global_masks)
        num_local = len(local_masks) if local_masks else 0
        num_levels = self.num_hierarchies

        # Create subplots: rows for levels, columns for crops
        fig, axes = plt.subplots(
            num_levels * 2,  # context + targets for each level
            num_global + num_local,
            figsize=figsize
        )

        if num_global + num_local == 1:
            axes = axes.reshape(-1, 1)

        # Plot global crops
        for crop_idx in range(num_global):
            crop_key = f'crop_{crop_idx}'
            crop_masks = global_masks[crop_key]

            for level_idx in range(num_levels):
                level_key = f'level_{level_idx}'
                level_masks = crop_masks[level_key]

                context_mask = level_masks['context'][sample_idx].cpu().numpy()
                target_masks = level_masks['targets'][sample_idx].cpu().numpy()

                # Reshape to 2D
                context_2d = context_mask.reshape(
                    self.global_num_patches_h,
                    self.global_num_patches_w
                )

                # Plot context
                row_ctx = level_idx * 2
                axes[row_ctx, crop_idx].imshow(context_2d, cmap='Blues', vmin=0, vmax=1)
                axes[row_ctx, crop_idx].set_title(
                    f'G{crop_idx} L{level_idx} Context',
                    fontsize=8
                )
                axes[row_ctx, crop_idx].axis('off')

                # Plot targets
                target_union = np.zeros((
                    self.global_num_patches_h,
                    self.global_num_patches_w
                ))
                for i, tgt_mask in enumerate(target_masks):
                    tgt_2d = tgt_mask.reshape(
                        self.global_num_patches_h,
                        self.global_num_patches_w
                    )
                    target_union = np.maximum(target_union, tgt_2d * (i + 1))

                row_tgt = level_idx * 2 + 1
                axes[row_tgt, crop_idx].imshow(
                    target_union,
                    cmap='Set1',
                    vmin=0,
                    vmax=self.num_target_masks
                )
                axes[row_tgt, crop_idx].set_title(
                    f'G{crop_idx} L{level_idx} Targets',
                    fontsize=8
                )
                axes[row_tgt, crop_idx].axis('off')

        # Plot local crops if available
        if local_masks:
            for crop_idx in range(num_local):
                col_idx = num_global + crop_idx
                crop_key = f'crop_{crop_idx}'
                crop_masks = local_masks[crop_key]

                if 'level_0' in crop_masks:
                    # Masked local crop
                    for level_idx in range(num_levels):
                        level_key = f'level_{level_idx}'
                        level_masks = crop_masks[level_key]

                        context_mask = level_masks['context'][sample_idx].cpu().numpy()
                        context_2d = context_mask.reshape(
                            self.local_num_patches_h,
                            self.local_num_patches_w
                        )

                        row_ctx = level_idx * 2
                        axes[row_ctx, col_idx].imshow(context_2d, cmap='Blues', vmin=0, vmax=1)
                        axes[row_ctx, col_idx].set_title(f'L{crop_idx} Context', fontsize=8)
                        axes[row_ctx, col_idx].axis('off')

                        # Targets
                        target_masks = level_masks['targets'][sample_idx].cpu().numpy()
                        target_union = np.zeros((
                            self.local_num_patches_h,
                            self.local_num_patches_w
                        ))
                        for i, tgt_mask in enumerate(target_masks):
                            tgt_2d = tgt_mask.reshape(
                                self.local_num_patches_h,
                                self.local_num_patches_w
                            )
                            target_union = np.maximum(target_union, tgt_2d * (i + 1))

                        row_tgt = level_idx * 2 + 1
                        axes[row_tgt, col_idx].imshow(
                            target_union,
                            cmap='Set1',
                            vmin=0,
                            vmax=self.num_target_masks
                        )
                        axes[row_tgt, col_idx].set_title(f'L{crop_idx} Targets', fontsize=8)
                        axes[row_tgt, col_idx].axis('off')
                else:
                    # Unmasked local crop (pure context)
                    context_mask = crop_masks['context'][sample_idx].cpu().numpy()
                    context_2d = context_mask.reshape(
                        self.local_num_patches_h,
                        self.local_num_patches_w
                    )

                    for level_idx in range(num_levels):
                        row_ctx = level_idx * 2
                        axes[row_ctx, col_idx].imshow(context_2d, cmap='Greens', vmin=0, vmax=1)
                        axes[row_ctx, col_idx].set_title(f'L{crop_idx} Context', fontsize=8)
                        axes[row_ctx, col_idx].axis('off')

                        row_tgt = level_idx * 2 + 1
                        axes[row_tgt, col_idx].text(
                            0.5, 0.5, 'No Targets',
                            ha='center', va='center',
                            fontsize=8
                        )
                        axes[row_tgt, col_idx].axis('off')

        plt.suptitle(
            f'Multi-Crop Masking Strategy: {self.masking_strategy}',
            fontsize=14
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def demo():
    """Demonstration of MultiCropMaskGenerator."""
    print("Multi-Crop Mask Generator Demo")
    print("=" * 70)

    strategies = ['global_only', 'global_with_local_context', 'cross_crop_prediction']

    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        print("-" * 70)

        mask_gen = MultiCropMaskGenerator(
            global_crop_size=224,
            local_crop_size=96,
            num_global_crops=2,
            num_local_crops=6,
            num_hierarchies=3,
            num_target_masks=4,
            masking_strategy=strategy,
        )

        crop_info = mask_gen.get_crop_info()
        print("Crop configuration:")
        for key, value in crop_info.items():
            print(f"  {key}: {value}")

        # Generate masks
        batch_size = 4
        masks = mask_gen(batch_size=batch_size, device='cpu')

        print(f"\nGenerated masks for batch_size={batch_size}:")
        print(f"  Strategy: {masks['strategy']}")
        print(f"  Global crops: {masks['num_global_crops']}")
        print(f"  Local crops: {masks['num_local_crops']}")

        if masks['global_masks']:
            print(f"  Global mask structure: {list(masks['global_masks'].keys())}")

        if masks['local_masks']:
            print(f"  Local mask structure: {list(masks['local_masks'].keys())}")

        # Visualize
        save_path = f'/tmp/multicrop_masks_{strategy}.png'
        fig = mask_gen.visualize_multicrop_masks(
            masks,
            sample_idx=0,
            save_path=save_path
        )
        plt.close(fig)
        print(f"  Visualization saved to {save_path}")

    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    demo()

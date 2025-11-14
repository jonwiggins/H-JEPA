"""
Hierarchical Joint-Embedding Predictive Architecture (H-JEPA).

This module implements the main H-JEPA model that combines context encoder,
target encoder, and predictor for hierarchical self-supervised learning.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .encoder import ContextEncoder, TargetEncoder, create_encoder
from .predictor import Predictor, create_predictor


class HJEPA(nn.Module):
    """
    Hierarchical Joint-Embedding Predictive Architecture.

    Combines context encoder, target encoder (with EMA), and predictor to learn
    hierarchical visual representations through prediction of masked regions.

    Args:
        encoder_type: Vision Transformer type from timm
        img_size: Input image size
        embed_dim: Embedding dimension
        predictor_depth: Number of layers in predictor
        predictor_num_heads: Number of attention heads in predictor
        predictor_mlp_ratio: MLP ratio in predictor
        num_hierarchies: Number of hierarchical levels (2-4)
        ema_momentum: Initial EMA momentum
        ema_momentum_end: Final EMA momentum
        ema_warmup_steps: Warmup steps for EMA
        pretrained: Whether to use pretrained encoder
        drop_path_rate: Stochastic depth rate

    Attributes:
        context_encoder: Encoder for context (visible) patches
        target_encoder: Encoder for target (full image) with EMA
        predictor: Predictor network
        num_hierarchies: Number of hierarchical levels
        hierarchy_projections: Projection layers for each hierarchy level
    """

    def __init__(
        self,
        encoder_type: str = "vit_base_patch16_224",
        img_size: int = 224,
        embed_dim: int = 768,
        predictor_depth: int = 6,
        predictor_num_heads: int = 12,
        predictor_mlp_ratio: float = 4.0,
        num_hierarchies: int = 3,
        ema_momentum: float = 0.996,
        ema_momentum_end: float = 1.0,
        ema_warmup_steps: int = 1000,
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        if not 2 <= num_hierarchies <= 4:
            raise ValueError(f"num_hierarchies must be between 2 and 4, got {num_hierarchies}")

        self.num_hierarchies = num_hierarchies
        self.embed_dim = embed_dim
        self.img_size = img_size

        # Create encoders
        self.context_encoder, self.target_encoder = create_encoder(
            encoder_type=encoder_type,
            img_size=img_size,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )

        # Override EMA parameters for target encoder
        self.target_encoder.momentum = ema_momentum
        self.target_encoder.ema_momentum_end = ema_momentum_end
        self.target_encoder.ema_warmup_steps = ema_warmup_steps

        # Create predictor
        self.predictor = create_predictor(
            embed_dim=embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=predictor_mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

        # Hierarchical projection heads
        # Each level projects to different semantic granularity
        self.hierarchy_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            for _ in range(num_hierarchies)
        ])

        # Pooling layers for hierarchical representations
        self.hierarchy_pooling = nn.ModuleList([
            self._create_pooling_layer(level)
            for level in range(num_hierarchies)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _create_pooling_layer(self, level: int) -> nn.Module:
        """
        Create pooling layer for hierarchical level.

        Args:
            level: Hierarchy level (0 = finest, higher = coarser)

        Returns:
            Pooling module
        """
        if level == 0:
            # Finest level: no pooling
            return nn.Identity()
        else:
            # Coarser levels: average pooling with increasing kernel size
            kernel_size = 2 ** level
            return nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size)

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        return_all_levels: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through H-JEPA.

        Args:
            images: Input images [B, C, H, W]
            mask: Binary mask for patches [B, N] where 1 indicates masked position
            return_all_levels: Whether to return predictions for all hierarchy levels

        Returns:
            Dictionary containing:
                - 'predictions': List of predictions for each hierarchy level
                - 'targets': List of target representations for each hierarchy level
                - 'context_features': Encoded context features
                - 'target_features': Encoded target features (full image)
        """
        B = images.shape[0]

        # Encode context (visible patches)
        context_features = self.context_encoder(images, mask=mask)

        # Encode target (full image) with no gradient
        with torch.no_grad():
            target_features = self.target_encoder(images)

        # Get mask indices for prediction
        mask_bool = mask.bool()
        mask_indices = mask_bool.nonzero(as_tuple=True)[1].view(B, -1)

        # Get positional embeddings from context encoder
        pos_embed = self.context_encoder.vit.pos_embed

        # Predict masked representations
        predicted_features = self.predictor(
            context_features=context_features[:, 1:, :],  # Exclude CLS token
            mask_indices=mask_indices,
            pos_embed=pos_embed[:, 1:, :],  # Exclude CLS position
        )

        # Extract target features for masked positions
        mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        target_masked = torch.gather(
            target_features[:, 1:, :],  # Exclude CLS token
            1,
            mask_indices_expanded
        )

        if not return_all_levels:
            # Return only finest level
            return {
                'predictions': [predicted_features],
                'targets': [target_masked],
                'context_features': context_features,
                'target_features': target_features,
            }

        # Compute hierarchical predictions and targets
        predictions_hierarchy = []
        targets_hierarchy = []

        for level in range(self.num_hierarchies):
            # Project features to hierarchy-specific space
            pred_projected = self.hierarchy_projections[level](predicted_features)
            target_projected = self.hierarchy_projections[level](target_masked)

            # Apply pooling for coarser levels
            if level > 0:
                # Rearrange for 1D pooling: [B, N, D] -> [B, D, N]
                pred_projected = rearrange(pred_projected, 'b n d -> b d n')
                target_projected = rearrange(target_projected, 'b n d -> b d n')

                # Apply pooling
                pred_projected = self.hierarchy_pooling[level](pred_projected)
                target_projected = self.hierarchy_pooling[level](target_projected)

                # Rearrange back: [B, D, N'] -> [B, N', D]
                pred_projected = rearrange(pred_projected, 'b d n -> b n d')
                target_projected = rearrange(target_projected, 'b d n -> b n d')

            predictions_hierarchy.append(pred_projected)
            targets_hierarchy.append(target_projected)

        return {
            'predictions': predictions_hierarchy,
            'targets': targets_hierarchy,
            'context_features': context_features,
            'target_features': target_features,
        }

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        level: int = 0,
        use_target_encoder: bool = True,
    ) -> torch.Tensor:
        """
        Extract features at a specific hierarchy level.

        Args:
            images: Input images [B, C, H, W]
            level: Hierarchy level to extract (0 = finest)
            use_target_encoder: Whether to use target encoder (True) or context encoder (False)

        Returns:
            Features at specified level [B, N', D]
        """
        if level >= self.num_hierarchies:
            raise ValueError(f"Level {level} exceeds num_hierarchies {self.num_hierarchies}")

        # Encode image
        if use_target_encoder:
            features = self.target_encoder(images)
        else:
            features = self.context_encoder(images)

        # Exclude CLS token
        features = features[:, 1:, :]

        # Project to hierarchy level
        features = self.hierarchy_projections[level](features)

        # Apply pooling for coarser levels
        if level > 0:
            features = rearrange(features, 'b n d -> b d n')
            features = self.hierarchy_pooling[level](features)
            features = rearrange(features, 'b d n -> b n d')

        return features

    def update_target_encoder(self, current_step: int) -> float:
        """
        Update target encoder using EMA from context encoder.

        Args:
            current_step: Current training step

        Returns:
            Current EMA momentum value
        """
        return self.target_encoder.update_from_context_encoder(
            self.context_encoder, current_step
        )

    def get_num_patches(self) -> int:
        """Get number of patches (excluding CLS token)."""
        return self.context_encoder.num_patches

    def get_patch_size(self) -> int:
        """Get patch size."""
        return self.context_encoder.patch_size


def create_hjepa(
    encoder_type: str = "vit_base_patch16_224",
    img_size: int = 224,
    embed_dim: int = 768,
    predictor_depth: int = 6,
    predictor_num_heads: int = 12,
    predictor_mlp_ratio: float = 4.0,
    num_hierarchies: int = 3,
    ema_momentum: float = 0.996,
    ema_momentum_end: float = 1.0,
    ema_warmup_steps: int = 1000,
    pretrained: bool = False,
    drop_path_rate: float = 0.0,
) -> HJEPA:
    """
    Factory function to create H-JEPA model.

    Args:
        encoder_type: Vision Transformer type from timm
        img_size: Input image size
        embed_dim: Embedding dimension
        predictor_depth: Number of layers in predictor
        predictor_num_heads: Number of attention heads in predictor
        predictor_mlp_ratio: MLP ratio in predictor
        num_hierarchies: Number of hierarchical levels (2-4)
        ema_momentum: Initial EMA momentum
        ema_momentum_end: Final EMA momentum
        ema_warmup_steps: Warmup steps for EMA
        pretrained: Whether to use pretrained encoder
        drop_path_rate: Stochastic depth rate

    Returns:
        H-JEPA model
    """
    return HJEPA(
        encoder_type=encoder_type,
        img_size=img_size,
        embed_dim=embed_dim,
        predictor_depth=predictor_depth,
        predictor_num_heads=predictor_num_heads,
        predictor_mlp_ratio=predictor_mlp_ratio,
        num_hierarchies=num_hierarchies,
        ema_momentum=ema_momentum,
        ema_momentum_end=ema_momentum_end,
        ema_warmup_steps=ema_warmup_steps,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
    )


def create_hjepa_from_config(config: Dict) -> HJEPA:
    """
    Create H-JEPA model from configuration dictionary.

    Args:
        config: Configuration dictionary with 'model' section

    Returns:
        H-JEPA model
    """
    model_config = config.get('model', {})
    training_config = config.get('training', {})

    # Calculate warmup steps from epochs
    ema_warmup_epochs = model_config.get('ema', {}).get('momentum_warmup_epochs', 30)
    # Assume ~1000 steps per epoch for step calculation (adjust based on dataset size)
    ema_warmup_steps = ema_warmup_epochs * 1000

    return create_hjepa(
        encoder_type=model_config.get('encoder_type', 'vit_base_patch16_224'),
        img_size=config.get('data', {}).get('image_size', 224),
        embed_dim=model_config.get('embed_dim', 768),
        predictor_depth=model_config.get('predictor', {}).get('depth', 6),
        predictor_num_heads=model_config.get('predictor', {}).get('num_heads', 12),
        predictor_mlp_ratio=model_config.get('predictor', {}).get('mlp_ratio', 4.0),
        num_hierarchies=model_config.get('num_hierarchies', 3),
        ema_momentum=model_config.get('ema', {}).get('momentum', 0.996),
        ema_momentum_end=model_config.get('ema', {}).get('momentum_end', 1.0),
        ema_warmup_steps=ema_warmup_steps,
        pretrained=False,
        drop_path_rate=training_config.get('drop_path_rate', 0.0),
    )

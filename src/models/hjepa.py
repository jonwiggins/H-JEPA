"""
Hierarchical Joint-Embedding Predictive Architecture (H-JEPA).

This module implements the main H-JEPA model that combines context encoder,
target encoder, and predictor for hierarchical self-supervised learning.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from einops import rearrange

from .encoder import create_encoder
from .predictor import create_predictor


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
        use_fpn: Whether to use Feature Pyramid Networks
        fpn_feature_dim: Feature dimension for FPN (default: same as embed_dim)
        fpn_fusion_method: Feature fusion method ('add' or 'concat')
        use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency
        use_layerscale: Whether to apply LayerScale regularization for training stability
        layerscale_init: Initial value for LayerScale parameters (default: 1e-5)
        use_flash_attention: Whether to use Flash Attention for 2-5x speedup

    Attributes:
        context_encoder: Encoder for context (visible) patches
        target_encoder: Encoder for target (full image) with EMA
        predictor: Predictor network
        num_hierarchies: Number of hierarchical levels
        hierarchy_projections: Projection layers for each hierarchy level
        use_fpn: Whether FPN is enabled
        fpn_lateral_convs: Lateral 1x1 convolutions for FPN
        fpn_top_down_convs: Top-down convolutions for FPN
        use_gradient_checkpointing: Flag for gradient checkpointing
        use_layerscale: Flag for LayerScale regularization
        use_flash_attention: Flag for Flash Attention
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
        use_fpn: bool = False,
        fpn_feature_dim: Optional[int] = None,
        fpn_fusion_method: str = "add",
        use_gradient_checkpointing: bool = False,
        use_layerscale: bool = False,
        layerscale_init: float = 1e-5,
        use_flash_attention: bool = True,
    ) -> None:
        super().__init__()

        if not 2 <= num_hierarchies <= 4:
            raise ValueError(f"num_hierarchies must be between 2 and 4, got {num_hierarchies}")

        if fpn_fusion_method not in ["add", "concat"]:
            raise ValueError(
                f"fpn_fusion_method must be 'add' or 'concat', got {fpn_fusion_method}"
            )

        self.num_hierarchies = num_hierarchies
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.use_fpn = use_fpn
        self.fpn_fusion_method = fpn_fusion_method
        self.fpn_feature_dim = fpn_feature_dim if fpn_feature_dim is not None else embed_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_layerscale = use_layerscale
        self.use_flash_attention = use_flash_attention

        # Create encoders
        self.context_encoder, self.target_encoder = create_encoder(
            encoder_type=encoder_type,
            img_size=img_size,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            use_flash_attention=use_flash_attention,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
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
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Hierarchical projection heads
        # Each level projects to different semantic granularity
        if use_fpn:
            # When using FPN, the output dimension is always fpn_feature_dim
            # (fusion convs reduce concat back to fpn_feature_dim)
            final_dim = self.fpn_feature_dim
        else:
            final_dim = embed_dim

        self.hierarchy_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(final_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                )
                for _ in range(num_hierarchies)
            ]
        )

        # Pooling layers for hierarchical representations
        self.hierarchy_pooling = nn.ModuleList(
            [self._create_pooling_layer(level) for level in range(num_hierarchies)]
        )

        # Feature Pyramid Network components
        if use_fpn:
            self._build_fpn()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
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
            kernel_size = 2**level
            return nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size)

    def _build_fpn(self) -> None:
        """
        Build Feature Pyramid Network components.

        FPN consists of:
        1. Lateral connections: 1x1 convolutions that reduce channel dimensions
           at each pyramid level to a uniform feature dimension.
        2. Top-down pathway: Upsampling from coarser to finer levels.
        3. Feature fusion: Combining lateral and top-down features via
           addition or concatenation.

        The FPN enables multi-scale feature learning by allowing information
        to flow both bottom-up (through the backbone) and top-down (through
        the pyramid), creating semantically strong features at all scales.
        """
        # Lateral connections: 1x1 convolutions to project features to FPN dimension
        # These operate on features at each hierarchy level before pooling
        self.fpn_lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dim, self.fpn_feature_dim),
                    nn.LayerNorm(self.fpn_feature_dim),
                )
                for _ in range(self.num_hierarchies)
            ]
        )

        # Top-down pathway: convolutions applied after upsampling
        # These smooth the upsampled features before fusion
        self.fpn_top_down_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.fpn_feature_dim, self.fpn_feature_dim),
                    nn.LayerNorm(self.fpn_feature_dim),
                )
                for _ in range(self.num_hierarchies - 1)  # No top-down for coarsest level
            ]
        )

        # If using concatenation fusion, we need additional convolutions
        # to combine the concatenated features
        if self.fpn_fusion_method == "concat":
            self.fpn_fusion_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self.fpn_feature_dim, self.fpn_feature_dim),
                        nn.LayerNorm(self.fpn_feature_dim),
                        nn.ReLU(inplace=True),
                    )
                    for _ in range(self.num_hierarchies - 1)
                ]
            )

    def _apply_fpn(
        self,
        features: torch.Tensor,
        is_prediction: bool = False,
    ) -> List[torch.Tensor]:
        """
        Apply Feature Pyramid Network to hierarchical features.

        Args:
            features: Input features [B, N, D] where N is number of tokens
            is_prediction: Whether this is for prediction (affects handling)

        Returns:
            List of FPN-enhanced features for each hierarchy level,
            ordered from finest (level 0) to coarsest (level N-1)

        FPN Process:
        1. Create pyramid levels using pooling (bottom-up pathway)
        2. Apply lateral connections to each level
        3. Build top-down pathway:
           - Start from coarsest level
           - Upsample and combine with finer levels
        4. Return enhanced multi-scale features
        """
        B, N, D = features.shape

        # Step 1: Create pyramid levels using pooling (bottom-up pathway)
        pyramid_features = []
        for level in range(self.num_hierarchies):
            level_features = features

            # Apply pooling for coarser levels
            if level > 0:
                # Rearrange for 1D pooling: [B, N, D] -> [B, D, N]
                level_features = rearrange(level_features, "b n d -> b d n")
                level_features = self.hierarchy_pooling[level](level_features)
                # Rearrange back: [B, D, N'] -> [B, N', D]
                level_features = rearrange(level_features, "b d n -> b n d")

            pyramid_features.append(level_features)

        # Step 2: Apply lateral connections (1x1 conv to uniform dimension)
        lateral_features = [
            self.fpn_lateral_convs[level](pyramid_features[level])
            for level in range(self.num_hierarchies)
        ]

        # Step 3: Build top-down pathway
        # Start from coarsest level and propagate to finer levels
        fpn_features = [None] * self.num_hierarchies

        # Initialize coarsest level (no top-down input)
        fpn_features[-1] = lateral_features[-1]

        # Propagate from coarse to fine
        for level in range(self.num_hierarchies - 2, -1, -1):
            # Get top-down features from coarser level
            top_down = fpn_features[level + 1]

            # Upsample top-down features to match current level resolution
            # For 1D sequence, we use interpolation
            current_n = lateral_features[level].shape[1]
            top_down_n = top_down.shape[1]  # type: ignore[attr-defined]

            if top_down_n != current_n:
                # Rearrange for interpolation: [B, N, D] -> [B, D, N]
                top_down = rearrange(top_down, "b n d -> b d n")
                # Upsample using linear interpolation
                top_down = torch.nn.functional.interpolate(
                    top_down,
                    size=current_n,
                    mode="linear",
                    align_corners=False,
                )
                # Rearrange back: [B, D, N] -> [B, N, D]
                top_down = rearrange(top_down, "b d n -> b n d")

            # Apply top-down convolution for smoothing
            top_down = self.fpn_top_down_convs[level](top_down)

            # Fuse lateral and top-down features
            if self.fpn_fusion_method == "add":
                # Element-wise addition
                fpn_features[level] = lateral_features[level] + top_down
            else:  # concat
                # Concatenate along feature dimension
                fused = torch.cat([lateral_features[level], top_down], dim=-1)
                # Apply fusion convolution to reduce dimension
                fpn_features[level] = self.fpn_fusion_convs[level](fused)

        return fpn_features  # type: ignore[return-value]

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
                - 'masks_valid': Validity mask(s) indicating which positions are valid (not padding)
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
        # mask shape: [B, N] where N is number of patches
        # We need to handle variable number of masked patches per sample
        mask_bool = mask.bool()

        # Find the number of masked patches per sample
        num_masked_per_sample = mask_bool.sum(dim=1)
        max_masked = num_masked_per_sample.max().item()

        # Create padded mask indices tensor [B, max_masked]
        mask_indices = torch.zeros((B, max_masked), dtype=torch.long, device=mask.device)  # type: ignore[arg-type]

        # Create validity mask to track which indices are actual (not padding)
        # This fixes the bug where padded zeros would gather from patch 0 repeatedly
        mask_valid = torch.zeros((B, max_masked), dtype=torch.bool, device=mask.device)

        # Fill in the actual mask indices for each sample
        for i in range(B):
            sample_mask_indices = mask_bool[i].nonzero(as_tuple=True)[0]
            num_masked = len(sample_mask_indices)
            mask_indices[i, :num_masked] = sample_mask_indices
            mask_valid[i, :num_masked] = True

        # Get positional embeddings from context encoder
        # pos_embed is [1, N+1, D], we need to expand to [B, N, D] (excluding CLS)
        pos_embed = self.context_encoder.vit.pos_embed[:, 1:, :].expand(B, -1, -1)

        # Predict masked representations
        predicted_features = self.predictor(
            context_features=context_features[:, 1:, :],  # Exclude CLS token
            mask_indices=mask_indices,
            pos_embed=pos_embed,  # Already excludes CLS and expanded to batch size
        )

        # Extract target features for masked positions
        mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        target_masked = torch.gather(
            target_features[:, 1:, :], 1, mask_indices_expanded  # Exclude CLS token
        )

        if not return_all_levels:
            # Return only finest level
            return {
                "predictions": [predicted_features],
                "targets": [target_masked],
                "mask_valid": mask_valid,  # Validity mask for padded positions
                "context_features": context_features,
                "target_features": target_features,
            }

        # Compute hierarchical predictions and targets
        predictions_hierarchy = []
        targets_hierarchy = []
        masks_valid_hierarchy = []

        if self.use_fpn:
            # Apply FPN to create multi-scale features with top-down pathway
            pred_fpn_features = self._apply_fpn(predicted_features, is_prediction=True)
            target_fpn_features = self._apply_fpn(target_masked, is_prediction=False)

            # Process mask_valid through FPN pooling as well
            # Convert to float and expand to embedding dimension for FPN processing
            mask_valid_float = mask_valid.unsqueeze(-1).float()  # [B, N, 1]
            # Expand to match embedding dimension expected by FPN
            mask_valid_expanded = mask_valid_float.expand(
                -1, -1, self.embed_dim
            )  # [B, N, embed_dim]
            mask_fpn_features = self._apply_fpn(mask_valid_expanded, is_prediction=False)

            # Project FPN features to final embedding space
            for level in range(self.num_hierarchies):
                pred_projected = self.hierarchy_projections[level](pred_fpn_features[level])
                target_projected = self.hierarchy_projections[level](target_fpn_features[level])

                # Convert pooled mask back to bool (average across embedding dim, then threshold)
                # Take mean across embedding dimension to get back mask shape
                mask_valid_level = mask_fpn_features[level].mean(dim=-1) > 0.5

                predictions_hierarchy.append(pred_projected)
                targets_hierarchy.append(target_projected)
                masks_valid_hierarchy.append(mask_valid_level)
        else:
            # Original hierarchical pooling without FPN
            for level in range(self.num_hierarchies):
                # Project features to hierarchy-specific space
                pred_projected = self.hierarchy_projections[level](predicted_features)
                target_projected = self.hierarchy_projections[level](target_masked)

                # Process mask_valid with same pooling
                mask_valid_level = mask_valid.clone()

                # Apply pooling for coarser levels
                if level > 0:
                    # Rearrange for 1D pooling: [B, N, D] -> [B, D, N]
                    pred_projected = rearrange(pred_projected, "b n d -> b d n")
                    target_projected = rearrange(target_projected, "b n d -> b d n")

                    # Apply pooling
                    pred_projected = self.hierarchy_pooling[level](pred_projected)
                    target_projected = self.hierarchy_pooling[level](target_projected)

                    # Rearrange back: [B, D, N'] -> [B, N', D]
                    pred_projected = rearrange(pred_projected, "b d n -> b n d")
                    target_projected = rearrange(target_projected, "b d n -> b n d")

                    # Pool mask_valid as well (convert to float, pool, threshold back to bool)
                    mask_valid_float = rearrange(mask_valid_level.float(), "b n -> b 1 n")
                    mask_valid_float = self.hierarchy_pooling[level](mask_valid_float)
                    mask_valid_level = rearrange(mask_valid_float, "b 1 n -> b n") > 0.5

                predictions_hierarchy.append(pred_projected)
                targets_hierarchy.append(target_projected)
                masks_valid_hierarchy.append(mask_valid_level)

        return {
            "predictions": predictions_hierarchy,
            "targets": targets_hierarchy,
            "masks_valid": masks_valid_hierarchy,  # Validity masks for each hierarchy level
            "context_features": context_features,
            "target_features": target_features,
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

        if self.use_fpn:
            # Apply FPN to get multi-scale features
            fpn_features = self._apply_fpn(features, is_prediction=False)
            # Get features at requested level
            features = fpn_features[level]
            # Project to final embedding space
            features = self.hierarchy_projections[level](features)
        else:
            # Original method: project then pool
            features = self.hierarchy_projections[level](features)

            # Apply pooling for coarser levels
            if level > 0:
                features = rearrange(features, "b n d -> b d n")
                features = self.hierarchy_pooling[level](features)
                features = rearrange(features, "b d n -> b n d")

        return features  # type: ignore[no-any-return]

    def update_target_encoder(self, current_step: int) -> float:
        """
        Update target encoder using EMA from context encoder.

        Args:
            current_step: Current training step

        Returns:
            Current EMA momentum value
        """
        return self.target_encoder.update_from_context_encoder(self.context_encoder, current_step)

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
    use_fpn: bool = False,
    fpn_feature_dim: Optional[int] = None,
    fpn_fusion_method: str = "add",
    use_gradient_checkpointing: bool = False,
    use_layerscale: bool = False,
    layerscale_init: float = 1e-5,
    use_flash_attention: bool = True,
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
        use_fpn: Whether to use Feature Pyramid Networks
        fpn_feature_dim: Feature dimension for FPN (default: same as embed_dim)
        fpn_fusion_method: Feature fusion method ('add' or 'concat')
        use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency
        use_layerscale: Whether to apply LayerScale regularization for training stability
        layerscale_init: Initial value for LayerScale parameters (default: 1e-5)
        use_flash_attention: Whether to use Flash Attention for 2-5x speedup

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
        use_fpn=use_fpn,
        fpn_feature_dim=fpn_feature_dim,
        fpn_fusion_method=fpn_fusion_method,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_layerscale=use_layerscale,
        layerscale_init=layerscale_init,
        use_flash_attention=use_flash_attention,
    )


def create_hjepa_from_config(config: Dict[str, Any]) -> HJEPA:
    """
    Create H-JEPA model from configuration dictionary.

    Args:
        config: Configuration dictionary with 'model' section

    Returns:
        H-JEPA model
    """
    model_config = config.get("model", {})
    training_config = config.get("training", {})

    # Calculate warmup steps from epochs
    ema_warmup_epochs = model_config.get("ema", {}).get("momentum_warmup_epochs", 30)
    # Assume ~1000 steps per epoch for step calculation (adjust based on dataset size)
    ema_warmup_steps = ema_warmup_epochs * 1000

    # Get FPN configuration
    fpn_config = model_config.get("fpn", {})
    use_fpn = fpn_config.get("use_fpn", False)
    fpn_feature_dim = fpn_config.get("feature_dim", None)
    fpn_fusion_method = fpn_config.get("fusion_method", "add")

    # Get gradient checkpointing configuration
    use_gradient_checkpointing = training_config.get("use_gradient_checkpointing", False)

    # Get LayerScale configuration
    layerscale_config = model_config.get("layerscale", {})
    use_layerscale = layerscale_config.get("use_layerscale", False)
    layerscale_init = layerscale_config.get("init_value", 1e-5)

    # Get Flash Attention configuration
    use_flash_attention = model_config.get("use_flash_attention", True)

    return create_hjepa(
        encoder_type=model_config.get("encoder_type", "vit_base_patch16_224"),
        img_size=config.get("data", {}).get("image_size", 224),
        embed_dim=model_config.get("embed_dim", 768),
        predictor_depth=model_config.get("predictor", {}).get("depth", 6),
        predictor_num_heads=model_config.get("predictor", {}).get("num_heads", 12),
        predictor_mlp_ratio=model_config.get("predictor", {}).get("mlp_ratio", 4.0),
        num_hierarchies=model_config.get("num_hierarchies", 3),
        ema_momentum=model_config.get("ema", {}).get("momentum", 0.996),
        ema_momentum_end=model_config.get("ema", {}).get("momentum_end", 1.0),
        ema_warmup_steps=ema_warmup_steps,
        pretrained=False,
        drop_path_rate=training_config.get("drop_path_rate", 0.0),
        use_fpn=use_fpn,
        fpn_feature_dim=fpn_feature_dim,
        fpn_fusion_method=fpn_fusion_method,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_layerscale=use_layerscale,
        layerscale_init=layerscale_init,
        use_flash_attention=use_flash_attention,
    )

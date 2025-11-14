"""
Vision Transformer encoders for H-JEPA.

This module implements the Context and Target encoders with EMA update mechanism.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import timm
from einops import rearrange


class ContextEncoder(nn.Module):
    """
    Context encoder using Vision Transformer from timm.

    Processes the visible/context patches of the input image.

    Args:
        encoder_type: Model name from timm (e.g., 'vit_base_patch16_224')
        img_size: Input image size
        pretrained: Whether to load pretrained weights
        drop_path_rate: Stochastic depth rate

    Attributes:
        vit: Vision Transformer model
        embed_dim: Embedding dimension
        num_patches: Number of patches per image
        patch_size: Size of each patch
    """

    def __init__(
        self,
        encoder_type: str = "vit_base_patch16_224",
        img_size: int = 224,
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Create Vision Transformer using timm
        self.vit = timm.create_model(
            encoder_type,
            pretrained=pretrained,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )

        # Get model properties
        self.embed_dim = self.vit.embed_dim
        self.num_patches = self.vit.patch_embed.num_patches
        self.patch_size = self.vit.patch_embed.patch_size[0]

        # Remove classification head (we don't need it for JEPA)
        if hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the context encoder.

        Args:
            x: Input images [B, C, H, W]
            mask: Optional mask for patches [B, N] where True indicates masked patches

        Returns:
            Encoded features [B, N, D] where N is number of patches, D is embed_dim
        """
        # Get patch embeddings
        x = self.vit.patch_embed(x)

        # Add class token
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embeddings
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        # Apply mask if provided (set masked patches to zero)
        if mask is not None:
            # mask shape: [B, N], we need [B, N+1, 1] to account for cls token
            mask_with_cls = torch.cat([
                torch.zeros(mask.shape[0], 1, device=mask.device, dtype=mask.dtype),
                mask
            ], dim=1).unsqueeze(-1)
            x = x * (1 - mask_with_cls)

        # Pass through transformer blocks
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        return x

    def get_num_patches(self, img_size: int) -> int:
        """Calculate number of patches for given image size."""
        return (img_size // self.patch_size) ** 2

    def get_patch_size(self) -> int:
        """Return the patch size."""
        return self.patch_size


class TargetEncoder(nn.Module):
    """
    Target encoder with Exponential Moving Average (EMA) updates.

    Processes the full image to generate target representations.
    Updates weights via EMA from the context encoder.

    Args:
        encoder_type: Model name from timm (e.g., 'vit_base_patch16_224')
        img_size: Input image size
        ema_momentum: Initial EMA momentum (tau)
        ema_momentum_end: Final EMA momentum
        ema_warmup_steps: Number of warmup steps for EMA schedule
        pretrained: Whether to load pretrained weights
        drop_path_rate: Stochastic depth rate

    Attributes:
        vit: Vision Transformer model (updated via EMA)
        momentum: Current EMA momentum value
        ema_momentum_end: Target momentum value
        ema_warmup_steps: Warmup steps for momentum schedule
    """

    def __init__(
        self,
        encoder_type: str = "vit_base_patch16_224",
        img_size: int = 224,
        ema_momentum: float = 0.996,
        ema_momentum_end: float = 1.0,
        ema_warmup_steps: int = 1000,
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Create Vision Transformer using timm
        self.vit = timm.create_model(
            encoder_type,
            pretrained=pretrained,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )

        # Get model properties
        self.embed_dim = self.vit.embed_dim
        self.num_patches = self.vit.patch_embed.num_patches
        self.patch_size = self.vit.patch_embed.patch_size[0]

        # Remove classification head
        if hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()

        # EMA parameters
        self.momentum = ema_momentum
        self.ema_momentum_end = ema_momentum_end
        self.ema_warmup_steps = ema_warmup_steps

        # Disable gradient computation for target encoder
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the target encoder.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Encoded features [B, N, D] where N is number of patches, D is embed_dim
        """
        # Get patch embeddings
        x = self.vit.patch_embed(x)

        # Add class token
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embeddings
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        # Pass through transformer blocks
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        return x

    @torch.no_grad()
    def update_from_context_encoder(
        self,
        context_encoder: ContextEncoder,
        current_step: int,
    ) -> float:
        """
        Update target encoder weights using EMA from context encoder.

        Implements cosine schedule for momentum:
        tau(t) = tau_base + (tau_end - tau_base) * (1 + cos(pi * t / T)) / 2

        Args:
            context_encoder: Context encoder to copy weights from
            current_step: Current training step for momentum scheduling

        Returns:
            Current momentum value
        """
        # Calculate momentum with cosine schedule
        if current_step < self.ema_warmup_steps:
            # Cosine warmup schedule
            momentum = self.momentum + (self.ema_momentum_end - self.momentum) * (
                1 + math.cos(math.pi * current_step / self.ema_warmup_steps)
            ) / 2
        else:
            momentum = self.ema_momentum_end

        # Update weights: θ_target = momentum * θ_target + (1 - momentum) * θ_context
        for param_target, param_context in zip(
            self.vit.parameters(), context_encoder.vit.parameters()
        ):
            param_target.data.mul_(momentum).add_(
                param_context.data, alpha=1 - momentum
            )

        return momentum

    @torch.no_grad()
    def copy_from_context_encoder(self, context_encoder: ContextEncoder):
        """
        Initialize target encoder with context encoder weights.

        Args:
            context_encoder: Context encoder to copy weights from
        """
        for param_target, param_context in zip(
            self.vit.parameters(), context_encoder.vit.parameters()
        ):
            param_target.data.copy_(param_context.data)


def create_encoder(
    encoder_type: str = "vit_base_patch16_224",
    img_size: int = 224,
    pretrained: bool = False,
    drop_path_rate: float = 0.0,
) -> Tuple[ContextEncoder, TargetEncoder]:
    """
    Factory function to create context and target encoders.

    Args:
        encoder_type: Model name from timm
        img_size: Input image size
        pretrained: Whether to load pretrained weights
        drop_path_rate: Stochastic depth rate

    Returns:
        Tuple of (context_encoder, target_encoder)
    """
    context_encoder = ContextEncoder(
        encoder_type=encoder_type,
        img_size=img_size,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
    )

    target_encoder = TargetEncoder(
        encoder_type=encoder_type,
        img_size=img_size,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
    )

    # Initialize target encoder with context encoder weights
    target_encoder.copy_from_context_encoder(context_encoder)

    return context_encoder, target_encoder

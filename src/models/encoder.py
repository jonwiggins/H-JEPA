"""
Vision Transformer encoders for H-JEPA.

This module implements the Context and Target encoders with EMA update mechanism.
Includes support for Rotary Position Embeddings (RoPE) for improved positional encoding.
"""

import math
from typing import Any, Optional, Tuple, cast

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# Check if Flash Attention is available via PyTorch's scaled_dot_product_attention
# Available in PyTorch 2.0+ with CUDA 7.5+ or MPS
FLASH_ATTENTION_AVAILABLE = (
    hasattr(F, "scaled_dot_product_attention")
    and torch.__version__ >= "2.0.0"
    and (torch.cuda.is_available() or torch.backends.mps.is_available())
)


class VisionRoPE2D(nn.Module):
    """
    2D Rotary Position Embeddings (RoPE) for Vision Transformers.

    RoPE encodes position information by rotating query and key vectors based on their
    2D spatial positions in the image. This provides several advantages over absolute
    position embeddings:

    1. Better generalization to different resolutions
    2. Relative position encoding (distance between patches matters)
    3. No learnable parameters needed
    4. Used in V-JEPA 2 and modern ViTs

    The rotation is applied separately to pairs of dimensions in Q and K embeddings:
    - For 2D images, we decompose position into (x, y) coordinates
    - Each coordinate gets its own set of rotation frequencies
    - Rotation angle = position * base_frequency^(-2i/d) where i is the dimension index

    Args:
        dim: Dimension of the embeddings (must be divisible by 4 for 2D)
        patch_size: Size of image patches
        num_patches_per_side: Number of patches along one side of the image
        theta: Base frequency for rotation (default: 10000.0, as in original RoPE)

    Attributes:
        dim: Embedding dimension
        theta: Base rotation frequency
        freqs_h: Precomputed frequency bands for height dimension
        freqs_w: Precomputed frequency bands for width dimension
    """

    def __init__(
        self,
        dim: int,
        patch_size: int = 16,
        num_patches_per_side: int = 14,
        theta: float = 10000.0,
    ):
        super().__init__()

        # Validate that dim is divisible by 4 (need 2D positions, each with 2 components)
        if dim % 4 != 0:
            raise ValueError(f"Embedding dimension {dim} must be divisible by 4 for 2D RoPE")

        self.dim = dim
        self.theta = theta
        self.patch_size = patch_size
        self.num_patches_per_side = num_patches_per_side

        # For 2D RoPE, we split the dimension into two halves:
        # - First half encodes x (width) position
        # - Second half encodes y (height) position
        half_dim = dim // 2

        # Compute frequency bands: theta^(-2i/d) for i in [0, d/4)
        # Standard RoPE formula with clearer implementation
        # For 2D, we need half_dim // 2 frequencies per dimension
        num_freqs = half_dim // 2
        freq_bands = torch.arange(0, num_freqs, dtype=torch.float32)
        freq_bands = 1.0 / (theta ** (2.0 * freq_bands / half_dim))

        # Create 2D position grid for patches
        # Each patch has a (y, x) coordinate in [0, num_patches_per_side)
        y_pos = torch.arange(num_patches_per_side, dtype=torch.float32)
        x_pos = torch.arange(num_patches_per_side, dtype=torch.float32)

        # Create meshgrid for 2D positions
        # y_grid shape: [num_patches_per_side, num_patches_per_side]
        # x_grid shape: [num_patches_per_side, num_patches_per_side]
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing="ij")

        # Flatten to [num_patches, 1]
        y_grid = y_grid.flatten()[:, None]
        x_grid = x_grid.flatten()[:, None]

        # Compute position * frequency for each dimension
        # Shape: [num_patches, half_dim // 2]
        freqs_y = y_grid * freq_bands[None, :]
        freqs_x = x_grid * freq_bands[None, :]

        # Register as buffers (not parameters, but part of state_dict)
        self.register_buffer("freqs_h", freqs_y, persistent=False)
        self.register_buffer("freqs_w", freqs_x, persistent=False)

    def _compute_rope_rotation(self, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin components for RoPE rotation.

        Args:
            freqs: Frequency tensor [num_patches, num_freqs]

        Returns:
            Tuple of (cos, sin) tensors for rotation
        """
        # Create cos and sin rotation matrices
        # We interleave the rotation pairs for efficient application
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        return cos, sin

    def _apply_rope_rotation(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.

        RoPE rotation for a pair of dimensions (x1, x2):
        [x1']   [cos  -sin] [x1]
        [x2'] = [sin   cos] [x2]

        Args:
            x: Input tensor [batch, seq_len, dim]
            cos: Cosine component [seq_len, dim//2]
            sin: Sine component [seq_len, dim//2]

        Returns:
            Rotated tensor [batch, seq_len, dim]
        """
        # Split x into pairs (for rotation matrix application)
        # x shape: [batch, seq_len, dim]
        # We need to interleave dimensions as pairs
        x1 = x[..., 0::2]  # Even indices [batch, seq_len, dim//2]
        x2 = x[..., 1::2]  # Odd indices [batch, seq_len, dim//2]

        # Apply rotation
        # x1' = x1 * cos - x2 * sin
        # x2' = x1 * sin + x2 * cos
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos

        # Interleave back
        x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)
        x_rotated = x_rotated.flatten(-2)  # Merge last two dimensions

        return x_rotated

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        num_patches_h: Optional[int] = None,
        num_patches_w: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D RoPE to query and key tensors.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            num_patches_h: Optional height in patches (for dynamic resolution)
            num_patches_w: Optional width in patches (for dynamic resolution)

        Returns:
            Tuple of rotated (q, k) tensors with same shape as input
        """
        # Handle dynamic resolution by recomputing frequencies if needed
        if num_patches_h is not None and num_patches_w is not None:
            if (
                num_patches_h != self.num_patches_per_side
                or num_patches_w != self.num_patches_per_side
            ):
                freqs_h, freqs_w = self._compute_freqs_dynamic(num_patches_h, num_patches_w)
            else:
                freqs_h = cast(torch.Tensor, self.freqs_h)
                freqs_w = cast(torch.Tensor, self.freqs_w)
        else:
            freqs_h = cast(torch.Tensor, self.freqs_h)
            freqs_w = cast(torch.Tensor, self.freqs_w)

        # Compute rotation components
        cos_h, sin_h = self._compute_rope_rotation(freqs_h)
        cos_w, sin_w = self._compute_rope_rotation(freqs_w)

        # Combine height and width rotations
        # Concatenate along the dimension axis
        cos = torch.cat([cos_h, cos_w], dim=-1)  # [num_patches, head_dim]
        sin = torch.cat([sin_h, sin_w], dim=-1)

        # Reshape for broadcasting: [1, 1, num_patches, head_dim]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # Apply rotation to queries and keys
        # q, k shape: [batch, num_heads, seq_len, head_dim]
        q_rotated = self._apply_rope_rotation(q, cos, sin)
        k_rotated = self._apply_rope_rotation(k, cos, sin)

        return q_rotated, k_rotated

    def _compute_freqs_dynamic(
        self,
        num_patches_h: int,
        num_patches_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute frequencies for dynamic image resolution.

        This enables better generalization to different image sizes during inference.

        Args:
            num_patches_h: Number of patches in height
            num_patches_w: Number of patches in width

        Returns:
            Tuple of (freqs_h, freqs_w) for the new resolution
        """
        half_dim = self.dim // 2
        freqs_h_device = cast(torch.Tensor, self.freqs_h).device
        freqs_w_device = cast(torch.Tensor, self.freqs_w).device

        # Use same formula as initialization for consistency
        num_freqs = half_dim // 2
        freq_bands = torch.arange(0, num_freqs, dtype=torch.float32, device=freqs_h_device)
        freq_bands = 1.0 / (self.theta ** (2.0 * freq_bands / half_dim))

        # Create position grids for new resolution
        y_pos = torch.arange(num_patches_h, dtype=torch.float32, device=freqs_h_device)
        x_pos = torch.arange(num_patches_w, dtype=torch.float32, device=freqs_w_device)

        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing="ij")
        y_grid = y_grid.flatten()[:, None]
        x_grid = x_grid.flatten()[:, None]

        freqs_y = y_grid * freq_bands[None, :]
        freqs_x = x_grid * freq_bands[None, :]

        return freqs_y, freqs_x


class RoPEAttentionWrapper(nn.Module):
    """
    Wrapper around timm's attention module to inject RoPE.

    This wrapper intercepts the attention computation and applies RoPE to
    query and key embeddings before computing attention scores.

    Args:
        attn_module: Original attention module from timm
        rope_module: RoPE module to apply
        use_flash_attention: Whether to use Flash Attention (F.scaled_dot_product_attention)

    Attributes:
        attn: Original attention module
        rope: RoPE module for position encoding
        use_flash: Whether Flash Attention is enabled and available
    """

    def __init__(
        self,
        attn_module: nn.Module,
        rope_module: VisionRoPE2D,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.attn = attn_module
        self.rope = rope_module
        self.use_flash = use_flash_attention and FLASH_ATTENTION_AVAILABLE

        # Copy all attributes from original attention (typed as Any to handle dynamic attributes)
        self.num_heads: int = getattr(attn_module, "num_heads")
        self.head_dim: int = getattr(attn_module, "head_dim")
        self.scale: float = getattr(attn_module, "scale")
        self.qkv: nn.Module = getattr(attn_module, "qkv")
        self.q_norm: Optional[nn.Module] = getattr(attn_module, "q_norm", None)
        self.k_norm: Optional[nn.Module] = getattr(attn_module, "k_norm", None)
        self.attn_drop: nn.Module = getattr(attn_module, "attn_drop")
        self.proj: nn.Module = getattr(attn_module, "proj")
        self.proj_drop: nn.Module = getattr(attn_module, "proj_drop")

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with RoPE applied to Q and K.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attn_mask: Optional attention mask (currently ignored with Flash Attention)

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        B, N, C = x.shape

        # Compute Q, K, V using original module
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each is [batch, num_heads, seq_len, head_dim]

        # Apply Q and K normalization if present (used in some ViT variants)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # Calculate grid dimensions (excluding CLS token)
        num_patches = N - 1 if N > 1 else N  # Account for CLS token
        grid_size = int(math.sqrt(num_patches))

        # Apply RoPE to Q and K (skip CLS token if present)
        if N > 1:  # Has CLS token
            q_cls, q_patches = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patches = k[:, :, :1, :], k[:, :, 1:, :]

            # Apply RoPE only to patch tokens
            q_patches_rope, k_patches_rope = self.rope(
                q_patches, k_patches, num_patches_h=grid_size, num_patches_w=grid_size
            )

            # Concatenate CLS token back
            q = torch.cat([q_cls, q_patches_rope], dim=2)
            k = torch.cat([k_cls, k_patches_rope], dim=2)
        else:
            # No CLS token, apply RoPE to all tokens
            q, k = self.rope(q, k, num_patches_h=grid_size, num_patches_w=grid_size)

        # Compute attention using Flash Attention if available
        if self.use_flash:
            # Use PyTorch's Flash Attention implementation
            # scaled_dot_product_attention automatically selects the best backend
            dropout_p = float(getattr(self.attn_drop, "p", 0.0)) if self.training else 0.0
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                scale=self.scale,
            )
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            # Standard attention fallback
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # Apply attention to values
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Project output
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ContextEncoder(nn.Module):
    """
    Context encoder using Vision Transformer from timm.

    Processes the visible/context patches of the input image.

    Args:
        encoder_type: Model name from timm (e.g., 'vit_base_patch16_224')
        img_size: Input image size
        pretrained: Whether to load pretrained weights
        drop_path_rate: Stochastic depth rate
        use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency
        use_rope: Whether to use Rotary Position Embeddings (RoPE)
        rope_theta: Base frequency for RoPE (default: 10000.0)

    Attributes:
        vit: Vision Transformer model
        embed_dim: Embedding dimension
        num_patches: Number of patches per image
        patch_size: Size of each patch
        use_gradient_checkpointing: Flag for gradient checkpointing
        use_rope: Flag for using RoPE
        rope: RoPE module (if use_rope=True)
    """

    def __init__(
        self,
        encoder_type: str = "vit_base_patch16_224",
        img_size: int = 224,
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
        use_gradient_checkpointing: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        # Create Vision Transformer using timm
        self.vit: Any = timm.create_model(
            encoder_type,
            pretrained=pretrained,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )

        # Get model properties
        self.embed_dim: int = self.vit.embed_dim  # type: ignore[assignment]
        self.num_patches: int = self.vit.patch_embed.num_patches  # type: ignore[assignment, union-attr]
        self.patch_size: int = self.vit.patch_embed.patch_size[0]  # type: ignore[assignment, index, union-attr]

        # Calculate grid size
        self.grid_size: int = int(math.sqrt(self.num_patches))

        # Remove classification head (we don't need it for JEPA)
        if hasattr(self.vit, "head"):
            self.vit.head = nn.Identity()

        # Gradient checkpointing flag
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Flash Attention configuration
        self.use_flash_attention = use_flash_attention

        # RoPE configuration
        self.use_rope = use_rope
        if use_rope:
            # Get head dimension from first attention block
            num_heads: int = self.vit.blocks[0].attn.num_heads  # type: ignore[index, union-attr]
            head_dim = self.embed_dim // num_heads

            # Create RoPE module
            self.rope = VisionRoPE2D(
                dim=head_dim,
                patch_size=self.patch_size,
                num_patches_per_side=self.grid_size,
                theta=rope_theta,
            )

            # Wrap all attention layers with RoPE
            for block in self.vit.blocks:  # type: ignore[union-attr]
                block.attn = RoPEAttentionWrapper(
                    block.attn, self.rope, use_flash_attention=use_flash_attention
                )

            # When using RoPE, we can optionally reduce or remove absolute position embeddings
            # Here we keep them but they can be set to zero or removed
            # self.vit.pos_embed.data.zero_()  # Uncomment to disable absolute pos embeddings

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
        # Note: When using RoPE, absolute embeddings are less critical
        # but we keep them for backward compatibility and hybrid approaches
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        # Apply mask if provided (set masked patches to zero)
        if mask is not None:
            # mask shape: [B, N], we need [B, N+1, 1] to account for cls token
            # Determine dtype for zeros tensor
            zero_dtype = torch.bool if mask.dtype == torch.bool else mask.dtype
            mask_with_cls = torch.cat(
                [torch.zeros(mask.shape[0], 1, device=mask.device, dtype=zero_dtype), mask], dim=1
            ).unsqueeze(-1)
            # Convert boolean mask to float for multiplication
            if mask_with_cls.dtype == torch.bool:
                mask_with_cls = mask_with_cls.float()
            x = x * (1 - mask_with_cls)

        # Pass through transformer blocks with optional gradient checkpointing
        # Gradient checkpointing trades computation for memory by recomputing
        # intermediate activations during backward pass instead of storing them
        if self.use_gradient_checkpointing and self.training:
            for block in self.vit.blocks:
                # torch.utils.checkpoint requires use_reentrant=False for compatibility
                # with distributed training and certain edge cases
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
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
        use_rope: Whether to use Rotary Position Embeddings (RoPE)
        rope_theta: Base frequency for RoPE (default: 10000.0)

    Attributes:
        vit: Vision Transformer model (updated via EMA)
        momentum: Current EMA momentum value
        ema_momentum_end: Target momentum value
        ema_warmup_steps: Warmup steps for momentum schedule
        use_rope: Flag for using RoPE
        rope: RoPE module (if use_rope=True)
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
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        # Create Vision Transformer using timm
        self.vit: Any = timm.create_model(
            encoder_type,
            pretrained=pretrained,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )

        # Get model properties
        self.embed_dim: int = self.vit.embed_dim  # type: ignore[assignment]
        self.num_patches: int = self.vit.patch_embed.num_patches  # type: ignore[assignment, union-attr]
        self.patch_size: int = self.vit.patch_embed.patch_size[0]  # type: ignore[assignment, index, union-attr]

        # Calculate grid size
        self.grid_size: int = int(math.sqrt(self.num_patches))

        # Remove classification head
        if hasattr(self.vit, "head"):
            self.vit.head = nn.Identity()

        # EMA parameters
        self.momentum = ema_momentum
        self.ema_momentum_end = ema_momentum_end
        self.ema_warmup_steps = ema_warmup_steps

        # Flash Attention configuration
        self.use_flash_attention = use_flash_attention

        # RoPE configuration
        self.use_rope = use_rope
        if use_rope:
            # Get head dimension from first attention block
            num_heads: int = self.vit.blocks[0].attn.num_heads  # type: ignore[index, union-attr]
            head_dim = self.embed_dim // num_heads

            # Create RoPE module
            self.rope = VisionRoPE2D(
                dim=head_dim,
                patch_size=self.patch_size,
                num_patches_per_side=self.grid_size,
                theta=rope_theta,
            )

            # Wrap all attention layers with RoPE
            for block in self.vit.blocks:  # type: ignore[union-attr]
                block.attn = RoPEAttentionWrapper(
                    block.attn, self.rope, use_flash_attention=use_flash_attention
                )

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

        Implements linear schedule for momentum as per I-JEPA paper:
        tau(t) = tau_base + (tau_end - tau_base) * min(1.0, t / T)

        Args:
            context_encoder: Context encoder to copy weights from
            current_step: Current training step for momentum scheduling

        Returns:
            Current momentum value
        """
        # Calculate momentum with linear schedule
        progress = min(1.0, current_step / self.ema_warmup_steps)
        momentum = self.momentum + (self.ema_momentum_end - self.momentum) * progress

        # Update weights: θ_target = momentum * θ_target + (1 - momentum) * θ_context
        for param_target, param_context in zip(
            self.vit.parameters(), context_encoder.vit.parameters()
        ):
            param_target.data.mul_(momentum).add_(param_context.data, alpha=1 - momentum)

        return momentum

    @torch.no_grad()
    def copy_from_context_encoder(self, context_encoder: ContextEncoder) -> None:
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
    use_rope: bool = False,
    rope_theta: float = 10000.0,
    use_flash_attention: bool = False,
    use_layerscale: bool = False,
    layerscale_init: float = 1e-5,
) -> Tuple[ContextEncoder, TargetEncoder]:
    """
    Factory function to create context and target encoders.

    Args:
        encoder_type: Model name from timm
        img_size: Input image size
        pretrained: Whether to load pretrained weights
        drop_path_rate: Stochastic depth rate
        use_rope: Whether to use Rotary Position Embeddings
        rope_theta: Base frequency for RoPE rotation
        use_flash_attention: Whether to use Flash Attention (PyTorch 2.0+ scaled_dot_product_attention)
        use_layerscale: Whether to use LayerScale (TODO: not implemented yet)
        layerscale_init: Initial value for LayerScale (TODO: not implemented yet)

    Returns:
        Tuple of (context_encoder, target_encoder)

    Note:
        Flash Attention provides 2-5x speedup for attention computation using PyTorch's
        F.scaled_dot_product_attention, which automatically selects the best backend
        (Flash Attention, memory-efficient, or standard). Requires PyTorch 2.0+ and
        CUDA 7.5+ or MPS for best performance. Compatible with RoPE and all other features.
    """
    # TODO: LayerScale integration
    # LayerScale provides training stability for deep networks
    # Currently these parameters are accepted but not used
    # Implementation would require:
    # 1. Add LayerScale layers after attention and MLP in each block
    # 2. Initialize with small values (layerscale_init)

    # Warn user if LayerScale is requested but not implemented
    import warnings

    if use_layerscale:
        warnings.warn(
            "LayerScale is not yet implemented. The use_layerscale and layerscale_init "
            "parameters are accepted but currently ignored. The model will be created "
            "without LayerScale. To implement LayerScale, add learnable scale parameters "
            "after attention and MLP in each transformer block.",
            UserWarning,
            stacklevel=2,
        )

    context_encoder = ContextEncoder(
        encoder_type=encoder_type,
        img_size=img_size,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        use_rope=use_rope,
        rope_theta=rope_theta,
        use_flash_attention=use_flash_attention,
    )

    target_encoder = TargetEncoder(
        encoder_type=encoder_type,
        img_size=img_size,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        use_rope=use_rope,
        rope_theta=rope_theta,
        use_flash_attention=use_flash_attention,
    )

    # Initialize target encoder with context encoder weights
    target_encoder.copy_from_context_encoder(context_encoder)

    return context_encoder, target_encoder

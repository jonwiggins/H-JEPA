"""
Predictor network for H-JEPA.

This module implements a lightweight Vision Transformer that predicts target
representations from context features and masked token positions.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange, repeat


class PredictorBlock(nn.Module):
    """
    Transformer block for the predictor.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
        dropout: Dropout probability
        drop_path: Stochastic depth rate
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, N, D]

        Returns:
            Output tensor [B, N, D]
        """
        # Self-attention with residual connection
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_out)

        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample for residual blocks.

    Args:
        drop_prob: Probability of dropping path
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Predictor(nn.Module):
    """
    Predictor network for H-JEPA.

    Takes context features and predicts target representations for masked regions.
    Implemented as a lightweight Vision Transformer.

    Args:
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
        dropout: Dropout probability
        drop_path_rate: Maximum stochastic depth rate
        use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency

    Attributes:
        mask_token: Learnable token for masked positions
        blocks: Transformer blocks
        norm: Final layer normalization
        head: Prediction head
        use_gradient_checkpointing: Flag for gradient checkpointing
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional encoding for mask tokens
        self.pos_embed_predictor = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.pos_embed_predictor, std=0.02)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Predictor transformer blocks
        self.blocks = nn.ModuleList(
            [
                PredictorBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction head (projects to target embedding space)
        self.head = nn.Linear(embed_dim, embed_dim)

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

    def forward(
        self,
        context_features: torch.Tensor,
        mask_indices: torch.Tensor,
        pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            context_features: Encoded context features [B, N_context, D]
            mask_indices: Indices of masked positions [B, N_mask]
            pos_embed: Optional positional embeddings for all patches [B, N_total, D]

        Returns:
            Predicted representations for masked positions [B, N_mask, D]
        """
        B, N_context, D = context_features.shape
        N_mask = mask_indices.shape[1]

        # Create mask tokens for each masked position
        mask_tokens = repeat(self.mask_token, "1 1 d -> b n d", b=B, n=N_mask)

        # Add positional embeddings to mask tokens if provided
        if pos_embed is not None:
            # Gather positional embeddings for masked positions
            # mask_indices: [B, N_mask], pos_embed: [B, N_total, D]
            mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, D)
            mask_pos_embed = torch.gather(pos_embed, 1, mask_indices_expanded)
            mask_tokens = mask_tokens + mask_pos_embed  # type: ignore[assignment]
        else:
            # Use default positional embedding
            mask_tokens = mask_tokens + self.pos_embed_predictor  # type: ignore[assignment]

        # Concatenate context features with mask tokens
        # [B, N_context + N_mask, D]
        x = torch.cat([context_features, mask_tokens], dim=1)

        # Pass through transformer blocks with optional gradient checkpointing
        # Gradient checkpointing saves memory by recomputing activations during
        # the backward pass instead of storing them during the forward pass
        if self.use_gradient_checkpointing and self.training:
            for block in self.blocks:
                # Use non-reentrant checkpointing for better compatibility with
                # distributed training and edge cases
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)

        # Extract only the predicted mask tokens
        x = x[:, N_context:, :]

        # Apply final normalization and prediction head
        x = self.norm(x)
        x = self.head(x)

        return x  # type: ignore[no-any-return]

    def forward_with_full_sequence(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Alternative forward pass that takes full sequence with mask.

        Args:
            features: Full encoded features [B, N, D] (masked positions zeroed)
            mask: Binary mask [B, N] where 1 indicates masked position
            pos_embed: Optional positional embeddings [B, N, D]

        Returns:
            Predicted representations for masked positions [B, N_mask, D]
        """
        B, N, D = features.shape

        # Split into context and mask tokens
        mask_bool = mask.bool()

        # Get context features (non-masked)
        context_features = features[~mask_bool].view(B, -1, D)

        # Get indices of masked positions
        mask_indices = mask_bool.nonzero(as_tuple=True)[1].view(B, -1)

        # Use standard forward pass
        return self.forward(context_features, mask_indices, pos_embed)


def create_predictor(
    embed_dim: int,
    depth: int = 6,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    drop_path_rate: float = 0.0,
    use_gradient_checkpointing: bool = False,
) -> Predictor:
    """
    Factory function to create a predictor.

    Args:
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
        dropout: Dropout probability
        drop_path_rate: Maximum stochastic depth rate
        use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency

    Returns:
        Predictor model
    """
    return Predictor(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

"""
Action-conditioned latent predictor with AdaLN-Zero modulation.

Implements the LeWorldModel-style predictor: a transformer that takes a
sequence of frame embeddings plus per-step actions and autoregressively
predicts the next-frame embedding using AdaLN-Zero conditioning. The
zero-initialized modulation MLPs ensure that early in training the action
has no effect (the block is an identity), so action conditioning ramps up
smoothly as training progresses.

Reference: LeWorldModel (https://arxiv.org/abs/2603.19312)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaLNZeroBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero (DiT-style) action conditioning.

    Each block predicts six modulation vectors from the action embedding
    (scale + shift + gate, for both the attention and MLP sub-blocks). The
    modulation linear is zero-initialized so that at init the block is an
    identity in the action input — the predictor reduces to an unconditional
    transformer until training warms up the action pathway.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        action_embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Six per-block modulation vectors: scale1, shift1, gate1, scale2, shift2, gate2
        self.modulation = nn.Linear(action_embed_dim, 6 * embed_dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        action_emb: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: token embeddings [B, T, D]
            action_emb: action conditioning. Either [B, T, A] for per-step
                conditioning or [B, A] for sequence-level conditioning.
            attn_mask: optional [T, T] additive mask for causal attention.
        """
        if action_emb.ndim == 2:
            action_emb = action_emb.unsqueeze(1)  # [B, 1, A] broadcasts over T

        mods = self.modulation(action_emb)  # [B, T_or_1, 6D]
        scale1, shift1, gate1, scale2, shift2, gate2 = mods.chunk(6, dim=-1)

        # Attention sub-block
        h = self.norm1(x) * (1 + scale1) + shift1
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + gate1 * attn_out

        # MLP sub-block
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)

        return x


class ActionPredictor(nn.Module):
    """
    Latent next-state predictor with per-step action conditioning.

    Given a sequence of frame embeddings and the actions taken at each step,
    predicts the next-frame embedding at every position via causal attention.
    Designed for use inside an LeWM-style world model alongside an encoder.

    Args:
        embed_dim: Frame embedding dimension (must match encoder output).
        action_dim: Raw action vector dimension.
        depth: Number of AdaLN-Zero transformer blocks.
        num_heads: Number of attention heads per block.
        mlp_ratio: MLP hidden expansion ratio.
        dropout: Dropout probability inside attention and MLP sub-blocks.
        action_embed_dim: Hidden size of the action embedding MLP.
            Defaults to ``embed_dim`` if not set.
        max_seq_len: Maximum supported sequence length (for the learned
            positional embedding).
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        depth: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        action_embed_dim: int | None = None,
        max_seq_len: int = 256,
    ):
        super().__init__()

        if action_embed_dim is None:
            action_embed_dim = embed_dim

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.depth = depth
        self.max_seq_len = max_seq_len

        # Action embedding MLP. SiLU keeps non-saturating gradients near zero
        # which pairs well with the zero-init AdaLN modulation.
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.SiLU(),
            nn.Linear(action_embed_dim, action_embed_dim),
        )

        # Learned temporal positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                AdaLNZeroBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    action_embed_dim=action_embed_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)
        # Projection head matches the encoder projector convention so that
        # predictor outputs live in the same space as encoder outputs.
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, T, D] frame embeddings (z_1, ..., z_T).
            actions:    [B, T, A] actions (a_1, ..., a_T) where a_t is the
                        action that drives z_t to z_{t+1}.

        Returns:
            [B, T, D] predicted next-step embeddings. The output at index t
            is ẑ_{t+1}; train against ``embeddings[:, 1:]`` shifted by one
            and the predictor's ``[:, :-1]`` slice.
        """
        if embeddings.ndim != 3:
            raise ValueError(f"embeddings must be [B, T, D], got shape {embeddings.shape}")
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B, T, A], got shape {actions.shape}")
        if embeddings.shape[:2] != actions.shape[:2]:
            raise ValueError(
                f"Batch/time dims must match: embeddings {embeddings.shape[:2]} "
                f"vs actions {actions.shape[:2]}"
            )

        B, T, D = embeddings.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        x = embeddings + self.pos_embed[:, :T, :]
        action_emb = self.action_embed(actions)  # [B, T, action_embed_dim]

        # Additive causal mask: position t can only attend to positions ≤ t.
        # Build float mask filled with -inf above the diagonal.
        causal_mask = torch.zeros((T, T), device=x.device, dtype=x.dtype)
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )

        for block in self.blocks:
            x = block(x, action_emb, attn_mask=causal_mask)

        x = self.final_norm(x)
        x = self.head(x)
        return x  # type: ignore[no-any-return]


def create_action_predictor(
    embed_dim: int,
    action_dim: int,
    depth: int = 6,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    action_embed_dim: int | None = None,
    max_seq_len: int = 256,
) -> ActionPredictor:
    """Factory function for ActionPredictor."""
    return ActionPredictor(
        embed_dim=embed_dim,
        action_dim=action_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        action_embed_dim=action_embed_dim,
        max_seq_len=max_seq_len,
    )

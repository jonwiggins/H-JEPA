"""
LeWorldModel (LeWM) end-to-end world model.

Composes a frame encoder, an action-conditioned predictor, and a projector
into a single module trained end-to-end for next-latent prediction. Collapse
is prevented by SIGReg on the encoder outputs (configured via the loss);
no EMA target encoder, no stop-gradient.

Reference: https://arxiv.org/abs/2603.19312
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from .action_predictor import ActionPredictor, create_action_predictor
from .encoder import ContextEncoder
from .hjepa import BatchNorm1dForTokens, _make_projection_norm


class FrameEncoder(nn.Module):
    """
    Wraps ``ContextEncoder`` to produce a single [B, D] embedding per frame.

    LeWM uses the [CLS] token followed by a 1-layer MLP + BatchNorm projector.
    BatchNorm is preferred because LayerNorm strips the per-feature variance
    that SIGReg shapes — the same rationale that drives ``projection_norm``
    on ``HJEPA``.
    """

    def __init__(
        self,
        encoder_type: str = "vit_tiny_patch16_224",
        img_size: int = 224,
        embed_dim: int = 192,
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
        use_flash_attention: bool = False,
        use_layerscale: bool = False,
        projection_norm: Literal["layernorm", "batchnorm", "none"] = "batchnorm",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = ContextEncoder(
            encoder_type=encoder_type,
            img_size=img_size,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            use_flash_attention=use_flash_attention,
            use_layerscale=use_layerscale,
        )
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            _make_projection_norm(embed_dim, projection_norm),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W]

        Returns:
            [B, D] frame embedding from CLS token + projector.
        """
        features = self.encoder(images)  # [B, N+1, D]
        cls = features[:, 0, :]
        return self.projector(cls)  # type: ignore[no-any-return]


class LeWM(nn.Module):
    """
    LeWorldModel: end-to-end joint-embedding predictive world model.

    Args:
        frame_encoder: Module mapping ``[B, C, H, W] -> [B, D]``.
        action_predictor: Module mapping ``([B, T, D], [B, T, A]) -> [B, T, D]``
            of next-step predicted embeddings.
        embed_dim: Embedding dimension (must match encoder + predictor).
        action_dim: Action vector dimension.

    Forward returns a dict containing:
        - ``"embeddings"``: encoder outputs ``[B, T, D]`` (with gradients).
        - ``"predictions"``: predictor outputs ``[B, T, D]`` (with gradients).
        - ``"target_embeddings"``: a slice ``embeddings[:, 1:]`` representing
          the next-frame targets for use in a prediction loss.
        - ``"prediction_inputs"``: a slice ``predictions[:, :-1]`` representing
          the predicted embeddings to compare against the targets.

    The trainer composes the prediction MSE loss against ``target_embeddings``
    plus a SIGReg term on ``embeddings`` (flattened over batch and time).
    """

    def __init__(
        self,
        frame_encoder: nn.Module,
        action_predictor: nn.Module,
        embed_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.action_predictor = action_predictor
        self.embed_dim = embed_dim
        self.action_dim = action_dim

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode every frame in a [B, T, C, H, W] sequence to [B, T, D]."""
        if frames.ndim != 5:
            raise ValueError(f"frames must be [B, T, C, H, W], got shape {frames.shape}")
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
        embs = self.frame_encoder(flat)
        return embs.reshape(B, T, -1)  # type: ignore[no-any-return]

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B, T, A], got shape {actions.shape}")
        if frames.shape[:2] != actions.shape[:2]:
            raise ValueError(
                f"Batch/time dims must match: frames {frames.shape[:2]} "
                f"vs actions {actions.shape[:2]}"
            )

        embeddings = self.encode_sequence(frames)  # [B, T, D]
        predictions = self.action_predictor(embeddings, actions)  # [B, T, D]

        return {
            "embeddings": embeddings,
            "predictions": predictions,
            # Pair (prediction_inputs, target_embeddings) for the loss:
            # at index t we predict z_{t+1} from (z_0..z_t, a_0..a_t).
            "prediction_inputs": predictions[:, :-1, :],
            "target_embeddings": embeddings[:, 1:, :],
        }


def create_lewm(
    encoder_type: str = "vit_tiny_patch16_224",
    img_size: int = 224,
    embed_dim: int = 192,
    action_dim: int = 4,
    predictor_depth: int = 6,
    predictor_num_heads: int = 16,
    predictor_mlp_ratio: float = 4.0,
    predictor_dropout: float = 0.1,
    predictor_max_seq_len: int = 256,
    pretrained: bool = False,
    use_flash_attention: bool = False,
    use_layerscale: bool = False,
    projection_norm: Literal["layernorm", "batchnorm", "none"] = "batchnorm",
) -> LeWM:
    """Factory that builds a LeWM model with LeWorldModel-paper defaults."""
    encoder = FrameEncoder(
        encoder_type=encoder_type,
        img_size=img_size,
        embed_dim=embed_dim,
        pretrained=pretrained,
        use_flash_attention=use_flash_attention,
        use_layerscale=use_layerscale,
        projection_norm=projection_norm,
    )
    predictor: ActionPredictor = create_action_predictor(
        embed_dim=embed_dim,
        action_dim=action_dim,
        depth=predictor_depth,
        num_heads=predictor_num_heads,
        mlp_ratio=predictor_mlp_ratio,
        dropout=predictor_dropout,
        max_seq_len=predictor_max_seq_len,
    )
    return LeWM(
        frame_encoder=encoder,
        action_predictor=predictor,
        embed_dim=embed_dim,
        action_dim=action_dim,
    )


def create_lewm_from_config(config: dict[str, Any]) -> LeWM:
    """Build a LeWM model from a YAML-style config dict."""
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    return create_lewm(
        encoder_type=model_config.get("encoder_type", "vit_tiny_patch16_224"),
        img_size=data_config.get("image_size", 224),
        embed_dim=model_config.get("embed_dim", 192),
        action_dim=model_config.get("action_dim", 4),
        predictor_depth=model_config.get("predictor", {}).get("depth", 6),
        predictor_num_heads=model_config.get("predictor", {}).get("num_heads", 16),
        predictor_mlp_ratio=model_config.get("predictor", {}).get("mlp_ratio", 4.0),
        predictor_dropout=model_config.get("predictor", {}).get("dropout", 0.1),
        predictor_max_seq_len=model_config.get("predictor", {}).get("max_seq_len", 256),
        pretrained=model_config.get("pretrained", False),
        use_flash_attention=model_config.get("use_flash_attention", False),
        use_layerscale=model_config.get("use_layerscale", False),
        projection_norm=model_config.get("projection_norm", "batchnorm"),
    )


# Re-export for convenience.
__all__ = [
    "FrameEncoder",
    "LeWM",
    "BatchNorm1dForTokens",
    "create_lewm",
    "create_lewm_from_config",
]

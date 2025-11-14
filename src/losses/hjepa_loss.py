"""
H-JEPA Loss: Hierarchical Joint-Embedding Predictive Architecture Loss

This module implements the core loss function for H-JEPA training, which computes
the prediction loss between predicted and target representations across multiple
hierarchical levels.

Mathematical Formulation:
    L_HJEPA = Σ_h w_h * L(pred_h, target_h)

    where:
    - h: hierarchy level index
    - w_h: weight for hierarchy level h
    - L: base loss function (MSE, SmoothL1, or Huber)
    - pred_h: predicted representation at level h
    - target_h: target representation at level h

References:
    - I-JEPA: https://arxiv.org/abs/2301.08243
    - VICReg: https://arxiv.org/abs/2105.04906
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Optional, Literal


class HJEPALoss(nn.Module):
    """
    Hierarchical JEPA Loss for self-supervised learning.

    Computes prediction loss between predicted and target representations
    across multiple hierarchical levels with configurable weights.

    Args:
        loss_type: Type of base loss function. One of:
            - 'mse': Mean Squared Error (L2)
            - 'smoothl1': Smooth L1 Loss (Huber with β=1)
            - 'huber': Huber Loss with configurable delta
        hierarchy_weights: Weights for each hierarchy level. If a single float,
            same weight is used for all levels. If list, must match num_hierarchies.
        num_hierarchies: Number of hierarchical levels (default: 3)
        reduction: Reduction method ('mean', 'sum', or 'none')
        normalize_embeddings: Whether to L2-normalize embeddings before computing loss
        huber_delta: Delta parameter for Huber loss (only used if loss_type='huber')
        eps: Small constant for numerical stability

    Example:
        >>> loss_fn = HJEPALoss(
        ...     loss_type='smoothl1',
        ...     hierarchy_weights=[1.0, 0.5, 0.25],
        ...     num_hierarchies=3,
        ...     normalize_embeddings=True
        ... )
        >>> # predictions and targets are lists of tensors, one per hierarchy
        >>> predictions = [torch.randn(32, 196, 768) for _ in range(3)]
        >>> targets = [torch.randn(32, 196, 768) for _ in range(3)]
        >>> loss_dict = loss_fn(predictions, targets)
        >>> total_loss = loss_dict['loss']
    """

    def __init__(
        self,
        loss_type: Literal['mse', 'smoothl1', 'huber'] = 'smoothl1',
        hierarchy_weights: Union[float, List[float]] = 1.0,
        num_hierarchies: int = 3,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        normalize_embeddings: bool = True,
        huber_delta: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.num_hierarchies = num_hierarchies
        self.reduction = reduction
        self.normalize_embeddings = normalize_embeddings
        self.huber_delta = huber_delta
        self.eps = eps

        # Process hierarchy weights
        if isinstance(hierarchy_weights, (int, float)):
            self.hierarchy_weights = [float(hierarchy_weights)] * num_hierarchies
        else:
            assert len(hierarchy_weights) == num_hierarchies, (
                f"Length of hierarchy_weights ({len(hierarchy_weights)}) must match "
                f"num_hierarchies ({num_hierarchies})"
            )
            self.hierarchy_weights = list(hierarchy_weights)

        # Register as buffer so it moves to correct device
        self.register_buffer(
            '_hierarchy_weights',
            torch.tensor(self.hierarchy_weights, dtype=torch.float32)
        )

        # Validate loss type
        valid_loss_types = ['mse', 'smoothl1', 'huber']
        assert loss_type in valid_loss_types, (
            f"loss_type must be one of {valid_loss_types}, got {loss_type}"
        )

    def _compute_base_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute base loss between predictions and targets.

        Args:
            predictions: Predicted representations [B, N, D]
            targets: Target representations [B, N, D]

        Returns:
            Loss tensor with shape depending on reduction parameter
        """
        if self.loss_type == 'mse':
            # MSE: (pred - target)^2
            loss = F.mse_loss(predictions, targets, reduction=self.reduction)

        elif self.loss_type == 'smoothl1':
            # Smooth L1: Huber loss with β=1
            loss = F.smooth_l1_loss(predictions, targets, reduction=self.reduction)

        elif self.loss_type == 'huber':
            # Huber loss with configurable delta
            loss = F.huber_loss(
                predictions,
                targets,
                reduction=self.reduction,
                delta=self.huber_delta
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        L2 normalize embeddings along the feature dimension.

        Args:
            x: Input tensor [B, N, D]

        Returns:
            Normalized tensor [B, N, D]
        """
        # Normalize along the last dimension (feature dimension)
        return F.normalize(x, p=2, dim=-1, eps=self.eps)

    def forward(
        self,
        predictions: Union[List[torch.Tensor], torch.Tensor],
        targets: Union[List[torch.Tensor], torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical JEPA loss.

        Args:
            predictions: Predicted representations. Either:
                - List of tensors [B, N, D], one per hierarchy level
                - Single tensor [B, N, D] (treated as single hierarchy)
            targets: Target representations. Same format as predictions.
            masks: Optional binary masks [B, N] indicating which patches to include
                in loss computation. One per hierarchy level or None.

        Returns:
            Dictionary containing:
                - 'loss': Total weighted hierarchical loss
                - 'loss_h{i}': Loss at hierarchy level i (for each level)
                - 'loss_unweighted': Unweighted total loss (for logging)

        Raises:
            AssertionError: If input shapes or types are invalid
        """
        # Convert single tensor to list
        if isinstance(predictions, torch.Tensor):
            predictions = [predictions]
        if isinstance(targets, torch.Tensor):
            targets = [targets]

        # Validate inputs
        assert len(predictions) == len(targets), (
            f"Number of predictions ({len(predictions)}) must match "
            f"number of targets ({len(targets)})"
        )

        num_levels = len(predictions)
        assert num_levels == self.num_hierarchies, (
            f"Expected {self.num_hierarchies} hierarchy levels, "
            f"but got {num_levels}"
        )

        # Validate shapes
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            assert pred.shape == target.shape, (
                f"Prediction and target shapes must match at level {i}. "
                f"Got pred: {pred.shape}, target: {target.shape}"
            )
            assert pred.ndim == 3, (
                f"Predictions must be 3D tensors [B, N, D], "
                f"got shape {pred.shape} at level {i}"
            )

        # Process masks
        if masks is not None:
            if isinstance(masks, torch.Tensor):
                masks = [masks]
            assert len(masks) == num_levels, (
                f"Number of masks ({len(masks)}) must match "
                f"number of hierarchy levels ({num_levels})"
            )

        # Compute loss at each hierarchy level
        losses = []
        loss_dict = {}

        for i, (pred, target) in enumerate(zip(predictions, targets)):
            # Normalize embeddings if requested
            if self.normalize_embeddings:
                pred = self._normalize(pred)
                target = self._normalize(target)

            # Detach targets (stop gradient)
            target = target.detach()

            # Compute base loss
            if masks is not None and masks[i] is not None:
                # Apply mask: compute loss only on masked patches
                mask = masks[i]  # [B, N]
                assert mask.shape[:2] == pred.shape[:2], (
                    f"Mask shape {mask.shape} incompatible with "
                    f"prediction shape {pred.shape} at level {i}"
                )

                # Expand mask to match feature dimension
                mask = mask.unsqueeze(-1)  # [B, N, 1]

                # Masked loss computation
                if self.reduction == 'none':
                    base_loss = self._compute_base_loss(pred, target)
                    level_loss = (base_loss * mask).sum() / (mask.sum() + self.eps)
                else:
                    # Apply mask before loss computation
                    masked_pred = pred * mask
                    masked_target = target * mask
                    base_loss = self._compute_base_loss(masked_pred, masked_target)

                    # Normalize by number of masked elements
                    if self.reduction == 'mean':
                        level_loss = base_loss * pred.numel() / (mask.sum() + self.eps)
                    else:  # sum
                        level_loss = base_loss
            else:
                # No masking
                level_loss = self._compute_base_loss(pred, target)

            losses.append(level_loss)
            loss_dict[f'loss_h{i}'] = level_loss

        # Stack losses for vectorized weighting
        losses_tensor = torch.stack(losses)  # [num_hierarchies]

        # Compute weighted sum
        weighted_losses = losses_tensor * self._hierarchy_weights
        total_loss = weighted_losses.sum()

        # Add to loss dictionary
        loss_dict['loss'] = total_loss
        loss_dict['loss_unweighted'] = losses_tensor.mean()

        return loss_dict

    def extra_repr(self) -> str:
        """String representation for print/logging."""
        return (
            f"loss_type={self.loss_type}, "
            f"num_hierarchies={self.num_hierarchies}, "
            f"hierarchy_weights={self.hierarchy_weights}, "
            f"normalize={self.normalize_embeddings}, "
            f"reduction={self.reduction}"
        )

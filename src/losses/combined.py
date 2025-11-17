"""
Combined Loss: H-JEPA + VICReg

This module implements a flexible combined loss function that integrates the
hierarchical prediction loss from H-JEPA with VICReg regularization to prevent
representation collapse.

Mathematical Formulation:
    L_total = Σ_h [w_h * L_JEPA(pred_h, target_h) + α_h * L_VICReg(pred_h, target_h)]

    where:
    - h: hierarchy level index
    - w_h: JEPA loss weight for level h
    - α_h: VICReg loss weight for level h
    - L_JEPA: Hierarchical prediction loss
    - L_VICReg: VICReg regularization loss

This combination provides:
    1. Strong prediction signal from JEPA loss
    2. Collapse prevention from VICReg regularization
    3. Hierarchical multi-scale learning

References:
    - I-JEPA: https://arxiv.org/abs/2301.08243
    - VICReg: https://arxiv.org/abs/2105.04906
"""

from typing import Any, Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn

from .hjepa_loss import HJEPALoss
from .sigreg import SIGRegLoss
from .vicreg import VICRegLoss


class CombinedLoss(nn.Module):
    """
    Combined H-JEPA and VICReg Loss.

    Integrates hierarchical prediction loss with variance-invariance-covariance
    regularization for robust self-supervised learning.

    Args:
        # H-JEPA Loss arguments
        jepa_loss_type: Type of base JEPA loss ('mse', 'smoothl1', 'huber')
        jepa_hierarchy_weights: Weights for JEPA loss at each hierarchy level
        num_hierarchies: Number of hierarchical levels
        normalize_embeddings: Whether to normalize embeddings in JEPA loss

        # VICReg Loss arguments
        vicreg_weight: Global weight for VICReg loss (can be scalar or list per hierarchy)
        vicreg_invariance_weight: Weight for VICReg invariance term
        vicreg_variance_weight: Weight for VICReg variance term
        vicreg_covariance_weight: Weight for VICReg covariance term
        vicreg_variance_threshold: Target variance threshold

        # Combined loss options
        apply_vicreg_per_level: If True, apply VICReg at each hierarchy level separately
        vicreg_on_targets: If True, apply VICReg to target representations (for regularization)
        reduction: Reduction method for base losses
        eps: Numerical stability constant

    Example:
        >>> loss_fn = CombinedLoss(
        ...     jepa_loss_type='smoothl1',
        ...     jepa_hierarchy_weights=[1.0, 0.5, 0.25],
        ...     num_hierarchies=3,
        ...     vicreg_weight=0.1,
        ... )
        >>> predictions = [torch.randn(32, 196, 768) for _ in range(3)]
        >>> targets = [torch.randn(32, 196, 768) for _ in range(3)]
        >>> loss_dict = loss_fn(predictions, targets)
        >>> total_loss = loss_dict['loss']
    """

    def __init__(
        self,
        # H-JEPA parameters
        jepa_loss_type: Literal["mse", "smoothl1", "huber"] = "smoothl1",
        jepa_hierarchy_weights: Union[float, List[float]] = 1.0,
        num_hierarchies: int = 3,
        normalize_embeddings: bool = True,
        huber_delta: float = 1.0,
        # VICReg parameters
        vicreg_weight: Union[float, List[float]] = 0.1,
        vicreg_invariance_weight: float = 25.0,
        vicreg_variance_weight: float = 25.0,
        vicreg_covariance_weight: float = 1.0,
        vicreg_variance_threshold: float = 1.0,
        # Combined loss options
        apply_vicreg_per_level: bool = True,
        vicreg_on_targets: bool = False,
        reduction: Literal["mean", "sum", "none"] = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.num_hierarchies = num_hierarchies
        self.apply_vicreg_per_level = apply_vicreg_per_level
        self.vicreg_on_targets = vicreg_on_targets
        self.eps = eps

        # Initialize H-JEPA loss
        self.jepa_loss = HJEPALoss(
            loss_type=jepa_loss_type,
            hierarchy_weights=jepa_hierarchy_weights,
            num_hierarchies=num_hierarchies,
            reduction=reduction,
            normalize_embeddings=normalize_embeddings,
            huber_delta=huber_delta,
            eps=eps,
        )

        # Initialize VICReg loss
        self.vicreg_loss = VICRegLoss(
            invariance_weight=vicreg_invariance_weight,
            variance_weight=vicreg_variance_weight,
            covariance_weight=vicreg_covariance_weight,
            variance_threshold=vicreg_variance_threshold,
            eps=eps,
            flatten_patches=True,
        )

        # Process VICReg weights
        if isinstance(vicreg_weight, (int, float)):
            self.vicreg_weights = [float(vicreg_weight)] * num_hierarchies
        else:
            assert len(vicreg_weight) == num_hierarchies, (
                f"Length of vicreg_weight ({len(vicreg_weight)}) must match "
                f"num_hierarchies ({num_hierarchies})"
            )
            self.vicreg_weights = list(vicreg_weight)

        # Register as buffer
        self._vicreg_weights: torch.Tensor
        self.register_buffer(
            "_vicreg_weights", torch.tensor(self.vicreg_weights, dtype=torch.float32)
        )

    def forward(
        self,
        predictions: Union[List[torch.Tensor], torch.Tensor],
        targets: Union[List[torch.Tensor], torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined H-JEPA + VICReg loss.

        Args:
            predictions: Predicted representations. Either:
                - List of tensors [B, N, D], one per hierarchy level
                - Single tensor [B, N, D] (treated as single hierarchy)
            targets: Target representations. Same format as predictions.
            masks: Optional binary masks [B, N] for JEPA loss

        Returns:
            Dictionary containing:
                - 'loss': Total combined loss
                - 'jepa_loss': Total JEPA loss
                - 'vicreg_loss': Total VICReg loss
                - 'loss_h{i}': JEPA loss at level i
                - 'vicreg_h{i}': VICReg loss at level i
                - Plus all sub-components from individual losses
        """
        # Convert to lists if needed
        if isinstance(predictions, torch.Tensor):
            predictions = [predictions]
        if isinstance(targets, torch.Tensor):
            targets = [targets]

        # 1. Compute JEPA loss
        jepa_dict = self.jepa_loss(predictions, targets, masks)

        # 2. Compute VICReg loss
        loss_dict = {}

        if self.apply_vicreg_per_level:
            # Apply VICReg at each hierarchy level
            vicreg_losses = []

            for i, (pred, target) in enumerate(zip(predictions, targets)):
                # Choose which representations to regularize
                if self.vicreg_on_targets:
                    # Regularize target representations (encourage diversity)
                    # Split target into two views for VICReg
                    # This assumes targets come from different augmentations
                    vicreg_dict = self.vicreg_loss(target, pred)
                else:
                    # Regularize predicted representations
                    # Create pseudo-views by splitting batch
                    if pred.shape[0] >= 2:
                        # Split predictions into two views
                        mid = pred.shape[0] // 2
                        pred_a = pred[:mid]
                        pred_b = pred[mid : 2 * mid]
                        vicreg_dict = self.vicreg_loss(pred_a, pred_b)
                    else:
                        # Not enough samples, use prediction-target pair
                        vicreg_dict = self.vicreg_loss(pred, target)

                vicreg_loss = vicreg_dict["loss"]
                vicreg_losses.append(vicreg_loss)

                # Log per-level VICReg components
                loss_dict[f"vicreg_h{i}"] = vicreg_loss
                loss_dict[f"vicreg_invariance_h{i}"] = vicreg_dict["invariance_loss"]
                loss_dict[f"vicreg_variance_h{i}"] = vicreg_dict["variance_loss"]
                loss_dict[f"vicreg_covariance_h{i}"] = vicreg_dict["covariance_loss"]

            # Weighted sum of VICReg losses
            vicreg_losses_tensor = torch.stack(vicreg_losses)
            weighted_vicreg = (vicreg_losses_tensor * self._vicreg_weights).sum()
            loss_dict["vicreg_loss"] = weighted_vicreg

        else:
            # Apply VICReg only at the last (finest) hierarchy level
            last_pred = predictions[-1]
            last_target = targets[-1]

            if self.vicreg_on_targets:
                vicreg_dict = self.vicreg_loss(last_target, last_pred)
            else:
                if last_pred.shape[0] >= 2:
                    mid = last_pred.shape[0] // 2
                    pred_a = last_pred[:mid]
                    pred_b = last_pred[mid : 2 * mid]
                    vicreg_dict = self.vicreg_loss(pred_a, pred_b)
                else:
                    vicreg_dict = self.vicreg_loss(last_pred, last_target)

            weighted_vicreg = self.vicreg_weights[-1] * vicreg_dict["loss"]
            loss_dict["vicreg_loss"] = weighted_vicreg

            # Log VICReg components
            for key, value in vicreg_dict.items():
                loss_dict[f"vicreg_{key}"] = value

        # 3. Combine losses
        total_loss = jepa_dict["loss"] + weighted_vicreg

        # Add to loss dictionary
        loss_dict["loss"] = total_loss
        loss_dict["jepa_loss"] = jepa_dict["loss"]

        # Add all JEPA components
        for key, value in jepa_dict.items():
            if key not in loss_dict:
                loss_dict[key] = value

        return loss_dict

    def get_loss_summary(
        self,
        loss_dict: Dict[str, torch.Tensor],
    ) -> str:
        """
        Generate a formatted summary of all loss components.

        Args:
            loss_dict: Dictionary returned by forward()

        Returns:
            Formatted string with loss breakdown
        """
        lines = ["Loss Summary:"]
        lines.append(f"  Total Loss: {loss_dict['loss'].item():.6f}")
        lines.append(f"  JEPA Loss:  {loss_dict['jepa_loss'].item():.6f}")
        lines.append(f"  VICReg Loss: {loss_dict['vicreg_loss'].item():.6f}")

        # Per-level breakdown
        lines.append("\nPer-Level Breakdown:")
        for i in range(self.num_hierarchies):
            jepa_key = f"loss_h{i}"
            vicreg_key = f"vicreg_h{i}"

            jepa_val = loss_dict.get(jepa_key, torch.tensor(0.0)).item()
            vicreg_val = loss_dict.get(vicreg_key, torch.tensor(0.0)).item()

            lines.append(f"  Level {i}: JEPA={jepa_val:.6f}, VICReg={vicreg_val:.6f}")

        return "\n".join(lines)

    def extra_repr(self) -> str:
        """String representation for print/logging."""
        return (
            f"num_hierarchies={self.num_hierarchies}, "
            f"vicreg_weights={self.vicreg_weights}, "
            f"apply_vicreg_per_level={self.apply_vicreg_per_level}"
        )


class HierarchicalCombinedLoss(CombinedLoss):
    """
    Advanced Combined Loss with hierarchy-specific VICReg configurations.

    Allows different VICReg configurations at different hierarchy levels,
    useful for multi-scale learning where different levels may need different
    regularization strengths.

    Args:
        Same as CombinedLoss, plus:
        vicreg_configs: List of dicts, one per hierarchy level, with VICReg parameters
            Example: [
                {'invariance_weight': 25.0, 'variance_weight': 25.0},
                {'invariance_weight': 15.0, 'variance_weight': 15.0},
            ]
    """

    def __init__(
        self,
        jepa_loss_type: Literal["mse", "smoothl1", "huber"] = "smoothl1",
        jepa_hierarchy_weights: Union[float, List[float]] = 1.0,
        num_hierarchies: int = 3,
        normalize_embeddings: bool = True,
        vicreg_weight: Union[float, List[float]] = 0.1,
        vicreg_configs: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        # Initialize base class with default VICReg parameters
        super().__init__(
            jepa_loss_type=jepa_loss_type,
            jepa_hierarchy_weights=jepa_hierarchy_weights,
            num_hierarchies=num_hierarchies,
            normalize_embeddings=normalize_embeddings,
            vicreg_weight=vicreg_weight,
            **kwargs,
        )

        # Create separate VICReg losses for each hierarchy level if configs provided
        if vicreg_configs is not None:
            assert len(vicreg_configs) == num_hierarchies, (
                f"Number of vicreg_configs ({len(vicreg_configs)}) must match "
                f"num_hierarchies ({num_hierarchies})"
            )

            self.vicreg_losses = nn.ModuleList([VICRegLoss(**config) for config in vicreg_configs])
        else:
            # Use same VICReg loss for all levels
            self.vicreg_losses = nn.ModuleList([self.vicreg_loss for _ in range(num_hierarchies)])

    def forward(
        self,
        predictions: Union[List[torch.Tensor], torch.Tensor],
        targets: Union[List[torch.Tensor], torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with hierarchy-specific VICReg."""
        if isinstance(predictions, torch.Tensor):
            predictions = [predictions]
        if isinstance(targets, torch.Tensor):
            targets = [targets]

        # 1. Compute JEPA loss
        jepa_dict = self.jepa_loss(predictions, targets, masks)
        loss_dict = {}

        # 2. Compute VICReg loss with hierarchy-specific configurations
        vicreg_losses = []

        for i, (pred, target, vicreg_fn) in enumerate(
            zip(predictions, targets, self.vicreg_losses)
        ):
            # Apply hierarchy-specific VICReg
            if pred.shape[0] >= 2:
                mid = pred.shape[0] // 2
                pred_a = pred[:mid]
                pred_b = pred[mid : 2 * mid]
                vicreg_dict = vicreg_fn(pred_a, pred_b)
            else:
                vicreg_dict = vicreg_fn(pred, target)

            vicreg_loss = vicreg_dict["loss"]
            vicreg_losses.append(vicreg_loss)

            # Log components
            loss_dict[f"vicreg_h{i}"] = vicreg_loss
            for key, value in vicreg_dict.items():
                if key != "loss":
                    loss_dict[f"vicreg_{key}_h{i}"] = value

        # Weighted sum
        vicreg_losses_tensor = torch.stack(vicreg_losses)
        weighted_vicreg = (vicreg_losses_tensor * self._vicreg_weights).sum()

        # 3. Combine
        total_loss = jepa_dict["loss"] + weighted_vicreg

        loss_dict["loss"] = total_loss
        loss_dict["jepa_loss"] = jepa_dict["loss"]
        loss_dict["vicreg_loss"] = weighted_vicreg

        # Add JEPA components
        for key, value in jepa_dict.items():
            if key not in loss_dict:
                loss_dict[key] = value

        return loss_dict


def create_loss_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss from configuration dictionary.

    Args:
        config: Configuration dictionary with loss parameters

    Returns:
        Initialized loss module

    Example:
        >>> config = {
        ...     'type': 'combined',
        ...     'jepa_loss_type': 'smoothl1',
        ...     'hierarchy_weights': [1.0, 0.5, 0.25],
        ...     'num_hierarchies': 3,
        ...     'vicreg_weight': 0.1,
        ... }
        >>> loss_fn = create_loss_from_config(config)
        >>>
        >>> # For C-JEPA (Contrastive JEPA)
        >>> config = {
        ...     'type': 'cjepa',
        ...     'use_contrastive': True,
        ...     'contrastive_weight': 0.1,
        ...     'contrastive_temperature': 0.1,
        ... }
        >>> loss_fn = create_loss_from_config(config)
    """
    # Get loss parameters from config
    # Check both top-level and 'loss' section for compatibility
    loss_config = config.get("loss", config)
    model_config = config.get("model", {})

    loss_type = loss_config.get("type", "combined").lower()
    num_hierarchies = model_config.get("num_hierarchies", loss_config.get("num_hierarchies", 3))

    # Check if contrastive learning is enabled (for C-JEPA)
    use_contrastive = loss_config.get("use_contrastive", False)

    # Validate VICReg configuration - warn if VICReg fields are specified but won't be used
    if loss_type in ["hjepa", "jepa", "smoothl1", "mse"]:
        # Check for unused VICReg fields
        if "vicreg_weight" in loss_config or "use_vicreg" in loss_config or "vicreg" in loss_config:
            import warnings

            warnings.warn(
                f"VICReg fields found in config but loss type is '{loss_type}'. "
                f"VICReg regularization is only used with type='combined'. "
                f"The VICReg configuration will be ignored.",
                UserWarning,
            )

    if loss_type == "hjepa" or loss_type == "jepa" or loss_type == "smoothl1" or loss_type == "mse":
        # Determine jepa_loss_type with fallback logic
        if loss_type in ["smoothl1", "mse", "cosine"]:
            default_loss_type = loss_type
        else:
            default_loss_type = "smoothl1"

        jepa_loss = HJEPALoss(
            loss_type=loss_config.get("jepa_loss_type", default_loss_type),
            hierarchy_weights=loss_config.get("hierarchy_weights", 1.0),
            num_hierarchies=num_hierarchies,
            normalize_embeddings=loss_config.get("normalize_embeddings", True),
        )

        # Wrap with contrastive loss if enabled (C-JEPA)
        if use_contrastive:
            from .contrastive import ContrastiveJEPALoss

            return ContrastiveJEPALoss(
                jepa_loss=jepa_loss,
                jepa_weight=loss_config.get("jepa_weight", 1.0),
                contrastive_weight=loss_config.get("contrastive_weight", 0.1),
                contrastive_temperature=loss_config.get("contrastive_temperature", 0.1),
                use_cosine_similarity=loss_config.get("use_cosine_similarity", True),
                contrastive_on_context=loss_config.get("contrastive_on_context", False),
            )

        return jepa_loss

    elif loss_type == "vicreg":
        return VICRegLoss(
            invariance_weight=loss_config.get("vicreg_invariance_weight", 25.0),
            variance_weight=loss_config.get("vicreg_variance_weight", 25.0),
            covariance_weight=loss_config.get("vicreg_covariance_weight", 1.0),
            variance_threshold=loss_config.get("vicreg_variance_threshold", 1.0),
        )

    elif loss_type == "sigreg":
        return SIGRegLoss(
            num_slices=loss_config.get("sigreg_num_slices", 1024),
            num_test_points=loss_config.get("sigreg_num_test_points", 17),
            invariance_weight=loss_config.get("sigreg_invariance_weight", 25.0),
            sigreg_weight=loss_config.get("sigreg_weight", 25.0),
            eps=loss_config.get("eps", 1e-6),
            flatten_patches=loss_config.get("flatten_patches", True),
            fixed_slices=loss_config.get("sigreg_fixed_slices", False),
        )

    elif loss_type == "combined":
        return CombinedLoss(
            jepa_loss_type=loss_config.get("jepa_loss_type", "smoothl1"),
            jepa_hierarchy_weights=loss_config.get("hierarchy_weights", 1.0),
            num_hierarchies=num_hierarchies,
            normalize_embeddings=loss_config.get("normalize_embeddings", True),
            vicreg_weight=loss_config.get("vicreg_weight", 0.1),
            vicreg_invariance_weight=loss_config.get("vicreg_invariance_weight", 25.0),
            vicreg_variance_weight=loss_config.get("vicreg_variance_weight", 25.0),
            vicreg_covariance_weight=loss_config.get("vicreg_covariance_weight", 1.0),
        )

    elif loss_type == "hierarchical_combined":
        return HierarchicalCombinedLoss(
            jepa_loss_type=config.get("jepa_loss_type", "smoothl1"),
            jepa_hierarchy_weights=config.get("hierarchy_weights", 1.0),
            num_hierarchies=config.get("num_hierarchies", 3),
            vicreg_weight=config.get("vicreg_weight", 0.1),
            vicreg_configs=config.get("vicreg_configs", None),
        )

    elif loss_type == "cjepa" or loss_type == "contrastive_jepa":
        # C-JEPA: Contrastive JEPA hybrid
        from .contrastive import ContrastiveJEPALoss

        # Create base JEPA loss
        jepa_loss = HJEPALoss(
            loss_type=loss_config.get("jepa_loss_type", "smoothl1"),
            hierarchy_weights=loss_config.get("hierarchy_weights", 1.0),
            num_hierarchies=num_hierarchies,
            normalize_embeddings=loss_config.get("normalize_embeddings", True),
        )

        # Wrap with contrastive loss
        return ContrastiveJEPALoss(
            jepa_loss=jepa_loss,
            jepa_weight=loss_config.get("jepa_weight", 1.0),
            contrastive_weight=loss_config.get("contrastive_weight", 0.1),
            contrastive_temperature=loss_config.get("contrastive_temperature", 0.1),
            use_cosine_similarity=loss_config.get("use_cosine_similarity", True),
            contrastive_on_context=loss_config.get("contrastive_on_context", False),
        )

    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Must be one of ['hjepa', 'vicreg', 'sigreg', 'combined', 'hierarchical_combined', 'cjepa', 'mse', 'smoothl1']"
        )

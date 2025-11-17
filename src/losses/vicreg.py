"""
VICReg Loss: Variance-Invariance-Covariance Regularization

This module implements the VICReg loss function for self-supervised learning,
which prevents representation collapse through three complementary regularization terms.

Mathematical Formulation:
    L_VICReg = λ * L_inv + μ * L_var + ν * L_cov

    where:
    L_inv (Invariance) = MSE(Z_a, Z_b)
        Encourages consistency between different views/representations

    L_var (Variance) = Σ_d max(0, γ - sqrt(Var(Z_d) + ε))
        Maintains variance above threshold γ for each dimension d

    L_cov (Covariance) = (1/D) * Σ_{i≠j} Cov(Z_i, Z_j)^2
        Decorrelates different dimensions to prevent redundancy

References:
    VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning
    https://arxiv.org/abs/2105.04906
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """
    VICReg (Variance-Invariance-Covariance Regularization) Loss.

    Prevents representation collapse in self-supervised learning through three terms:
    1. Invariance: Consistency between representations
    2. Variance: Maintains sufficient variance in each dimension
    3. Covariance: Decorrelates different dimensions

    Args:
        invariance_weight: Weight λ for invariance term (default: 25.0)
        variance_weight: Weight μ for variance term (default: 25.0)
        covariance_weight: Weight ν for covariance term (default: 1.0)
        variance_threshold: Target variance γ (default: 1.0)
        eps: Small constant for numerical stability (default: 1e-4)
        flatten_patches: If True, flattens batch and patch dimensions before computing
            variance/covariance (useful for patch-based models like ViT)

    Example:
        >>> loss_fn = VICRegLoss(
        ...     invariance_weight=25.0,
        ...     variance_weight=25.0,
        ...     covariance_weight=1.0
        ... )
        >>> # Two views of the same data
        >>> z_a = torch.randn(32, 196, 768)  # [B, N, D]
        >>> z_b = torch.randn(32, 196, 768)
        >>> loss_dict = loss_fn(z_a, z_b)
        >>> total_loss = loss_dict['loss']
    """

    def __init__(
        self,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        variance_threshold: float = 1.0,
        eps: float = 1e-4,
        flatten_patches: bool = True,
    ) -> None:
        super().__init__()

        self.invariance_weight: Union[float, nn.Parameter] = invariance_weight
        self.variance_weight: Union[float, nn.Parameter] = variance_weight
        self.covariance_weight: Union[float, nn.Parameter] = covariance_weight
        self.variance_threshold = variance_threshold
        self.eps = eps
        self.flatten_patches = flatten_patches

        # Validate weights
        assert invariance_weight >= 0, f"invariance_weight must be >= 0, got {invariance_weight}"
        assert variance_weight >= 0, f"variance_weight must be >= 0, got {variance_weight}"
        assert covariance_weight >= 0, f"covariance_weight must be >= 0, got {covariance_weight}"
        assert variance_threshold > 0, f"variance_threshold must be > 0, got {variance_threshold}"

    def _invariance_loss(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute invariance loss (MSE between representations).

        Args:
            z_a: First representation [B, N, D] or [B, D]
            z_b: Second representation [B, N, D] or [B, D]

        Returns:
            Scalar invariance loss
        """
        return F.mse_loss(z_a, z_b)

    def _variance_loss(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute variance loss to maintain variance above threshold.

        For each dimension d, penalizes if std(z_d) < γ:
            loss_d = max(0, γ - sqrt(Var(z_d) + ε))

        Args:
            z: Representations [N, D] where N is batch (or batch*patches)

        Returns:
            Scalar variance loss
        """
        # Compute standard deviation for each dimension
        std = torch.sqrt(z.var(dim=0) + self.eps)  # [D]

        # Hinge loss: penalize if std < threshold
        variance_loss = torch.mean(F.relu(self.variance_threshold - std))

        return variance_loss

    def _covariance_loss(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute covariance loss to decorrelate dimensions.

        Penalizes off-diagonal elements of the covariance matrix:
            loss = (1/D) * Σ_{i≠j} Cov(z_i, z_j)^2

        Args:
            z: Representations [N, D] where N is batch (or batch*patches)

        Returns:
            Scalar covariance loss
        """
        N, D = z.shape

        # Center the features (zero mean)
        z = z - z.mean(dim=0, keepdim=True)  # [N, D]

        # Compute covariance matrix: Cov = (1/N) * Z^T @ Z
        cov = (z.T @ z) / (N - 1)  # [D, D]

        # Extract off-diagonal elements and compute their squared sum
        # We want to minimize off-diagonal covariances
        off_diagonal_mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
        off_diagonal_cov = cov[off_diagonal_mask]

        # Mean of squared off-diagonal elements
        covariance_loss: torch.Tensor = (off_diagonal_cov**2).mean()

        return covariance_loss

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VICReg loss.

        Args:
            z_a: First representation. Shape [B, D] or [B, N, D]
                If z_b is None, z_a is split along batch dimension into two views
            z_b: Optional second representation. Same shape as z_a.
                If None, z_a is assumed to contain both views concatenated

        Returns:
            Dictionary containing:
                - 'loss': Total weighted VICReg loss
                - 'invariance_loss': Invariance term
                - 'variance_loss': Variance term
                - 'covariance_loss': Covariance term
                - 'variance_loss_a': Variance loss for first view
                - 'variance_loss_b': Variance loss for second view

        Raises:
            AssertionError: If input shapes are invalid
        """
        # Handle single input case (both views concatenated)
        if z_b is None:
            assert z_a.shape[0] % 2 == 0, (
                "If only one input is provided, batch size must be even "
                "(first half = view A, second half = view B)"
            )
            batch_size = z_a.shape[0] // 2
            z_a, z_b = z_a[:batch_size], z_a[batch_size:]

        # Validate input shapes
        assert z_a.shape == z_b.shape, (
            f"z_a and z_b must have the same shape. " f"Got z_a: {z_a.shape}, z_b: {z_b.shape}"
        )
        assert z_a.ndim in [
            2,
            3,
        ], f"Inputs must be 2D [B, D] or 3D [B, N, D], got shape {z_a.shape}"

        # Flatten patch dimension if needed
        if z_a.ndim == 3 and self.flatten_patches:
            # [B, N, D] -> [B*N, D]
            B, N, D = z_a.shape
            z_a_flat = z_a.reshape(B * N, D)
            z_b_flat = z_b.reshape(B * N, D)
        else:
            z_a_flat = z_a.reshape(z_a.shape[0], -1)  # [B, D]
            z_b_flat = z_b.reshape(z_b.shape[0], -1)

        # 1. Invariance Loss: MSE between the two views
        inv_loss = self._invariance_loss(z_a_flat, z_b_flat)

        # 2. Variance Loss: Maintain variance for both views
        var_loss_a = self._variance_loss(z_a_flat)
        var_loss_b = self._variance_loss(z_b_flat)
        var_loss = (var_loss_a + var_loss_b) / 2

        # 3. Covariance Loss: Decorrelate dimensions for both views
        cov_loss_a = self._covariance_loss(z_a_flat)
        cov_loss_b = self._covariance_loss(z_b_flat)
        cov_loss = (cov_loss_a + cov_loss_b) / 2

        # Compute weighted total loss
        total_loss = (
            self.invariance_weight * inv_loss
            + self.variance_weight * var_loss
            + self.covariance_weight * cov_loss
        )

        # Return detailed loss dictionary
        loss_dict = {
            "loss": total_loss,
            "invariance_loss": inv_loss,
            "variance_loss": var_loss,
            "covariance_loss": cov_loss,
            "variance_loss_a": var_loss_a,
            "variance_loss_b": var_loss_b,
            "covariance_loss_a": cov_loss_a,
            "covariance_loss_b": cov_loss_b,
        }

        return loss_dict

    def extra_repr(self) -> str:
        """String representation for print/logging."""
        return (
            f"inv_weight={self.invariance_weight}, "
            f"var_weight={self.variance_weight}, "
            f"cov_weight={self.covariance_weight}, "
            f"var_threshold={self.variance_threshold}, "
            f"flatten_patches={self.flatten_patches}"
        )


class AdaptiveVICRegLoss(VICRegLoss):
    """
    Adaptive VICReg Loss with learnable or scheduled weights.

    Extends VICReg to support dynamic weight adjustment during training,
    which can help balance the three loss terms more effectively.

    Args:
        Same as VICRegLoss, plus:
        adaptive_weights: If True, weights become learnable parameters
        weight_momentum: EMA momentum for adaptive weight updates (0.9-0.999)
    """

    def __init__(
        self,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        variance_threshold: float = 1.0,
        eps: float = 1e-4,
        flatten_patches: bool = True,
        adaptive_weights: bool = False,
        weight_momentum: float = 0.99,
    ) -> None:
        super().__init__(
            invariance_weight=invariance_weight,
            variance_weight=variance_weight,
            covariance_weight=covariance_weight,
            variance_threshold=variance_threshold,
            eps=eps,
            flatten_patches=flatten_patches,
        )

        self.adaptive_weights = adaptive_weights
        self.weight_momentum = weight_momentum

        if adaptive_weights:
            # Convert weights to learnable parameters
            self.invariance_weight = nn.Parameter(
                torch.tensor(invariance_weight, dtype=torch.float32)
            )
            self.variance_weight = nn.Parameter(torch.tensor(variance_weight, dtype=torch.float32))
            self.covariance_weight = nn.Parameter(
                torch.tensor(covariance_weight, dtype=torch.float32)
            )

    def update_weights(
        self,
        loss_dict: Dict[str, torch.Tensor],
    ) -> None:
        """
        Update weights based on loss magnitudes (EMA).

        This can help balance the three terms automatically.

        Args:
            loss_dict: Dictionary containing individual loss terms
        """
        if not self.adaptive_weights:
            return

        with torch.no_grad():
            # Get current loss magnitudes
            inv = loss_dict["invariance_loss"].item()
            var = loss_dict["variance_loss"].item()
            cov = loss_dict["covariance_loss"].item()

            # Compute target weights (inverse of loss magnitude)
            total = inv + var + cov + self.eps
            target_inv = (1.0 - inv / total) * 30
            target_var = (1.0 - var / total) * 30
            target_cov = (1.0 - cov / total) * 2

            # EMA update (only for Parameters, not floats)
            if isinstance(self.invariance_weight, nn.Parameter):
                self.invariance_weight.data = (
                    self.weight_momentum * self.invariance_weight.data
                    + (1 - self.weight_momentum) * target_inv
                )
            if isinstance(self.variance_weight, nn.Parameter):
                self.variance_weight.data = (
                    self.weight_momentum * self.variance_weight.data
                    + (1 - self.weight_momentum) * target_var
                )
            if isinstance(self.covariance_weight, nn.Parameter):
                self.covariance_weight.data = (
                    self.weight_momentum * self.covariance_weight.data
                    + (1 - self.weight_momentum) * target_cov
                )

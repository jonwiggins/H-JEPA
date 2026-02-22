"""
SIGReg Loss: Sketched Isotropic Gaussian Regularization

This module implements SIGReg from the LeJEPA paper, which provides improved
training stability over standard VICReg through a theoretically grounded approach
to preventing representation collapse.

Mathematical Formulation:
    SIGReg uses random projections (slicing) to test if embeddings follow an
    isotropic Gaussian distribution. For M random directions a_i ∈ S^{K-1}:

    L_SIGReg = (1/M) * Σ_{i=1}^M T({a_i^T z_n}_{n=1}^N)

    where T is a univariate statistical test (Epps-Pulley) that measures
    the distance from a standard 1D Gaussian distribution.

Key Differences from VICReg:
    1. Computational Efficiency: O(K) complexity vs O(K^2) for covariance
    2. Single Hyperparameter: num_slices vs 3 separate weights in VICReg
    3. Theoretical Foundation: Based on Cramér-Wold theorem and optimal
       isotropic Gaussian distribution
    4. Improved Stability: Sign consistency and better variance handling
    5. Scalability: Linear memory/time complexity in dimension and samples

Epps-Pulley Test:
    The Epps-Pulley test is a smooth, differentiable test for comparing
    distributions. For projected samples y = a^T z:

    EP(y) = (1/N^2) * Σ_{i,j} ψ(y_i - y_j) - 2/(N*K) * Σ_{i,k} ψ(y_i - g_k)
            + (1/K^2) * Σ_{k,l} ψ(g_k - g_l)

    where ψ is a smooth kernel function and g_k are reference Gaussian samples.

References:
    LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics
    https://arxiv.org/abs/2511.08544

    VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning
    https://arxiv.org/abs/2105.04906
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EppsPulleyTest(nn.Module):
    """
    Epps-Pulley statistical test for measuring distance from standard Gaussian.

    This is a smooth, differentiable test that compares the distribution of
    1D projections to a standard Gaussian distribution. It uses characteristic
    functions and is suitable for gradient-based optimization.

    Args:
        num_points: Number of reference Gaussian points to use (default: 17)
            Higher values give more accurate tests but increase computation.
            LeJEPA paper uses 17 as a good balance.
        eps: Small constant for numerical stability (default: 1e-6)

    Note:
        The test statistic is minimized when the input distribution matches
        a standard Gaussian (mean=0, std=1).
    """

    def __init__(
        self,
        num_points: int = 17,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_points = num_points
        self.eps = eps

        # Pre-compute reference Gaussian samples (evenly spaced quantiles)
        # These serve as reference points for the standard Gaussian distribution
        quantiles = torch.linspace(0.01, 0.99, num_points)
        # Convert quantiles to Gaussian samples using inverse CDF
        # For standard normal: mean=0, std=1
        reference_points = torch.erfinv(2 * quantiles - 1) * math.sqrt(2)
        self.reference_points: torch.Tensor
        self.register_buffer("reference_points", reference_points)

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Smooth kernel function ψ for Epps-Pulley test.

        Uses a characteristic function based kernel:
        ψ(x, y) = exp(-0.5 * (x - y)^2)

        This is a Gaussian kernel that is smooth and differentiable.

        Args:
            x: First set of values [..., N]
            y: Second set of values [..., M]

        Returns:
            Kernel values [..., N, M]
        """
        # Compute pairwise differences
        diff = x.unsqueeze(-1) - y.unsqueeze(-2)  # [..., N, M]
        # Apply Gaussian kernel
        return torch.exp(-0.5 * diff**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Epps-Pulley test statistic for 1D samples.

        The test statistic measures how far the distribution of x is from
        a standard Gaussian N(0, 1). Lower values indicate closer match.

        Args:
            x: 1D projected samples [N] or batched [B, N]

        Returns:
            Scalar test statistic (or [B] if batched)
        """
        # Handle batched or single input
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, N]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, N = x.shape
        K = self.num_points

        # Standardize input to have mean=0, std=1 for fair comparison
        # This ensures we're comparing shape, not just location/scale
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True) + self.eps
        x_standardized = (x - x_mean) / x_std  # [B, N]

        # Get reference Gaussian points
        g = self.reference_points.unsqueeze(0).expand(batch_size, -1)  # [B, K]

        # Compute three terms of Epps-Pulley statistic:

        # 1. Self-interaction of empirical samples: (1/N^2) * Σ_{i,j} ψ(x_i - x_j)
        kernel_xx = self._kernel(x_standardized, x_standardized)  # [B, N, N]
        term1 = kernel_xx.sum(dim=(-2, -1)) / (N**2)  # [B]

        # 2. Cross-interaction with reference: -2/(N*K) * Σ_{i,k} ψ(x_i - g_k)
        kernel_xg = self._kernel(x_standardized, g)  # [B, N, K]
        term2 = -2 * kernel_xg.sum(dim=(-2, -1)) / (N * K)  # [B]

        # 3. Self-interaction of reference: (1/K^2) * Σ_{k,l} ψ(g_k - g_l)
        kernel_gg = self._kernel(g, g)  # [B, K, K]
        term3 = kernel_gg.sum(dim=(-2, -1)) / (K**2)  # [B]

        # Combine terms
        test_statistic = term1 + term2 + term3  # [B]

        # Return scalar if input was 1D
        if squeeze_output:
            return test_statistic.squeeze(0)

        return test_statistic


class SIGRegLoss(nn.Module):
    """
    SIGReg (Sketched Isotropic Gaussian Regularization) Loss.

    Improved regularization for self-supervised learning that prevents
    representation collapse through random slicing and statistical testing.

    Benefits over VICReg:
        - Single hyperparameter (num_slices) vs 3 weights
        - O(K) complexity vs O(K^2) for covariance computation
        - Theoretically grounded in optimal Gaussian distribution
        - Better training stability and scalability
        - Sign consistency through statistical testing

    Args:
        num_slices: Number of random 1D projections (default: 1024)
            Higher values give more thorough testing but increase computation.
            LeJEPA paper uses 1024 as default.
        num_test_points: Number of reference Gaussian points (default: 17)
        invariance_weight: Weight for invariance (MSE) term (default: 25.0)
        sigreg_weight: Weight for SIGReg regularization term (default: 25.0)
        eps: Small constant for numerical stability (default: 1e-6)
        flatten_patches: If True, flattens batch and patch dimensions (default: True)
        fixed_slices: If True, uses fixed random slices (default: False)
            Setting to True can improve reproducibility and reduce variance.

    Example:
        >>> loss_fn = SIGRegLoss(
        ...     num_slices=1024,
        ...     invariance_weight=25.0,
        ...     sigreg_weight=25.0
        ... )
        >>> z_a = torch.randn(32, 196, 768)  # [B, N, D]
        >>> z_b = torch.randn(32, 196, 768)
        >>> loss_dict = loss_fn(z_a, z_b)
        >>> total_loss = loss_dict['loss']
    """

    def __init__(
        self,
        num_slices: int = 1024,
        num_test_points: int = 17,
        invariance_weight: float = 25.0,
        sigreg_weight: float = 25.0,
        eps: float = 1e-6,
        flatten_patches: bool = True,
        fixed_slices: bool = False,
    ):
        super().__init__()

        self.num_slices = num_slices
        self.num_test_points = num_test_points
        self.invariance_weight = invariance_weight
        self.sigreg_weight = sigreg_weight
        self.eps = eps
        self.flatten_patches = flatten_patches
        self.fixed_slices = fixed_slices

        # Validate parameters
        assert num_slices > 0, f"num_slices must be > 0, got {num_slices}"
        assert num_test_points > 0, f"num_test_points must be > 0, got {num_test_points}"
        assert invariance_weight >= 0, f"invariance_weight must be >= 0, got {invariance_weight}"
        assert sigreg_weight >= 0, f"sigreg_weight must be >= 0, got {sigreg_weight}"

        # Initialize Epps-Pulley test
        self.univariate_test = EppsPulleyTest(
            num_points=num_test_points,
            eps=eps,
        )

        # Pre-allocated fixed random slices if requested
        self._fixed_random_slices: torch.Tensor | None = None

    def _generate_random_slices(
        self,
        embedding_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate random unit vectors for slicing.

        Samples M random directions uniformly from the unit sphere S^{K-1}.

        Args:
            embedding_dim: Dimension K of embeddings
            device: Device to create tensors on

        Returns:
            Random unit vectors [M, K]
        """
        if self.fixed_slices and self._fixed_random_slices is not None:
            # Use cached fixed slices
            if self._fixed_random_slices.shape == (self.num_slices, embedding_dim):
                return self._fixed_random_slices.to(device)

        # Sample from standard Gaussian
        slices = torch.randn(
            self.num_slices,
            embedding_dim,
            device=device,
        )

        # Normalize to unit vectors (project onto unit sphere)
        slices = F.normalize(slices, p=2, dim=1)  # [M, K]

        # Cache if using fixed slices
        if self.fixed_slices:
            self._fixed_random_slices = slices.detach().cpu()

        return slices

    def _invariance_loss(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute invariance loss (MSE between representations).

        This encourages consistency between different views of the same input.

        Args:
            z_a: First representation [N, D]
            z_b: Second representation [N, D]

        Returns:
            Scalar invariance loss
        """
        return F.mse_loss(z_a, z_b)

    def _sigreg_loss(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SIGReg loss using random slicing and Epps-Pulley test.

        Projects embeddings onto M random 1D directions and tests if each
        projection follows a standard Gaussian distribution.

        Mathematical formulation:
            L_SIGReg = (1/M) * Σ_{i=1}^M EP({a_i^T z_n}_{n=1}^N)

        Args:
            z: Representations [N, D] where N is batch (or batch*patches)

        Returns:
            Scalar SIGReg loss
        """
        N, D = z.shape

        # Generate random slicing directions
        random_slices = self._generate_random_slices(D, z.device)  # [M, D]

        # Project embeddings onto random directions: y_i = a_i^T @ z
        # z: [N, D], random_slices: [M, D]
        # Result: [N, M] where each column is a 1D projection
        projections = z @ random_slices.T  # [N, M]

        # Compute Epps-Pulley test for each projection
        # We want to minimize the test statistic (distance from Gaussian)
        test_statistics = []

        for i in range(self.num_slices):
            projection = projections[:, i]  # [N]
            stat = self.univariate_test(projection)
            test_statistics.append(stat)

        # Average over all slices
        sigreg_loss = torch.stack(test_statistics).mean()

        return sigreg_loss

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute SIGReg loss.

        Args:
            z_a: First representation. Shape [B, D] or [B, N, D]
                If z_b is None, z_a is split along batch dimension into two views
            z_b: Optional second representation. Same shape as z_a.
                If None, z_a is assumed to contain both views concatenated

        Returns:
            Dictionary containing:
                - 'loss': Total weighted SIGReg loss
                - 'invariance_loss': Invariance (MSE) term
                - 'sigreg_loss': SIGReg regularization term
                - 'sigreg_loss_a': SIGReg loss for first view
                - 'sigreg_loss_b': SIGReg loss for second view

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

        # 2. SIGReg Loss: Test if embeddings follow isotropic Gaussian
        sigreg_loss_a = self._sigreg_loss(z_a_flat)
        sigreg_loss_b = self._sigreg_loss(z_b_flat)
        sigreg_loss = (sigreg_loss_a + sigreg_loss_b) / 2

        # Compute weighted total loss
        total_loss = self.invariance_weight * inv_loss + self.sigreg_weight * sigreg_loss

        # Return detailed loss dictionary
        loss_dict = {
            "loss": total_loss,
            "invariance_loss": inv_loss,
            "sigreg_loss": sigreg_loss,
            "sigreg_loss_a": sigreg_loss_a,
            "sigreg_loss_b": sigreg_loss_b,
        }

        return loss_dict

    def extra_repr(self) -> str:
        """String representation for print/logging."""
        return (
            f"num_slices={self.num_slices}, "
            f"num_test_points={self.num_test_points}, "
            f"inv_weight={self.invariance_weight}, "
            f"sigreg_weight={self.sigreg_weight}, "
            f"flatten_patches={self.flatten_patches}, "
            f"fixed_slices={self.fixed_slices}"
        )


class HybridVICRegSIGRegLoss(nn.Module):
    """
    Hybrid loss combining VICReg and SIGReg.

    This allows gradual transition from VICReg to SIGReg during training,
    or using both for maximum stability. Useful for ablation studies.

    Args:
        vicreg_weight: Weight for VICReg component (default: 1.0)
        sigreg_weight: Weight for SIGReg component (default: 1.0)
        invariance_weight: Weight for invariance term (default: 25.0)
        variance_weight: VICReg variance weight (default: 25.0)
        covariance_weight: VICReg covariance weight (default: 1.0)
        num_slices: SIGReg number of slices (default: 1024)
        num_test_points: SIGReg number of test points (default: 17)
        variance_threshold: VICReg variance threshold (default: 1.0)
        eps: Small constant for numerical stability (default: 1e-6)
        flatten_patches: If True, flattens batch and patch dimensions (default: True)

    Example:
        >>> # Start with VICReg, gradually add SIGReg
        >>> loss_fn = HybridVICRegSIGRegLoss(
        ...     vicreg_weight=1.0,
        ...     sigreg_weight=0.0,  # Gradually increase this
        ... )
    """

    def __init__(
        self,
        vicreg_weight: float = 1.0,
        sigreg_weight: float = 1.0,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        num_slices: int = 1024,
        num_test_points: int = 17,
        variance_threshold: float = 1.0,
        eps: float = 1e-6,
        flatten_patches: bool = True,
    ):
        super().__init__()

        from .vicreg import VICRegLoss

        self.vicreg_weight = vicreg_weight
        self.sigreg_weight = sigreg_weight

        # VICReg component
        self.vicreg_loss = VICRegLoss(
            invariance_weight=invariance_weight,
            variance_weight=variance_weight,
            covariance_weight=covariance_weight,
            variance_threshold=variance_threshold,
            eps=eps,
            flatten_patches=flatten_patches,
        )

        # SIGReg component
        self.sigreg_loss = SIGRegLoss(
            num_slices=num_slices,
            num_test_points=num_test_points,
            invariance_weight=invariance_weight,
            sigreg_weight=sigreg_weight,
            eps=eps,
            flatten_patches=flatten_patches,
        )

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute hybrid VICReg + SIGReg loss.

        Args:
            z_a: First representation [B, D] or [B, N, D]
            z_b: Optional second representation

        Returns:
            Dictionary with combined loss components
        """
        # Compute both losses
        vicreg_dict = self.vicreg_loss(z_a, z_b)
        sigreg_dict = self.sigreg_loss(z_a, z_b)

        # Combine
        total_loss = (
            self.vicreg_weight * vicreg_dict["loss"] + self.sigreg_weight * sigreg_dict["loss"]
        )

        # Merge dictionaries
        loss_dict = {
            "loss": total_loss,
            "vicreg_loss": vicreg_dict["loss"],
            "sigreg_loss": sigreg_dict["loss"],
            "invariance_loss": vicreg_dict["invariance_loss"],
            "variance_loss": vicreg_dict["variance_loss"],
            "covariance_loss": vicreg_dict["covariance_loss"],
            "sigreg_regularization": sigreg_dict["sigreg_loss"],
        }

        return loss_dict

    def extra_repr(self) -> str:
        """String representation for print/logging."""
        return f"vicreg_weight={self.vicreg_weight}, " f"sigreg_weight={self.sigreg_weight}"

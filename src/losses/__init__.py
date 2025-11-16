"""
Loss functions for H-JEPA training including hierarchical consistency losses.

Available loss functions:
    - HJEPALoss: Hierarchical prediction loss for JEPA training
    - VICRegLoss: Variance-Invariance-Covariance regularization
    - AdaptiveVICRegLoss: VICReg with learnable/adaptive weights
    - SIGRegLoss: Sketched Isotropic Gaussian Regularization (improved stability)
    - HybridVICRegSIGRegLoss: Hybrid combining VICReg and SIGReg
    - CombinedLoss: Combines H-JEPA and VICReg losses
    - HierarchicalCombinedLoss: Advanced combined loss with per-level configs
    - NTXentLoss: NT-Xent (InfoNCE) contrastive loss
    - ContrastiveJEPALoss: C-JEPA hybrid combining JEPA with contrastive learning
    - create_loss_from_config: Factory function for creating losses from config
    - create_cjepa_loss_from_config: Factory function for creating C-JEPA loss

Example:
    >>> from src.losses import CombinedLoss
    >>> loss_fn = CombinedLoss(
    ...     jepa_loss_type='smoothl1',
    ...     jepa_hierarchy_weights=[1.0, 0.5, 0.25],
    ...     num_hierarchies=3,
    ...     vicreg_weight=0.1
    ... )
    >>>
    >>> # For SIGReg (improved training stability)
    >>> from src.losses import SIGRegLoss
    >>> sigreg_loss = SIGRegLoss(
    ...     num_slices=1024,
    ...     invariance_weight=25.0,
    ...     sigreg_weight=25.0
    ... )
    >>>
    >>> # For C-JEPA (Contrastive JEPA)
    >>> from src.losses import ContrastiveJEPALoss, HJEPALoss
    >>> jepa_loss = HJEPALoss(loss_type='smoothl1', num_hierarchies=3)
    >>> cjepa_loss = ContrastiveJEPALoss(
    ...     jepa_loss=jepa_loss,
    ...     contrastive_weight=0.1,
    ...     contrastive_temperature=0.1
    ... )
"""

from .combined import (
    CombinedLoss,
    HierarchicalCombinedLoss,
    create_loss_from_config,
)
from .contrastive import (
    ContrastiveJEPALoss,
    NTXentLoss,
    create_cjepa_loss_from_config,
)
from .hjepa_loss import HJEPALoss
from .sigreg import EppsPulleyTest, HybridVICRegSIGRegLoss, SIGRegLoss
from .vicreg import AdaptiveVICRegLoss, VICRegLoss

__all__ = [
    "HJEPALoss",
    "VICRegLoss",
    "AdaptiveVICRegLoss",
    "SIGRegLoss",
    "EppsPulleyTest",
    "HybridVICRegSIGRegLoss",
    "CombinedLoss",
    "HierarchicalCombinedLoss",
    "NTXentLoss",
    "ContrastiveJEPALoss",
    "create_loss_from_config",
    "create_cjepa_loss_from_config",
]

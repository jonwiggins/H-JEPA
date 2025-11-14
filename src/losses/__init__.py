"""
Loss functions for H-JEPA training including hierarchical consistency losses.

Available loss functions:
    - HJEPALoss: Hierarchical prediction loss for JEPA training
    - VICRegLoss: Variance-Invariance-Covariance regularization
    - AdaptiveVICRegLoss: VICReg with learnable/adaptive weights
    - CombinedLoss: Combines H-JEPA and VICReg losses
    - HierarchicalCombinedLoss: Advanced combined loss with per-level configs
    - create_loss_from_config: Factory function for creating losses from config

Example:
    >>> from src.losses import CombinedLoss
    >>> loss_fn = CombinedLoss(
    ...     jepa_loss_type='smoothl1',
    ...     jepa_hierarchy_weights=[1.0, 0.5, 0.25],
    ...     num_hierarchies=3,
    ...     vicreg_weight=0.1
    ... )
"""

from .hjepa_loss import HJEPALoss
from .vicreg import VICRegLoss, AdaptiveVICRegLoss
from .combined import (
    CombinedLoss,
    HierarchicalCombinedLoss,
    create_loss_from_config,
)

__all__ = [
    'HJEPALoss',
    'VICRegLoss',
    'AdaptiveVICRegLoss',
    'CombinedLoss',
    'HierarchicalCombinedLoss',
    'create_loss_from_config',
]

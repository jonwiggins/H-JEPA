"""
Masking strategies for multi-block masking in H-JEPA.
"""

from .multi_block import MultiBlockMaskGenerator
from .hierarchical import HierarchicalMaskGenerator

__all__ = [
    'MultiBlockMaskGenerator',
    'HierarchicalMaskGenerator',
]

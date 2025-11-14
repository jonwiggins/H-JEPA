"""
Visualization utilities for H-JEPA.

This package provides comprehensive visualization tools for:
- Attention maps and patterns
- Multi-block masking strategies
- Predictions and feature spaces
- Training metrics and analysis
"""

from .attention_viz import (
    visualize_attention_maps,
    visualize_multihead_attention,
    visualize_attention_rollout,
    visualize_hierarchical_attention,
)
from .masking_viz import (
    visualize_masking_strategy,
    visualize_masked_image,
    visualize_context_target_regions,
    animate_masking_process,
    compare_masking_strategies,
    visualize_multi_block_masking,
    plot_masking_statistics,
)
from .prediction_viz import (
    visualize_predictions,
    visualize_hierarchical_predictions,
    visualize_feature_space,
    visualize_nearest_neighbors,
    visualize_reconstruction,
    visualize_embedding_distribution,
)
from .training_viz import (
    plot_training_curves,
    plot_hierarchical_losses,
    visualize_loss_landscape,
    visualize_gradient_flow,
    plot_collapse_metrics,
    plot_ema_momentum,
    load_training_logs,
)

__all__ = [
    # Attention visualization
    'visualize_attention_maps',
    'visualize_multihead_attention',
    'visualize_attention_rollout',
    'visualize_hierarchical_attention',
    # Masking visualization
    'visualize_masking_strategy',
    'visualize_masked_image',
    'visualize_context_target_regions',
    'animate_masking_process',
    'compare_masking_strategies',
    'visualize_multi_block_masking',
    'plot_masking_statistics',
    # Prediction visualization
    'visualize_predictions',
    'visualize_hierarchical_predictions',
    'visualize_feature_space',
    'visualize_nearest_neighbors',
    'visualize_reconstruction',
    'visualize_embedding_distribution',
    # Training visualization
    'plot_training_curves',
    'plot_hierarchical_losses',
    'visualize_loss_landscape',
    'visualize_gradient_flow',
    'plot_collapse_metrics',
    'plot_ema_momentum',
    'load_training_logs',
]

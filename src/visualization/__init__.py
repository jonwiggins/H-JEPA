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
    visualize_attention_rollout,
    visualize_hierarchical_attention,
    visualize_multihead_attention,
)
from .masking_viz import (
    animate_masking_process,
    compare_masking_strategies,
    plot_masking_statistics,
    visualize_context_target_regions,
    visualize_masked_image,
    visualize_masking_strategy,
    visualize_multi_block_masking,
)
from .prediction_viz import (
    visualize_embedding_distribution,
    visualize_feature_space,
    visualize_hierarchical_predictions,
    visualize_nearest_neighbors,
    visualize_predictions,
    visualize_reconstruction,
)
from .training_viz import (
    load_training_logs,
    plot_collapse_metrics,
    plot_ema_momentum,
    plot_hierarchical_losses,
    plot_training_curves,
    visualize_gradient_flow,
    visualize_loss_landscape,
)

__all__ = [
    # Attention visualization
    "visualize_attention_maps",
    "visualize_multihead_attention",
    "visualize_attention_rollout",
    "visualize_hierarchical_attention",
    # Masking visualization
    "visualize_masking_strategy",
    "visualize_masked_image",
    "visualize_context_target_regions",
    "animate_masking_process",
    "compare_masking_strategies",
    "visualize_multi_block_masking",
    "plot_masking_statistics",
    # Prediction visualization
    "visualize_predictions",
    "visualize_hierarchical_predictions",
    "visualize_feature_space",
    "visualize_nearest_neighbors",
    "visualize_reconstruction",
    "visualize_embedding_distribution",
    # Training visualization
    "plot_training_curves",
    "plot_hierarchical_losses",
    "visualize_loss_landscape",
    "visualize_gradient_flow",
    "plot_collapse_metrics",
    "plot_ema_momentum",
    "load_training_logs",
]

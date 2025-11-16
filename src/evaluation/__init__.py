"""
Evaluation framework for H-JEPA.

This module provides comprehensive evaluation protocols for self-supervised
learning models, including:

- Linear probe evaluation: Train linear classifier on frozen features
- k-NN evaluation: k-nearest neighbors classification
- Feature quality analysis: Rank, variance, isotropy metrics
- Transfer learning: Fine-tuning and few-shot learning

Example usage:
    >>> from src.evaluation import linear_probe_eval, knn_eval
    >>> from src.models.hjepa import create_hjepa
    >>>
    >>> # Load model
    >>> model = create_hjepa()
    >>> model.load_state_dict(checkpoint['model_state_dict'])
    >>>
    >>> # Linear probe evaluation
    >>> metrics = linear_probe_eval(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     num_classes=10,
    ...     hierarchy_level=0,
    ... )
    >>>
    >>> # k-NN evaluation
    >>> metrics = knn_eval(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     test_loader=test_loader,
    ...     num_classes=10,
    ...     k=20,
    ... )
"""

# Feature quality
from .feature_quality import (
    FeatureQualityAnalyzer,
    analyze_feature_quality,
    compare_hierarchy_levels,
    print_quality_report,
)

# k-NN evaluation
from .knn_eval import (
    KNNEvaluator,
    knn_eval,
    sweep_knn_params,
)

# Linear probe
from .linear_probe import (
    LinearProbe,
    LinearProbeEvaluator,
    linear_probe_eval,
)

# Transfer learning
from .transfer import (
    FewShotEvaluator,
    FineTuneEvaluator,
    TransferHead,
    few_shot_eval,
    fine_tune_eval,
)

__all__ = [
    # Linear probe
    "LinearProbe",
    "LinearProbeEvaluator",
    "linear_probe_eval",
    # k-NN
    "KNNEvaluator",
    "knn_eval",
    "sweep_knn_params",
    # Feature quality
    "FeatureQualityAnalyzer",
    "analyze_feature_quality",
    "print_quality_report",
    "compare_hierarchy_levels",
    # Transfer learning
    "TransferHead",
    "FineTuneEvaluator",
    "FewShotEvaluator",
    "fine_tune_eval",
    "few_shot_eval",
]

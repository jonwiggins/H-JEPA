# H-JEPA Evaluation Module

Comprehensive evaluation protocols for H-JEPA models.

## Module Structure

```
src/evaluation/
├── __init__.py              # Module exports
├── linear_probe.py          # Linear probe evaluation
├── knn_eval.py             # k-NN classification
├── feature_quality.py      # Representation quality metrics
└── transfer.py             # Transfer learning & few-shot
```

## Quick Import

```python
from src.evaluation import (
    # Linear probe
    linear_probe_eval,
    LinearProbeEvaluator,
    
    # k-NN
    knn_eval,
    KNNEvaluator,
    
    # Feature quality
    analyze_feature_quality,
    print_quality_report,
    
    # Transfer learning
    fine_tune_eval,
    few_shot_eval,
)
```

## See Also

- **EVALUATION_GUIDE.md**: Comprehensive usage guide
- **examples/evaluation_examples.py**: Code examples
- **scripts/evaluate.py**: Main evaluation script


# H-JEPA Evaluation Framework

Comprehensive evaluation framework for H-JEPA (Hierarchical Joint-Embedding Predictive Architecture) models.

## Overview

This framework provides multiple evaluation protocols to assess the quality of learned representations:

1. **Linear Probe**: Train a linear classifier on frozen features
2. **k-NN Classification**: k-nearest neighbors evaluation
3. **Feature Quality Analysis**: Representation quality metrics (rank, variance, isotropy)
4. **Transfer Learning**: Fine-tuning on downstream tasks
5. **Few-Shot Learning**: Learning from limited examples

## Quick Start

### Run All Evaluations

```bash
python scripts/evaluate.py \
    --checkpoint results/checkpoints/best_model.pth \
    --dataset cifar10 \
    --data-path ./data \
    --hierarchy-levels 0 1 2
```

### Run Specific Evaluation

```bash
# Linear probe only
python scripts/evaluate.py \
    --checkpoint model.pth \
    --dataset cifar10 \
    --eval-type linear_probe

# k-NN only
python scripts/evaluate.py \
    --checkpoint model.pth \
    --dataset cifar10 \
    --eval-type knn \
    --knn-k 20

# Feature quality analysis
python scripts/evaluate.py \
    --checkpoint model.pth \
    --dataset cifar10 \
    --eval-type feature_quality
```

## Evaluation Protocols

### 1. Linear Probe Evaluation

Trains a linear classifier on top of frozen H-JEPA features. This is the standard protocol for evaluating self-supervised representations.

**Usage:**
```bash
python scripts/evaluate.py \
    --checkpoint model.pth \
    --dataset cifar10 \
    --eval-type linear_probe \
    --linear-probe-epochs 100 \
    --linear-probe-lr 0.1
```

**Python API:**
```python
from src.evaluation import linear_probe_eval
from src.models.hjepa import create_hjepa

model = create_hjepa()
model.load_state_dict(torch.load('model.pth'))

metrics = linear_probe_eval(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=10,
    hierarchy_level=0,
    epochs=100,
    lr=0.1,
)

print(f"Accuracy: {metrics['accuracy']:.2f}%")
print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.2f}%")
```

**Features:**
- Frozen feature extraction
- Support for different hierarchy levels
- k-fold cross-validation
- Accuracy and top-k accuracy metrics
- Confusion matrix computation

### 2. k-NN Evaluation

Evaluates features using k-nearest neighbors classification. No training required!

**Usage:**
```bash
python scripts/evaluate.py \
    --checkpoint model.pth \
    --dataset cifar10 \
    --eval-type knn \
    --knn-k 20 \
    --knn-temperature 0.07
```

**Python API:**
```python
from src.evaluation import knn_eval

metrics = knn_eval(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    num_classes=10,
    hierarchy_level=0,
    k=20,
    distance_metric='cosine',
    temperature=0.07,
)

print(f"k-NN Accuracy: {metrics['accuracy']:.2f}%")
```

**Features:**
- No training required
- Distance metrics: cosine, euclidean
- Temperature-based weighting
- Top-k accuracy
- Parameter sweeping for hyperparameter tuning

### 3. Feature Quality Analysis

Analyzes representation quality using various metrics.

**Usage:**
```bash
python scripts/evaluate.py \
    --checkpoint model.pth \
    --dataset cifar10 \
    --eval-type feature_quality
```

**Python API:**
```python
from src.evaluation import analyze_feature_quality, print_quality_report

metrics = analyze_feature_quality(
    model=model,
    dataloader=val_loader,
    hierarchy_level=0,
    max_samples=10000,
)

print_quality_report(metrics)
```

**Metrics Computed:**

- **Rank Analysis**
  - Effective rank (SVD-based)
  - Rank ratio
  - Number of components for 99% variance
  - Singular value distribution

- **Feature Statistics**
  - Per-dimension variance
  - Mean/std of features
  - Covariance structure
  - Correlation analysis

- **Isotropy**
  - Feature similarity distribution
  - Uniformity measure
  - Alignment metrics

- **Collapse Detection**
  - Rank collapse indicators
  - Variance collapse detection
  - Dimension collapse checking

### 4. Transfer Learning (Fine-tuning)

Fine-tunes the model on downstream tasks.

**Usage:**
```bash
# Full fine-tuning
python scripts/evaluate.py \
    --checkpoint model.pth \
    --dataset cifar10 \
    --eval-type fine_tune \
    --fine-tune-epochs 50 \
    --fine-tune-lr 1e-3

# Frozen encoder (linear head only)
python scripts/evaluate.py \
    --checkpoint model.pth \
    --dataset cifar10 \
    --eval-type fine_tune \
    --freeze-encoder
```

**Python API:**
```python
from src.evaluation import fine_tune_eval

# Full fine-tuning
metrics = fine_tune_eval(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=10,
    hierarchy_level=0,
    freeze_encoder=False,
    epochs=50,
    lr=1e-3,
)

print(f"Fine-tuned Accuracy: {metrics['accuracy']:.2f}%")
```

**Features:**
- Full fine-tuning or frozen encoder
- Multi-layer classification heads
- Different learning rates for encoder and head
- Training history tracking

### 5. Few-Shot Learning

Evaluates the model's ability to learn from very few examples.

**Usage:**
```bash
python scripts/evaluate.py \
    --checkpoint model.pth \
    --dataset cifar10 \
    --eval-type few_shot \
    --few-shot-n-way 5 \
    --few-shot-k-shots 1 5 10 \
    --few-shot-episodes 100
```

**Python API:**
```python
from src.evaluation import few_shot_eval

results = few_shot_eval(
    model=model,
    dataset=val_dataset,
    num_classes=10,
    n_way=5,
    k_shot_list=[1, 5, 10],
    n_episodes=100,
    hierarchy_level=0,
)

for k_shot, metrics in results.items():
    print(f"{k_shot}-shot: {metrics['accuracy']:.2f}% "
          f"± {metrics['confidence_interval']:.2f}%")
```

**Features:**
- N-way K-shot evaluation
- Episode-based sampling
- Confidence intervals
- Nearest centroid classification

## Hierarchy Level Evaluation

H-JEPA learns features at multiple hierarchical levels. Evaluate all levels:

```bash
python scripts/evaluate.py \
    --checkpoint model.pth \
    --dataset cifar10 \
    --hierarchy-levels 0 1 2 \
    --eval-type all
```

This will run all evaluation protocols on hierarchy levels 0, 1, and 2, allowing you to compare representation quality across different granularities.

## Supported Datasets

- **CIFAR-10**: 10 classes, 50k train, 10k test
- **CIFAR-100**: 100 classes, 50k train, 10k test
- **STL-10**: 10 classes, 5k train, 8k test
- **ImageNet-100**: 100 classes (subset of ImageNet)
- **ImageNet**: 1000 classes (requires manual download)

## Output

All results are saved to JSON files in the output directory:

```
results/evaluation/
├── evaluation_results.json    # All metrics in JSON format
└── visualizations/            # t-SNE/UMAP plots (if enabled)
```

### Results Format

```json
{
  "level_0": {
    "linear_probe": {
      "accuracy": 85.4,
      "top_5_accuracy": 97.2
    },
    "knn": {
      "accuracy": 82.1,
      "top_1_accuracy": 82.1,
      "top_5_accuracy": 95.8
    },
    "feature_quality": {
      "rank": {
        "effective_rank": 512.3,
        "rank_ratio": 0.667
      },
      "isotropy": {
        "uniformity": -2.34
      }
    }
  },
  "metadata": {
    "checkpoint": "model.pth",
    "dataset": "cifar10",
    "timestamp": "2025-11-14T12:00:00"
  }
}
```

## Advanced Usage

### Compare Multiple Hierarchy Levels

```python
from src.evaluation import compare_hierarchy_levels

results = compare_hierarchy_levels(
    model=model,
    dataloader=val_loader,
    num_levels=3,
    max_samples=10000,
)

# Results contain metrics for each level
for level, metrics in results.items():
    print(f"Level {level}:")
    print(f"  Effective Rank: {metrics['rank']['effective_rank']:.2f}")
    print(f"  Uniformity: {metrics['isotropy']['uniformity']:.6f}")
```

### Hyperparameter Sweeping for k-NN

```python
from src.evaluation import sweep_knn_params

results = sweep_knn_params(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    num_classes=10,
    k_values=[10, 20, 50, 100],
    temperatures=[0.01, 0.05, 0.07, 0.1],
    distance_metrics=['cosine', 'euclidean'],
)

# Find best configuration
best_config = max(results.items(),
                  key=lambda x: x[1]['metrics']['accuracy'])
print(f"Best: {best_config}")
```

### k-Fold Cross-Validation

```python
from src.evaluation import LinearProbeEvaluator

evaluator = LinearProbeEvaluator(
    model=model,
    num_classes=10,
    input_dim=768,
    hierarchy_level=0,
)

results = evaluator.k_fold_cross_validation(
    dataset=train_dataset,
    k_folds=5,
    epochs=100,
    batch_size=256,
)

print(f"Mean Accuracy: {results['mean_accuracy']:.2f}%")
print(f"Std Accuracy: {results['std_accuracy']:.2f}%")
```

### Feature Visualization

```python
from src.evaluation import FeatureQualityAnalyzer

analyzer = FeatureQualityAnalyzer(model, hierarchy_level=0)

# Extract features
features, labels = analyzer.extract_features(val_loader, max_samples=5000)

# t-SNE
tsne_embeddings = analyzer.visualize_features_tsne(
    features, labels, perplexity=30
)

# UMAP (requires umap-learn package)
umap_embeddings = analyzer.visualize_features_umap(
    features, labels, n_neighbors=15
)

# PCA
pca_embeddings, pca_obj = analyzer.compute_pca(
    features, n_components=50
)
```

## Best Practices

1. **Reproducibility**: Always set a random seed
   ```bash
   --seed 42
   ```

2. **Multiple Runs**: Run evaluations multiple times with different seeds for robust estimates

3. **Hierarchy Levels**: Evaluate all levels to understand multi-scale representations

4. **Dataset Size**: Use sufficient samples (>10k) for feature quality analysis

5. **Batch Size**: Use larger batch sizes for faster evaluation
   ```bash
   --batch-size 512
   ```

6. **Save Results**: Always save results for later comparison
   ```bash
   --output-dir results/eval_$(date +%Y%m%d_%H%M%S)
   ```

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
--batch-size 128
```

For feature quality analysis, reduce max samples:
```bash
# Edit in Python code
max_samples=5000
```

### Slow Evaluation

- Use fewer workers if CPU-bound: `--num-workers 2`
- Reduce number of epochs for probes: `--linear-probe-epochs 50`
- Evaluate single hierarchy level: `--hierarchy-levels 0`

### Missing Dependencies

Install evaluation dependencies:
```bash
pip install scikit-learn scipy
pip install umap-learn  # For UMAP visualization (optional)
```

## Metrics Interpretation

### Good Representations

- **Linear Probe Accuracy**: >80% on CIFAR-10, >65% on CIFAR-100
- **Effective Rank**: >50% of feature dimension
- **Uniformity**: <-2.0 (more negative is better)
- **k-NN Accuracy**: Close to linear probe accuracy

### Warning Signs

- **Low Effective Rank**: <10% indicates collapse
- **Low Variance**: <0.01 indicates collapse
- **High Uniformity**: >0 indicates features are too similar
- **Large gap between k-NN and linear probe**: May indicate features need non-linear classification

## Citation

If you use this evaluation framework, please cite:

```bibtex
@software{hjepa_eval,
  title={H-JEPA Evaluation Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/H-JEPA}
}
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## License

This evaluation framework is part of the H-JEPA project. See LICENSE for details.

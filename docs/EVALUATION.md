# H-JEPA Evaluation Guide

Comprehensive guide to evaluating trained H-JEPA models.

## Quick Start

```bash
# 1. Linear Probing (best metric for SSL)
python3.11 scripts/eval_linear_probe.py \
  --checkpoint results/my_exp/checkpoints/best_model.pt \
  --dataset cifar10 \
  --device mps

# 2. k-NN Evaluation (fast, no training)
python3.11 scripts/eval_knn.py \
  --checkpoint results/my_exp/checkpoints/best_model.pt \
  --dataset cifar10 \
  --k 1 5 10 20

# 3. Transfer Learning (comprehensive)
python3.11 scripts/eval_transfer.py \
  --checkpoint results/my_exp/checkpoints/best_model.pt \
  --datasets cifar10 cifar100 stl10 \
  --linear-probe \
  --knn
```

---

## Evaluation Metrics

### 1. Linear Probing

**What it measures:** Quality of learned features for supervised tasks

**How it works:**
1. Freeze pretrained encoder
2. Train a linear classifier on top
3. Measure classification accuracy

**Interpretation:**
- **70-80%:** Good representations
- **80-85%:** Very good representations
- **85%+:** Excellent representations
(on CIFAR-10)

**Usage:**
```bash
python3.11 scripts/eval_linear_probe.py \
  --checkpoint PATH \
  --dataset cifar10 \
  --epochs 100 \
  --lr 0.001 \
  --hierarchy-level -1  # Use top hierarchy
```

**Output:**
```
Best validation accuracy: 82.45%
Final validation accuracy: 82.31%
Results saved to results/linear_probe/checkpoint_cifar10_results.json
```

### 2. k-NN Classification

**What it measures:** Clustering quality in feature space

**How it works:**
1. Extract features from train/test sets
2. For each test sample, find k nearest neighbors
3. Predict via majority vote

**Interpretation:**
- No training required (fast!)
- Pure measure of representation quality
- Lower than linear probe (expected)

**Usage:**
```bash
python3.11 scripts/eval_knn.py \
  --checkpoint PATH \
  --dataset cifar10 \
  --k 1 5 10 20 \
  --temperature 0.07  # Softmax temperature
```

**Output:**
```
k= 1: 75.32%
k= 5: 78.91%
k=10: 79.45%
k=20: 79.12%
```

**Best k value:** Typically 10-20 for CIFAR, 5-10 for ImageNet

### 3. Transfer Learning

**What it measures:** Generalization to different datasets

**How it works:**
1. Pretrain on dataset A (e.g., CIFAR-10)
2. Evaluate on dataset B (e.g., STL-10)
3. Compare performance

**Usage:**
```bash
python3.11 scripts/eval_transfer.py \
  --checkpoint PATH \
  --datasets cifar10 cifar100 stl10 \
  --linear-probe \
  --knn
```

**Output:**
```
Dataset        Linear Probe  k-NN (k=20)
CIFAR-10       82.45%        79.12%
CIFAR-100      54.32%        48.91%
STL-10         85.67%        82.34%
```

---

## Visualization Tools

### 1. Model Explorer

Interactive exploration of trained models.

```bash
python3.11 scripts/explore_model.py \
  --checkpoint PATH \
  --device mps \
  --output-dir results/exploration \
  --sample-idx 0
```

**Generates:**
- `attention_maps.png` - Multi-head attention patterns
- `hierarchical_representations.png` - Multi-scale features
- `masked_prediction.png` - Prediction demo
- `embedding_similarity.png` - Feature space analysis

### 2. Feature Visualization

Understand what features detect.

```bash
python3.11 scripts/visualize_features.py \
  --checkpoint PATH \
  --dataset cifar10 \
  --hierarchy 0 1 2 \
  --num-samples 200
```

**Generates:**
- Feature activation maps per channel
- Top activating patches
- Feature statistics and distributions

### 3. Attention Rollout

Visualize aggregated attention across layers.

```bash
python3.11 scripts/visualize_attention_rollout.py \
  --checkpoint PATH \
  --sample-idx 0 10 20 \
  --discard-ratio 0.1
```

**Generates:**
- Attention rollout heatmaps
- Layer-by-layer attention comparison
- Attention overlays on images

**Note:** Disables Flash Attention to extract weights

### 4. Interactive Notebook

Jupyter notebook for hands-on exploration.

```bash
cd notebooks
jupyter notebook explore_hjepa.ipynb
```

**Features:**
- Interactive sample browser
- Attention visualization
- Similarity search
- Quick k-NN evaluation
- Feature export

---

## Benchmark Results

### Expected Performance (ViT-Base/16)

**CIFAR-10:**
- Linear Probe: 78-85%
- k-NN (k=20): 75-80%

**CIFAR-100:**
- Linear Probe: 50-60%
- k-NN (k=20): 45-55%

**STL-10:**
- Linear Probe: 80-88%
- k-NN (k=20): 78-85%

**ImageNet-1K:**
- Linear Probe: 65-75%
- k-NN (k=200): 55-65%

### Comparison to Other Methods

**CIFAR-10 Linear Probe (ViT-Base):**
```
Method          Accuracy
SimCLR          76.5%
DINO            81.3%
MAE             78.2%
H-JEPA (ours)   82.5%  ← Target
```

**Training efficiency:**
- H-JEPA: 100 epochs → 82%
- MAE: 200 epochs → 78%
- DINO: 300 epochs → 81%

---

## Evaluation Best Practices

### 1. Use Multiple Metrics

Don't rely on a single metric:

```bash
# Comprehensive evaluation
python3.11 scripts/eval_transfer.py \
  --checkpoint PATH \
  --datasets cifar10 stl10 \
  --linear-probe \  # Supervised metric
  --knn            # Unsupervised metric
```

### 2. Test on Multiple Datasets

Evaluate generalization:

```bash
# Train on CIFAR-10, test on multiple
python3.11 scripts/eval_transfer.py \
  --checkpoint pretrained_cifar10.pt \
  --datasets cifar10 cifar100 stl10
```

**Good model:** Performance holds across datasets
**Overfit model:** Drops significantly on new data

### 3. Compare Hierarchies

Test each hierarchy level:

```bash
for level in -3 -2 -1; do
  python3.11 scripts/eval_linear_probe.py \
    --checkpoint PATH \
    --hierarchy-level $level \
    --output-dir results/hierarchy_$level
done
```

**Expected:**
- Level 1 (fine): Good for texture tasks
- Level 2 (mid): Good for parts
- Level 3 (coarse): Good for objects/scenes

### 4. Track Over Training

Evaluate checkpoints during training:

```bash
# Automatic evaluation every epoch
./scripts/watch_and_explore.sh
```

Or manually:

```bash
for epoch in 10 20 30 40 50; do
  python3.11 scripts/eval_knn.py \
    --checkpoint results/checkpoints/checkpoint_epoch_$epoch.pt \
    --dataset cifar10
done
```

**Plot results:**
- Accuracy vs. Epoch
- Find optimal checkpoint

### 5. Document Everything

Create a results file:

```yaml
# results/my_experiment/results.yaml
experiment:
  name: my_experiment
  date: 2025-01-17
  checkpoint: checkpoint_epoch_100.pt

config:
  model: vit_base_patch16_224
  dataset: cifar10
  epochs: 100
  batch_size: 64
  lr: 0.001

evaluation:
  cifar10:
    linear_probe: 82.45%
    knn_k20: 79.12%
  stl10:
    linear_probe: 85.67%
    knn_k20: 82.34%

notes: |
  - Flash Attention enabled
  - RoPE position embeddings
  - 3 hierarchies
  - Training time: 4.5 hours on M1 Max
```

---

## Troubleshooting

### Low Linear Probe Accuracy (<70%)

**Possible causes:**
1. Undertrained - train longer
2. Bad hyperparameters - tune LR, weight decay
3. Too small model - use larger ViT
4. Wrong hierarchy - try different levels

**Solutions:**
```bash
# 1. Train longer
python3.11 scripts/train.py --config CONFIG --epochs 200

# 2. Tune hyperparameters
python3.11 scripts/train.py --config CONFIG --lr 0.003 --weight-decay 0.1

# 3. Larger model
# Edit config.yaml: encoder_type: vit_large_patch16_224

# 4. Try all hierarchies
for h in -3 -2 -1; do
  python3.11 scripts/eval_linear_probe.py --hierarchy-level $h
done
```

### k-NN Much Lower Than Linear Probe

**Expected:** k-NN is typically 3-5% lower

**Too large gap (>10%):**
- Features not well normalized
- Need more neighbors (try k=50)
- Temperature tuning needed

**Solutions:**
```bash
# Try more neighbors
python3.11 scripts/eval_knn.py --k 10 20 50 100

# Tune temperature
python3.11 scripts/eval_knn.py --temperature 0.05  # or 0.1, 0.2
```

### Out of Memory During Evaluation

**Solutions:**
```bash
# Reduce batch size
python3.11 scripts/eval_linear_probe.py --batch-size 64  # default: 256

# Use CPU for linear probe (still fast)
python3.11 scripts/eval_linear_probe.py --device cpu

# Extract features once, train classifier separately
# (Features are cached in eval scripts)
```

---

## Creating Publication-Quality Results

### 1. Systematic Evaluation

```bash
#!/bin/bash
# evaluate_all.sh

CHECKPOINT="results/final_model/checkpoints/best_model.pt"

# Linear probing on all datasets
for dataset in cifar10 cifar100 stl10; do
  python3.11 scripts/eval_linear_probe.py \
    --checkpoint $CHECKPOINT \
    --dataset $dataset \
    --epochs 100 \
    --output-dir results/paper/linear_probe
done

# k-NN evaluation
python3.11 scripts/eval_knn.py \
  --checkpoint $CHECKPOINT \
  --dataset cifar10 \
  --k 1 5 10 20 50 100 \
  --output-dir results/paper/knn

# Transfer learning
python3.11 scripts/eval_transfer.py \
  --checkpoint $CHECKPOINT \
  --datasets cifar10 cifar100 stl10 \
  --linear-probe \
  --knn \
  --output-dir results/paper/transfer

# Visualizations
python3.11 scripts/visualize_features.py \
  --checkpoint $CHECKPOINT \
  --output-dir results/paper/visualizations

python3.11 scripts/visualize_attention_rollout.py \
  --checkpoint $CHECKPOINT \
  --sample-idx 0 10 20 30 40 \
  --output-dir results/paper/attention
```

### 2. Ablation Studies

```bash
# Test each component
for config in baseline +rope +flash +hierarchies; do
  python3.11 scripts/train.py --config configs/ablation_$config.yaml
  python3.11 scripts/eval_linear_probe.py \
    --checkpoint results/ablation_$config/checkpoints/best.pt \
    --dataset cifar10
done
```

### 3. Generate Tables

```python
# scripts/generate_tables.py
import pandas as pd
import json

results = {
    'Method': ['Baseline', '+RoPE', '+Flash', '+Hierarchies'],
    'CIFAR-10': [78.2, 79.5, 79.4, 82.5],
    'CIFAR-100': [52.1, 53.8, 53.9, 56.2],
    'STL-10': [82.3, 83.9, 84.1, 85.7],
}

df = pd.DataFrame(results)
print(df.to_latex(index=False))  # For paper
print(df.to_markdown(index=False))  # For README
```

---

## Next Steps

After evaluation:

1. **Document results** - Create results summary
2. **Compare to baselines** - How does it stack up?
3. **Identify improvements** - What could be better?
4. **Share checkpoints** - Make models available
5. **Publish findings** - Blog post, paper, etc.

**Further Reading:**
- `docs/TRAINING_GUIDE.md` - Training best practices
- `docs/ARCHITECTURE.md` - Understanding the model
- `notebooks/explore_hjepa.ipynb` - Interactive exploration

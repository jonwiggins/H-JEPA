# Building a Foundation Model with H-JEPA

## 🎯 What is a Foundation Model?

Foundation models (like **CLIP, DINOv2, SAM**) are trained on **massive, diverse datasets** to learn general-purpose visual representations that transfer well to many downstream tasks.

**Key characteristics:**
1. **Multiple datasets** - Not just ImageNet!
2. **Massive scale** - Millions to billions of images
3. **Diverse sources** - Photos, art, diagrams, medical images, etc.
4. **Strong transfer** - Works well on unseen tasks

## 🏗️ Foundation Model Strategy for H-JEPA

### Why Multiple Datasets?

**Single dataset (e.g., ImageNet only):**
- ❌ Biased toward photographic images
- ❌ Limited domain coverage
- ❌ Poor generalization to specialized domains

**Multiple datasets (foundation model):**
- ✅ Diverse visual knowledge
- ✅ Better transfer learning
- ✅ Robust to domain shifts
- ✅ More generalizable features

## 📊 Dataset Mixture Strategies

### Strategy 1: Weighted Sampling (Recommended)

Sample from datasets with different probabilities:

```yaml
datasets:
  - name: "imagenet"
    weight: 0.7      # 70% ImageNet (core visual knowledge)
  - name: "coco"
    weight: 0.2      # 20% COCO (diverse scenes, contexts)
  - name: "places365"
    weight: 0.1      # 10% Places (scene understanding)
```

**Pros:**
- Control dataset contribution
- Prevent small datasets from being undersampled
- Standard approach for foundation models

**When to use:** When you want balanced representation from diverse sources

### Strategy 2: Balanced Sampling

Equal samples from each dataset per epoch:

```yaml
sampling_strategy: "balanced"
datasets:
  - name: "imagenet"    # 1.28M images → sample 100K
  - name: "stl10"       # 100K images → sample 100K (all)
  - name: "cifar100"    # 50K images → sample 100K (with repetition)
```

**Pros:**
- Every dataset contributes equally
- Good for domain-specific expertise

**When to use:** When all datasets are equally important

### Strategy 3: Simple Concatenation

Just combine all datasets:

```yaml
sampling_strategy: "concat"
datasets:
  - name: "imagenet"
  - name: "stl10"
```

**Pros:**
- Simple, no tuning
- Natural dataset proportions

**Cons:**
- Large datasets dominate
- Small datasets rarely seen

**When to use:** When datasets are similar size

## 🎓 Pre-Configured Foundation Models

I've created 3 ready-to-use configs:

### 1. **Mini Foundation Model** (Recommended for M1 Max)

**Datasets:**
- ImageNet-100: 126K images (60%)
- STL-10: 100K images (30%)
- CIFAR-100: 50K images (10%)

**Total:** ~276K images

**Training time:** ~18-24 hours (100 epochs on M1 Max)

**Expected results:**
- Linear probe: 65-75%
- Better than any single dataset
- Good transfer learning

**To train:**
```bash
# Download all datasets
./scripts/download_imagenet100.sh
python3.11 scripts/download_data.py --dataset stl10 --dataset cifar100

# Train foundation model
python3.11 scripts/train.py --config configs/foundation_model_mini.yaml
```

### 2. **Medium Foundation Model**

**Datasets:**
- ImageNet (full): 1.28M images (90%)
- STL-10: 100K images (10%)

**Total:** ~1.38M images

**Training time:** ~7-10 days (100 epochs on M1 Max)

**Expected results:**
- Linear probe: 70-78%
- Close to published I-JEPA results
- Excellent transfer learning

**To train:**
```bash
# Download ImageNet (requires ~150GB)
python3.11 scripts/download_data.py --dataset imagenet --dataset stl10

# Train
python3.11 scripts/train.py --config configs/foundation_model_medium.yaml
```

### 3. **Custom Foundation Model**

Build your own mixture:

```bash
python3.11 scripts/create_foundation_model.py \
    --datasets imagenet100 stl10 cifar100 places365 \
    --weights 0.5 0.2 0.2 0.1 \
    --output-config configs/my_foundation_model.yaml

# Then train
python3.11 scripts/train.py --config configs/my_foundation_model.yaml
```

## 🌍 Real-World Foundation Model Examples

### CLIP (OpenAI)
- **Datasets:** 400M image-text pairs from internet
- **Size:** 400M images
- **Training:** Weeks on hundreds of GPUs

### DINOv2 (Meta)
- **Datasets:** LVD-142M (curated web images)
- **Size:** 142M images
- **Training:** Days on 64 GPUs
- **Results:** SOTA self-supervised learning

### SAM (Meta)
- **Datasets:** SA-1B (segmentation dataset)
- **Size:** 1B images with masks
- **Training:** Custom dataset + massive compute

## 📈 Scaling Laws for Foundation Models

Based on SSL research:

| Total Images | Expected Linear Probe | Transfer Performance | Training Time (M1 Max) |
|--------------|---------------------|---------------------|----------------------|
| 50K (CIFAR-10) | 30-50% | Poor | 2-3 hours |
| 250K (Mini) | 65-75% | Good | 18-24 hours |
| 500K | 68-78% | Very good | 2-3 days |
| 1M+ (Medium) | 70-80% | Excellent | 7-10 days |
| 10M+ | 75-85% | SOTA | Months (impractical on M1 Max) |

**Key insight:** Diminishing returns after ~1M images for single-GPU training.

## 🎯 Recommended Path for Your Use Case

### Goal: Quick Experimentation
**Use:** ImageNet-100 only
- Fast (10-15 hours)
- Good enough for testing
- Easy to iterate

### Goal: Strong Performance
**Use:** Mini Foundation Model
- ImageNet-100 + STL-10 + CIFAR-100
- Diverse knowledge
- ~18-24 hours
- Best bang for buck on M1 Max

### Goal: Publication/Production
**Use:** Medium Foundation Model
- Full ImageNet + STL-10
- Week-long training
- Competitive with published results
- Strong transfer learning

### Goal: Maximum Performance (Multi-GPU)
**Use:** Custom large-scale
- ImageNet + COCO + Places365 + CC3M
- Requires multiple GPUs or cloud
- Weeks of training
- SOTA performance

## 🔧 Implementation Details

### Weighted Sampling in Practice

```python
from src.data.multi_dataset import create_foundation_model_dataset

# Create mini foundation model dataset
dataset = create_foundation_model_dataset(
    scale="mini",  # or "medium"
    data_path="./data",
    split="train",
    image_size=224,
)

# The dataset automatically:
# 1. Loads all specified datasets
# 2. Applies weighted sampling
# 3. Balances across datasets
# 4. Tracks statistics
```

### Dataset Statistics Tracking

The multi-dataset system tracks:
- Samples from each dataset per epoch
- Effective dataset contribution
- Sampling efficiency

```python
stats = dataset.get_dataset_stats()
# {
#   'imagenet100': {'size': 126689, 'weight': 0.6, 'expected_samples': 165600},
#   'stl10': {'size': 100000, 'weight': 0.3, 'expected_samples': 82800},
#   'cifar100': {'size': 50000, 'weight': 0.1, 'expected_samples': 27600}
# }
```

## 💡 Best Practices

### 1. Dataset Selection

**Core dataset (70-90%):**
- ImageNet or ImageNet-100
- Provides foundational visual knowledge

**Diversity datasets (10-30%):**
- STL-10: Unlabeled, SSL-friendly
- COCO: Complex scenes, multiple objects
- Places365: Scene understanding
- CIFAR-100: Fine-grained categories

### 2. Weight Tuning

Start with these ratios:
```yaml
# Balanced approach
- ImageNet: 60-70%
- Scene/Context dataset: 20-30%
- Fine-grained dataset: 10%
```

Adjust based on:
- Your target domain
- Dataset quality
- Observed performance

### 3. Training Strategy

**Phase 1: Validation (1-5 epochs)**
- Verify all datasets load correctly
- Check sampling distribution
- Ensure no crashes

**Phase 2: Short run (20 epochs)**
- Evaluate multi-dataset benefit
- Tune weights if needed
- Compare to single-dataset baseline

**Phase 3: Full run (100+ epochs)**
- Train to convergence
- Save checkpoints frequently
- Monitor per-dataset metrics

### 4. Evaluation

Test on **diverse downstream tasks**:
- ImageNet classification (standard)
- COCO object detection (transfer)
- Places365 scene recognition (domain shift)
- Fine-grained datasets (detail preservation)

## 📊 Expected Performance Improvements

**Single dataset (ImageNet-100):**
- Linear probe: 60-70%
- Transfer to COCO: Moderate
- Transfer to scenes: Poor

**Mini foundation model (3 datasets):**
- Linear probe: 65-75% (+5-10%)
- Transfer to COCO: Good (+15-20%)
- Transfer to scenes: Moderate (+10-15%)
- **More robust and generalizable**

**Medium foundation model (ImageNet + more):**
- Linear probe: 70-80% (+10-15%)
- Transfer to COCO: Excellent (+20-30%)
- Transfer to scenes: Good (+15-25%)
- **Near publication quality**

## 🚀 Quick Start Commands

### Mini Foundation Model (Recommended)

```bash
# 1. Download datasets (~20GB total)
./scripts/download_imagenet100.sh
python3.11 scripts/download_data.py --dataset stl10 --dataset cifar100

# 2. Train
python3.11 scripts/train.py --config configs/foundation_model_mini.yaml

# Expected: ~18-24 hours, 65-75% accuracy
```

### Medium Foundation Model

```bash
# 1. Download datasets (~155GB total)
python3.11 scripts/download_data.py --dataset imagenet --dataset stl10

# 2. Train
python3.11 scripts/train.py --config configs/foundation_model_medium.yaml

# Expected: ~7-10 days, 70-78% accuracy
```

### Custom Foundation Model

```bash
# 1. Create custom config
python3.11 scripts/create_foundation_model.py \
    --datasets imagenet100 stl10 places365 \
    --weights 0.6 0.3 0.1 \
    --output-config configs/custom_foundation.yaml

# 2. Train
python3.11 scripts/train.py --config configs/custom_foundation.yaml
```

## 🎓 Further Reading

**Papers on multi-dataset training:**
- CLIP (OpenAI, 2021): Learning Transferable Visual Models From Natural Language Supervision
- DINOv2 (Meta, 2023): Learning Robust Visual Features without Supervision
- ALIGN (Google, 2021): Scaling Up Visual and Vision-Language Representation Learning

**Key insights from research:**
1. **Diversity > Scale** (sometimes): 500K diverse images can outperform 1M similar images
2. **Dataset balance matters**: Weighted sampling improves over simple concatenation
3. **Longer training helps**: Foundation models benefit from 100+ epochs
4. **Transfer is the goal**: Optimize for downstream tasks, not just pretraining loss

## ✅ Foundation Model Checklist

Before starting:
- [ ] Decided on scale (mini, medium, large)
- [ ] Downloaded all datasets
- [ ] Verified disk space (~20GB for mini, ~155GB for medium)
- [ ] Set appropriate training time (18-24 hours for mini)
- [ ] Prepared for monitoring (TensorBoard, logs)

During training:
- [ ] Monitor dataset sampling distribution
- [ ] Check loss curves for each hierarchy level
- [ ] Verify no collapse (effective rank checks)
- [ ] Save checkpoints regularly

After training:
- [ ] Evaluate on multiple downstream tasks
- [ ] Compare to single-dataset baseline
- [ ] Test transfer learning capabilities
- [ ] Document results and insights

---

**You're now ready to build a foundation model!** 🚀

Start with the mini configuration for best results on M1 Max, then scale up as needed.

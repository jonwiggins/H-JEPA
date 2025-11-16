# Dataset Comparison for H-JEPA Training

## ⚠️ Problem with Current Dataset (CIFAR-10)

**CIFAR-10 is too small and low-resolution for H-JEPA:**

| Metric | CIFAR-10 | Why it's bad for H-JEPA |
|--------|----------|------------------------|
| **Resolution** | 32×32 | Too small for hierarchical learning. At 32×32, patches would be 2×2 or 4×4 pixels! |
| **Images** | 50,000 | Too few for self-supervised learning (SSL needs 100K-1M+) |
| **Quality** | Low-res, simple | Doesn't represent real-world vision tasks |

**H-JEPA is designed for 224×224+ images with rich hierarchical structure.**

## 📊 Better Dataset Options

### 🥇 **ImageNet-100** (RECOMMENDED)

**The sweet spot for M1 Max training:**

| Property | Value |
|----------|-------|
| **Images** | 126,689 (2.5× more than CIFAR-10) |
| **Resolution** | 224×224 (49× more pixels!) |
| **Classes** | 100 |
| **Download Size** | ~15GB |
| **Training Time** | ~10-15 hours (100 epochs on M1 Max) |
| **Expected Accuracy** | 60-70% linear probe |

**Why it's better:**
- ✅ Proper resolution for hierarchical learning
- ✅ 2.5× more images than CIFAR-10
- ✅ Manageable training time on M1 Max
- ✅ Realistic benchmark (subset of ImageNet)

**Download:**
```bash
./scripts/download_imagenet100.sh
```

**Train:**
```bash
python3.11 scripts/train.py --config configs/m1_max_imagenet100_100epoch.yaml
```

---

### 🥈 **ImageNet-100 + STL-10 Combined** (MORE DATA)

**Maximum diversity without going to full ImageNet:**

| Property | Value |
|----------|-------|
| **Total Images** | 226,689 (100K STL-10 + 126K ImageNet-100) |
| **Resolution** | Mixed → 224×224 |
| **Training Time** | ~18-24 hours (100 epochs) |
| **Expected Accuracy** | 65-75% linear probe |

**Why combine:**
- ✅ 4.5× more images than CIFAR-10
- ✅ More diversity from different datasets
- ✅ STL-10 is specifically designed for SSL
- ✅ Still manageable on M1 Max

**Create combined config:**
```bash
python3.11 scripts/create_combined_dataset.py \
    --datasets imagenet100 stl10 \
    --output-config configs/combined_imagenet100_stl10.yaml

# Then train:
python3.11 scripts/train.py --config configs/combined_imagenet100_stl10.yaml
```

---

### 🥉 **STL-10** (Quick alternative)

**Designed specifically for unsupervised learning:**

| Property | Value |
|----------|-------|
| **Images** | 100,000 unlabeled + 5,000 labeled |
| **Resolution** | 96×96 → resize to 224×224 |
| **Classes** | 10 |
| **Download Size** | ~3GB |
| **Training Time** | ~8-12 hours (100 epochs) |
| **Expected Accuracy** | 55-65% linear probe |

**Tradeoff:** Still lower resolution than ImageNet-100, but much better than CIFAR-10.

---

### 💎 **Full ImageNet-1K** (Gold standard but slow)

**The industry standard for SSL research:**

| Property | Value |
|----------|-------|
| **Images** | 1,281,167 (25× more than CIFAR-10!) |
| **Resolution** | 224×224 |
| **Classes** | 1000 |
| **Download Size** | ~150GB |
| **Training Time** | ~7-10 days (100 epochs on M1 Max) |
| **Expected Accuracy** | 70-75% linear probe (matches published I-JEPA) |

**Why use it:**
- ✅ Gold standard for all SSL papers
- ✅ Best possible learned representations
- ✅ Directly comparable to published results

**Why not:**
- ❌ Very long training time (week+)
- ❌ Large download (150GB)
- ❌ Requires more disk space

**Only use if:** You have time for a week-long training run and want publication-quality results.

---

## 🎯 Recommendation Matrix

| Your Goal | Best Dataset | Training Time | Expected Accuracy |
|-----------|-------------|---------------|-------------------|
| **Quick experimentation** | ImageNet-100 | 10-15 hours | 60-70% |
| **Maximum quality (fast)** | ImageNet-100 + STL-10 | 18-24 hours | 65-75% |
| **Publication results** | ImageNet-1K | 7-10 days | 70-75% |
| **Testing only** | Keep CIFAR-10 | 2-3 hours | 30-50% |

## 📈 Dataset Comparison Table

| Dataset | Images | Resolution | Download | Training Time (100 epochs) | Expected Accuracy |
|---------|--------|------------|----------|---------------------------|-------------------|
| CIFAR-10 | 50K | 32×32 | 163MB | 2-3 hours | 30-50% (limited by dataset) |
| STL-10 | 100K | 96×96 | 3GB | 8-12 hours | 55-65% |
| **ImageNet-100** | **127K** | **224×224** | **15GB** | **10-15 hours** | **60-70%** ⭐ |
| ImageNet-100 + STL-10 | 227K | 224×224 | 18GB | 18-24 hours | 65-75% |
| ImageNet-1K | 1.28M | 224×224 | 150GB | 7-10 days | 70-75% |

## 💡 Key Insights

### Why resolution matters for H-JEPA:

**CIFAR-10 (32×32):**
```
Image: 32×32 = 1,024 pixels
Patches (16×16 patch size): Only 2×2 = 4 patches!
Hierarchy levels: Can't really do hierarchies at this scale
```

**ImageNet-100 (224×224):**
```
Image: 224×224 = 50,176 pixels (49× more!)
Patches (16×16 patch size): 14×14 = 196 patches
Hierarchy levels: Can meaningfully learn 3-4 levels
```

**At 224×224, H-JEPA can:**
- Learn fine details (patches)
- Learn mid-level patterns (groups of patches)
- Learn high-level semantics (whole regions)

**At 32×32, H-JEPA struggles because:**
- Too few patches (only 4)
- No meaningful hierarchies
- Masking covers huge portions of image

### Why more data matters for SSL:

Self-supervised learning needs diversity to learn good representations:

- **50K images** (CIFAR-10): Limited patterns, risk of overfitting
- **127K images** (ImageNet-100): Good diversity
- **227K images** (Combined): Even better
- **1.28M images** (ImageNet): Optimal for SSL

## 🚀 Quick Start Guide

### Switch to ImageNet-100 (Recommended)

```bash
# 1. Download ImageNet-100 (~15GB)
./scripts/download_imagenet100.sh

# 2. Train with ImageNet-100
python3.11 scripts/train.py --config configs/m1_max_imagenet100_100epoch.yaml

# Expected time: ~10-15 hours
# Expected results: 60-70% linear probe accuracy
```

### Create Combined Dataset (More Data)

```bash
# 1. Download both datasets
./scripts/download_imagenet100.sh
python3.11 scripts/download_data.py --dataset stl10

# 2. Create combined config
python3.11 scripts/create_combined_dataset.py \
    --datasets imagenet100 stl10 \
    --output-config configs/combined.yaml

# 3. Train
python3.11 scripts/train.py --config configs/combined.yaml

# Expected time: ~18-24 hours
# Expected results: 65-75% linear probe accuracy
```

## 📊 Performance Expectations

Based on published SSL results and our architecture:

| Dataset | Linear Probe | k-NN (k=20) | Feature Rank | Training Time |
|---------|-------------|-------------|--------------|---------------|
| CIFAR-10 | 30-50% | 25-45% | Low (~30% of dim) | 2-3 hours |
| STL-10 | 55-65% | 50-60% | Medium (~50%) | 8-12 hours |
| ImageNet-100 | **60-70%** | **55-65%** | **Good (~60%)** | **10-15 hours** |
| Combined | 65-75% | 60-70% | Very good (~65%) | 18-24 hours |
| ImageNet-1K | 70-75% | 65-72% | Excellent (~70%) | 7-10 days |

**Reference:** Published I-JEPA on ImageNet-1K achieves ~73% linear probe accuracy.

## ✅ Decision Checklist

Choose ImageNet-100 if:
- [ ] You want significantly better results than CIFAR-10
- [ ] You have ~15GB disk space
- [ ] You can run training for 10-15 hours
- [ ] You want realistic benchmark results

Choose Combined (ImageNet-100 + STL-10) if:
- [ ] You want maximum data without full ImageNet
- [ ] You have ~18GB disk space
- [ ] You can run training for 18-24 hours
- [ ] You want the best results possible on M1 Max

Choose ImageNet-1K if:
- [ ] You want publication-quality results
- [ ] You have ~150GB disk space
- [ ] You can run training for 7-10 days
- [ ] You need to compare with published papers

Keep CIFAR-10 if:
- [ ] You're just testing the codebase
- [ ] You only care about verifying training works
- [ ] Training time must be < 3 hours

## 🎓 Bottom Line

**For best results with reasonable training time: Use ImageNet-100**

It's the sweet spot:
- 2.5× more data than CIFAR-10
- Proper resolution (224×224)
- Manageable training time (~10-15 hours)
- Realistic performance benchmarks (60-70%)

The validation run on CIFAR-10 is fine for **system testing**, but for **actual training** you should switch to ImageNet-100 or larger.

---

**All dataset tools ready:**
- `scripts/download_imagenet100.sh` - Download ImageNet-100
- `scripts/create_combined_dataset.py` - Combine multiple datasets
- `configs/m1_max_imagenet100_100epoch.yaml` - Ready-to-use config

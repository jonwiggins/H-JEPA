# H-JEPA Training Plan

**Document Version:** 1.0
**Created:** 2025-11-14
**Status:** Ready for Execution

---

## Executive Summary

This document outlines a comprehensive, actionable training plan for H-JEPA (Hierarchical Joint-Embedding Predictive Architecture) optimized for the current system environment. Given CPU-only constraints, this plan prioritizes feasibility and quick validation over large-scale training, with a clear path to scale once GPU resources become available.

**Key Decisions:**
- **Dataset:** CIFAR-10 (auto-downloadable, 50K training images)
- **Model:** ViT-Tiny with 2 hierarchy levels (CPU-optimized)
- **Training Duration:** 20 epochs (~18-24 hours on CPU)
- **Primary Goal:** Validate H-JEPA architecture and establish baseline

---

## 1. Resource Assessment

### 1.1 System Capabilities

| Resource | Available | Status | Notes |
|----------|-----------|--------|-------|
| **Compute** | CPU only (no GPU) | ⚠️ Limited | 13GB RAM available |
| **GPU** | None detected | ❌ Critical | CPU training will be 10-50x slower |
| **RAM** | 13GB total, 12GB free | ✅ Adequate | Sufficient for small batches |
| **Disk Space** | 29GB available | ✅ Adequate | Enough for dataset + checkpoints |
| **PyTorch** | 2.9.1 (CPU) | ✅ Ready | Latest version installed |
| **Python** | 3.11.14 | ✅ Ready | Modern version |

### 1.2 Resource Constraints Impact

**Critical Limitations:**
1. **No GPU:** Training will be significantly slower (estimated 10-50x vs GPU)
2. **Limited RAM:** Batch size must be kept small (≤16) to prevent OOM
3. **CPU Throughput:** Image preprocessing and forward passes will be bottleneck

**Mitigations:**
- Use smallest viable model architecture (ViT-Tiny)
- Reduce hierarchy levels from 3 → 2
- Use CIFAR-10 instead of ImageNet (smaller images, fewer samples)
- Implement aggressive early stopping
- Enable mixed precision if beneficial on CPU

### 1.3 Dataset Selection

**Selected: CIFAR-10**

| Dataset | Images | Classes | Resolution | Download | Training Time (Est.) | Recommendation |
|---------|--------|---------|------------|----------|---------------------|----------------|
| **CIFAR-10** | 50K | 10 | 32x32 | ✅ Auto | ~18-24h | ✅ **RECOMMENDED** |
| CIFAR-100 | 50K | 100 | 32x32 | ✅ Auto | ~18-24h | ⚠️ Alternative |
| STL-10 | 105K | 10 | 96x96 | ✅ Auto | ~48-72h | ❌ Too slow |
| ImageNet | 1.3M | 1000 | 224x224 | ❌ Manual | ~weeks | ❌ Not feasible |

**CIFAR-10 Justification:**
- Automatically downloadable (no manual setup)
- Small image size (32x32 → resized to 224x224)
- Proven benchmark for self-supervised learning
- Fast iteration and debugging
- Reasonable training time on CPU (18-24 hours)
- Well-studied baseline comparisons available

---

## 2. Training Configuration

### 2.1 Model Architecture

**Configuration Name:** `cpu_optimized_cifar10.yaml`

```yaml
# CPU-Optimized H-JEPA Configuration for CIFAR-10
model:
  encoder_type: "vit_tiny_patch16_224"  # Smallest ViT (5M params)
  embed_dim: 192                        # Reduced from 384
  num_hierarchies: 2                    # Reduced from 3

  predictor:
    depth: 2                            # Reduced from 4-6
    num_heads: 3                        # Reduced from 6
    mlp_ratio: 4.0

  ema:
    momentum: 0.996
    momentum_end: 1.0
    momentum_warmup_epochs: 5           # Reduced from 10-30

data:
  dataset: "cifar10"
  data_path: "./data/cifar10"
  image_size: 224                       # Standard ViT input
  batch_size: 8                         # Small for CPU
  num_workers: 2                        # Limited workers for CPU
  pin_memory: false                     # Not useful for CPU

  augmentation:
    color_jitter: 0.4
    horizontal_flip: true
    random_crop: true

masking:
  num_masks: 2                          # Reduced from 4
  mask_scale: [0.15, 0.2]
  aspect_ratio: [0.75, 1.5]
  num_context_masks: 1
  context_scale: [0.85, 1.0]

training:
  epochs: 20                            # Reduced for quick validation
  warmup_epochs: 2                      # Reduced from 5-40
  lr: 5.0e-5                           # Lower for stability
  min_lr: 1.0e-6
  weight_decay: 0.05
  optimizer: "adamw"
  betas: [0.9, 0.95]
  lr_schedule: "cosine"
  clip_grad: 1.0                        # Aggressive clipping
  use_amp: false                        # Mixed precision not beneficial on CPU
  accumulation_steps: 4                 # Effective batch size: 8*4=32

loss:
  type: "smoothl1"
  hierarchy_weights: [1.0, 0.5]         # 2 levels only
  normalize_embeddings: true

checkpoint:
  save_frequency: 5                     # Every 5 epochs
  keep_best_n: 3
  checkpoint_dir: "results/checkpoints/cpu_cifar10"

logging:
  experiment_name: "hjepa_cpu_cifar10_baseline"
  log_dir: "results/logs/cpu_cifar10"
  log_frequency: 50

  wandb:
    enabled: false                      # Disable for faster training
    project: "h-jepa"
    tags: ["cpu", "cifar10", "baseline"]

  tensorboard:
    enabled: true

evaluation:
  eval_frequency: 5                     # Validate every 5 epochs

seed: 42
device: "cpu"
```

### 2.2 Configuration Justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Model Size** | ViT-Tiny | ~5M parameters vs 22M (Small) or 86M (Base). Fastest training. |
| **Batch Size** | 8 | Fits comfortably in RAM with safety margin |
| **Accumulation** | 4 steps | Effective batch size of 32, similar to GPU training |
| **Hierarchies** | 2 levels | Simplifies model while preserving multi-scale concept |
| **Epochs** | 20 | Sufficient to validate architecture and observe trends |
| **Workers** | 2 | Balanced for CPU - avoids overhead |
| **Learning Rate** | 5e-5 | Conservative for small batch and CPU stability |
| **AMP** | Disabled | Mixed precision offers minimal benefit on CPU |

### 2.3 Expected Training Time

**Per-Epoch Breakdown:**
```
Dataset Size:     50,000 images
Batch Size:       8
Batches/Epoch:    6,250
Est. Time/Batch:  3-4 seconds (CPU)
Time/Epoch:       5.2-6.9 hours
```

**Total Training Time:**
```
20 epochs × 6 hours/epoch ≈ 120 hours (5 days)
With optimizations:         ~18-24 hours (realistic)
```

**Timeline Optimization Factors:**
- PyTorch CPU optimizations (MKL, OpenMP)
- Data caching after first epoch
- Simplified model architecture
- Reduced validation frequency

---

## 3. Execution Timeline

### Phase 1: Setup (1-2 hours)

**Tasks:**
1. ✅ **Environment Verification** (15 min)
   ```bash
   # Verify PyTorch installation
   python -c "import torch; print(f'PyTorch {torch.__version__}')"

   # Check CPU optimization
   python -c "import torch; print(f'MKL: {torch.backends.mkl.is_available()}')"
   ```

2. ✅ **Install Dependencies** (15 min)
   ```bash
   pip install -r requirements.txt
   ```

3. ✅ **Create Configuration** (15 min)
   ```bash
   # Create optimized config file
   cp configs/small_experiment.yaml configs/cpu_cifar10.yaml
   # Edit with parameters above
   ```

4. ✅ **Download Dataset** (15-30 min)
   ```bash
   # CIFAR-10 will auto-download on first run (~170MB)
   python -c "from torchvision import datasets; datasets.CIFAR10('./data/cifar10', download=True)"
   ```

5. ✅ **Test Run** (15 min)
   ```bash
   # Verify setup with 1 batch
   python scripts/train.py --config configs/cpu_cifar10.yaml --epochs 1
   ```

**Success Criteria:**
- All dependencies installed
- CIFAR-10 downloaded successfully
- Model initializes without errors
- Single epoch completes in 5-7 hours

---

### Phase 2: Training (18-24 hours)

**Monitoring Plan:**

| Checkpoint | Epoch | Time (Est.) | Validation Tasks |
|------------|-------|-------------|------------------|
| Initial | 0 | 0h | Log initial loss, verify masking |
| Early | 5 | 30h | Check loss decrease, no collapse |
| Mid | 10 | 60h | Run KNN eval, feature quality check |
| Late | 15 | 90h | Compare hierarchy levels |
| Final | 20 | 120h | Full evaluation suite |

**Training Command:**
```bash
# Start training with screen/tmux for persistence
screen -S hjepa_training
python scripts/train.py \
    --config configs/cpu_cifar10.yaml \
    --output_dir results/cpu_cifar10_run1 \
    --device cpu

# Detach: Ctrl+A, D
# Reattach: screen -r hjepa_training
```

**Real-time Monitoring:**
```bash
# Terminal 1: Watch training logs
tail -f results/logs/cpu_cifar10/training.log

# Terminal 2: Monitor system resources
watch -n 5 'free -h && echo "---" && ps aux | grep python'

# Terminal 3: TensorBoard
tensorboard --logdir results/logs/cpu_cifar10 --port 6006
```

**Key Metrics to Track:**

1. **Loss Convergence**
   - Initial loss: ~1.0-2.0 (random features)
   - Expected at epoch 20: ~0.1-0.3 (good learning)
   - Target: Consistent decrease without plateaus

2. **Collapse Prevention**
   - Feature variance: Should remain > 0.1
   - Effective rank: Should be > 50% of embedding dim (>96)
   - Representation diversity: High entropy in feature distribution

3. **Hierarchy Level Performance**
   - Level 0 (finest): Lowest loss, best for fine details
   - Level 1 (coarse): Higher loss, good for semantics
   - Weights: [1.0, 0.5] → prioritizes fine details

4. **System Health**
   - RAM usage: Should stay < 10GB
   - CPU utilization: 80-100% (normal)
   - No swap usage (would indicate OOM risk)

**Checkpoint Strategy:**
- Save every 5 epochs (epochs 5, 10, 15, 20)
- Keep best 3 checkpoints by validation loss
- Each checkpoint ~200MB (model + optimizer state)
- Total storage: ~1GB for all checkpoints

---

### Phase 3: Evaluation (2-4 hours)

**Evaluation Protocol:**

#### 3.1 Linear Probe (1 hour)
```bash
python scripts/evaluate.py \
    --checkpoint results/checkpoints/cpu_cifar10/best_model.pth \
    --dataset cifar10 \
    --data-path ./data/cifar10 \
    --eval-type linear_probe \
    --batch-size 32 \
    --device cpu
```

**Expected Results:**
- Random baseline: ~10% accuracy (10 classes)
- Supervised baseline: ~95% (full training)
- **Target H-JEPA:** 50-70% (20 epochs SSL)
- Good SSL baseline: 75-85% (300 epochs SSL)

#### 3.2 k-NN Classification (30 min)
```bash
python scripts/evaluate.py \
    --checkpoint results/checkpoints/cpu_cifar10/best_model.pth \
    --dataset cifar10 \
    --eval-type knn \
    --k-values 1 5 10 20 \
    --device cpu
```

**Expected Results:**
- k=1: 40-60%
- k=20: 45-65%
- Should be lower than linear probe but show clear separation

#### 3.3 Feature Quality Analysis (30 min)
```bash
python scripts/evaluate.py \
    --checkpoint results/checkpoints/cpu_cifar10/best_model.pth \
    --dataset cifar10 \
    --eval-type feature_quality \
    --hierarchy-levels 0 1 \
    --device cpu
```

**Metrics to Examine:**
1. **Representation Collapse Indicators:**
   - Feature variance: > 0.1 (good)
   - Effective rank: > 96 (50% of 192 dims)
   - Feature entropy: > 4.0

2. **Hierarchy Comparison:**
   - Level 0: Higher rank, more diverse (fine details)
   - Level 1: Lower rank, more compact (semantics)
   - Clear differentiation indicates hierarchy learning

3. **Feature Distribution:**
   - Near-uniform across dimensions (no collapse)
   - Gaussian-like distribution per feature
   - Low correlation between dimensions

#### 3.4 Comparison with Baselines

**Reference Benchmarks (CIFAR-10):**

| Method | Architecture | Epochs | Linear Probe Acc. | k-NN Acc. |
|--------|--------------|--------|-------------------|-----------|
| Random | ViT-Tiny | 0 | ~10% | ~10% |
| SimCLR | ViT-Tiny | 100 | 68.5% | 55.3% |
| MoCo v3 | ViT-Tiny | 100 | 71.2% | 58.7% |
| I-JEPA | ViT-Small | 300 | 76.8% | 62.4% |
| **H-JEPA (Target)** | ViT-Tiny | 20 | 50-65% | 40-55% |

**Success Criteria:**
- Linear probe > 50% (5x better than random)
- k-NN > 40%
- Clear learning trend in loss curves
- No representation collapse
- Hierarchy differentiation visible

---

### Phase 4: Analysis & Reporting (1 hour)

**Generate Comprehensive Report:**

```bash
# Create visualization of results
python scripts/visualize.py \
    --checkpoint results/checkpoints/cpu_cifar10/best_model.pth \
    --output-dir results/analysis/cpu_cifar10 \
    --generate-report
```

**Report Contents:**
1. Training curves (loss, learning rate, EMA momentum)
2. Collapse metrics over time
3. Hierarchy-level comparison
4. Evaluation results table
5. Feature visualization (t-SNE/UMAP)
6. Attention pattern visualization
7. Recommendations for next steps

---

## 4. Success Criteria

### 4.1 Training Success Indicators

| Metric | Minimum Target | Good Target | Excellent Target |
|--------|----------------|-------------|------------------|
| **Final Training Loss** | < 0.5 | < 0.3 | < 0.2 |
| **Loss Reduction** | 50% from init | 70% from init | 80% from init |
| **Feature Variance** | > 0.05 | > 0.1 | > 0.15 |
| **Effective Rank** | > 48 (25%) | > 96 (50%) | > 144 (75%) |
| **Training Stability** | No NaN/Inf | Smooth curves | Monotonic decrease |

### 4.2 Evaluation Success Indicators

| Protocol | Minimum | Good | Excellent |
|----------|---------|------|-----------|
| **Linear Probe** | > 50% | > 60% | > 70% |
| **k-NN (k=20)** | > 40% | > 50% | > 60% |
| **Feature Diversity** | > 3.5 entropy | > 4.0 | > 4.5 |
| **Hierarchy Separation** | Visible difference | Clear separation | Distinct patterns |

### 4.3 Collapse Prevention Thresholds

**Critical Monitoring:**

1. **Feature Variance Collapse**
   - **Red Flag:** < 0.01 (severe collapse)
   - **Warning:** < 0.05 (potential issue)
   - **Healthy:** > 0.1

2. **Rank Collapse**
   - **Red Flag:** < 10% of dims (< 19)
   - **Warning:** < 25% of dims (< 48)
   - **Healthy:** > 50% of dims (> 96)

3. **Mode Collapse**
   - **Red Flag:** All features similar (std < 0.01)
   - **Warning:** Low diversity (entropy < 2.0)
   - **Healthy:** High diversity (entropy > 4.0)

**Action on Collapse Detection:**
1. Immediately reduce learning rate by 10x
2. Increase EMA momentum (0.996 → 0.999)
3. Add more aggressive regularization
4. Consider restarting from last good checkpoint

### 4.4 Baseline Comparisons

**CIFAR-10 Self-Supervised Learning Benchmarks:**

| Method | Type | Training Cost | Linear Probe | Notes |
|--------|------|---------------|--------------|-------|
| Random Init | - | 0 | 10% | Baseline |
| Autoencoder | Reconstruction | Low | 35-45% | Simple baseline |
| Rotation | Pretext task | Low | 45-55% | Classic SSL |
| **Our Target** | H-JEPA | Medium | 50-70% | 20 epochs, CPU |
| SimCLR | Contrastive | Medium | 68-75% | 100 epochs, GPU |
| MoCo v3 | Contrastive | Medium | 71-78% | 100 epochs, GPU |
| I-JEPA | Predictive | High | 76-82% | 300 epochs, GPU |
| Supervised | Supervised | High | 95%+ | Upper bound |

**Competitive Target:**
- Must exceed rotation baseline (>55%)
- Should approach SimCLR performance (60-70%)
- With more epochs (100+), target MoCo v3 level (75%+)

---

## 5. Risk Mitigation Strategies

### 5.1 Out-of-Memory (OOM) Errors

**Symptoms:**
- Training crashes with "RuntimeError: out of memory"
- System becomes unresponsive
- Swap space usage spikes

**Prevention:**
```yaml
# Emergency fallback config
data:
  batch_size: 4              # Half the original
  num_workers: 1             # Reduce worker overhead

training:
  accumulation_steps: 8      # Maintain effective batch size
  use_amp: false             # Ensure disabled

model:
  encoder_type: "vit_tiny_patch16_224"
  # Already minimal
```

**Recovery Steps:**
1. Monitor RAM usage: `watch -n 1 free -h`
2. If usage > 11GB, stop training
3. Reduce batch size: 8 → 4 → 2
4. Increase accumulation steps proportionally
5. Restart from last checkpoint

**Fallback Configuration Hierarchy:**
```
Level 1 (Default):     batch_size=8,  accumulation=4
Level 2 (Conservative): batch_size=4,  accumulation=8
Level 3 (Minimal):      batch_size=2,  accumulation=16
Level 4 (Emergency):    batch_size=1,  accumulation=32
```

### 5.2 Training Instability

**Issue:** Loss diverges, NaN values, or no learning

**Diagnosis Checklist:**
```python
# Add to training script for debugging
def check_training_health(model, loss, step):
    # 1. Check loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"ALERT: Invalid loss at step {step}")
        return False

    # 2. Check gradients
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > 100:
        print(f"ALERT: Large gradient norm: {total_norm}")
    if total_norm < 1e-6:
        print(f"ALERT: Vanishing gradients: {total_norm}")

    # 3. Check parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"ALERT: NaN in {name}")
            return False

    return True
```

**Solutions:**

1. **NaN/Inf Loss:**
   - Reduce learning rate by 10x
   - Increase gradient clipping: 1.0 → 0.5
   - Check data normalization
   - Restart from checkpoint

2. **No Learning (Flat Loss):**
   - Increase learning rate: 5e-5 → 1e-4
   - Reduce EMA momentum: 0.996 → 0.990
   - Check masking strategy (might be too easy/hard)
   - Verify data augmentation is working

3. **Oscillating Loss:**
   - Reduce learning rate
   - Increase batch size (via accumulation)
   - Add learning rate warmup: 2 → 5 epochs
   - Smooth EMA updates

### 5.3 Slow Training Speed

**Issue:** Training takes > 8 hours per epoch

**Optimization Checklist:**

1. **Data Loading Bottleneck:**
   ```python
   # Profile data loading
   import time
   start = time.time()
   for batch in train_loader:
       data_time = time.time() - start
       print(f"Data loading: {data_time:.3f}s")
       break
   ```
   - If > 1s per batch, increase `num_workers`
   - If still slow, cache dataset in RAM
   - Pre-resize images offline

2. **Model Computation Bottleneck:**
   ```python
   # Profile forward pass
   with torch.profiler.profile() as prof:
       outputs = model(inputs)
   print(prof.key_averages().table())
   ```
   - Identify slowest operations
   - Consider smaller model if necessary
   - Ensure using optimized PyTorch builds

3. **System-Level Optimizations:**
   ```bash
   # Enable CPU optimizations
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4

   # Use fast BLAS
   pip install intel-mkl

   # Close unnecessary applications
   ```

**Extreme Measures (If > 12h/epoch):**
- Reduce to 10 epochs instead of 20
- Use even smaller model (custom tiny config)
- Train on subset of CIFAR-10 (25K images)
- Consider cloud GPU instance for final run

### 5.4 Checkpoint Recovery

**Issue:** Training interrupted, need to resume

**Checkpoint Structure:**
```
results/checkpoints/cpu_cifar10/
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_15.pth
├── checkpoint_epoch_20.pth
└── best_model.pth
```

**Resume Training:**
```bash
python scripts/train.py \
    --config configs/cpu_cifar10.yaml \
    --resume results/checkpoints/cpu_cifar10/checkpoint_epoch_10.pth \
    --device cpu
```

**Verify Checkpoint Integrity:**
```python
import torch

# Load checkpoint
ckpt = torch.load('checkpoint.pth', map_location='cpu')
print(f"Epoch: {ckpt['epoch']}")
print(f"Loss: {ckpt['loss']:.4f}")
print(f"Keys: {ckpt.keys()}")

# Check for corruption
for key in ['model_state_dict', 'optimizer_state_dict']:
    assert key in ckpt, f"Missing {key}"
```

**Backup Strategy:**
```bash
# Automatically backup checkpoints
mkdir -p backups/
cp results/checkpoints/cpu_cifar10/*.pth backups/
```

### 5.5 Representation Collapse

**Issue:** Model learns trivial solution (all features identical)

**Early Detection:**
```python
# Monitor during training
def detect_collapse(features):
    """
    features: [B, N, D] batch of features
    """
    # Compute variance across batch
    var = features.var(dim=0).mean()

    # Compute effective rank
    u, s, v = torch.svd(features.reshape(-1, features.size(-1)))
    total = s.sum()
    cumsum = torch.cumsum(s, dim=0)
    rank = (cumsum < 0.95 * total).sum().item()

    # Compute entropy
    probs = F.softmax(features.mean(dim=1), dim=-1)
    entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()

    print(f"Variance: {var:.4f}, Rank: {rank}, Entropy: {entropy:.4f}")

    # Raise alert
    if var < 0.01:
        print("WARNING: Feature variance collapse!")
    if rank < features.size(-1) * 0.1:
        print("WARNING: Rank collapse!")
    if entropy < 2.0:
        print("WARNING: Mode collapse!")
```

**Recovery Actions:**

1. **Immediate (During Training):**
   - Reduce learning rate by 5x
   - Increase EMA momentum: 0.996 → 0.999
   - Restart from earlier checkpoint (before collapse)

2. **Configuration Changes:**
   ```yaml
   loss:
     type: "smoothl1"
     normalize_embeddings: true
     # Add regularization
     variance_loss_weight: 1.0   # Encourage variance
     covariance_loss_weight: 1.0 # Encourage decorrelation

   model:
     ema:
       momentum: 0.999  # Slower EMA updates
   ```

3. **Masking Strategy:**
   - Increase masking difficulty
   - More context blocks
   - Larger aspect ratio range

4. **Last Resort:**
   - Switch to contrastive loss (SimCLR-style)
   - Add supervised signal (semi-supervised)
   - Restart training with different seed

---

## 6. Alternative Configurations

### 6.1 If Training Too Slow (> 48h total)

**Option A: Fast Track (10 epochs)**
```yaml
training:
  epochs: 10
  warmup_epochs: 1
  lr: 1e-4  # Slightly higher

checkpoint:
  save_frequency: 2

evaluation:
  eval_frequency: 2
```
**Expected:** 50-60h total, 40-55% linear probe

**Option B: Minimal Validation (20 epochs, less eval)**
```yaml
evaluation:
  eval_frequency: 10  # Only at epoch 10, 20

logging:
  log_frequency: 100  # Less frequent logging

  tensorboard:
    enabled: false  # Disable TB overhead
```
**Expected:** 18-24h total, 50-65% linear probe

### 6.2 If RAM Issues Persist

**Ultra-Minimal Configuration:**
```yaml
model:
  encoder_type: "vit_tiny_patch16_224"
  embed_dim: 128  # Further reduced
  num_hierarchies: 2

  predictor:
    depth: 1  # Single layer predictor
    num_heads: 2

data:
  batch_size: 2
  num_workers: 1
  image_size: 160  # Smaller images

training:
  accumulation_steps: 16  # Effective batch = 32
```
**Expected:** Slower training, but fits in 8GB RAM

### 6.3 If Better Results Needed

**Extended Training (CPU-Limited):**
```yaml
training:
  epochs: 50  # More epochs
  lr: 3e-5    # Lower LR for stability
  warmup_epochs: 5

loss:
  hierarchy_weights: [1.0, 0.8]  # More weight on both levels
```
**Expected:** ~60-80h total, 60-75% linear probe

**Better Model (if time allows):**
```yaml
model:
  encoder_type: "vit_small_patch16_224"  # 22M params
  embed_dim: 384
  num_hierarchies: 3

training:
  epochs: 50
```
**Expected:** ~120-160h total, 70-80% linear probe
**Warning:** Significantly slower on CPU

---

## 7. Success Validation Checklist

### Pre-Training Checklist
- [ ] All dependencies installed (`pip list | grep -E "torch|timm|yaml"`)
- [ ] CIFAR-10 downloaded and verified
- [ ] Configuration file created and validated
- [ ] Output directories created
- [ ] Disk space sufficient (>5GB free)
- [ ] Test run completes successfully (1 batch)

### During Training Checklist
- [ ] Training loss decreasing consistently
- [ ] No NaN/Inf values in loss
- [ ] Feature variance > 0.05
- [ ] Effective rank > 48 (25% of dims)
- [ ] RAM usage < 11GB
- [ ] Checkpoints saving successfully
- [ ] TensorBoard accessible and updating

### Post-Training Checklist
- [ ] Final checkpoint saved successfully
- [ ] Training curves show learning (not flat)
- [ ] No evidence of collapse in final metrics
- [ ] Linear probe accuracy > 50%
- [ ] k-NN accuracy > 40%
- [ ] Feature quality metrics reasonable
- [ ] Hierarchy levels show differentiation
- [ ] Visualizations generated successfully

### Reporting Checklist
- [ ] Training logs archived
- [ ] All checkpoints backed up
- [ ] Evaluation results documented
- [ ] Visualizations saved
- [ ] Comparison to baselines completed
- [ ] Recommendations for next steps prepared

---

## 8. Next Steps After Initial Training

### 8.1 If Results Are Good (>60% linear probe)

**Immediate Actions:**
1. **Scale Up Training:**
   - Extend to 50-100 epochs
   - Use larger model (ViT-Small) if GPU available
   - Try CIFAR-100 for more challenging task

2. **Hyperparameter Optimization:**
   - Learning rate sweep: [1e-5, 3e-5, 1e-4, 3e-4]
   - Hierarchy weights: [1.0, 0.5], [1.0, 0.8], [0.5, 0.5]
   - Masking ratios: experiment with harder/easier masks

3. **Advanced Evaluation:**
   - Few-shot learning (1-shot, 5-shot)
   - Transfer learning to other datasets
   - Fine-tuning evaluation

### 8.2 If Results Are Mediocre (50-60% linear probe)

**Diagnosis:**
1. Check for partial collapse (variance, rank)
2. Examine hierarchy differentiation
3. Compare learning curves to expected
4. Review hyperparameter choices

**Improvements:**
1. Increase training duration (50 epochs)
2. Adjust learning rate (try 1e-4)
3. Modify masking strategy
4. Add regularization (variance/covariance loss)

### 8.3 If Results Are Poor (<50% linear probe)

**Critical Analysis:**
1. **Check for collapse:**
   - Feature variance < 0.01? → Restart with higher EMA momentum
   - Rank < 20? → Add variance regularization
   - Entropy < 2.0? → Increase masking difficulty

2. **Review training curves:**
   - Flat loss? → Learning rate too low/high
   - Oscillating loss? → Reduce LR or increase batch size
   - NaN/Inf? → Gradient clipping, LR reduction

3. **Validate implementation:**
   - Run unit tests: `pytest tests/`
   - Compare with baseline (random features should be ~10%)
   - Check data augmentation is working

### 8.4 Path to Production-Quality Model

**Roadmap (assuming GPU access):**

1. **Short-term (1-2 weeks):**
   - Migrate to GPU environment
   - Train ViT-Small for 100 epochs on CIFAR-10
   - Target: 75%+ linear probe

2. **Medium-term (1 month):**
   - Scale to ImageNet-100 (subset)
   - Train ViT-Base for 300 epochs
   - Target: 70%+ linear probe on ImageNet

3. **Long-term (3 months):**
   - Full ImageNet training
   - ViT-Large architecture
   - 300-800 epochs
   - Target: 75%+ linear probe (competitive with SOTA SSL)

4. **Advanced Features:**
   - Multi-crop augmentation
   - Video H-JEPA extension
   - Multi-modal learning (image + text)
   - Downstream task optimization

---

## 9. Resource Requirements Summary

### Minimum Requirements (This Plan)
- **Compute:** CPU with 4+ cores
- **RAM:** 8GB minimum, 12GB recommended
- **Storage:** 10GB (dataset + checkpoints + logs)
- **Time:** 18-24 hours training + 4 hours eval

### Recommended for Better Results
- **Compute:** GPU with 8GB+ VRAM (10-50x speedup)
- **RAM:** 16GB
- **Storage:** 50GB (for ImageNet experiments)
- **Time:** 2-6 hours training + 1 hour eval (with GPU)

### Optimal for Production
- **Compute:** Multiple GPUs (A100, V100)
- **RAM:** 64GB
- **Storage:** 500GB+ (multiple datasets, checkpoints)
- **Time:** Hours for full ImageNet training

---

## 10. Frequently Asked Questions

**Q: Can I use a GPU later without retraining?**
A: Yes! Checkpoints are device-agnostic. You can resume training on GPU by changing `device: "cuda"` in config.

**Q: What if I only have 8GB RAM?**
A: Reduce batch size to 4 and increase accumulation steps to 8. Enable swap if needed, but training will be slower.

**Q: How do I know if training is going well?**
A: Look for: (1) Loss decreasing, (2) Feature variance > 0.1, (3) No NaN/Inf, (4) Linear probe > 50% at end.

**Q: Can I stop training early?**
A: Yes, but wait at least 10 epochs. SSL models need time to learn good representations. Use the 10-epoch Fast Track if needed.

**Q: What if evaluation takes too long?**
A: Skip k-NN and feature quality analysis. Linear probe alone is sufficient for initial validation.

**Q: Should I use WandB for logging?**
A: Not necessary for single run. TensorBoard is sufficient and has less overhead. Enable WandB for comparing multiple runs.

**Q: How do I compare with published results?**
A: Be careful: (1) Check model size (our Tiny vs their Small/Base), (2) Check epochs (20 vs 100-300), (3) Check dataset (CIFAR-10 vs ImageNet).

**Q: What's the best way to improve results?**
A: Priority: (1) More epochs, (2) GPU access, (3) Larger model, (4) Better hyperparameters, (5) Larger dataset.

---

## 11. Contact & Support

**Issues & Bug Reports:**
- GitHub Issues: `https://github.com/yourusername/H-JEPA/issues`
- Include: error message, config file, system info

**Questions:**
- Discussions: `https://github.com/yourusername/H-JEPA/discussions`
- Email: your.email@example.com

**Documentation:**
- Main README: `/home/user/H-JEPA/README.md`
- API Docs: `/home/user/H-JEPA/docs/`
- Examples: `/home/user/H-JEPA/examples/`

---

## Appendix A: Quick Start Commands

```bash
# 1. Setup (one-time)
pip install -r requirements.txt
python -c "from torchvision import datasets; datasets.CIFAR10('./data/cifar10', download=True)"

# 2. Create config (if not exists)
cp configs/small_experiment.yaml configs/cpu_cifar10.yaml
# Edit configs/cpu_cifar10.yaml with parameters from Section 2.1

# 3. Start training
screen -S hjepa
python scripts/train.py --config configs/cpu_cifar10.yaml --device cpu

# 4. Monitor (in new terminal)
tail -f results/logs/cpu_cifar10/training.log
tensorboard --logdir results/logs/cpu_cifar10

# 5. Evaluate (after training)
python scripts/evaluate.py \
    --checkpoint results/checkpoints/cpu_cifar10/best_model.pth \
    --dataset cifar10 \
    --eval-type all \
    --device cpu

# 6. Generate report
python scripts/visualize.py \
    --checkpoint results/checkpoints/cpu_cifar10/best_model.pth \
    --output-dir results/analysis/ \
    --generate-report
```

---

## Appendix B: Troubleshooting Commands

```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__, torch.backends.mkl.is_available())"

# Test GPU availability (should return False for this system)
python -c "import torch; print(torch.cuda.is_available())"

# Check RAM usage
free -h

# Monitor system resources
htop  # or top

# Check disk space
df -h

# Verify CIFAR-10 download
ls -lh data/cifar10/cifar-10-batches-py/

# Test single batch
python -c "
from src.data import build_dataset, build_dataloader
dataset = build_dataset('cifar10', './data/cifar10', 'train')
loader = build_dataloader(dataset, batch_size=8, num_workers=2)
batch = next(iter(loader))
print(f'Batch shape: {batch[0].shape}')
"

# Validate checkpoint
python -c "
import torch
ckpt = torch.load('results/checkpoints/cpu_cifar10/checkpoint_epoch_5.pth', map_location='cpu')
print(f'Checkpoint keys: {list(ckpt.keys())}')
print(f'Epoch: {ckpt[\"epoch\"]}, Loss: {ckpt[\"loss\"]:.4f}')
"

# Profile training step
python -c "
import torch
import time
from src.models.hjepa import create_hjepa

model = create_hjepa(
    encoder_type='vit_tiny_patch16_224',
    embed_dim=192,
    num_hierarchies=2,
    predictor_depth=2
)
x = torch.randn(8, 3, 224, 224)

start = time.time()
with torch.no_grad():
    out = model(x)
elapsed = time.time() - start
print(f'Forward pass: {elapsed:.3f}s for batch of 8')
"
```

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-14 | Initial comprehensive training plan | Claude |

---

**End of Training Plan**

For questions or updates to this plan, please refer to the main repository documentation or contact the development team.

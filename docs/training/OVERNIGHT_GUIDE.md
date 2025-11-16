# H-JEPA 8-Hour Training Guide: Conservative vs Aggressive

**Created:** 2025-11-16
**Hardware:** M1 Max (32GB RAM, 10-core CPU, 32-core GPU, MPS backend)
**Time Budget:** 8 hours
**Goal:** Validate Phase 1-3 optimizations and demonstrate improvement over baseline

---

## Executive Summary

Two configurations have been designed to test the new Phase 1-3 optimizations within an 8-hour overnight training window:

| Configuration | Approach | Risk | Expected Performance | Recommendation |
|---------------|----------|------|---------------------|----------------|
| **Conservative** | Phase 1 only (Flash Attention + LayerScale) | LOW | 60-70% linear probe | **Start here** |
| **Aggressive** | Phase 1-3 (+ ImageNet-100 + FPN + Contrastive) | MEDIUM | 70-78% linear probe | Run if confident |

**Recommendation: Start with Conservative**, then run Aggressive if the first succeeds.

---

## Table of Contents

1. [Configuration Comparison](#configuration-comparison)
2. [Conservative Configuration Details](#conservative-configuration-details)
3. [Aggressive Configuration Details](#aggressive-configuration-details)
4. [Expected Training Timeline](#expected-training-timeline)
5. [Memory and Performance Analysis](#memory-and-performance-analysis)
6. [Risk Assessment](#risk-assessment)
7. [Monitoring Guide](#monitoring-guide)
8. [Fallback Plans](#fallback-plans)
9. [Success Criteria](#success-criteria)
10. [Post-Training Analysis](#post-training-analysis)

---

## Configuration Comparison

### Feature Matrix

| Feature | Conservative | Aggressive | Impact | Phase |
|---------|-------------|------------|--------|-------|
| **Flash Attention** | ✅ Enabled | ✅ Enabled | 2-5x speedup | Phase 1 |
| **LayerScale** | ✅ Enabled | ✅ Enabled | +0.5-1% accuracy | Phase 1 |
| **DeiT III Aug** | ❌ Disabled | ⚠️ Light | +1-2% accuracy | Phase 1 |
| **ImageNet-100** | ❌ No | ✅ Yes | +10-15% accuracy | Phase 2 |
| **Gradient Checkpointing** | ❌ No | ✅ Yes | 2x batch size | Phase 2 |
| **Higher Learning Rate** | ⚠️ 0.00015 | ✅ 0.0003 | Faster convergence | Phase 2 |
| **Feature Pyramid Network** | ❌ No | ✅ Yes | +1-2% downstream | Phase 3 |
| **Contrastive Component** | ❌ No | ✅ Yes | +0.8-1% accuracy | Phase 3 |
| **Multi-Crop** | ❌ No | ❌ No | +2-4% accuracy | Phase 3* |
| **RoPE** | ❌ No | ❌ No | +0.5-1.5% | Phase 2* |

*Deferred to future runs due to time constraints

### Dataset Comparison

| Aspect | Conservative | Aggressive |
|--------|-------------|------------|
| **Datasets** | CIFAR-10 + STL-10 | ImageNet-100 |
| **Total Images** | ~155K | ~127K |
| **Native Resolution** | 32×32, 96×96 (upscaled) | 224×224 (native) |
| **Quality** | Good | Excellent |
| **Loading Speed** | Fast | Moderate |
| **Expected Gain** | Baseline | +10-15% vs CIFAR |

### Training Parameters Comparison

| Parameter | Conservative | Aggressive | Rationale |
|-----------|-------------|------------|-----------|
| **Model** | ViT-Small | ViT-Small | Same capacity |
| **Epochs** | 50 | 40 | Fewer epochs, better data |
| **Batch Size** | 32 | 16 × 4 acc = 64 | Larger effective batch |
| **Learning Rate** | 0.00015 | 0.0003 | Scale with batch size |
| **Warmup** | 5 epochs | 4 epochs | 10% of training |
| **Gradient Clip** | 3.0 | 1.0 | Tighter for higher LR |

---

## Conservative Configuration Details

### Philosophy

**"Prove the optimizations work"**

The conservative configuration focuses on validating that Phase 1 optimizations (Flash Attention and LayerScale) function correctly on M1 Max hardware without introducing additional complexity.

### Key Decisions

1. **Phase 1 Only**
   - Flash Attention: Critical for speed, well-tested
   - LayerScale: Proven stability improvement from DeiT III
   - No advanced features that could introduce bugs

2. **Proven Dataset**
   - CIFAR-10 + STL-10: Well-understood baseline
   - Fast iteration: Quick loading, small images
   - Easy debugging: If issues arise, dataset isn't the problem

3. **Conservative Hyperparameters**
   - Learning rate: 0.00015 (safe baseline)
   - Batch size: 32 (comfortable for 32GB RAM)
   - Standard training recipe: No surprises

### What You're Testing

1. **Flash Attention Speedup**
   - Hypothesis: 2-3x faster than baseline
   - Measurement: Compare epoch time with/without
   - Success: < 10 min/epoch vs ~20-30 min baseline

2. **LayerScale Stability**
   - Hypothesis: Smoother training curves
   - Measurement: Loss variance, gradient norms
   - Success: No divergence, clean convergence

3. **Basic Functionality**
   - All Phase 1 features work on M1 Max MPS
   - No compatibility issues
   - Establishes baseline for aggressive run

### Expected Outcomes

**Time:**
- Per epoch: 9-10 minutes
- Total: 50 epochs = 7.5-8 hours
- Margin: Should finish with 30-60 min buffer

**Performance:**
- Linear probe: 60-70%
- k-NN (k=20): 55-65%
- Better than baseline (50-60%) due to optimizations

**Memory:**
- Peak RAM: 12-16 GB
- MPS usage: 8-10 GB
- Well within 32GB limit

### When to Choose Conservative

Choose this configuration if:
- ✅ First time using Phase 1 optimizations
- ✅ Want to establish a reliable baseline
- ✅ Prefer stability over maximum performance
- ✅ Need guaranteed results in 8 hours
- ✅ Planning to run aggressive next

---

## Aggressive Configuration Details

### Philosophy

**"Push performance boundaries"**

The aggressive configuration combines Phase 1-3 optimizations to maximize performance within the 8-hour window, accepting higher risk for potentially superior results.

### Key Decisions

1. **Phase 1-3 Optimizations**
   - Flash Attention + LayerScale (proven)
   - ImageNet-100 (native resolution)
   - Gradient checkpointing (larger batches)
   - FPN (multi-scale features)
   - Contrastive component (hybrid learning)

2. **High-Quality Dataset**
   - ImageNet-100: Native 224×224 resolution
   - Major expected improvement: +10-15% over CIFAR
   - Slower loading but worth the quality gain

3. **Optimized Hyperparameters**
   - Higher LR (0.0003): Faster convergence
   - Larger batch (64 effective): Better gradients
   - Gradient accumulation: Trade memory for quality

### What You're Testing

1. **Dataset Quality Impact**
   - Hypothesis: ImageNet-100 >> CIFAR for SSL
   - Measurement: Linear probe improvement
   - Success: +10-15% over conservative

2. **Advanced Features**
   - FPN: Better multi-scale representations
   - Contrastive: +0.8-1% proven improvement
   - Combined effect: Hopefully additive

3. **Hyperparameter Optimization**
   - Higher LR with larger batch
   - Faster convergence in fewer epochs
   - Success: 40 epochs rivals 50+ on CIFAR

### Expected Outcomes

**Time:**
- Per epoch: 10-12 minutes (ImageNet-100 slower)
- Total: 40 epochs = 6.7-8 hours
- Evaluation overhead: +30-60 min
- Total: 7.3-8.5 hours

**Performance (Optimistic):**
- Linear probe: 70-78%
- k-NN (k=20): 65-73%
- Significant improvement over conservative

**Performance (Realistic):**
- Linear probe: 65-72%
- k-NN (k=20): 60-68%
- Solid improvement, validates approach

**Memory:**
- Peak RAM: 18-24 GB (gradient checkpointing helps)
- MPS usage: 10-14 GB
- Should fit with headroom

### When to Choose Aggressive

Choose this configuration if:
- ✅ Conservative run succeeded
- ✅ Want maximum performance
- ✅ Have ImageNet-100 dataset ready
- ✅ Comfortable with higher risk
- ✅ Can afford to retry if issues arise

**Warning:** If this is your first overnight run, start with conservative!

---

## Expected Training Timeline

### Conservative Timeline (50 epochs)

```
Hour 0:00 - Start training
Hour 0:09 - Epoch 1 complete (baseline established)
Hour 0:45 - Epoch 5 complete (warmup done)
Hour 1:30 - Epoch 10 complete (first checkpoint, k-NN eval)
Hour 3:45 - Epoch 25 complete (mid-training evaluation)
Hour 6:00 - Epoch 40 complete (near-final performance)
Hour 7:30 - Epoch 50 complete (training done)
Hour 8:00 - Final evaluation complete
```

**Critical Checkpoints:**
- Epoch 10: Loss should be decreasing steadily
- Epoch 25: k-NN should show 40-50% accuracy
- Epoch 40: Final performance nearly reached
- Epoch 50: Full linear probe evaluation

### Aggressive Timeline (40 epochs)

```
Hour 0:00 - Start training
Hour 0:11 - Epoch 1 complete (baseline, slower due to ImageNet)
Hour 0:44 - Epoch 4 complete (warmup done)
Hour 0:55 - Epoch 5 complete (first evaluation)
Hour 1:50 - Epoch 10 complete (checkpoint)
Hour 3:30 - Epoch 20 complete (halfway)
Hour 5:15 - Epoch 30 complete (near-final)
Hour 7:00 - Epoch 40 complete (training done)
Hour 8:00 - Final evaluation + comparison
```

**Critical Checkpoints:**
- Epoch 5: Contrastive loss should be working
- Epoch 10: FPN features should look good
- Epoch 20: k-NN > 50% (on track)
- Epoch 30: Final performance visible
- Epoch 40: Comprehensive evaluation

### Parallel Runs (If Resources Allow)

If you have enough RAM and CPU to run two instances:

```
Terminal 1: Conservative (lower priority)
Terminal 2: Aggressive (higher priority)
```

This gives you:
- Conservative: Guaranteed stable results
- Aggressive: Push for best performance
- Comparison: Direct A/B test of optimizations

**Resource Requirements:**
- RAM: 2 × 16 GB = 32 GB (at limit)
- CPU: 10 cores shared (may slow down)
- **Recommendation:** Run sequentially unless you have >64GB RAM

---

## Memory and Performance Analysis

### Memory Breakdown

**Conservative (Peak: 12-16 GB)**

| Component | Memory | Notes |
|-----------|--------|-------|
| Base model (ViT-Small) | 2 GB | 22M params × 4 bytes × 2 (gradients) |
| Activations | 4-6 GB | Batch 32, depends on depth |
| Optimizer state | 2 GB | AdamW momentum buffers |
| Data loading | 2-3 GB | 4 workers, prefetch |
| Flash Attention savings | -2 GB | vs standard attention |
| System overhead | 2 GB | OS, Python, logging |
| **Total** | **12-16 GB** | **Safe for 32 GB RAM** |

**Aggressive (Peak: 18-24 GB)**

| Component | Memory | Notes |
|-----------|--------|-------|
| Base model (ViT-Small) | 2 GB | Same as conservative |
| Activations | 3-4 GB | Smaller batch but checkpointing |
| Gradient checkpointing | -2 GB | Recompute saves memory |
| FPN components | +1 GB | Additional lateral connections |
| Optimizer state | 2 GB | Same as conservative |
| Data loading | 3-4 GB | More workers, larger images |
| ImageNet-100 cache | 4-6 GB | Larger dataset in memory |
| Contrastive buffers | +1 GB | Negative sampling |
| Flash Attention savings | -2 GB | Critical for fitting |
| System overhead | 2 GB | Same as conservative |
| **Total** | **18-24 GB** | **Should fit in 32 GB** |

### Performance Analysis

**Flash Attention Speedup:**

```
Baseline attention: O(N²) memory, ~20-30 sec/batch
Flash Attention:    O(N) memory, ~6-10 sec/batch
Speedup:           2-5x depending on sequence length
```

For ViT-Small (196 patches):
- Conservative: 9-10 min/epoch (vs 25-30 min baseline)
- Aggressive: 10-12 min/epoch (ImageNet overhead)

**Gradient Checkpointing Trade-off:**

```
Without checkpointing:
- Memory: 8-10 GB activations
- Speed: 100% (baseline)
- Max batch: 16-24

With checkpointing:
- Memory: 3-4 GB activations (-50%)
- Speed: 80-85% (recompute overhead)
- Max batch: 32-48 (2x larger)

Net benefit: 2x batch size × 0.85 speed = 1.7x throughput
```

**Effective Throughput:**

Conservative:
```
32 samples/batch × 6.25 batches/epoch × 6.3 epochs/hour
= 1,260 samples/hour
= 63,000 samples in 50 epochs
```

Aggressive:
```
64 samples/batch (effective) × 2.0 batches/epoch × 5.2 epochs/hour
= 666 samples/hour
= 26,640 samples in 40 epochs

But higher quality data (ImageNet-100) compensates
```

---

## Risk Assessment

### Conservative Configuration Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Flash Attention fails on MPS | LOW | HIGH | Fallback: disable, accept slower training |
| Memory issues | VERY LOW | MEDIUM | Plenty of headroom (12-16 GB vs 32 GB) |
| Training divergence | VERY LOW | MEDIUM | Conservative LR, proven hyperparameters |
| Dataset loading issues | VERY LOW | LOW | Auto-downloadable datasets |
| Time overrun | LOW | LOW | 50 min buffer in 8-hour window |

**Overall Risk: LOW** ✅

### Aggressive Configuration Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| ImageNet-100 path issues | MEDIUM | HIGH | Verify dataset exists first |
| Memory overflow | MEDIUM | HIGH | Gradient checkpointing, batch size tuning |
| Feature interaction bugs | MEDIUM | HIGH | Disable features one by one if needed |
| Training instability | MEDIUM | MEDIUM | Monitor gradients, reduce LR if needed |
| Contrastive loss issues | LOW | MEDIUM | Fallback to JEPA-only |
| FPN initialization problems | LOW | MEDIUM | Well-tested separately |
| Time overrun | MEDIUM | LOW | May need 8.5-9 hours |

**Overall Risk: MEDIUM** ⚠️

### Risk Mitigation Strategy

**Before Training:**
1. Verify ImageNet-100 dataset exists and is accessible
2. Test Flash Attention on small batch (1 epoch test)
3. Check available RAM: `free -h` or Activity Monitor
4. Ensure 10+ GB free disk space for checkpoints
5. Close unnecessary applications

**During Training (First Hour):**
1. Watch memory usage closely
2. Check loss is decreasing (not NaN/Inf)
3. Verify epoch time is within expected range
4. Monitor for any error messages
5. If issues arise, abort and switch to fallback

**Fallback Decision Tree:**

```
Issue detected?
  ├─ Memory overflow → Reduce batch size by 25%
  ├─ Too slow (>15 min/epoch) → Reduce epochs by 20%
  ├─ Training diverges → Reduce LR by 50%
  ├─ Flash Attention error → Disable, continue
  ├─ FPN error → Disable FPN only
  └─ Multiple issues → Abort, use conservative config
```

---

## Monitoring Guide

### Key Metrics to Track

**Every Epoch:**

```bash
# Watch training logs in real-time
tail -f results/overnight_*/logs/training.log

# Look for these patterns:
# ✓ Loss: decreasing smoothly
# ✓ LR: following schedule
# ✓ Time: 9-12 min/epoch
# ✗ NaN/Inf: immediate abort
```

**Critical Metrics:**

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| **Loss** | Decreasing | Flat for 5 epochs | Increasing |
| **Time/Epoch** | 9-12 min | 13-15 min | >15 min |
| **Memory** | <20 GB | 20-25 GB | >28 GB |
| **Gradient Norm** | <5 | 5-50 | >50 |
| **Feature Variance** | >0.1 | 0.05-0.1 | <0.05 |

### TensorBoard Monitoring

```bash
# Start TensorBoard
tensorboard --logdir results/overnight_conservative/logs --port 6006
# Open http://localhost:6006

# Or for aggressive
tensorboard --logdir results/overnight_aggressive/logs --port 6007
```

**What to Watch:**

1. **Scalars Tab:**
   - `train/loss`: Should decrease smoothly
   - `train/loss_level_0/1/2`: Hierarchy losses
   - `train/feature_variance`: Should stay >0.1
   - `train/grad_norm`: Should be <5 (clipped)
   - `train/lr`: Should follow schedule
   - `eval/knn_accuracy`: Should increase

2. **Images Tab (if enabled):**
   - Input images with masks
   - Attention maps (aggressive only)
   - FPN feature visualizations (aggressive only)

3. **Histograms Tab:**
   - Parameter distributions
   - Gradient distributions
   - Should stay roughly Gaussian

### Checkpoint Verification

```bash
# Check checkpoint size (should be ~800 MB for ViT-Small)
ls -lh results/overnight_*/checkpoints/

# Verify checkpoint integrity
python -c "
import torch
ckpt = torch.load('results/overnight_conservative/checkpoints/checkpoint_epoch_10.pth', map_location='cpu')
print(f'Epoch: {ckpt.get(\"epoch\", \"N/A\")}')
print(f'Loss: {ckpt.get(\"loss\", \"N/A\"):.4f}')
print(f'Keys: {list(ckpt.keys())}')
"
```

### Real-Time Monitoring Commands

**Terminal 1: Training Log**
```bash
tail -f results/overnight_*/logs/training.log | grep -E "(Epoch|Loss|Time)"
```

**Terminal 2: System Resources**
```bash
# macOS
watch -n 5 'top -l 1 | grep -E "PhysMem|Python"'

# Or use Activity Monitor GUI
open -a "Activity Monitor"
```

**Terminal 3: Quick Stats**
```bash
# Every 5 minutes, check progress
watch -n 300 '
echo "=== Training Progress ==="
tail -1 results/overnight_*/logs/training.log
echo ""
echo "=== System Resources ==="
free -h | grep Mem
echo ""
echo "=== Disk Space ==="
df -h | grep disk
'
```

---

## Fallback Plans

### Conservative Fallback Plan

**Issue: Out of Memory**
```yaml
# Reduce batch size
batch_size: 32 → 24 → 16

# Reduce workers
num_workers: 4 → 2
```

**Issue: Flash Attention Error**
```yaml
# Disable Flash Attention
use_flash_attention: false

# Accept slower training (12-15 min/epoch)
# May need to reduce epochs: 50 → 40
```

**Issue: Too Slow (>12 min/epoch)**
```yaml
# Reduce epochs to fit time
epochs: 50 → 40 → 35

# Reduce logging overhead
log_frequency: 50 → 100
log_images: false
log_attention: false
```

**Issue: Training Diverges**
```yaml
# Reduce learning rate
lr: 0.00015 → 0.0001

# Increase warmup
warmup_epochs: 5 → 10

# Tighter gradient clipping
clip_grad: 3.0 → 1.0
```

### Aggressive Fallback Plan

**Issue: ImageNet-100 Not Found**
```yaml
# Fallback to multi-dataset
use_multi_dataset: true
datasets:
  - name: cifar10
    weight: 0.5
  - name: stl10
    weight: 0.5

# Keep all other optimizations
# Expected: 65-70% instead of 70-78%
```

**Issue: Memory Overflow**
```yaml
# Reduce batch size
batch_size: 16 → 12 → 8

# Increase accumulation to maintain effective batch
accumulation_steps: 4 → 6 → 8

# Or disable gradient checkpointing (paradoxically saves memory in some cases)
use_gradient_checkpointing: false
batch_size: 24  # Increase again without checkpointing
```

**Issue: Feature Interaction Bugs**
```yaml
# Disable features one at a time:

# First, try disabling contrastive
use_contrastive: false

# If still broken, disable FPN
use_fpn: false

# If still broken, disable gradient checkpointing
use_gradient_checkpointing: false
batch_size: 24

# Last resort: reduce to Phase 1 only (like conservative)
use_flash_attention: true
use_layerscale: true
# All others: false
```

**Issue: Training Unstable**
```yaml
# Reduce learning rate
lr: 0.0003 → 0.0002 → 0.00015

# Reduce contrastive weight
contrastive_weight: 0.1 → 0.05

# Increase warmup
warmup_epochs: 4 → 8

# Balance hierarchy weights more
hierarchy_weights: [1.0, 0.8, 0.6] → [1.0, 0.7, 0.5]
```

**Issue: Too Slow to Finish**
```yaml
# Reduce epochs
epochs: 40 → 35 → 30

# Reduce evaluation frequency
eval_frequency: 5 → 10

# Reduce workers (paradoxically sometimes faster)
num_workers: 6 → 4

# Disable attention logging
log_attention: false
```

---

## Success Criteria

### Conservative Success Criteria

**Minimum (Must Have):**
- ✅ Training completes in <8.5 hours
- ✅ No NaN/Inf losses at any point
- ✅ Flash Attention runs without errors
- ✅ Final loss <1.0
- ✅ Linear probe accuracy >55%
- ✅ Feature variance >0.05 (no collapse)

**Target (Should Have):**
- ✅ Training completes in <8 hours
- ✅ Flash Attention shows 2x+ speedup vs baseline
- ✅ Smooth loss curves (no oscillation)
- ✅ Final loss <0.5
- ✅ Linear probe accuracy >60%
- ✅ k-NN accuracy >50%
- ✅ Feature variance >0.1

**Stretch (Nice to Have):**
- ✅ Training completes in <7.5 hours
- ✅ Flash Attention shows 3x+ speedup
- ✅ Final loss <0.3
- ✅ Linear probe accuracy >65%
- ✅ k-NN accuracy >60%
- ✅ Clear hierarchy differentiation visible

### Aggressive Success Criteria

**Minimum (Must Have):**
- ✅ Training completes in <9 hours
- ✅ No catastrophic failures
- ✅ Linear probe accuracy >60% (better than CIFAR baseline)
- ✅ All Phase 1-3 features run without errors
- ✅ Memory stays <28 GB

**Target (Should Have):**
- ✅ Training completes in <8.5 hours
- ✅ Linear probe accuracy >68%
- ✅ Improvement over conservative >5%
- ✅ FPN features look good in visualizations
- ✅ Contrastive accuracy >0.9
- ✅ Final loss <0.3
- ✅ k-NN accuracy >60%

**Stretch (Nice to Have):**
- ✅ Training completes in <8 hours
- ✅ Linear probe accuracy >73%
- ✅ Improvement over conservative >10%
- ✅ Match or exceed I-JEPA baseline (75%)
- ✅ Strong transfer to downstream tasks
- ✅ Clear multi-scale hierarchy visible

### Comparison Success Criteria

If both configs run:

**Must Show:**
- Aggressive > Conservative in linear probe
- ImageNet-100 benefit visible (+5-10%)
- Phase 1 features work in both

**Should Show:**
- FPN improves multi-scale features
- Contrastive adds instance discrimination
- Higher LR with larger batch converges faster

**Would Be Impressive:**
- Aggressive: >10% better than Conservative
- Both: Better than published baselines
- Clear path to 75%+ with more training

---

## Post-Training Analysis

### Immediate Analysis (While Model is Warm)

```bash
# 1. Run comprehensive evaluation
python scripts/evaluate.py \
  --checkpoint results/overnight_*/checkpoints/best_model.pth \
  --config configs/overnight_training_*.yaml \
  --eval-types all \
  --output results/overnight_*/evaluation

# 2. Generate visualizations
python scripts/visualize.py \
  --checkpoint results/overnight_*/checkpoints/best_model.pth \
  --output-dir results/overnight_*/visualizations \
  --generate-report

# 3. Compare configurations (if both ran)
python scripts/compare_runs.py \
  --run1 results/overnight_conservative \
  --run2 results/overnight_aggressive \
  --output results/overnight_comparison
```

### Key Questions to Answer

**1. Performance:**
- What was the final linear probe accuracy?
- How does it compare to baseline/target?
- Which configuration performed better?
- Was the improvement worth the complexity?

**2. Optimization Impact:**
- What speedup did Flash Attention provide?
  - Measure: Compare epoch time with/without
  - Expected: 2-5x speedup

- Did LayerScale improve stability?
  - Measure: Loss curve variance, gradient norms
  - Expected: Smoother training

- Did ImageNet-100 justify the cost?
  - Measure: Linear probe improvement
  - Expected: +10-15% over CIFAR

- Was FPN beneficial?
  - Measure: Multi-scale feature quality
  - Expected: Better downstream performance

- Did Contrastive help?
  - Measure: Contrastive accuracy, final performance
  - Expected: +0.8-1% improvement

**3. Efficiency:**
- Actual time per epoch: _____ minutes
- Total training time: _____ hours
- Memory peak: _____ GB
- Throughput: _____ samples/hour

**4. Stability:**
- Any NaN/Inf during training? Yes/No
- Any features that caused issues? List
- Would you change anything? Notes

### Detailed Metrics Checklist

**Training Metrics:**
- [ ] Final training loss: _____
- [ ] Loss reduction from init: _____%
- [ ] Training time: _____ hours
- [ ] Average epoch time: _____ minutes
- [ ] Peak memory usage: _____ GB
- [ ] Gradient norm (avg): _____
- [ ] Feature variance (final): _____
- [ ] Effective rank: _____

**Evaluation Metrics:**
- [ ] Linear probe accuracy: _____%
- [ ] k-NN accuracy (k=1): _____%
- [ ] k-NN accuracy (k=5): _____%
- [ ] k-NN accuracy (k=20): _____%
- [ ] Few-shot (1-shot): _____%
- [ ] Few-shot (5-shot): _____%

**Hierarchy Analysis:**
- [ ] Level 0 loss: _____
- [ ] Level 1 loss: _____
- [ ] Level 2 loss: _____
- [ ] Hierarchy differentiation: Clear/Moderate/Weak
- [ ] Multi-scale features: Good/Moderate/Poor

**Feature Quality:**
- [ ] Representation collapse: No/Partial/Yes
- [ ] Feature diversity (entropy): _____
- [ ] Attention patterns: Good/Moderate/Poor
- [ ] Transfer quality: Good/Moderate/Poor

### Decision Matrix for Next Steps

Based on results, decide next action:

```
Results: Excellent (>70% linear probe)
├─ Action: Scale to full ImageNet-1K
├─ Config: Use aggressive as base
├─ Changes: 300 epochs, add multi-crop
└─ Expected: 73-78% linear probe

Results: Good (65-70% linear probe)
├─ Action: Extend current training
├─ Config: Resume from checkpoint
├─ Changes: Train 50-100 more epochs
└─ Expected: 68-73% linear probe

Results: Moderate (60-65% linear probe)
├─ Action: Tune hyperparameters
├─ Config: Grid search on LR, batch size
├─ Changes: Optimize before scaling
└─ Expected: 65-70% linear probe

Results: Poor (<60% linear probe)
├─ Action: Debug issues
├─ Config: Analyze what went wrong
├─ Changes: Fix bugs, verify implementation
└─ Expected: Identify and resolve problems
```

### Report Template

Create a report file: `results/overnight_training_report.md`

```markdown
# Overnight Training Results

**Date:** YYYY-MM-DD
**Configuration:** Conservative / Aggressive
**Hardware:** M1 Max (32GB RAM, MPS)

## Summary

- Final Performance: XX% linear probe
- Training Time: X.X hours
- Success: Yes/No (vs criteria)

## Training Details

- Epochs completed: XX
- Final loss: X.XXX
- Time per epoch: XX.X minutes
- Peak memory: XX GB

## Optimization Results

### Flash Attention
- Speedup achieved: Xx
- Issues: None / [list]

### LayerScale
- Stability improvement: Yes/No
- Impact: [describe]

### [Other features...]

## Performance Breakdown

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Linear Probe | XX% | 60%+ | ✅/❌ |
| k-NN (k=20) | XX% | 50%+ | ✅/❌ |
| Training Time | X.Xh | <8h | ✅/❌ |
| Memory Usage | XXG | <28G | ✅/❌ |

## Lessons Learned

1. [What worked well]
2. [What didn't work]
3. [Surprises]
4. [Would change next time]

## Next Steps

- [ ] Action 1
- [ ] Action 2
- [ ] Action 3

## Appendix

- Checkpoint: `results/overnight_*/checkpoints/best_model.pth`
- Logs: `results/overnight_*/logs/`
- Visualizations: `results/overnight_*/visualizations/`
```

---

## Quick Reference

### Start Training

**Conservative:**
```bash
python scripts/train.py \
  --config configs/overnight_training_conservative.yaml \
  --device mps

# Monitor
tensorboard --logdir results/overnight_conservative/logs --port 6006
tail -f results/overnight_conservative/logs/training.log
```

**Aggressive:**
```bash
# First, verify ImageNet-100 exists
ls -la data/imagenet/train/ | head -20

# Start training
python scripts/train.py \
  --config configs/overnight_training_aggressive.yaml \
  --device mps

# Monitor
tensorboard --logdir results/overnight_aggressive/logs --port 6007
tail -f results/overnight_aggressive/logs/training.log
```

### Emergency Stop

```bash
# Find training process
ps aux | grep train.py

# Kill gracefully (allows checkpoint save)
kill -SIGINT <PID>

# Force kill (if frozen)
kill -9 <PID>
```

### Resume from Checkpoint

```bash
python scripts/train.py \
  --config configs/overnight_training_*.yaml \
  --resume results/overnight_*/checkpoints/checkpoint_epoch_X.pth \
  --device mps
```

### Quick Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint results/overnight_*/checkpoints/best_model.pth \
  --eval-type linear_probe \
  --dataset cifar10  # or imagenet100
```

---

## Conclusion

You now have two well-designed configurations:

1. **Conservative**: Low-risk validation of Phase 1 optimizations
2. **Aggressive**: High-performance test of Phase 1-3 combined

**Recommendation:**
1. Start with **Conservative** to establish baseline
2. If successful, run **Aggressive** for maximum performance
3. Compare results to guide future training

**Expected Timeline:**
- Conservative: 7.5-8 hours
- Aggressive: 7.3-8.5 hours
- Both sequentially: 15-17 hours (overnight + next day)

**Success Criteria:**
- Conservative: >60% linear probe, Flash Attention working
- Aggressive: >68% linear probe, ImageNet-100 benefit visible

**Next Steps After Success:**
- Scale to full ImageNet (300 epochs)
- Add multi-crop training
- Test on downstream tasks

Good luck with your overnight training! 🚀

---

**Document Version:** 1.0
**Last Updated:** 2025-11-16
**Contact:** See main README for support

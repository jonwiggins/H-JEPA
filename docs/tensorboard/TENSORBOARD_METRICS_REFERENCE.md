# H-JEPA TensorBoard Metrics Reference

Quick reference guide for all proposed TensorBoard enhancements with metric names, interpretations, and expected values.

## 1. Loss Metrics

### Hierarchical Loss Dashboard
Track prediction loss across 3 hierarchy levels with different semantic granularities.

```
train/loss/total                           Total weighted hierarchical loss
train/loss/h0                              Finest level loss (14x14 patches)
train/loss/h1                              Intermediate level loss (7x7 patches)
train/loss/h2                              Coarsest level loss (4x4 patches)
train/loss/unweighted                      Mean of all hierarchy losses
train/loss/contribution_h0                 % of total loss from h0 (~50-70%)
train/loss/contribution_h1                 % of total loss from h1 (~25-40%)
train/loss/contribution_h2                 % of total loss from h2 (~5-15%)

Convergence Tracking:
train/convergence/rate_h0                  Loss reduction rate (level 0)
train/convergence/rate_h1                  Loss reduction rate (level 1)
train/convergence/rate_h2                  Loss reduction rate (level 2)
```

**Expected Behavior:**
- All losses should decrease monotonically
- h0 and h1 should converge faster than h2
- Contributions should reflect configured hierarchy_weights
- If contributions are imbalanced, adjust config weights

---

## 2. EMA (Exponential Moving Average) Dynamics

Target encoder is updated via EMA from context encoder. This is critical for JEPA-style training.

```
train/ema/momentum_current                 Current EMA momentum (0.996 -> 1.0)
train/ema/momentum_target                  Target final momentum (1.0)
train/ema/avg_parameter_divergence         Average weight difference ||w_ctx - w_tgt||
train/ema/weight_magnitude_ratio           ||w_tgt|| / ||w_ctx|| (should ~= 1.0)

Interpretation:
- momentum_current: Should increase smoothly from 0.996 to ~1.0
- parameter_divergence: Should increase then plateau (target becomes frozen)
- weight_magnitude_ratio: Should remain close to 1.0 (similar magnitudes)
```

**Alert Thresholds:**
- If momentum not increasing: Check EMA scheduler
- If divergence very high (>10x): Target encoder deviating too much
- If weight ratio < 0.8 or > 1.2: Magnitude mismatch, check learning rate

---

## 3. Masking Strategy

Validates that masking is generating appropriate context regions and mask randomization works.

```
train/masking/mask_ratio                   Fraction of patches masked (0.0-1.0)
train/masking/num_masked_patches           Average masked patches per sample
train/masking/num_unmasked_patches         Average unmasked (visible) patches
```

**Expected Behavior:**
- mask_ratio should be consistent across batches
- If mask_ratio changes suddenly: Check mask generator
- Typical range: 0.4-0.7 (40-70% masked)

**Configuration:**
```yaml
masking:
  num_masks: 4                # Number of rectangular masks
  mask_scale: [0.05, 0.15]   # Min/max scale of masked region
  aspect_ratio: [0.75, 1.5]  # Aspect ratio range
```

---

## 4. Prediction Quality

Core objective: predict masked patches accurately.

```
train/prediction/level{i}_cosine_sim_mean           Mean cosine similarity pred vs target
train/prediction/level{i}_cosine_sim_std            Std of similarity (should be > 0.1)
train/prediction/level{i}_cosine_sim_min            Minimum similarity (easiest patches)
train/prediction/level{i}_cosine_sim_max            Maximum similarity (hardest patches)
train/prediction/level{i}_normalized_mse            MSE between normalized embeddings
train/prediction/level{i}_l2_distance               L2 distance between embeddings
```

**Interpretation:**
- cosine_sim_mean > 0.7: Good predictions
- cosine_sim_std > 0.1: Diverse prediction quality (some hard, some easy patches)
- cosine_sim_max - cosine_sim_min: Prediction difficulty range
- Level 0 should have highest similarity (finer details easier)
- Level 2 may have lower similarity (coarser features harder to distinguish)

**Expected Trend:**
```
Epoch 1:  cosine_sim ≈ 0.4
Epoch 50: cosine_sim ≈ 0.7
Epoch 100: cosine_sim ≈ 0.85 (saturates)
```

---

## 5. Representational Collapse Monitoring

**CRITICAL**: Detect if features collapse to constant values or low-rank approximations.

### Collapse Indicators (watch all three)

```
train/collapse/level{i}_mean_std_per_dim              Mean std of features per dimension
train/collapse/level{i}_min_std_per_dim               Minimum std (worst dimension)
train/collapse/level{i}_effective_rank                Effective rank of covariance matrix
train/collapse/level{i}_mean_similarity               Mean pairwise cosine similarity
train/collapse/level{i}_max_similarity                Max pairwise cosine similarity
train/collapse/level{i}_std_similarity                Std of pairwise similarities
```

**Alert Thresholds:**

| Metric | HEALTHY | WARNING | COLLAPSE |
|--------|---------|---------|----------|
| mean_std_per_dim | > 0.05 | 0.02-0.05 | < 0.01 |
| min_std_per_dim | > 0.005 | 0.001-0.005 | < 0.001 |
| effective_rank | > 500 (for 768-D) | 200-500 | < 200 |
| mean_similarity | < 0.3 | 0.3-0.7 | > 0.7 |

**If Collapse Detected:**
1. Increase learning rate (2x)
2. Reduce EMA momentum (0.996 -> 0.99)
3. Reduce batch size (lower update magnitude)
4. Add stronger augmentation
5. Check data distribution (class imbalance?)
6. Review masking strategy (too aggressive?)

---

## 6. Gradient Flow Analysis

Ensures gradients flow properly through the network without vanishing or exploding.

```
train/gradient_flow/global_norm                      Total gradient norm (all params)
train/gradient_flow/context_encoder_mean             Avg gradient magnitude in context encoder
train/gradient_flow/context_encoder_max              Max gradient in context encoder
train/gradient_flow/predictor_mean                   Avg gradient magnitude in predictor
train/gradient_flow/predictor_max                    Max gradient in predictor
train/gradient_flow/target_encoder_mean              Avg gradient in target (should be 0, detached)
train/gradient/global_norm_before_clip               Gradient norm before clipping
train/gradient/global_norm_after_clip                Gradient norm after clipping
train/gradient/clipping_ratio                        after_clip / before_clip
train/gradient/was_clipped                           1 if clipped, 0 otherwise
train/gradient/clipping_percentage                   % of steps where clipping applied
```

**Expected Behavior:**
- global_norm should be stable (not exploding)
- context_encoder_mean > predictor_mean: Context encoder has stronger gradients
- target_encoder should have near-zero gradients (no_grad context)
- clipping_percentage < 5-10%: Gradient clipping working but not excessive

**Alert Thresholds:**
- global_norm > 10: Potentially unstable gradients
- clipping_percentage > 50%: Learning rate likely too high
- clipping_percentage = 0%: Clipping threshold may be too high

**Histograms:**
- `gradients/{layer}_histogram`: Distribution of gradient values per layer
- Watch for: Bimodal distributions, heavy tails, clusters at zero

---

## 7. Learning Rate Schedule

```
train/learning_rate/base_lr                         Base learning rate
train/learning_rate/param_group_{i}                 LR for param group i (if using differential LRs)
train/learning_rate/avg_param_update                Average parameter update magnitude
```

**Typical Cosine Annealing Schedule:**
```
Warmup phase (0-5 epochs):  0.0 -> 1e-3 linearly
Main phase (5-100 epochs):  1e-3 -> 1e-5 via cosine
End phase (100+ epochs):    1e-5 (plateaued)
```

**Expected Pattern:**
- Warm up linearly from 0 to peak_lr
- Decay smoothly with cosine schedule
- No sudden jumps (indicates scheduler issue)

---

## 8. Hierarchy Level Feature Analysis

Understanding how features differ across hierarchy levels.

```
train/hierarchy/level{i}_feat_mean                  Mean of embeddings at level i
train/hierarchy/level{i}_feat_std                   Std of embeddings at level i
train/hierarchy/level{i}_feat_range                 Max - min of embeddings
train/hierarchy/level{i}_feat_norm                  Average L2 norm of embeddings
train/hierarchy/level{i}_num_patches                Number of patches at level i (resolution)

Example for 224x224 image with patch_size=16:
- Level 0: 196 patches (14x14)
- Level 1:  98 patches (~7x7 after pooling)
- Level 2:  49 patches (~5x5 after pooling)
```

**Expected Pattern:**
- feat_std should be similar across levels (normalized)
- feat_norm should be consistent (learnable scaling)
- num_patches decreases as hierarchy level increases
- Coarser levels may have slightly different feature distributions

---

## 9. Training Stability

Detect training issues early via loss landscape smoothness.

```
train/stability/loss_smoothness                     Variance of loss over recent steps
train/stability/loss_trend_slope                    Slope of loss curve (should be negative)
train/stability/loss_spike_ratio                    max_loss / min_loss in recent window
train/stability/val_train_loss_ratio                val_loss / train_loss (overfitting indicator)
```

**Interpretation:**
- loss_smoothness: High variance = unstable training
- loss_trend_slope: Negative = converging, near-zero = plateaued
- loss_spike_ratio > 1.5: Loss spikes detected, potentially unstable
- val_train_loss_ratio > 1.1: Possible overfitting

**Healthy Training Pattern:**
```
Step 1:    loss_smoothness ≈ 0.5, trend_slope ≈ -0.01
Step 500:  loss_smoothness ≈ 0.1, trend_slope ≈ -0.001
Step 1000: loss_smoothness ≈ 0.05, trend_slope ≈ 0.0 (converged)
```

---

## 10. Performance and Efficiency

Monitor computational efficiency and bottlenecks.

```
train/performance/forward_time_ms                   Forward pass time in milliseconds
train/performance/backward_time_ms                  Backward pass time in milliseconds
train/performance/total_time_ms                     Forward + backward time
train/performance/samples_per_second                Throughput (samples/sec)
train/performance/forward_backward_ratio            Forward / backward ratio
train/performance/gpu_memory_allocated_gb           GPU memory allocated (in use)
train/performance/gpu_memory_reserved_gb            GPU memory reserved (allocated + headroom)
```

**Expected Ratios:**
- forward_backward_ratio: 0.5-1.5 (backward usually 1-2x forward)
- If ratio > 2: Backward very expensive (check gradient computation)
- If ratio < 0.3: Unusual, check timing code

**Throughput Example:**
- Batch size 64, RTX A100: ~200-400 samples/sec
- Batch size 32, RTX A6000: ~100-200 samples/sec
- If 10x lower: Check for bottleneck (data loading? synchronization?)

**Memory Usage:**
- ViT-Base (768-D): ~8GB allocated
- With gradients (32-bit float): +8GB more
- Total: ~16GB for batch_size=64 on single GPU
- If allocated > 40GB: Memory leak or incorrect batch size

---

## 11. Quick Diagnostic Flowchart

```
Training issue detected? Follow this flowchart:

├─ Loss not decreasing?
│  ├─ Check: learning_rate/base_lr (should be > 1e-5)
│  ├─ Check: gradient_flow/global_norm (should be non-zero)
│  ├─ Check: stability/loss_smoothness (should be < 1.0)
│  └─ Action: Increase LR or check data loading
│
├─ Collapse detected? (mean_std < 0.01)
│  ├─ Check: prediction/levelX_cosine_sim_mean (< 0.5 means collapse)
│  ├─ Check: gradient/clipping_percentage (too high?)
│  ├─ Check: ema/parameter_divergence (too large?)
│  └─ Action: Reduce LR, increase EMA momentum, stronger augmentation
│
├─ Poor prediction quality?
│  ├─ Check: loss/contribution_hX (imbalanced?)
│  ├─ Check: hierarchy/levelX_feat_std (low variance?)
│  ├─ Check: masking/mask_ratio (too aggressive?)
│  └─ Action: Adjust hierarchy_weights, check masking config
│
├─ Gradient issues?
│  ├─ Check: gradient/clipping_percentage (> 50%?)
│  ├─ Check: gradient_flow/global_norm (exploding?)
│  ├─ Check: gradient_flow/predictor_max (much higher than others?)
│  └─ Action: Reduce LR, increase clip_grad threshold, check layer scales
│
└─ Slow training?
   ├─ Check: performance/forward_time_ms (GPU limited?)
   ├─ Check: performance/gpu_memory_reserved_gb (near capacity?)
   ├─ Check: performance/samples_per_second (< 100?)
   └─ Action: Reduce batch size, profile code, check data loading
```

---

## 12. Metric Logging Frequency Recommendations

| Metric Category | Frequency | Notes |
|-----------------|-----------|-------|
| Loss metrics | Every step | Low overhead, essential |
| Masking stats | Every 100 steps | Quick to compute |
| Prediction quality | Every 100-500 steps | Medium overhead |
| Collapse metrics | Every 100-500 steps | Can be expensive (SVD) |
| Gradient flow | Every 100 steps | Always available post-backward |
| Learning rate | Every 100 steps | Very cheap |
| Hierarchy stats | Every 100 steps | Cheap |
| Stability metrics | Every step | Cheap (window-based) |
| Performance | Every 100 steps | Timing overhead < 1% |
| EMA dynamics | Every 500-1000 steps | Expensive (parameter divergence) |

---

## 13. Batch TensorBoard Commands

```bash
# Launch TensorBoard with all enhancements
tensorboard --logdir=results/logs/tensorboard --reload_multifile=true

# View only scalars (fastest)
tensorboard --logdir=results/logs/tensorboard --samples_per_plugin scalars=100

# Specify port
tensorboard --logdir=results/logs/tensorboard --port=6006

# View on remote server
# SSH: ssh -L 6006:localhost:6006 user@server
# Then open: http://localhost:6006

# Export metrics to CSV
# TensorBoard -> Download as CSV (via Web UI)
# Or use: tensorboard.plugin.scalar.metadata.SESSION_TAG
```

---

## 14. Configuration for Maximum Insights

Add to `config.yaml`:

```yaml
logging:
  log_dir: results/logs
  log_frequency: 100              # Log every 100 steps
  experiment_name: hjepa_run_1

  tensorboard:
    enabled: true

wandb:
  enabled: false                  # Optional

# TensorBoard will automatically track:
# - All metrics logged via metrics_logger
# - Scalars organized by prefix (train/, val/)
# - Update frequency based on log_frequency
```

---

## 15. Expected Metric Ranges (Good Training)

```
Epoch 10/100:
  loss/total:                    0.5-1.0
  loss/h0:                       0.3-0.6
  loss/h1:                       0.2-0.4
  loss/h2:                       0.1-0.2
  prediction/level0_cosine_sim:  0.4-0.6
  collapse/level0_mean_std:      0.05-0.15
  gradient_flow/global_norm:     0.5-2.0
  learning_rate/base_lr:         3e-4-5e-4
  performance/samples_per_second: 100-400
  ema/momentum_current:          0.9970-0.9990

Epoch 50/100:
  loss/total:                    0.1-0.3
  loss/h0:                       0.05-0.15
  prediction/level0_cosine_sim:  0.7-0.85
  collapse/level0_mean_std:      0.08-0.2
  gradient_flow/global_norm:     0.1-0.5
  learning_rate/base_lr:         1e-4-3e-4
  ema/momentum_current:          0.999-0.9999

Epoch 100/100:
  loss/total:                    0.05-0.15
  loss/h0:                       0.02-0.10
  prediction/level0_cosine_sim:  0.85-0.95
  collapse/level0_mean_std:      0.10-0.25
  gradient_flow/global_norm:     0.01-0.1
  learning_rate/base_lr:         1e-5-1e-4
  ema/momentum_current:          1.0 (saturated)
```

---

## 16. Common Patterns and Their Meanings

### Pattern: Loss stuck high (not decreasing)
**Likely cause:** Learning rate too low, or data issue
**Check:** learning_rate/base_lr, gradient_flow/global_norm
**Fix:** Increase LR by 2x or check data loading

### Pattern: Loss decreases then increases (divergence)
**Likely cause:** Learning rate too high, or EMA momentum too low
**Check:** gradient/clipping_percentage (if > 50%), ema/momentum
**Fix:** Reduce LR, increase EMA momentum

### Pattern: Collapse around epoch 30
**Likely cause:** Model collapsed to low-rank solution
**Check:** collapse metrics all at alert thresholds
**Fix:** Restart with better augmentation or different masking

### Pattern: Loss very noisy / high variance
**Likely cause:** Unstable training, possibly bad batch composition
**Check:** stability/loss_smoothness, masking/mask_ratio
**Fix:** Reduce LR, check data augmentation strength

### Pattern: Prediction quality plateaus early
**Likely cause:** EMA momentum too high (target encoder not updating)
**Check:** prediction/levelX_cosine_sim_mean stagnates
**Fix:** Reduce EMA momentum starting value

---

## Summary

These metrics provide complete visibility into H-JEPA training dynamics. Monitor the **collapse metrics** and **prediction quality** most closely, as these directly indicate model health. Use **gradient flow** and **learning rate** to debug convergence issues.

For production runs, focus on:
1. `loss/total` - primary objective
2. `collapse/level{i}_*` - prevent collapse
3. `prediction/level{i}_cosine_sim_mean` - validate learning
4. `gradient/clipping_percentage` - detect training instability


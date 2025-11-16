# Validation Run Learnings & Optimization Guide

This document tracks key learnings from the validation run to optimize the larger training run.

## 🎯 Validation Run Objectives

1. **Verify System Stability**: Ensure MPS compatibility and no crashes
2. **Measure Training Speed**: Characterize M1 Max performance
3. **Validate Loss Convergence**: Confirm model is learning
4. **Identify Bottlenecks**: Find performance optimization opportunities
5. **Estimate Full Training Time**: Project completion times for larger runs

## 📊 Key Metrics to Track

### 1. Training Speed
- **Target**: ~3.2 it/s (based on initial observations)
- **Actual**: [To be measured]
- **Stability**: Check variance in iteration speed
- **Bottlenecks**: CPU, GPU, or I/O bound?

**Questions to answer:**
- Is speed consistent throughout training?
- Are there periodic slowdowns?
- Does speed degrade over time?

### 2. Loss Convergence
- **Initial Loss**: [To be measured]
- **Final Loss**: [To be measured]
- **Convergence Rate**: How quickly does loss decrease?
- **Stability**: Is loss curve smooth or noisy?

**Questions to answer:**
- Is the learning rate appropriate? (loss should decrease steadily)
- Is warmup schedule working? (no sudden spikes)
- Any signs of divergence? (loss increasing)
- Is the model plateauing too early?

### 3. Memory Usage
- **Peak Memory**: [To be measured]
- **Average Memory**: [To be measured]
- **Stability**: Any memory leaks?

**Questions to answer:**
- Can we increase batch size?
- Is there room for a larger model?
- Any memory pressure issues?

### 4. Feature Quality (Post-Training)
- **Representation Collapse**: Check effective rank
- **Feature Variance**: Ensure features are diverse
- **Isotropy**: Features should be well-distributed

**Red flags:**
- Effective rank < 10% of embedding dimension
- Very low feature variance
- High cosine similarity between all samples

## 🔍 MPS-Specific Observations

### What We've Fixed
1. ✅ `.cpu()` calls before `.numpy()` conversions
2. ✅ Visualization code MPS compatibility
3. ✅ Evaluation code MPS compatibility

### What to Monitor
- [ ] Any new MPS compatibility errors
- [ ] Performance compared to CUDA (if benchmarks available)
- [ ] MPS fallback operations (SVD operations fall back to CPU)

### Known MPS Limitations
1. **SVD Operations**: Fall back to CPU (warning in logs)
   - Impact: Minor slowdown during visualization/analysis
   - Solution: This is expected, doesn't affect training
2. **Mixed Precision**: GradScaler disabled (MPS doesn't support AMP)
   - Impact: No mixed precision speedup
   - Solution: Accepted tradeoff for M1 Max

## 📈 Optimization Opportunities

### Potential Improvements for Full Training

#### 1. Batch Size
- **Current**: 32 (validation)
- **Options**:
  - Increase to 48-64 if memory allows
  - Decrease to 24 if using larger model (ViT-Small)
- **Expected Impact**: Larger batch = better GPU utilization but may affect convergence

#### 2. Number of Workers
- **Current**: 4 (validation)
- **Options**:
  - Increase to 6-8 (M1 Max has 10 cores)
  - Decrease to 2 if CPU bound
- **Expected Impact**: Better data loading if I/O bound

#### 3. Learning Rate
- **Current**: 1e-4 (validation)
- **Observations**: [Monitor warmup and convergence]
- **Adjustments**:
  - If converging too slowly: increase to 1.5e-4 or 2e-4
  - If loss is noisy: decrease to 5e-5
  - If diverging: decrease significantly

#### 4. Warmup Epochs
- **Current**: 1 epoch (validation)
- **For 20-epoch run**: 2-4 epochs
- **For 100-epoch run**: 5-10 epochs
- **Rationale**: Longer training needs longer warmup

#### 5. Model Architecture
- **Current**: ViT-Tiny (validation)
- **Options**:
  - ViT-Small: Better accuracy, ~2x slower
  - ViT-Base: Best accuracy, ~4x slower
- **Decision**: Based on validation speed and available time

## 🚀 Decision Matrix for Full Training

### Option A: Quick Baseline (20 epochs)
**Choose if:**
- Validation loss converges well
- Speed is ~3.2 it/s or better
- Want results within 2-3 hours

**Configuration:**
```yaml
model:
  encoder_type: "vit_tiny_patch16_224"
  num_hierarchies: 2
  predictor:
    depth: 4  # Deeper than validation

training:
  epochs: 20
  warmup_epochs: 4  # Longer warmup
```

**Expected Results:**
- Linear probe: 70-78%
- Training time: ~2.5 hours
- Good baseline for comparison

### Option B: Competitive Results (100 epochs)
**Choose if:**
- Validation shows stable training
- Can run overnight (~12-14 hours)
- Want publication-quality results

**Configuration:**
```yaml
model:
  encoder_type: "vit_small_patch16_224"  # Upgrade model
  num_hierarchies: 3  # More hierarchy levels
  predictor:
    depth: 6

training:
  epochs: 100
  warmup_epochs: 10
  batch_size: 24  # Reduce for larger model
```

**Expected Results:**
- Linear probe: 80-85%
- Training time: ~12-14 hours
- Competitive with published results

### Option C: Extended Baseline (50 epochs)
**Choose if:**
- Want better than quick baseline
- Don't have time for 100 epochs
- Validation shows good convergence

**Configuration:**
```yaml
model:
  encoder_type: "vit_tiny_patch16_224"
  num_hierarchies: 2

training:
  epochs: 50
  warmup_epochs: 5
```

**Expected Results:**
- Linear probe: 75-80%
- Training time: ~6-7 hours
- Good middle ground

## 📝 Post-Validation Checklist

Before starting full training, verify:

### Training Stability
- [ ] No crashes or errors during validation
- [ ] Loss decreased steadily
- [ ] No memory issues
- [ ] MPS compatibility confirmed

### Performance Acceptable
- [ ] Training speed ≥ 2.5 it/s (minimum)
- [ ] Speed variance < 20%
- [ ] No major bottlenecks identified

### Model Learning
- [ ] Loss improved by ≥ 20% from initial
- [ ] No divergence observed
- [ ] Warmup schedule working properly

### System Optimization
- [ ] num_workers tuned
- [ ] batch_size optimized
- [ ] No unnecessary logging slowing things down

## 🎓 Lessons Learned Template

After validation run, document:

### What Worked Well
-
-
-

### What Needs Improvement
-
-
-

### Unexpected Findings
-
-
-

### Recommendations for Full Training
1.
2.
3.

## 📚 Reference Baselines

For CIFAR-10 self-supervised learning (from literature):

| Method | Architecture | Epochs | Linear Probe Accuracy |
|--------|-------------|--------|----------------------|
| SimCLR | ResNet-50 | 1000 | 90.6% |
| MoCo v2 | ResNet-50 | 800 | 89.0% |
| BYOL | ResNet-50 | 1000 | 91.3% |
| I-JEPA | ViT-B | 300 | ~85% |
| H-JEPA (ours) | ViT-Tiny | 100 | 80-85% (target) |
| H-JEPA (ours) | ViT-Small | 100 | 85-88% (target) |

**Note**: Our targets are reasonable given smaller models and fewer epochs.

## 🔧 Debugging Guide

### If Loss Doesn't Decrease
1. Check learning rate (may be too low)
2. Verify gradients are flowing (check logs)
3. Check mask generator (ensure valid masks)
4. Verify data augmentation working

### If Loss Explodes
1. Reduce learning rate immediately
2. Increase warmup epochs
3. Check for NaN/Inf in gradients
4. Reduce batch size

### If Training Slows Down
1. Check system resources (Activity Monitor)
2. Reduce num_workers if CPU bound
3. Check disk I/O if data loading is slow
4. Monitor memory pressure

### If Representations Collapse
1. Increase VICReg regularization weight
2. Add more data augmentation
3. Adjust EMA decay (target encoder)
4. Check predictor capacity

## 📊 Analysis Scripts Available

After training completes, run:

```bash
# Automated analysis
./scripts/auto_analyze_and_recommend.sh training_run.log

# Manual analysis
python3.11 scripts/analyze_validation_run.py --log training_run.log

# Comprehensive evaluation
./scripts/quick_eval_after_training.sh results/checkpoints/best_checkpoint.pt
```

## 🎯 Success Criteria

The validation run is successful if:

1. ✅ **Stability**: No crashes, completed all 5 epochs
2. ✅ **Performance**: Speed ≥ 2.5 it/s
3. ✅ **Learning**: Loss decreased by ≥ 20%
4. ✅ **Quality**: No representation collapse
5. ✅ **Compatibility**: All MPS issues resolved

If all criteria met → Proceed to full training!

---

**Last Updated**: [To be filled after validation run]
**Validation Status**: In Progress
**Next Action**: Monitor to completion, then analyze

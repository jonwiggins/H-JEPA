# H-JEPA Training Plan - Executive Summary

**Document Reference:** See `/home/user/H-JEPA/TRAINING_PLAN.md` for full details

---

## Quick Overview

**System Environment:** CPU-only, 13GB RAM, 29GB disk
**Target:** Validate H-JEPA architecture with quick, feasible training
**Timeline:** ~24-30 hours total (setup + training + evaluation)
**Expected Result:** 50-70% linear probe accuracy on CIFAR-10

---

## Key Decisions

### 1. Dataset Selection: CIFAR-10 ✅

**Why CIFAR-10:**
- Auto-downloadable (no manual setup)
- Small enough for CPU training (50K images)
- Well-studied baseline comparisons
- Training time: 18-24 hours (feasible)

**Alternatives Considered:**
- ImageNet: Too large for CPU (weeks of training)
- STL-10: Larger images = slower training
- CIFAR-100: Similar to CIFAR-10, more challenging

### 2. Model Architecture: ViT-Tiny + 2 Hierarchies

**Configuration:**
```yaml
model:
  encoder_type: "vit_tiny_patch16_224"    # 5M parameters
  embed_dim: 192                          # Reduced from 384
  num_hierarchies: 2                       # Reduced from 3
  predictor_depth: 2                       # Simplified
```

**Rationale:**
- Smallest viable ViT (5M params vs 22M Small, 86M Base)
- Fast forward/backward passes on CPU
- Still captures hierarchical learning concept
- Proven architecture (timm library)

### 3. Training Configuration: Conservative & Stable

**Key Parameters:**
```yaml
training:
  epochs: 20                    # Quick validation
  batch_size: 8                 # CPU memory safe
  accumulation_steps: 4         # Effective batch: 32
  lr: 5.0e-5                    # Conservative
  warmup_epochs: 2              # Fast warmup
  use_amp: false                # No benefit on CPU
```

**Design Philosophy:**
- Small batch size (8) to avoid OOM
- Gradient accumulation (4 steps) for effective batch of 32
- Conservative learning rate for stability
- Disabled mixed precision (CPU doesn't benefit)

### 4. Training Duration: 20 Epochs

**Time Breakdown:**
```
Setup Phase:         1-2 hours
Training Phase:      18-24 hours (6h per epoch × 20)
Evaluation Phase:    2-4 hours
Analysis Phase:      1 hour
─────────────────────────────────
Total:              ~24-30 hours
```

**Rationale:**
- 20 epochs sufficient to validate architecture
- Shows learning trends without excessive time
- Checkpoints every 5 epochs for early stopping
- Can extend to 50+ epochs if results promising

---

## Expected Outcomes

### Success Metrics

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| **Training Loss** | < 0.5 | < 0.3 | < 0.2 |
| **Linear Probe Acc.** | > 50% | > 60% | > 70% |
| **k-NN Acc. (k=20)** | > 40% | > 50% | > 60% |
| **Feature Variance** | > 0.05 | > 0.1 | > 0.15 |
| **Effective Rank** | > 48 | > 96 | > 144 |

### Baseline Comparisons

| Method | Architecture | Epochs | Linear Probe | Notes |
|--------|--------------|--------|--------------|-------|
| Random | - | 0 | ~10% | Lower bound |
| Rotation | ViT-Tiny | 100 | 45-55% | Classic SSL |
| **H-JEPA (Ours)** | ViT-Tiny | 20 | 50-70% | Target |
| SimCLR | ViT-Tiny | 100 | 68-75% | SOTA contrastive |
| Supervised | ViT-Tiny | 100 | 95%+ | Upper bound |

**Interpretation:**
- Must beat random baseline (>50% ✓)
- Should approach rotation baseline (55%)
- With more epochs, target SimCLR level (70%+)

---

## Risk Mitigation

### Top 3 Risks & Solutions

#### 1. Out-of-Memory (OOM) Errors
**Symptoms:** Training crashes, system freeze, swap usage
**Solution:**
```yaml
# Emergency fallback config
data:
  batch_size: 4         # Half the size
  num_workers: 1        # Reduce overhead
training:
  accumulation_steps: 8 # Maintain effective batch
```

#### 2. Training Too Slow (> 8h/epoch)
**Symptoms:** Progress bar estimates > 8h
**Solutions:**
- Reduce to 10 epochs instead of 20
- Cache dataset in RAM
- Use smaller image size (160 instead of 224)
- Enable CPU optimizations (MKL, OpenMP)

#### 3. Representation Collapse
**Symptoms:** Flat loss, feature variance < 0.01
**Solutions:**
- Reduce learning rate by 5x
- Increase EMA momentum (0.996 → 0.999)
- Restart from earlier checkpoint
- Add variance regularization

---

## Quick Start Commands

### 1. Automated Setup & Test
```bash
cd /home/user/H-JEPA
./run_training.sh
```

### 2. Manual Training
```bash
# In screen/tmux for persistence
screen -S hjepa_training
python scripts/train.py --config configs/cpu_cifar10.yaml --device cpu
# Detach: Ctrl+A, D
```

### 3. Monitor Progress
```bash
# Terminal 1: Watch logs
tail -f results/logs/cpu_cifar10/training.log

# Terminal 2: TensorBoard
tensorboard --logdir results/logs/cpu_cifar10 --port 6006

# Terminal 3: System resources
watch -n 5 'free -h && echo "---" && ps aux | grep python | head -1'
```

### 4. Evaluate After Training
```bash
python scripts/evaluate.py \
    --checkpoint results/checkpoints/cpu_cifar10/best_model.pth \
    --dataset cifar10 \
    --data-path ./data/cifar10 \
    --eval-type all \
    --device cpu
```

---

## File Locations

| File | Location | Purpose |
|------|----------|---------|
| **Full Training Plan** | `/home/user/H-JEPA/TRAINING_PLAN.md` | Complete documentation |
| **Configuration** | `/home/user/H-JEPA/configs/cpu_cifar10.yaml` | Training config |
| **Setup Script** | `/home/user/H-JEPA/run_training.sh` | Automated setup |
| **Training Script** | `/home/user/H-JEPA/scripts/train.py` | Main training |
| **Evaluation Script** | `/home/user/H-JEPA/scripts/evaluate.py` | Model evaluation |
| **Checkpoints** | `/home/user/H-JEPA/results/checkpoints/cpu_cifar10/` | Saved models |
| **Logs** | `/home/user/H-JEPA/results/logs/cpu_cifar10/` | Training logs |

---

## Checkpoints & Milestones

### Training Milestones

| Epoch | Time | Action | Success Criteria |
|-------|------|--------|------------------|
| 1 | 6h | First checkpoint | Loss decreasing, no errors |
| 5 | 30h | Early eval | Loss < 0.5, variance > 0.05 |
| 10 | 60h | Mid eval | Loss < 0.4, run k-NN test |
| 15 | 90h | Late eval | Loss < 0.3, stable training |
| 20 | 120h | Final eval | Full evaluation suite |

### Evaluation Timeline

**After Training Completes (2-4 hours):**
1. Linear Probe (1h): Train linear classifier on frozen features
2. k-NN Evaluation (30m): Nearest neighbor classification
3. Feature Quality (30m): Analyze collapse metrics
4. Generate Report (30m): Visualizations and summary

---

## Next Steps After Initial Training

### If Results Are Good (>60% linear probe)
1. ✅ **Scale up:** Train for 50-100 epochs
2. ✅ **Upgrade:** Use ViT-Small if GPU available
3. ✅ **Challenge:** Try CIFAR-100 or ImageNet-100
4. ✅ **Optimize:** Hyperparameter tuning

### If Results Are Mediocre (50-60%)
1. 🔍 **Diagnose:** Check for partial collapse
2. 🔧 **Adjust:** Modify learning rate or masking
3. ⏱️ **Extend:** Train longer (50 epochs)
4. 📊 **Compare:** Analyze hierarchy differentiation

### If Results Are Poor (<50%)
1. ⚠️ **Check collapse:** Variance, rank, entropy
2. 🐛 **Debug:** Run unit tests, verify data
3. 🔄 **Restart:** Different seed or config
4. 📖 **Review:** Revisit TRAINING_PLAN.md Section 5

---

## Critical Success Factors

### Must-Have for Success
1. ✅ Training loss consistently decreases
2. ✅ No NaN/Inf values in loss
3. ✅ Feature variance > 0.05 throughout
4. ✅ RAM usage < 11GB (no OOM)
5. ✅ Linear probe accuracy > 50%

### Nice-to-Have for Excellence
1. 🎯 Loss decreases smoothly (no oscillations)
2. 🎯 Feature variance > 0.1
3. 🎯 Effective rank > 96 (50% of dims)
4. 🎯 Clear hierarchy differentiation
5. 🎯 Linear probe accuracy > 60%

---

## Resource Requirements

### This Plan (Minimum)
- **Compute:** CPU with 4+ cores
- **RAM:** 8GB minimum, 12GB safe
- **Storage:** 10GB for everything
- **Time:** 24-30 hours total
- **Cost:** $0 (local hardware)

### Recommended (Better Results)
- **Compute:** GPU with 8GB+ VRAM
- **RAM:** 16GB
- **Storage:** 50GB
- **Time:** 2-6 hours total
- **Cost:** ~$5-20 cloud GPU

### Optimal (Production)
- **Compute:** Multi-GPU (A100/V100)
- **RAM:** 64GB
- **Storage:** 500GB+
- **Time:** Hours for ImageNet
- **Cost:** $50-500 depending on scale

---

## Key Takeaways

1. **This is a validation run**, not production training
   - Goal: Prove H-JEPA architecture works
   - 20 epochs is enough to see learning trends
   - Can scale up later with GPU

2. **CPU training is slow but viable**
   - 18-24 hours for 20 epochs is reasonable
   - Use screen/tmux to keep training alive
   - Monitor RAM to avoid OOM

3. **Success = >50% linear probe accuracy**
   - This proves features are meaningful
   - 5x better than random baseline
   - Foundation for further optimization

4. **Fallback plans exist for all risks**
   - OOM → reduce batch size
   - Too slow → reduce epochs
   - Collapse → adjust learning rate
   - See full plan for details

5. **Path to production is clear**
   - This run validates implementation
   - GPU training → 10-50x speedup
   - 100+ epochs → 70-80% accuracy
   - ImageNet training → SOTA results

---

## Questions? See Full Documentation

📄 **Full Training Plan:** `/home/user/H-JEPA/TRAINING_PLAN.md`
- Detailed rationale for all decisions
- Complete troubleshooting guide
- Alternative configurations
- FAQ and contact info

🚀 **Setup Script:** `/home/user/H-JEPA/run_training.sh`
- Automated setup and testing
- Verifies all requirements
- Provides next steps

📊 **Configuration:** `/home/user/H-JEPA/configs/cpu_cifar10.yaml`
- All training parameters
- Documented choices
- Ready to use

---

**Created:** 2025-11-14
**Status:** Ready for execution
**Estimated Time:** 24-30 hours total
**Expected Outcome:** 50-70% linear probe accuracy

**Good luck with training!** 🎯

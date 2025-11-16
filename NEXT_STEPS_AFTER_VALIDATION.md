# Next Steps After Validation Run

Quick reference for actions to take once the 5-epoch validation training completes.

## 📊 Step 1: Run Automated Analysis

```bash
# Automated analysis (recommended)
./scripts/auto_analyze_and_recommend.sh training_run.log

# Or manual analysis
python3.11 scripts/analyze_validation_run.py --log training_run.log --output-dir results/validation_analysis
```

**Output:**
- `results/validation_analysis/validation_report.md` - Full analysis report
- `results/validation_analysis/validation_training_curves.png` - Training curves
- `results/validation_analysis/analysis.json` - Detailed metrics

## 📈 Step 2: Review Key Metrics

Check these critical indicators:

### ✅ Success Criteria
- [ ] Loss decreased by ≥ 20% from initial value
- [ ] Training speed ≥ 2.5 it/s average
- [ ] No crashes or MPS errors
- [ ] Loss curve is smooth (not diverging)

### 📊 Key Numbers to Note
- **Initial Loss**: [from report]
- **Final Loss**: [from report]
- **Average Speed**: [from report] it/s
- **Total Time**: [from logs]

## 🎯 Step 3: Choose Next Training Run

Based on validation results, choose one option:

### Option A: Quick Baseline (20 epochs, ~2.5 hours)

**When to choose:**
- Validation converged well
- Want faster results
- Need baseline for comparison

```bash
python3.11 scripts/train.py --config configs/m1_max_full_20epoch.yaml
```

**Expected:** 70-78% linear probe accuracy

### Option B: Competitive Results (100 epochs, ~12-14 hours)

**When to choose:**
- Validation was stable
- Can run overnight
- Want publication-quality results

```bash
python3.11 scripts/train.py --config configs/m1_max_full_100epoch.yaml
```

**Expected:** 80-85% linear probe accuracy

### Option C: Continue Validation (50 epochs, ~6-7 hours)

**When to choose:**
- Validation still improving at epoch 5
- Want more data before committing to large run

```bash
# Modify config to start from checkpoint and continue to 50 epochs
python3.11 scripts/train.py \
    --config configs/m1_max_quick_val.yaml \
    --resume results/checkpoints/best_checkpoint.pt \
    --epochs 50
```

## 🔍 Step 4: Optional - Evaluate Validation Checkpoint

Before full training, you can evaluate the 5-epoch validation checkpoint:

```bash
# Quick evaluation
python3.11 scripts/evaluate.py \
    --checkpoint results/checkpoints/best_checkpoint.pt \
    --config configs/m1_max_quick_val.yaml \
    --eval-type linear_probe knn feature_quality

# Or use automated script
./scripts/quick_eval_after_training.sh results/checkpoints/best_checkpoint.pt
```

**This will show:**
- Linear probe accuracy (expected: 30-50% after only 5 epochs)
- k-NN accuracy
- Feature quality metrics (check for collapse)

## ⚙️ Step 5: Optimize Full Training Config (Optional)

If validation revealed issues, adjust before full training:

### If Loss Converged Too Slowly
```yaml
training:
  lr: 0.00015  # Increase from 0.0001
```

### If Loss Was Noisy
```yaml
training:
  lr: 0.00005  # Decrease from 0.0001
  batch_size: 48  # Increase from 32
```

### If Speed Was Variable
```yaml
data:
  num_workers: 2  # Decrease from 4
```

### If Memory Was Tight
```yaml
data:
  batch_size: 24  # Decrease from 32
```

## 🚀 Step 6: Launch Full Training

Once you've chosen your configuration:

### Monitor Training

```bash
# In one terminal - run training
python3.11 scripts/train.py --config configs/m1_max_full_20epoch.yaml 2>&1 | tee full_training.log

# In another terminal - monitor progress
watch -n 30 tail -20 full_training.log

# Or monitor TensorBoard
tensorboard --logdir results/logs/tensorboard
```

### Set Up Automated Evaluation

Create a script to auto-evaluate when done:

```bash
#!/bin/bash
# auto_eval_when_done.sh

while pgrep -f "train.py.*m1_max_full" > /dev/null; do
    sleep 60
done

echo "Training complete! Running evaluation..."
./scripts/quick_eval_after_training.sh results/checkpoints/best_checkpoint.pt
```

Run in background:
```bash
chmod +x auto_eval_when_done.sh
./auto_eval_when_done.sh &
```

## 📊 Step 7: Post-Training Evaluation

After full training completes:

```bash
# Comprehensive evaluation
./scripts/quick_eval_after_training.sh results/checkpoints/best_checkpoint.pt

# Or individual evaluations
python3.11 scripts/evaluate.py \
    --checkpoint results/checkpoints/best_checkpoint.pt \
    --config configs/m1_max_full_20epoch.yaml \
    --eval-type all
```

**Expected Outputs:**
- Linear probe: 70-85% (depending on epochs/model)
- k-NN: 65-80%
- Feature quality: No collapse detected
- Visualization: Attention maps, feature embeddings

## 📝 Step 8: Document Results

Update project documentation:

```bash
# Create results summary
cat >> RESULTS.md <<EOF

## Training Run: $(date +%Y-%m-%d)

**Configuration:** [config file]
**Architecture:** [encoder type]
**Epochs:** [number]
**Training Time:** [hours]

**Results:**
- Linear Probe Accuracy: [XX.X%]
- k-NN Accuracy (k=20): [XX.X%]
- Feature Effective Rank: [XX.X]

**Observations:**
- [Any notable findings]

**Checkpoint:** results/checkpoints/[name].pt
EOF
```

## 🎓 Tips for Successful Training

### During Training:
1. **Don't interrupt** unless there's an error
2. **Monitor disk space** (checkpoints can be large)
3. **Check logs periodically** for warnings
4. **Let it run uninterrupted** (especially overnight runs)

### If Training Fails:
1. Check `training_run.log` for errors
2. Review `results/logs/tensorboard` for anomalies
3. Resume from last checkpoint if possible:
   ```bash
   python3.11 scripts/train.py \
       --config configs/[config].yaml \
       --resume results/checkpoints/latest_checkpoint.pt
   ```

### System Optimization:
1. **Close other apps** to free resources
2. **Disable sleep** during training:
   ```bash
   caffeinate -i python3.11 scripts/train.py --config configs/[config].yaml
   ```
3. **Monitor temperature** (M1 Max should handle it fine)

## 🆘 Troubleshooting

### Loss Explodes (NaN/Inf)
- Reduce learning rate by 10x
- Check for data issues
- Resume from earlier checkpoint

### Out of Memory
- Reduce batch_size
- Reduce num_workers
- Use smaller model (ViT-Tiny instead of ViT-Small)

### Training Too Slow
- Verify MPS is being used (`device: mps` in logs)
- Reduce num_workers if CPU bound
- Close resource-intensive applications

### Checkpoints Not Saving
- Check disk space
- Verify write permissions in results/checkpoints/
- Check logs for errors

## 📚 Reference Baselines

For CIFAR-10 H-JEPA (your implementation):

| Config | Architecture | Epochs | Time | Expected Accuracy |
|--------|-------------|--------|------|-------------------|
| Validation | ViT-Tiny | 5 | 40 min | 30-50% |
| Quick Baseline | ViT-Tiny | 20 | 2.5 hrs | 70-78% |
| Medium | ViT-Tiny | 50 | 6-7 hrs | 75-80% |
| Full | ViT-Small | 100 | 12-14 hrs | 80-85% |

## ✅ Checklist Before Full Training

- [ ] Validation run completed successfully
- [ ] Analysis results reviewed
- [ ] Configuration chosen (20 or 100 epochs)
- [ ] System has enough disk space (>10GB free)
- [ ] No other heavy processes running
- [ ] Time allocated (for overnight runs)
- [ ] Monitoring plan in place

## 🎯 Success Metrics

Your full training is successful if:

**For 20-epoch run:**
- Linear probe: ≥ 70%
- k-NN: ≥ 65%
- No collapse detected
- Smooth loss curve

**For 100-epoch run:**
- Linear probe: ≥ 80%
- k-NN: ≥ 75%
- No collapse detected
- Effective rank > 50% of embedding dim

---

**Ready to proceed?** Start with the analysis, review results, then launch your full training!

Good luck! 🚀

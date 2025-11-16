# Overnight Training Quick Reference Card

**Print this page and keep it handy during your overnight run!**

---

## 🚀 Quick Start

### Conservative (Recommended First)
```bash
screen -S hjepa
python scripts/train.py --config configs/overnight_training_conservative.yaml --device mps
# Ctrl+A, D to detach
```

### Aggressive (After Conservative Succeeds)
```bash
# Verify dataset first!
ls data/imagenet/train/ | wc -l  # Should be ~100

screen -S hjepa
python scripts/train.py --config configs/overnight_training_aggressive.yaml --device mps
# Ctrl+A, D to detach
```

---

## ⏰ Expected Timeline

| Config | Time/Epoch | Total | Performance |
|--------|-----------|-------|-------------|
| **Conservative** | 9-10 min | 7.5-8h | 60-70% |
| **Aggressive** | 10-12 min | 7.3-8.5h | 70-78% |

---

## 📊 Success Criteria at a Glance

### Conservative
- ✅ Time: <8.5 hours
- ✅ Loss: <0.5
- ✅ Linear probe: >60%
- ✅ No NaN/Inf

### Aggressive
- ✅ Time: <9 hours
- ✅ Loss: <0.3
- ✅ Linear probe: >68%
- ✅ Memory: <28 GB

---

## 🔍 Monitoring Commands

### Check Status
```bash
# Reattach to screen
screen -r hjepa

# View last 20 lines
tail -20 results/overnight_*/logs/training.log

# Watch live
tail -f results/overnight_*/logs/training.log

# Check memory
top -l 1 | grep -E "PhysMem|Python"
```

### Check Progress
```bash
# Find current epoch
grep "Epoch" results/overnight_*/logs/training.log | tail -1

# Find latest loss
grep "Loss" results/overnight_*/logs/training.log | tail -1

# Check disk space
df -h | grep disk
```

---

## ⚠️ Warning Signs

| Issue | Symptom | Action |
|-------|---------|--------|
| **Too Slow** | >15 min/epoch | Reduce epochs or batch size |
| **Memory High** | >28 GB used | Kill and reduce batch_size |
| **NaN Loss** | Loss = NaN/Inf | Kill, reduce LR, restart |
| **No Progress** | Loss flat for 10 epochs | Check logs for errors |

---

## 🛠️ Emergency Procedures

### Stop Training Gracefully
```bash
# Find process
ps aux | grep train.py

# Stop gracefully (saves checkpoint)
kill -SIGINT <PID>
```

### Force Kill
```bash
# If frozen
kill -9 <PID>
```

### Resume from Checkpoint
```bash
python scripts/train.py \
  --config configs/overnight_training_*.yaml \
  --resume results/overnight_*/checkpoints/checkpoint_epoch_X.pth \
  --device mps
```

---

## 🔧 Quick Fixes

### Out of Memory
```yaml
# Edit config file:
batch_size: 32 → 16    # or 24
num_workers: 4 → 2
```

### Too Slow
```yaml
# Edit config file:
epochs: 50 → 40        # or 35
log_images: false
log_attention: false
```

### Training Unstable
```yaml
# Edit config file:
lr: 0.00015 → 0.0001
clip_grad: 3.0 → 1.0
warmup_epochs: 5 → 10
```

---

## 📈 Expected Metrics by Epoch

### Conservative (CIFAR+STL, 50 epochs)

| Epoch | Loss | k-NN | Time |
|-------|------|------|------|
| 1 | ~1.5 | ~15% | 0h 09m |
| 10 | ~0.8 | ~30% | 1h 30m |
| 25 | ~0.4 | ~50% | 3h 45m |
| 40 | ~0.3 | ~60% | 6h 00m |
| 50 | ~0.25 | ~65% | 7h 30m |

### Aggressive (ImageNet-100, 40 epochs)

| Epoch | Loss | k-NN | Time |
|-------|------|------|------|
| 1 | ~2.0 | ~10% | 0h 11m |
| 10 | ~0.6 | ~40% | 1h 50m |
| 20 | ~0.35 | ~58% | 3h 40m |
| 30 | ~0.25 | ~65% | 5h 30m |
| 40 | ~0.20 | ~70% | 7h 20m |

*Note: Your results may vary ±10% depending on exact hardware and conditions*

---

## 💾 Checkpoint Locations

```
Conservative:
results/overnight_conservative/checkpoints/
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
├── checkpoint_epoch_30.pth
├── checkpoint_epoch_40.pth
├── checkpoint_epoch_50.pth
└── best_model.pth

Aggressive:
results/overnight_aggressive/checkpoints/
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_15.pth
├── checkpoint_epoch_20.pth
├── checkpoint_epoch_25.pth
├── checkpoint_epoch_30.pth
├── checkpoint_epoch_35.pth
├── checkpoint_epoch_40.pth
└── best_model.pth
```

---

## 📱 Morning Checklist

When you wake up:

1. **Check if running:**
   ```bash
   ps aux | grep train.py
   ```

2. **Check completion:**
   ```bash
   tail -20 results/overnight_*/logs/training.log
   ```

3. **Check performance:**
   ```bash
   grep "Linear probe\|k-NN" results/overnight_*/logs/training.log | tail -3
   ```

4. **If successful:**
   ```bash
   # Run full evaluation
   python scripts/evaluate.py \
     --checkpoint results/overnight_*/checkpoints/best_model.pth \
     --eval-type all
   ```

5. **If failed:**
   - Check logs for error messages
   - Note what epoch it failed at
   - Review fallback plan in guide
   - Plan retry with fixes

---

## 📋 Optimization Impact Table

| Optimization | Speedup | Accuracy | Memory | Phase |
|--------------|---------|----------|--------|-------|
| Flash Attention | 2-5x | - | -40% | 1 |
| LayerScale | - | +0.5-1% | - | 1 |
| ImageNet-100 | 0.8x | +10-15% | - | 2 |
| Grad Checkpoint | 0.85x | - | -50% | 2 |
| Higher LR | 1.3x | - | - | 2 |
| FPN | 0.95x | +1-2% | +5% | 3 |
| Contrastive | 0.98x | +0.8-1% | +5% | 3 |

*Note: Effects are approximate and may combine non-linearly*

---

## 🎯 Target Performance

### Conservative (CIFAR+STL)
```
Minimum:  55-60% linear probe
Target:   60-65% linear probe
Stretch:  65-70% linear probe

Comparison to baseline: +10-15%
```

### Aggressive (ImageNet-100)
```
Minimum:  63-68% linear probe
Target:   68-73% linear probe
Stretch:  73-78% linear probe

Comparison to baseline: +20-25%
Comparison to conservative: +5-10%
```

---

## 🔄 Next Steps After Success

### If Conservative Succeeds (60-70%)
1. ✅ Phase 1 optimizations validated
2. → Run Aggressive next
3. → Compare results
4. → Plan 100+ epoch run

### If Aggressive Succeeds (70-78%)
1. ✅ All Phase 1-3 optimizations work
2. → Scale to full ImageNet
3. → Add multi-crop training
4. → Target 75%+ performance

### If Both Succeed
1. ✅ Complete validation of optimizations
2. → Quantify each feature's impact
3. → Write up results
4. → Plan publication-quality run

---

## 📞 Help & Resources

**Full Documentation:**
- `OVERNIGHT_TRAINING_GUIDE.md` - Complete guide
- `OVERNIGHT_TRAINING_RECOMMENDATION.md` - Decision guide
- Main `README.md` - General H-JEPA documentation

**Config Files:**
- `configs/overnight_training_conservative.yaml`
- `configs/overnight_training_aggressive.yaml`

**Troubleshooting:**
- Check logs: `results/overnight_*/logs/training.log`
- Check TensorBoard: `http://localhost:6006`
- GitHub Issues: (your repo URL)

---

## ✅ Pre-Flight Checklist

Before starting overnight run:

**Environment:**
- [ ] 20+ GB RAM free
- [ ] 10+ GB disk free
- [ ] No other heavy processes running
- [ ] Terminal won't sleep/close

**Dataset:**
- [ ] CIFAR-10 downloaded (Conservative)
- [ ] ImageNet-100 verified (Aggressive)
- [ ] Paths correct in config

**Testing:**
- [ ] 1 epoch test completed successfully
- [ ] Time per epoch acceptable (9-12 min)
- [ ] No errors in logs

**Monitoring:**
- [ ] Know how to check status
- [ ] Know how to kill if needed
- [ ] TensorBoard accessible (optional)

**Backup:**
- [ ] Fallback plan ready
- [ ] Config file backed up
- [ ] Can resume from checkpoint

---

## 🎉 Expected Final Results

After 8 hours, you should have:

**Conservative:**
- ✅ Trained model (50 epochs)
- ✅ Linear probe: 60-70%
- ✅ Validated Flash Attention + LayerScale work
- ✅ Established baseline for comparison

**Aggressive:**
- ✅ Trained model (40 epochs)
- ✅ Linear probe: 70-78%
- ✅ Validated Phase 1-3 optimizations
- ✅ Demonstrated ImageNet-100 benefit

**Both:**
- ✅ Direct A/B comparison
- ✅ Quantified optimization impact
- ✅ Clear path to 75%+ with more training
- ✅ Ready to scale to full ImageNet

---

**Print this page and keep it visible during your overnight run!**

**Quick reminder:** Start with **Conservative** unless you're absolutely confident everything works.

Good luck! 🚀

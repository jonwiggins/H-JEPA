# H-JEPA Quick Start - M1 Max Edition ⚡

**TL;DR:** Everything is ready. Pick your training profile and run.

---

## 🚀 Three Simple Commands

### 1. Quick Validation (40 min) - Currently Running ✅
```bash
python3.11 scripts/train.py --config configs/m1_max_quick_val.yaml
```
**Purpose:** System test
**When complete:** Proceed to option 2 or 3

---

### 2. Medium Training (2.5 hrs) - Recommended Next
```bash
python3.11 scripts/train.py --config configs/m1_max_full_20epoch.yaml
```
**Purpose:** Baseline performance (70-78% accuracy)
**After training:** Run evaluation (see below)

---

### 3. Full Training (13 hrs) - Overnight Run
```bash
python3.11 scripts/train.py --config configs/m1_max_full_100epoch.yaml
```
**Purpose:** Competitive results (80-85% accuracy)
**Best time:** Start before bed, results in morning

---

## 📊 After Training: Evaluate

```bash
# Automatic comprehensive evaluation
./scripts/quick_eval_after_training.sh

# Or manual evaluation
python3.11 scripts/evaluate.py \
    --checkpoint results/checkpoints/checkpoint_best.pth \
    --dataset cifar10 \
    --hierarchy-levels 0 1
```

---

## 📈 Monitor Training

```bash
# Real-time with TensorBoard
tensorboard --logdir results/logs

# Watch log file
tail -f results/logs/m1_max_*/training.log

# Check progress
ls -lh results/checkpoints/
```

---

## 🎯 Expected Results

| Training | Time | Accuracy | Status |
|----------|------|----------|--------|
| 5 epochs | 40 min | 40-60% | ✅ Running |
| 20 epochs | 2.5 hrs | 70-78% | 📋 Ready |
| 100 epochs | 13 hrs | 80-85% | 📋 Ready |

---

## 🔧 Common Adjustments

### Reduce batch size if OOM
```bash
python3.11 scripts/train.py \
    --config configs/m1_max_full_20epoch.yaml \
    --batch_size 24
```

### Resume interrupted training
```bash
python3.11 scripts/train.py \
    --config configs/m1_max_full_20epoch.yaml \
    --resume results/checkpoints/checkpoint_latest.pth
```

### Disable W&B logging
```bash
python3.11 scripts/train.py \
    --config configs/m1_max_full_20epoch.yaml \
    --no_wandb
```

---

## 📁 Key Files

- **Configs:** `configs/m1_max_*.yaml`
- **Results:** `results/checkpoints/`, `results/logs/`
- **Guides:** `M1_MAX_TRAINING_GUIDE.md`, `VALIDATION_RUN_ANALYSIS.md`
- **Evaluation:** `scripts/quick_eval_after_training.sh`

---

## ⚡ Performance

- **ViT-Tiny:** ~3.2 it/s, 8 min/epoch
- **ViT-Small:** ~2.0-2.5 it/s, 12-15 min/epoch
- **M1 Max GPU:** Fully utilized via MPS
- **Memory:** 10-12GB / 32GB available

---

## 🎓 Full Documentation

For deep dive, see: `M1_MAX_TRAINING_GUIDE.md`

---

**That's it! Everything is automated and ready to go.** 🚀

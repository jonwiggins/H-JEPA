# Configuration Files Quick Reference

## Status Overview

| Config | Type | Status | Recommendation |
|--------|------|--------|-----------------|
| **CORE CONFIGS** |
| `default.yaml` | Baseline | ✅ Good | Use as reference |
| `pure_ijepa.yaml` | Paper replica | ✅ Good | Use for I-JEPA experiments |
| `overnight_safe.yaml` | Safe 8h training | ✅ VERIFIED | Use for overnight training |
| **QUICK VALIDATION** |
| `quick_validation.yaml` | CPU test | ⚠️ Outdated | Use `m1_max_quick_val.yaml` instead |
| `m1_max_quick_val.yaml` | M1 test | ✅ Good | Use for 5-minute test |
| `small_experiment.yaml` | Debug | ✅ Good | Use for quick experiments |
| `validation_test.yaml` | Features test | ✅ Good | Use to validate latest features |
| **M1 MAX TRAINING** |
| `m1_max_full_20epoch.yaml` | M1 20-epoch | ✅ Good | ~2.5 hours |
| `m1_max_full_100epoch.yaml` | M1 100-epoch | ✅ Good | ~12-13 hours |
| `m1_max_imagenet100_100epoch.yaml` | M1 ImageNet | ✅ Good | ~10-15 hours |
| **CPU-ONLY** |
| `cpu_cifar10.yaml` | CPU training | ✅ Works | Slow, for testing only |
| **FOUNDATION MODELS** |
| `foundation_model_mini.yaml` | Multi-dataset | ⚠️ Redundant | Consolidate into imagenet100_* |
| `foundation_model_cifar_stl.yaml` | Multi-dataset | ✅ Good | Use for CIFAR+STL training |
| `imagenet100_multi_dataset.yaml` | Multi-dataset | ✅ Good | Use for ImageNet-100 based training |
| **FEATURES** |
| `rope_experiment.yaml` | RoPE position encoding | ✅ Valid | Use for position encoding experiments |
| `fpn_example.yaml` | Feature Pyramid Networks | ✅ Valid | Use for multi-scale feature learning |
| `fpn_concat_example.yaml` | FPN variant | ⚠️ Redundant | Use fpn_example with alt fusion_method |
| `cjepa_example.yaml` | Contrastive JEPA | ✅ Valid | Use for hybrid JEPA+Contrastive |
| **ARCHIVE (Incomplete/Broken)** |
| `overnight_training_conservative.yaml` | 8h training | ❌ BROKEN | **DO NOT USE** - Use overnight_safe.yaml |
| `overnight_training_aggressive.yaml` | 8h training | ❌ BROKEN | **DO NOT USE** - Use overnight_safe.yaml |
| `deit3_augmentation.yaml` | Strong augmentation | ⚠️ Not integrated | Archive - DeiT III not in pipeline |
| `multicrop_training.yaml` | Multi-crop | ⚠️ Unclear | Archive - implementation unclear |
| `sigreg_example.yaml` | SIGReg loss | ⚠️ Experimental | Archive - newer LeJEPA approach |

---

## Which Config Should I Use?

### I want to... | Use this config
---|---
Train baseline H-JEPA on ImageNet | `default.yaml`
Reproduce I-JEPA paper exactly | `pure_ijepa.yaml`
Quick 5-minute validation on M1 Max | `m1_max_quick_val.yaml`
Quick 1-hour debugging session | `small_experiment.yaml`
Overnight training (8 hours) on M1 Max | `overnight_safe.yaml`
Full training (20 hours) on M1 Max | `m1_max_full_100epoch.yaml`
Foundation model with multiple datasets | `foundation_model_mini.yaml` or `imagenet100_multi_dataset.yaml`
Test RoPE position encoding | `rope_experiment.yaml`
Test FPN multi-scale features | `fpn_example.yaml`
Test Contrastive JEPA | `cjepa_example.yaml`
Train on CPU only | `cpu_cifar10.yaml`
Test latest code changes | `validation_test.yaml`

---

## Critical Issues

### 🔴 DO NOT USE (Broken)
- **`overnight_training_conservative.yaml`** - Will crash with TypeError due to Flash Attention parameter mismatch
- **`overnight_training_aggressive.yaml`** - Will crash for same reason

**Fix:** Use `overnight_safe.yaml` instead for 8-hour overnight training

### ✅ USE INSTEAD
- **`overnight_safe.yaml`** - Verified working, same performance with verified features only

---

## Parameter Inconsistencies to Know About

### Naming Variations (You may see either)

**EMA Configuration:**
```yaml
# Variation A (most configs)
model:
  ema:
    momentum: 0.996
    momentum_end: 1.0
    momentum_warmup_epochs: 30

# Variation B (foundation configs)
model:
  target_encoder:
    ema_decay: 0.996
    ema_end_decay: 1.0
    ema_anneal_end_step: 300000
```

**Data Augmentation:**
```yaml
# Variation A (most configs)
data:
  augmentation:
    color_jitter: 0.4
    horizontal_flip: true
    random_crop: true

# Variation B (foundation configs)
data:
  transforms:
    crop_scale: [0.8, 1.0]
    horizontal_flip: true
    color_jitter: 0.1
```

**Experiment Naming:**
```yaml
# Variation A (most configs)
logging:
  experiment_name: "hjepa_default"

# Variation B (foundation configs)
experiment:
  name: "foundation_mini"
```

---

## Configuration for Different Hardware

### M1 Max (32GB Unified Memory)
**Recommended:** Any `m1_max_*.yaml` config
- `m1_max_quick_val.yaml` - 5-10 minute test
- `m1_max_full_20epoch.yaml` - 2.5 hour training
- `m1_max_full_100epoch.yaml` - 12 hour training
- `m1_max_imagenet100_100epoch.yaml` - 10-15 hour training

### NVIDIA GPU (16GB+)
**Recommended:** `default.yaml` with modifications:
```yaml
device: "cuda"  # Change from default
training:
  batch_size: 256  # Can increase for more GPU memory
```

### CPU Only
**Recommended:** `cpu_cifar10.yaml`
**Note:** Very slow, expect 18-24 hours for 20 epochs

---

## Key Parameter Values Across Configs

### Model Sizes Used

| Config | Encoder | Embed Dim | Parameters |
|--------|---------|-----------|-----------|
| Quick tests | ViT-Tiny | 192 | ~5M |
| Small exp | ViT-Small | 384 | ~22M |
| Default | ViT-Base | 768 | ~86M |

### Training Durations

| Config | Epochs | Expected Time |
|--------|--------|---------------|
| `quick_validation.yaml` | 5 | ~30 minutes |
| `m1_max_quick_val.yaml` | 5 | ~50 minutes |
| `small_experiment.yaml` | 50 | ~8 hours |
| `m1_max_full_20epoch.yaml` | 20 | ~2.5 hours |
| `m1_max_full_100epoch.yaml` | 100 | ~12-13 hours |
| `overnight_safe.yaml` | 45 | ~7.5-8 hours |

### Hierarchy Levels Used

| Config | Levels | Weights |
|--------|--------|---------|
| Pure I-JEPA | 1 | [1.0] |
| Quick tests | 2 | [1.0, 0.5] |
| Most configs | 3 | [1.0, 0.5, 0.25] |
| Some configs | 3 | [1.0, 0.7, 0.5] |

---

## Common Modifications

### Enable RoPE Position Encoding
```yaml
model:
  rope:
    use_rope: true  # Change from false
```
**Expected:** +1-2% accuracy improvement

### Enable Feature Pyramid Networks
```yaml
model:
  fpn:
    use_fpn: true  # Change from false
```
**Expected:** +2-3% accuracy improvement

### Enable Gradient Checkpointing (Save Memory)
```yaml
training:
  use_gradient_checkpointing: true
```
**Trade-off:** ~20% slower but saves ~30% memory

### Disable Mixed Precision (Stability)
```yaml
training:
  use_amp: false  # Change from true
```
**Trade-off:** ~20% slower but more stable

### Increase Batch Size (For Large GPU)
```yaml
data:
  batch_size: 256  # Increase from 128

training:
  lr: 3.0e-4  # Scale LR linearly: 1.5e-4 * (256/128)
```

---

## Security Checklist

- [ ] Check that `data_path` points to actual dataset location
- [ ] Never commit configs with `wandb.entity` field filled in
- [ ] Verify no absolute paths like `/path/to/dataset` are used
- [ ] Ensure `/tmp` paths are never used for important data
- [ ] Validate that checkpoint directories have appropriate permissions

---

## File Sizes and Complexity

| Config | File Size | Parameters | Complexity |
|--------|-----------|-----------|-----------|
| `quick_validation.yaml` | 2.3 KB | Basic (50+ params) | Low |
| `default.yaml` | 4.7 KB | Complete (80+ params) | Medium |
| `overnight_safe.yaml` | 11.8 KB | Complete + docs | Medium |
| `foundation_model_mini.yaml` | ~3 KB | Multi-dataset | Medium |
| `sigreg_example.yaml` | 5.5 KB | Complex loss (120+ params) | High |

---

## Troubleshooting Config Issues

### Error: "TypeError: create_encoder() got unexpected keyword argument"
**Cause:** Flash Attention or LayerScale enabled
**Solution:** Use `overnight_safe.yaml` instead or disable problematic parameters

### Error: "File not found: ./data"
**Cause:** Data path doesn't exist
**Solution:** Update `data_path` in config or download dataset first

### Error: "CUDA out of memory"
**Cause:** Batch size too large
**Solution:** Reduce `batch_size` in config or enable gradient checkpointing

### Error: "Config validation failed"
**Cause:** Inconsistent parameters (e.g., hierarchy_weights length doesn't match num_hierarchies)
**Solution:** Review config carefully - see full report for what values are valid

---

## Migration Guide (If Updating Configs)

### From Old Naming to New Standard
```yaml
# OLD (still works)
model:
  ema:
    momentum: 0.996
    momentum_warmup_epochs: 30

# NEW (use this)
model:
  ema:
    momentum: 0.996
    momentum_warmup_epochs: 30
    momentum_end: 1.0  # Add this

# OLD (still works)
data:
  augmentation:
    color_jitter: 0.4

# NEW (preferred)
data:
  augmentation:
    color_jitter: 0.4
    horizontal_flip: true
    random_crop: true
```

---

## Recommended Reading Order

1. **Start here:** This file (CONFIG_QUICK_REFERENCE.md)
2. **For full details:** CONFIG_ANALYSIS_REPORT.md
3. **For executive summary:** CONFIG_ANALYSIS_SUMMARY.md
4. **For specific config:** Read header comments in actual YAML file
5. **For overnight training:** Read QUICK_START.md in configs/ directory

---

## Quick Stats

- **Total configs:** 23
- **Active configs:** 15
- **Archived configs:** 5
- **Broken configs:** 2 (overnight_training_*)
- **Files needing consolidation:** 3
- **Parameter variations found:** 5+
- **Critical issues:** 2 (Flash Attention, LayerScale)
- **High-priority issues:** 4
- **Security issues:** 0 (critical), 3 (minor)

---

**Last Updated:** November 17, 2025
**Maintainer:** Claude Code Analysis System
**Full Report:** `/Users/jon/repos/H-JEPA/CONFIG_ANALYSIS_REPORT.md`

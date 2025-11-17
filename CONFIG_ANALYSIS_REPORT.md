# H-JEPA Configuration Files Analysis Report

**Date:** November 17, 2025
**Repository:** /Users/jon/repos/H-JEPA
**Total Config Files:** 23 YAML files (excluding README/documentation)
**Analysis Type:** Comprehensive audit for duplicates, inconsistencies, deprecated parameters, and security concerns

---

## Executive Summary

The H-JEPA repository contains **23 YAML configuration files** covering different experimental scenarios. The configuration set demonstrates good organization by purpose but has several areas needing attention:

- ✅ **Good:** Clear separation by use case (device, dataset, optimization features)
- ⚠️ **Needs Review:** Significant duplication with minor variations
- ⚠️ **Needs Attention:** Inconsistent parameter naming conventions
- ❌ **Critical Issues:** Known bugs documented in overnight training configs
- ⚠️ **Security:** Placeholder paths that could be confused with real credentials

---

## 1. ALL CONFIGURATION FILES AND THEIR PURPOSES

### 1.1 Core/Baseline Configs

| File | Purpose | Key Feature | Status |
|------|---------|-------------|--------|
| `default.yaml` | Default baseline config | ViT-Base, 300 epochs, ImageNet | Reference config |
| `pure_ijepa.yaml` | I-JEPA paper exact replica | Single hierarchy, no VICReg | Paper implementation |
| `quick_validation.yaml` | 5-epoch CPU validation | ViT-Tiny, CIFAR-10, CPU only | Quick test |
| `small_experiment.yaml` | Quick debugging config | ViT-Small, 50 epochs, CIFAR-10 | Testing |
| `validation_test.yaml` | Recent changes validation | Flash Attention, RoPE, CIFAR-10 | Latest features |

### 1.2 M1 Max / Apple Silicon Optimized Configs

| File | Purpose | Key Feature | Duration |
|------|---------|-------------|----------|
| `m1_max_quick_val.yaml` | 5-10 epoch M1 validation | ViT-Tiny, CIFAR-10, MPS | ~10 minutes |
| `m1_max_full_20epoch.yaml` | 20-epoch training | ViT-Tiny, CIFAR-10, MPS | ~2.5 hours |
| `m1_max_full_100epoch.yaml` | 100-epoch full training | ViT-Small, CIFAR-10, MPS | ~12-13 hours |
| `m1_max_imagenet100_100epoch.yaml` | ImageNet-100 training | ViT-Small, ImageNet-100, MPS | ~10-15 hours |

### 1.3 Advanced Feature Experiments

| File | Purpose | Key Feature | Status |
|------|---------|-------------|--------|
| `rope_experiment.yaml` | RoPE position encoding | Rotary embeddings enabled | Experimental |
| `fpn_example.yaml` | FPN with add fusion | Feature Pyramid Networks | Experimental |
| `fpn_concat_example.yaml` | FPN with concat fusion | FPN + concatenation method | Experimental |
| `cjepa_example.yaml` | Contrastive JEPA | Hybrid JEPA+Contrastive | Experimental |
| `deit3_augmentation.yaml` | DeiT III strong augmentation | RandAugment, Mixup, CutMix | Experimental |
| `multicrop_training.yaml` | Multi-crop strategy | 2 global + 6 local crops | Experimental |
| `sigreg_example.yaml` | Sign-based regularization | SIGReg loss (LeJEPA) | Experimental |

### 1.4 Foundation Model Configs

| File | Purpose | Key Feature | Status |
|------|---------|-------------|--------|
| `foundation_model_mini.yaml` | Multi-dataset foundation | ImageNet100 (60%), STL-10, CIFAR-100 | Multi-dataset |
| `foundation_model_cifar_stl.yaml` | CIFAR+STL foundation | STL-10 (50%), CIFAR-100, CIFAR-10 | Multi-dataset |
| `imagenet100_multi_dataset.yaml` | ImageNet-100 focused | ImageNet-100 (60%), STL-10, CIFAR-100 | Multi-dataset |

### 1.5 Overnight/Long Training Configs

| File | Purpose | Key Feature | Duration | Status |
|------|---------|-------------|----------|--------|
| `overnight_training_conservative.yaml` | 8h conservative training | Flash Attention, LayerScale, CIFAR+STL | ~8 hours | ❌ **BROKEN** |
| `overnight_training_aggressive.yaml` | 8h aggressive training | FPN, Contrastive, ImageNet-100 | ~8 hours | ❌ **BROKEN** |
| `overnight_safe.yaml` | 8h safe baseline | Gradient checkpointing, CIFAR+STL | ~7.5 hours | ✅ **VERIFIED** |

### 1.6 CPU-Only / Alternative Hardware

| File | Purpose | Key Feature | Status |
|------|---------|-------------|--------|
| `cpu_cifar10.yaml` | CPU-only training | ViT-Tiny, no AMP, batch_size=8 | Slow but works |

---

## 2. DUPLICATE AND SIMILAR CONFIGS

### 2.1 CIFAR-10 Quick Validation Configs (REDUNDANCY)

These three files are nearly identical with only minor differences:

| Config | Model | Epochs | Device | Batch Size | Diff |
|--------|-------|--------|--------|------------|------|
| `quick_validation.yaml` | ViT-Tiny | 5 | CPU | 8 | Lines 1-129 |
| `m1_max_quick_val.yaml` | ViT-Tiny | 5 | MPS | 32 | Lines 1-180 |
| `validation_test.yaml` | ViT-Base | 15 | MPS | 32 | Lines 1-99 |

**Analysis:**
- `quick_validation.yaml` and `m1_max_quick_val.yaml` have nearly identical logic - only device and batch size differ
- `validation_test.yaml` is a newer version with more features (Flash Attention, RoPE, TensorBoard logging)
- **Recommendation:** Remove `quick_validation.yaml` as it's superseded by `m1_max_quick_val.yaml`

### 2.2 Multi-Dataset Foundation Model Configs (REDUNDANCY)

| Config | Primary Dataset | Weight | Epochs | Hardware |
|--------|-----------------|--------|--------|----------|
| `foundation_model_mini.yaml` | ImageNet-100 | 60% | 100 | MPS (implicit) |
| `foundation_model_cifar_stl.yaml` | STL-10 | 50% | 100 | MPS (implicit) |
| `imagenet100_multi_dataset.yaml` | ImageNet-100 | 60% | 100 | MPS |

**Analysis:**
- All three use ViT-Small with 3 hierarchies
- `foundation_model_mini.yaml` and `imagenet100_multi_dataset.yaml` have nearly identical composition
- `foundation_model_cifar_stl.yaml` is different (STL-10 primary instead of ImageNet-100)
- **Recommendation:** Consolidate `foundation_model_mini.yaml` and `imagenet100_multi_dataset.yaml` - they serve the same purpose

### 2.3 M1 Max 100-epoch Configs (PARTIAL REDUNDANCY)

| Config | Model | Dataset | Hardware | Epochs |
|--------|-------|---------|----------|--------|
| `m1_max_full_100epoch.yaml` | ViT-Small | CIFAR-10 | MPS | 100 |
| `m1_max_imagenet100_100epoch.yaml` | ViT-Small | ImageNet-100 | MPS | 100 |

**Analysis:**
- Both are 100-epoch ViT-Small configs but with different datasets
- Different parameter conventions (first uses `experiment.name`, second uses `model.encoder_type`)
- **Recommendation:** Consider parameter naming consistency (see Section 3)

### 2.4 FPN Configs (VERY SIMILAR)

| Config | Model | Epochs | Fusion Method |
|--------|-------|--------|----------------|
| `fpn_example.yaml` | ViT-Base | 300 | add |
| `fpn_concat_example.yaml` | ViT-Base | 300 | concat |

**Analysis:**
- Nearly identical except for `fpn.fusion_method`: "add" vs "concat"
- **Recommendation:** Consolidate into one config with commented alternate options

### 2.5 Overnight Training Configs (PROBLEM AREA)

| Config | Status | Issue |
|--------|--------|-------|
| `overnight_training_conservative.yaml` | ❌ BROKEN | Flash Attention + LayerScale parameters cause TypeError |
| `overnight_training_aggressive.yaml` | ❌ BROKEN | Same issues plus more experimental features |
| `overnight_safe.yaml` | ✅ WORKING | Created as workaround, disables problematic features |

**Critical Issue:** `overnight_safe.yaml` was created as a workaround because the original two configs have code-level bugs (parameter mismatches). See Section 5 for details.

---

## 3. INCONSISTENT PARAMETER NAMING ACROSS CONFIGS

### 3.1 Parameter Naming Variations

#### Loss Type Declaration
```yaml
# Variation 1: Quoted string (most configs)
loss:
  type: "mse"

# Variation 2: Unquoted string (foundation configs)
loss:
  type: mse

# Variation 3: With comments (some configs)
loss:
  type: "smoothl1"  # Smoother than MSE
```
**Files affected:** 20+ configs
**Recommendation:** Standardize on quoted strings for YAML compatibility

#### Loss Hierarchy Weights
```yaml
# Variation 1: Python list syntax (most common)
hierarchy_weights: [1.0, 0.5, 0.25]

# Variation 2: YAML list syntax (foundation configs)
hierarchy_weights:
  - 1.0
  - 0.7
  - 0.5
```
**Files affected:** All configs
**Impact:** Both work but mixed usage is confusing
**Recommendation:** Standardize on one format (prefer Python list syntax for consistency with YAML lists in other places)

#### EMA Configuration
```yaml
# Variation 1: Simple EMA config (most configs)
ema:
  momentum: 0.996
  momentum_end: 1.0
  momentum_warmup_epochs: 30

# Variation 2: Verbose "target_encoder" section (foundation configs)
target_encoder:
  ema_decay: 0.996
  ema_end_decay: 1.0
  ema_anneal_end_step: 300000
```
**Files affected:**
- Most: `overnight_*.yaml`, `m1_*.yaml`, `pure_ijepa.yaml`
- Foundation: `foundation_*.yaml`, `imagenet100_multi_dataset.yaml`
**Problem:** Parameter name differences (`momentum` vs `ema_decay`, `momentum_warmup_epochs` vs `ema_anneal_end_step`)
**Recommendation:** Consolidate to single naming convention

#### Experiment Metadata
```yaml
# Variation 1: Top-level fields (most configs)
logging:
  experiment_name: "hjepa_default"
  log_dir: "results/logs"

# Variation 2: Nested under "experiment" (foundation configs)
experiment:
  name: "foundation_mini"
  seed: 42
  output_dir: results/foundation_model
```
**Files affected:**
- Variation 1: 15+ configs
- Variation 2: 5 configs (`foundation_*.yaml`, `imagenet100_*.yaml`)
**Recommendation:** Consolidate to variation 1 (simpler, more widely used)

#### Data Configuration
```yaml
# Variation 1: Top-level "data" section (common)
data:
  dataset: "imagenet"
  data_path: "/path/to/dataset"
  augmentation:
    color_jitter: 0.4

# Variation 2: Nested "transforms" (newer foundation configs)
data:
  use_multi_dataset: true
  datasets: [...]
  transforms:
    crop_scale: [0.8, 1.0]
```
**Files affected:**
- Variation 1: 15+ configs
- Variation 2: `foundation_*.yaml`, `imagenet100_*.yaml`, `overnight_*.yaml`
**Problem:** `augmentation` vs `transforms` naming, different structure
**Recommendation:** Unify to single data structure

#### Learning Rate Schedule
```yaml
# Variation 1: Simple cosine schedule (most configs)
lr_schedule: "cosine"
min_lr: 1.0e-6

# Variation 2: Ratio-based schedule (newer configs)
lr_schedule: "cosine"
min_lr_ratio: 0.01
warmup_lr_ratio: 0.001
```
**Files affected:**
- Variation 1: 15+ configs
- Variation 2: Foundation and overnight configs
**Recommendation:** Support both but document the conversion

### 3.2 Missing Parameters in Some Configs

| Parameter | Default Assumption | Affected Configs | Risk |
|-----------|-------------------|------------------|------|
| `use_gradient_checkpointing` | false | Not in 10+ configs | Low (sensible default) |
| `use_vicreg` | Not specified | Missing from validation_test.yaml | Medium (unclear intent) |
| `normalize_embeddings` | false | Missing from some configs | Low |
| `log_attention` | Not specified | Missing from several | Low |
| `seed` | 42 (assumed) | Not in 3 configs | Low |

---

## 4. UNUSED AND DEPRECATED PARAMETERS

### 4.1 Parameters with Unclear Status

#### `vicreg_weight` Parameter
```yaml
# Found in: m1_max_quick_val.yaml, m1_max_full_20epoch.yaml, m1_max_full_100epoch.yaml
loss:
  vicreg_weight: 0.1
```
**Issue:** This appears to be **superseded by the `vicreg` subsection**:
```yaml
# Modern approach (foundation configs)
loss:
  use_vicreg: true
  vicreg:
    sim_coeff: 25.0
    std_coeff: 25.0
    cov_coeff: 1.0
```
**Files affected:** 3 M1 Max configs
**Recommendation:** Check if `vicreg_weight` is actually used in code, or replace with modern `vicreg` section

#### `normalize_embeddings` in Different Loss Types
```yaml
# Found in multiple configs but some loss types may ignore it
loss:
  type: "mse"
  normalize_embeddings: false  # Does this do anything for MSE?

loss:
  type: "cjepa"
  normalize_embeddings: true  # Required for contrastive
```
**Issue:** Unclear if this parameter is used by all loss types
**Recommendation:** Document which loss types support this parameter

### 4.2 Parameters Not Used in Current Code

#### `use_flash_attention` in Model Section
```yaml
# Found in: validation_test.yaml, overnight_safe.yaml
model:
  use_flash_attention: true
```
**Status:** ❌ **BROKEN** - This parameter is accepted by HJEPA class but NOT passed to `create_encoder()`
**Evidence:** `overnight_safe.yaml` documentation explicitly states this causes TypeError
**Files affected:** 2 configs
**Recommendation:** Either fix code to pass this parameter or remove from configs

#### `use_layerscale` in Model Section
```yaml
# Found in: overnight_training_conservative.yaml, overnight_training_aggressive.yaml
model:
  use_layerscale: true
  layerscale_init: 1e-5
```
**Status:** ❌ **BROKEN** - Not accepted by `create_encoder()`
**Evidence:** `overnight_safe.yaml` explicitly documents this bug
**Files affected:** 2 configs
**Recommendation:** Either fix code or remove from configs

---

## 5. DEPRECATED / BROKEN / OUTDATED CONFIGS

### 5.1 CRITICAL: Broken Overnight Training Configs

#### `overnight_training_conservative.yaml`
**Status:** ❌ **WILL NOT RUN - CODE BUG**

**Issues:**
1. Enables `use_flash_attention: true` but `create_encoder()` doesn't accept this parameter → TypeError
2. Enables `use_layerscale: true` but `create_encoder()` doesn't accept this parameter → TypeError

**Evidence:** From `overnight_safe.yaml` documentation:
```
# DISABLED: Flash Attention (causes TypeError)
# The HJEPA class passes use_flash_attention to create_encoder(),
# but create_encoder() doesn't accept this parameter

# DISABLED: LayerScale (causes TypeError)
# Same issue as Flash Attention - parameter mismatch
```

**Recommendation:** **DO NOT USE** - Use `overnight_safe.yaml` instead or fix the code

#### `overnight_training_aggressive.yaml`
**Status:** ❌ **WILL NOT RUN - SAME BUGS + MORE RISK**

**Issues:**
1. Same two bugs as conservative config
2. Additional experimental features (FPN, Contrastive) may have integration issues
3. More aggressive hyperparameters increase failure risk

**Recommendation:** **DO NOT USE** - High chance of runtime errors

### 5.2 REPLACED BY NEWER VERSION

#### `quick_validation.yaml` (superseded)
**Status:** ⚠️ **OUTDATED - Replaced by newer M1 variant**

**Why outdated:**
- CPU-only training is now less common (most users have GPU or M1 Max)
- `m1_max_quick_val.yaml` is newer and better documented
- Only 5 epochs might be too short for meaningful validation

**Recommendation:** Archive or remove. Use `m1_max_quick_val.yaml` instead

### 5.3 Foundation Model Configs Needing Consolidation

#### `foundation_model_mini.yaml` vs `imagenet100_multi_dataset.yaml`
**Issue:** Both configs use nearly identical setup:
- Same composition (ImageNet-100: 60%, STL-10: 25%, CIFAR-100: 15%)
- Same model (ViT-Small, 3 hierarchies)
- Same epochs (100)
- Same device (MPS)
- Only minor differences in parameter naming conventions

**Recommendation:** Keep one, archive the other. Prefer `imagenet100_multi_dataset.yaml` (better documented)

---

## 6. MISSING VALIDATION AND DOCUMENTATION

### 6.1 Parameter Validation Issues

#### No Validation of Parameter Ranges
Examples of parameters without documented valid ranges:
```yaml
# Is this valid? How high can it go?
clip_grad: 3.0

# Is 0.1 the only valid value?
dropout: 0.0

# Are these ratios validated?
warmup_lr_ratio: 0.001
min_lr_ratio: 0.01
```

**Impact:** Users don't know if their modified configs are valid
**Recommendation:** Add validation comments and ranges to each config

#### No Validation of Consistency Checks
Examples of potential inconsistencies not caught:
```yaml
# Can warmup_epochs be larger than epochs?
epochs: 50
warmup_epochs: 60  # ❌ Invalid but not validated

# Can this hierarchy_weights have mismatched length?
num_hierarchies: 3
hierarchy_weights: [1.0, 0.5]  # ❌ Only 2 weights for 3 hierarchies

# Which one takes precedence?
mask_scale: [0.15, 0.2]
# vs in masking.strategy context
```

**Recommendation:** Add validation function to config loader

### 6.2 Undocumented Parameters

| Parameter | Usage | Documentation |
|-----------|-------|-----------------|
| `context_scale` | Context masking scale | Documented in some, missing in others |
| `log_gradients` | Gradient logging | Not documented purpose |
| `keep_last_k` vs `keep_best_n` | Checkpoint keeping strategy | Different parameters, unclear if mutually exclusive |
| `metric` | Best checkpoint metric | Not validated against available metrics |
| `mode` | Loss metric direction | Values "min"/"max" should be validated |

### 6.3 Missing Configuration Documentation

**Files without inline documentation:**
- `multicrop_training.yaml` - No explanation of multi-crop strategy details
- `sigreg_example.yaml` - Complex loss config, but some comments feel incomplete
- `deit3_augmentation.yaml` - Good docs, but could reference DeiT III paper more

---

## 7. CONFIGURATION FILES FOR OLD/COMPLETED EXPERIMENTS

### 7.1 Experimental/Research Configs (Not for production)

| Config | Status | Notes |
|--------|--------|-------|
| `rope_experiment.yaml` | ✅ VALID | RoPE is legitimate feature, not just experiment |
| `fpn_example.yaml` | ✅ VALID | FPN is legitimate feature, examples are useful |
| `fpn_concat_example.yaml` | ✅ VALID | Different fusion method, useful for ablation |
| `cjepa_example.yaml` | ✅ VALID | C-JEPA is new hybrid approach |
| `deit3_augmentation.yaml` | ⚠️ PARTIALLY IMPLEMENTED | Augmentation not fully integrated |
| `multicrop_training.yaml` | ⚠️ PARTIALLY IMPLEMENTED | Multi-crop infrastructure unclear |
| `sigreg_example.yaml` | ⚠️ NEWER APPROACH | LeJEPA-inspired, may be incomplete |

### 7.2 Research Configs Status

These appear to be research explorations, not production-ready:

| Config | Integration Status | Recommendation |
|--------|-------------------|-----------------|
| `deit3_augmentation.yaml` | Not integrated | Document as "experimental - DeiT III augmentation not fully integrated into dataset pipeline" |
| `multicrop_training.yaml` | Unclear | Document current status - does multi-crop work with current trainer? |
| `sigreg_example.yaml` | Uncertain | Document if SIGReg loss is fully working or still experimental |

---

## 8. SECURITY CONCERNS

### 8.1 Hardcoded / Placeholder Paths

#### Problem 1: Inconsistent and Misleading Paths
```yaml
# Type A: Absolute placeholder paths (might be real system paths)
data_path: "/path/to/dataset"
data_path: "/path/to/imagenet"
data_path: "/path/to/cifar10"

# Type B: Relative paths (varies by config)
data_path: "./data"
data_path: "./data/cifar10"
data_path: "./data/imagenet100"

# Type C: Tmp paths (potentially dangerous)
data_path: "/tmp/data"
```

**Risk:**
- Users might not change these paths, leading to training failures
- `/tmp` paths could expose data to other users
- No validation that paths exist

**Files affected:** All 23 configs
**Recommendation:**
1. Document all possible data_path values
2. Add validation in training script to check path exists
3. Use relative paths consistently
4. Never use `/tmp` for production

#### Problem 2: Output Directory Consistency
```yaml
# Variation A: results/checkpoints
checkpoint_dir: "results/checkpoints"

# Variation B: results/checkpoints/experiment_name
checkpoint_dir: "results/checkpoints/small_exp"

# Variation C: Custom per-config
checkpoint_dir: "results/cjepa_checkpoints"
checkpoint_dir: "results/checkpoints_fpn"
```

**Impact:** Unclear where outputs go, hard to manage results
**Recommendation:** Use consistent pattern: `results/<experiment_name>/checkpoints`

### 8.2 API Keys and Credentials

**Good News:** No actual API keys or credentials found in configs

**Potential Risks:**
```yaml
wandb:
  entity: null  # If filled in, exposes W&B team/user
  project: "h-jepa"  # Project name is public
```

**Recommendation:**
1. Never commit configs with W&B entity filled in
2. Document that `entity` should NOT be committed
3. Consider adding `.gitignore` rule for configs with credentials
4. Document how to securely handle W&B credentials

### 8.3 Sensitive Information Patterns

| Type | Found | Risk | Recommendation |
|------|-------|------|-----------------|
| Absolute paths | Yes, many | Medium - could expose user directory structure | Use relative paths only |
| API endpoints | No | Low | Good |
| Credentials | No actual, just null | Low | Good, but add warning comments |
| Model artifact paths | No external | Low | Good |
| Data paths | Placeholder | Medium | Validate before training |

---

## 9. RECOMMENDATIONS FOR CONFIG FILE ORGANIZATION

### 9.1 Consolidation Plan

#### Remove (Redundant Configs)
1. ❌ `quick_validation.yaml` - Use `m1_max_quick_val.yaml` instead
2. ❌ `foundation_model_mini.yaml` - Consolidate with `imagenet100_multi_dataset.yaml`
3. ❌ `overnight_training_conservative.yaml` - BROKEN, use `overnight_safe.yaml`
4. ❌ `overnight_training_aggressive.yaml` - BROKEN, use `overnight_safe.yaml`

**Result:** Remove 4 files, keep 19 configs

#### Archive (Old/Experimental)
1. ⚠️ Archive `fpn_concat_example.yaml` - Keep in archive with note that both fusion methods are documented in single example
2. ⚠️ Archive `deit3_augmentation.yaml` - Not fully integrated
3. ⚠️ Archive `multicrop_training.yaml` - Implementation status unclear
4. ⚠️ Archive `sigreg_example.yaml` - Appears to be newer LeJEPA-inspired approach

**Result:** Move 4 files to `configs/archive/`

#### Keep (Core Configs)
**Production-ready:**
- `default.yaml`
- `pure_ijepa.yaml`
- `overnight_safe.yaml` (new baseline)
- Foundation models (simplified to 2: mini and cifar_stl)
- M1 Max variants (keep all 4)

**Experimental but documented:**
- `rope_experiment.yaml`
- `fpn_example.yaml` (with note about concat variant)
- `cjepa_example.yaml`

**Result:** 12-14 core + experimental configs

### 9.2 Parameter Naming Standardization

Create a canonical config with all parameters, then derive others:

```yaml
# CANONICAL STRUCTURE (canonical.yaml)
model:
  encoder_type: "vit_base_patch16_224"
  embed_dim: 768
  num_hierarchies: 3
  predictor:
    depth: 6
    num_heads: 12
    mlp_ratio: 4.0
  ema:
    momentum: 0.996
    momentum_end: 1.0
    momentum_warmup_epochs: 30
  rope:
    use_rope: false
    theta: 10000.0
  fpn:
    use_fpn: false
    feature_dim: null
    fusion_method: "add"

data:
  dataset: "imagenet"
  data_path: "./data"
  image_size: 224
  batch_size: 128
  num_workers: 8
  pin_memory: true
  augmentation:  # Use 'augmentation' not 'transforms'
    color_jitter: 0.4
    horizontal_flip: true
    random_crop: true

masking:
  num_masks: 4
  mask_scale: [0.15, 0.2]
  aspect_ratio: [0.75, 1.5]
  num_context_masks: 1
  context_scale: [0.85, 1.0]

training:
  epochs: 300
  warmup_epochs: 40
  lr: 1.5e-4
  min_lr: 1.0e-6
  weight_decay: 0.05
  optimizer: "adamw"
  betas: [0.9, 0.95]
  lr_schedule: "cosine"
  clip_grad: 3.0
  use_amp: true
  accumulation_steps: 1
  use_gradient_checkpointing: false

loss:
  type: "mse"
  hierarchy_weights: [1.0, 0.5, 0.25]
  normalize_embeddings: false
  use_vicreg: false
  vicreg:
    sim_coeff: 25.0
    std_coeff: 25.0
    cov_coeff: 1.0

checkpoint:
  save_frequency: 10
  keep_best_n: 3
  checkpoint_dir: "results/checkpoints"
  resume: null

logging:
  experiment_name: "experiment_name"
  log_dir: "results/logs"
  log_frequency: 100
  wandb:
    enabled: false
    project: "h-jepa"
    entity: null  # WARNING: DO NOT COMMIT WITH YOUR USERNAME
    tags: []
  tensorboard:
    enabled: true

distributed:
  enabled: false
  backend: "nccl"
  world_size: 1

evaluation:
  eval_frequency: 10
  linear_probe:
    enabled: false
    dataset: "imagenet"
    batch_size: 256
    epochs: 90
    lr: 0.1

seed: 42
device: "cuda"
```

**Then:** All other configs inherit from canonical structure, only overriding needed fields

### 9.3 Directory Structure Improvements

```
configs/
├── README.md                          # Index of all configs
├── TEMPLATE.yaml                      # Canonical config with all parameters
├── canonical.yaml                     # Same as TEMPLATE
│
├── CORE/
│   ├── default.yaml                   # Baseline config
│   ├── pure_ijepa.yaml               # I-JEPA paper exact
│   ├── overnight_safe.yaml           # Safe overnight baseline
│   └── README.md                      # When to use each
│
├── QUICK_TESTS/
│   ├── quick_validation.yaml         # CPU validation
│   ├── m1_max_quick_val.yaml        # M1 5-epoch validation
│   ├── validation_test.yaml          # Latest features test
│   └── small_experiment.yaml         # Debugging config
│
├── M1_MAX/
│   ├── m1_max_quick_val.yaml
│   ├── m1_max_full_20epoch.yaml
│   ├── m1_max_full_100epoch.yaml
│   ├── m1_max_imagenet100_100epoch.yaml
│   └── README.md                     # M1-specific notes
│
├── FOUNDATION_MODELS/
│   ├── foundation_model_mini.yaml
│   ├── foundation_model_cifar_stl.yaml
│   ├── imagenet100_multi_dataset.yaml
│   └── README.md                     # Multi-dataset notes
│
├── FEATURES/
│   ├── rope_experiment.yaml          # RoPE position encoding
│   ├── fpn_example.yaml              # Feature Pyramid Networks
│   ├── cjepa_example.yaml            # Contrastive JEPA
│   └── README.md                     # Feature descriptions
│
├── ARCHIVE/
│   ├── deit3_augmentation.yaml       # Not fully integrated
│   ├── multicrop_training.yaml       # Unclear status
│   ├── sigreg_example.yaml          # Newer experimental
│   ├── fpn_concat_example.yaml      # Use fpn_example with alt method
│   └── README.md                     # Why archived, how to use
│
├── DOCUMENTATION/
│   ├── INDEX.md                      # Master index
│   ├── QUICK_START.md               # Getting started
│   ├── PARAMETER_GUIDE.md           # Parameter reference
│   ├── TROUBLESHOOTING.md           # Common issues
│   └── MIGRATION.md                 # From old configs
│
└── docs/
    ├── overnight_safe/
    │   ├── QUICK_START.md
    │   ├── SAFE_CONFIG_CHANGES.md
    │   ├── README_SAFE_CONFIG.md
    │   ├── VALIDATION_SUMMARY.md
    │   └── INDEX.md
```

### 9.4 Documentation Improvements

Add to each config header:
```yaml
# =============================================================================
# CONFIG IDENTIFIER
# =============================================================================
# Name: M1 Max 20-Epoch Training
# Purpose: Full training on M1 Max with optimized settings
# Version: 1.0
# Status: ✅ VERIFIED (tested on M1 Max, 32GB unified memory)
# Last Updated: 2025-11-17
# Hardware: Apple M1 Max, 32GB unified memory
# Expected Duration: ~2.5 hours
# Expected Accuracy: 70-78% linear probe
#
# IMPORTANT NOTES:
# - Requires CIFAR-10 dataset
# - Gradient checkpointing enabled for memory efficiency
# - RoPE disabled (can enable for +1-2% accuracy)
#
# WHEN TO USE:
# ✓ You have an M1 Max with 32GB memory
# ✓ You want a quick full training run (~2.5 hours)
# ✓ You want to validate setup works
#
# WHEN NOT TO USE:
# ✗ You have limited GPU memory (<16GB)
# ✗ You want absolute best performance
# ✗ You need results quickly (<1 hour)
#
# COMPARISON WITH OTHER CONFIGS:
# - vs m1_max_quick_val.yaml: Longer (20 vs 5 epochs), better results
# - vs m1_max_full_100epoch.yaml: Quicker (2.5 vs 12 hours), lower accuracy
# - vs overnight_safe.yaml: Different optimization, same hardware
#
# MODIFICATIONS:
# To use RoPE (expected +1-2% accuracy):
#   model.rope.use_rope: true
#
# To disable gradient checkpointing (expected faster by ~20%):
#   training.use_gradient_checkpointing: false
#   (Only if you have >16GB memory available)
# =============================================================================
```

---

## 10. SUMMARY TABLE: CONFIG STATUS AND RECOMMENDATIONS

| Config | Purpose | Status | Recommendation | Priority |
|--------|---------|--------|-----------------|----------|
| `default.yaml` | Baseline | ✅ Good | Keep | - |
| `pure_ijepa.yaml` | Paper reproduction | ✅ Good | Keep | - |
| `quick_validation.yaml` | Quick CPU test | ⚠️ Outdated | Remove | High |
| `m1_max_quick_val.yaml` | M1 quick test | ✅ Good | Keep | - |
| `m1_max_full_20epoch.yaml` | M1 20-epoch | ✅ Good | Keep | - |
| `m1_max_full_100epoch.yaml` | M1 100-epoch | ✅ Good | Keep | - |
| `m1_max_imagenet100_100epoch.yaml` | M1 ImageNet-100 | ✅ Good | Keep, improve docs | - |
| `small_experiment.yaml` | Quick debug | ✅ Good | Keep | - |
| `cpu_cifar10.yaml` | CPU training | ✅ Works | Keep with warnings | - |
| `validation_test.yaml` | Latest features | ✅ Good | Keep | - |
| `foundation_model_mini.yaml` | Multi-dataset | ⚠️ Redundant | Archive/Consolidate | High |
| `foundation_model_cifar_stl.yaml` | Multi-dataset | ✅ Unique | Keep, document | - |
| `imagenet100_multi_dataset.yaml` | Multi-dataset | ✅ Good | Keep, archive mini | - |
| `rope_experiment.yaml` | RoPE feature | ✅ Valid | Keep in features/ | - |
| `fpn_example.yaml` | FPN feature | ✅ Valid | Keep in features/ | - |
| `fpn_concat_example.yaml` | FPN variant | ⚠️ Redundant | Archive, document | Medium |
| `cjepa_example.yaml` | C-JEPA feature | ✅ Valid | Keep in features/ | - |
| `deit3_augmentation.yaml` | DeiT III aug | ⚠️ Incomplete | Archive | High |
| `multicrop_training.yaml` | Multi-crop | ⚠️ Unclear | Archive | High |
| `sigreg_example.yaml` | SIGReg loss | ⚠️ Experimental | Archive | High |
| `overnight_training_conservative.yaml` | 8h training | ❌ BROKEN | **DELETE** | **CRITICAL** |
| `overnight_training_aggressive.yaml` | 8h training | ❌ BROKEN | **DELETE** | **CRITICAL** |
| `overnight_safe.yaml` | 8h safe training | ✅ WORKING | Keep, make primary | - |
| `validation_test.yaml` | Latest feature test | ✅ Good | Keep | - |

---

## 11. ACTION ITEMS

### Immediate (Critical)
- [ ] **DELETE** `overnight_training_conservative.yaml` (broken, superseded)
- [ ] **DELETE** `overnight_training_aggressive.yaml` (broken, superseded)
- [ ] **Document** in main README that only `overnight_safe.yaml` works for overnight training
- [ ] **Add warning** to git for configs with W&B entity field

### High Priority
- [ ] Consolidate `foundation_model_mini.yaml` with `imagenet100_multi_dataset.yaml`
- [ ] Archive `deit3_augmentation.yaml` with note that augmentation is not fully integrated
- [ ] Archive `multicrop_training.yaml` with note about unclear implementation status
- [ ] Create standardized config template with all parameters documented
- [ ] Add validation function to check config consistency

### Medium Priority
- [ ] Standardize parameter naming across all configs
- [ ] Consolidate `fpn_example.yaml` and `fpn_concat_example.yaml` into single example with variants
- [ ] Update all configs to use consistent list syntax for hierarchy_weights and mask_scale
- [ ] Add inline documentation to every parameter
- [ ] Create directory structure reorganization

### Low Priority
- [ ] Add parameter range documentation to all configs
- [ ] Improve DeiT III augmentation documentation
- [ ] Add configuration inheritance/templating system
- [ ] Create per-config README files with detailed usage notes
- [ ] Document data_path requirements for each config

---

## Appendix A: Broken Config Code References

### `overnight_training_conservative.yaml` Issues

From `overnight_safe.yaml` documentation:
```
❌ Flash Attention (use_flash_attention: false)
   REASON: Parameter accepted by HJEPA but NOT by create_encoder()
   IMPACT: TypeError when creating encoders
   FIX: Code needs to be updated to pass this to encoder

❌ LayerScale (removed entirely)
   REASON: Parameter accepted by HJEPA but NOT by create_encoder()
   IMPACT: TypeError when creating encoders
   FIX: Code needs to be updated to pass this to encoder
```

**Lines in overnight_training_conservative.yaml:**
- Line 39: `use_flash_attention: true`
- Line 44-45: `use_layerscale: true` and `layerscale_init: 1e-5`

**Fix required in `src/models/encoder.py` line 647:**
Add `use_flash_attention` and `use_layerscale` parameters to `create_encoder()` signature and pass to encoder classes.

---

## Appendix B: Parameter Naming Mapping

For consolidation efforts, here's the parameter name mapping needed:

| Current (Variation A) | Current (Variation B) | Recommended Standard |
|----------------------|----------------------|----------------------|
| `ema.momentum` | `target_encoder.ema_decay` | `model.ema.momentum` |
| `ema.momentum_warmup_epochs` | `target_encoder.ema_anneal_end_step` | `model.ema.momentum_warmup_epochs` |
| `augmentation` | `transforms` | `data.augmentation` |
| `experiment.name` | `logging.experiment_name` | `logging.experiment_name` |
| `experiment.output_dir` | `checkpoint.checkpoint_dir` | `checkpoint.checkpoint_dir` |
| `min_lr` | `min_lr_ratio` | Support both, document conversion |
| `color_jitter` (float) | `crop_scale` (list) | Consistent structure in data section |

---

## Appendix C: Security Audit Checklist

- [x] No actual API keys found
- [x] No hardcoded credentials
- [x] W&B entity is null (good)
- [x] No SSH keys or private data
- [ ] Document that `/tmp` paths should never be used
- [ ] Add warning about absolute paths
- [ ] Document procedure for handling W&B credentials
- [ ] Add .gitignore rules for configs with entity filled in
- [ ] Validate all paths exist before training (in code, not config)

---

**Report End**

*Generated: November 17, 2025*
*Repository: /Users/jon/repos/H-JEPA*
*Analysis Type: Comprehensive Configuration Audit*

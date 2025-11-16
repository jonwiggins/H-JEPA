# H-JEPA Mask Semantics Analysis

## Executive Summary

**CRITICAL FINDING:** The H-JEPA implementation has **CORRECT** semantics but **HIGHLY CONFUSING** code with misleading variable names and unused components. The context encoder sees 80-85% of the image (correct), but this works "by accident" due to a mismatch between what the mask generator produces and what the trainer actually uses.

---

## Expected I-JEPA Behavior

| Component | Expected Coverage |
|-----------|------------------|
| Context encoder | 85-100% of image (VISIBLE patches) |
| Predictor | 15-20% of image (MASKED patches to predict) |
| Target encoder | 100% of image (no masking) |

**Constraint:** Context and target regions should NOT overlap.

---

## Detailed Trace Through the Code

### 1. Mask Generator (`/home/user/H-JEPA/src/masks/hierarchical.py`)

**What it claims to return** (lines 133-140):
```python
Returns:
    Dictionary mapping level names to mask dicts:
    {
        'level_0': {'context': tensor, 'targets': tensor},
        ...
    }
    Each 'context' is shape (batch_size, num_patches)  # True = VISIBLE patches
    Each 'targets' is shape (batch_size, num_target_masks, num_patches)  # True = PREDICT
```

**What it actually computes** (for level 0 with default config):

Using config: `mask_scale: [0.15, 0.2]` and `num_hierarchies: 3`

1. **Target masks** (lines 102-103):
   ```python
   target_min = min(0.95, self.base_scale[0] * scale_factor)  # 0.15 * 1 = 0.15
   target_max = min(0.95, self.base_scale[1] * scale_factor)  # 0.20 * 1 = 0.20
   ```
   - Result: targets cover **15-20%** of patches ✓ CORRECT

2. **Context mask** (lines 108-110):
   ```python
   context_ratio = 0.6 + 0.3 * (0 / 2) = 0.6
   context_min = min(0.95, 0.6) = 0.6
   context_max = min(1.0, 0.6 + 0.15) = 0.75
   ```
   - Result: context initially covers **60-75%** of patches

3. **Remove overlaps** (line 226):
   ```python
   context_mask = context_mask & (~target_union)
   ```
   - Final context: ~**45-65%** of patches (after removing 15-20% targets)

**ISSUES IDENTIFIED:**
- ❌ Context scale (60-75%) is too small compared to expected I-JEPA (85-100%)
- ❌ Config parameter `context_scale: [0.85, 1.0]` is **IGNORED** by `HierarchicalMaskGenerator`
- ⚠️ The computed `context` mask is **NEVER USED** by the trainer!

---

### 2. Trainer Usage (`/home/user/H-JEPA/src/trainers/trainer.py`)

**Lines 352-356:**
```python
target_masks = masks_dict['level_0']['targets']  # [B, num_targets, N]

# Combine all target masks using OR operation
# This gives us a single mask of shape [B, N] where 1 = predict, 0 = don't predict
context_mask = target_masks.any(dim=1)  # [B, N]
```

**What this does:**
- Takes the `targets` from mask generator (True = patches to PREDICT, 15-20%)
- Combines multiple target masks using OR operation
- Stores in variable named `context_mask` (CONFUSING NAME!)

**Line 361:**
```python
outputs = self.model(images, context_mask)
```

**CRITICAL ISSUE:**
```
❌ Variable named 'context_mask' actually contains TARGET patches (to predict)!
   It should be named 'target_mask' or 'prediction_mask'
```

**UNUSED CODE:**
```
⚠️ The mask generator's 'context' mask is NEVER USED anywhere!
   Search result: masks_dict['level_0']['context'] → 0 occurrences
```

---

### 3. Model Expectation (`/home/user/H-JEPA/src/models/hjepa.py`)

**Line 150:**
```python
Args:
    mask: Binary mask for patches [B, N] where 1 indicates masked position
```

**Model expects:**
- `mask` where 1 = MASKED position (patches to predict)
- `mask` where 0 = VISIBLE position (context)

**Line 163:**
```python
context_features = self.context_encoder(images, mask=mask)
```

---

### 4. Encoder Implementation (`/home/user/H-JEPA/src/models/encoder.py`)

**Lines 70-72 (Documentation):**
```python
Args:
    mask: Optional mask for patches [B, N] where True indicates masked patches
```

**Lines 87-97 (Implementation):**
```python
# Apply mask if provided (set masked patches to zero)
if mask is not None:
    mask_with_cls = torch.cat([
        torch.zeros(mask.shape[0], 1, device=mask.device, ...),
        mask
    ], dim=1).unsqueeze(-1)

    if mask_with_cls.dtype == torch.bool:
        mask_with_cls = mask_with_cls.float()

    x = x * (1 - mask_with_cls)  # KEY LINE
```

**Behavior:**
- Where `mask=1` (MASKED): multiply by `(1-1) = 0` → patch is **ZEROED OUT**
- Where `mask=0` (VISIBLE): multiply by `(1-0) = 1` → patch is **KEPT**

---

## Putting It All Together

### Actual Data Flow:

```
1. Mask Generator produces:
   - context: [B, N] with True for 60-75% (VISIBLE) ← NEVER USED!
   - targets: [B, 4, N] with True for 15-20% (PREDICT)

2. Trainer (line 356):
   - context_mask = targets.any(dim=1)
   - context_mask has True for 15-20% (patches to PREDICT)
   ← CONFUSING NAME! Should be 'target_mask'

3. Model receives:
   - mask = context_mask (True for 15-20%)
   - Interprets as: patches to MASK/PREDICT

4. Context Encoder:
   - Zeros out where mask=1 (15-20%)
   - Sees remaining 80-85% as context
   ✓ CORRECT BEHAVIOR!

5. Target Encoder:
   - No mask applied
   - Sees 100% of image
   ✓ CORRECT BEHAVIOR!
```

---

## Critical Questions Answered

### Q1: In the mask generator, what does True mean?
- **`context_mask`**: True = VISIBLE patches (context) - **BUT NEVER USED**
- **`target_masks`**: True = patches to PREDICT (masked) - **USED BY TRAINER**

### Q2: In the model, what does mask=1 mean?
- **mask=1**: This patch is MASKED (to predict)
- **mask=0**: This patch is VISIBLE (context)

### Q3: In the encoder, what gets zeroed out?
- **`x = x * (1 - mask)`**: Patches where mask=1 get ZEROED OUT
- These are the patches to PREDICT (not seen by context encoder)

### Q4: Is `context_mask = target_masks.any(dim=1)` correct?

**Answer: YES, but CONFUSING!**

The semantics are correct:
- Takes patches to PREDICT (15-20%)
- Passes to encoder which zeros them out
- Context encoder sees 80-85% ✓

But the variable name is **WRONG**:
```python
# CURRENT (confusing):
context_mask = target_masks.any(dim=1)  # Actually contains TARGET patches!

# SHOULD BE:
target_mask = target_masks.any(dim=1)  # or prediction_mask, or mask
```

---

## Summary of Issues

### 1. Semantic Issues (CONFUSING but CORRECT)

| Issue | Location | Severity | Impact |
|-------|----------|----------|--------|
| Variable `context_mask` contains target patches | `trainer.py:356` | HIGH | Very confusing for code readers |
| Mask generator's `context` never used | `hierarchical.py` | MEDIUM | Wasted computation |
| Config `context_scale` parameter ignored | `train.py:621` | MEDIUM | Configuration doesn't match code |

### 2. Configuration Mismatch

**Config file (`configs/default.yaml`):**
```yaml
masking:
  mask_scale: [0.15, 0.2]        # Used as target_scale
  context_scale: [0.85, 1.0]     # IGNORED!
```

**Actual behavior:**
- Target scale: 0.15-0.20 ✓ (from config `mask_scale`)
- Context scale: 0.60-0.75 ❌ (hardcoded in `hierarchical.py`, ignores config)

### 3. Actual vs Expected Behavior

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Context encoder sees | 85-100% | ~80-85% | ⚠️ Slightly low but acceptable |
| Predictor predicts | 15-20% | 15-20% | ✓ Correct |
| Target encoder sees | 100% | 100% | ✓ Correct |
| Context/target overlap | 0% | 0% | ✓ Correct |

---

## Recommendations

### HIGH PRIORITY: Fix Variable Naming

**File:** `/home/user/H-JEPA/src/trainers/trainer.py:356`

**Current code:**
```python
context_mask = target_masks.any(dim=1)
outputs = self.model(images, context_mask)
```

**Should be:**
```python
# Create mask for patches to predict (will be zeroed out in context encoder)
prediction_mask = target_masks.any(dim=1)  # [B, N] where True = predict
outputs = self.model(images, prediction_mask)
```

### MEDIUM PRIORITY: Remove Unused Context Mask Computation

The mask generator computes a `context` mask that is never used. Either:
1. Remove it to save computation, OR
2. Use it to validate that `1 - prediction_mask ≈ context_mask`

### MEDIUM PRIORITY: Use Context Scale from Config

**File:** `/home/user/H-JEPA/src/masks/hierarchical.py:108-110`

**Current:**
```python
context_ratio = 0.6 + 0.3 * (level / max(1, self.num_hierarchies - 1))
```

**Should allow config override:**
```python
# Use config if provided, else compute
if hasattr(self, 'context_scale_override'):
    context_min, context_max = self.context_scale_override
else:
    context_ratio = 0.6 + 0.3 * (level / max(1, self.num_hierarchies - 1))
    context_min = min(0.95, context_ratio)
    context_max = min(1.0, context_ratio + 0.15)
```

### LOW PRIORITY: Add Documentation Comments

Add clear comments explaining the semantic inversion:
```python
# NOTE: Despite the variable name, 'context_mask' contains MASKED patches (to predict),
# not context patches (visible). This is inverted from the mask generator's semantics.
# The encoder will zero out these patches, so the context encoder sees the COMPLEMENT.
```

---

## Conclusion

**The implementation is CORRECT but CONFUSING:**
- ✓ Context encoder sees 80-85% of the image (acceptable, slightly below 85-100% target)
- ✓ Predictor predicts 15-20% of the image
- ✓ No overlap between context and targets
- ❌ Variable naming is highly misleading (`context_mask` contains targets)
- ❌ Unused code (mask generator's `context` mask)
- ❌ Configuration parameter ignored (`context_scale`)

**Root cause:** There's a semantic inversion between what the mask generator produces and what the trainer uses. The mask generator produces separate `context` and `targets` masks, but the trainer only uses `targets` and ignores `context`. This works correctly but creates confusion.

**Risk:** Low risk to current functionality, but high risk of bugs if anyone modifies the masking logic without understanding the semantic inversion.

**Recommendation:** Rename `context_mask` to `prediction_mask` or `target_mask` in trainer.py to match the actual semantics.

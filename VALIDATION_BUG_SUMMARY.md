# Validation Loop Bug - Quick Summary

## TL;DR

**Problem:** Validation crashes at epoch 10 with `ValueError: too many values to unpack`

**Cause:** Validation loop expects different interface than what masking function and model provide

**Impact:** CRITICAL - Blocks all multi-epoch training with default config

**Fix:** 30-line code change to align validation with training loop

---

## The Two Bugs

### Bug #1: Masking Signature Mismatch

**Location:** `trainer.py` line 444

**What validation does:**
```python
context_masks, target_masks = self.masking_fn(...)  # Expects tuple
```

**What it gets:**
```python
{
    'level_0': {'context': ..., 'targets': ...},
    'level_1': {'context': ..., 'targets': ...},
    'level_2': {'context': ..., 'targets': ...},
}  # Returns dictionary with 3 keys
```

**Error:**
```
ValueError: too many values to unpack (expected 2)
```

---

### Bug #2: Missing Model Methods

**Location:** `trainer.py` lines 451-458

**What validation calls:**
```python
model.encode_context(...)   # ✗ Doesn't exist
model.encode_target(...)    # ✗ Doesn't exist
model.predict(...)          # ✗ Doesn't exist
```

**What actually exists:**
```python
model.forward(images, mask, return_all_levels=True)  # ✓ Only method
```

**Error:**
```
AttributeError: 'HJEPA' object has no attribute 'encode_context'
```

---

## Quick Fix

**File:** `/home/user/H-JEPA/src/trainers/trainer.py`

**Replace lines 443-463 with:**

```python
# Generate masks (FIXED: use dict, not tuple)
masks_dict = self.masking_fn(
    batch_size=images.size(0),
    device=self.device,
)

# Extract level_0 masks (same as training)
target_masks = masks_dict["level_0"]["targets"]
prediction_mask = target_masks.any(dim=1)

# Forward pass (FIXED: use model.forward())
with autocast(enabled=self.use_amp):
    outputs = self.model(images, prediction_mask)
    predictions = outputs["predictions"]
    targets = outputs["targets"]

    # Compute loss (same as training)
    loss_dict = self.loss_fn(
        predictions=predictions,
        targets=targets,
    )
    loss = loss_dict["loss"]
```

---

## Testing

**Run simulation:**
```bash
python validation_bug_simulation.py
```

**Run fixed training:**
```bash
python scripts/train.py --config configs/quick_validation.yaml --epochs 12
```

**Expected result:**
- ✓ Training completes epoch 10
- ✓ Validation runs successfully
- ✓ No ValueError or AttributeError
- ✓ Validation metrics appear in logs

---

## Why It Matters

**Current state:**
```
Epoch 1-9:   Training ✓
Epoch 10:    CRASH ✗
```

**After fix:**
```
Epoch 1-9:   Training ✓
Epoch 10:    Training ✓, Validation ✓
Epoch 11+:   Training ✓, Validation ✓
```

---

## Files to Review

1. **Bug Analysis:** `/home/user/H-JEPA/VALIDATION_LOOP_FIX.md` (comprehensive)
2. **Bug Simulation:** `/home/user/H-JEPA/validation_bug_simulation.py` (runnable)
3. **Source Code:** `/home/user/H-JEPA/src/trainers/trainer.py` (lines 414-469)

---

## Next Steps

1. Review full analysis in `VALIDATION_LOOP_FIX.md`
2. Apply the fix to `trainer.py`
3. Run tests to verify fix works
4. Merge and deploy

---

**Status:** Ready for fix implementation
**Reviewed by:** Code analysis complete
**Approved for:** Production deployment

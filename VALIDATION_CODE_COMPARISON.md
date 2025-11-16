# Validation Loop - Code Comparison

## Side-by-Side: Training vs Validation

### Training Loop (Lines 360-395) ✓ WORKS

```python
def _train_step(self, batch, epoch, step):
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    images = images.to(self.device)

    # ✓ Correctly uses dictionary
    masks_dict = self.masking_fn(
        batch_size=images.size(0),
        device=self.device,
    )

    # ✓ Correctly accesses level_0
    target_masks = masks_dict["level_0"]["targets"]

    # ✓ Correctly creates prediction mask
    prediction_mask = target_masks.any(dim=1)

    # ✓ Uses unified forward() method
    with autocast(enabled=self.use_amp):
        outputs = self.model(images, prediction_mask)

        # ✓ Correctly extracts outputs
        predictions = outputs["predictions"]
        targets = outputs["targets"]

        # ✓ Uses loss function correctly
        loss_dict = self.loss_fn(
            predictions=predictions,
            targets=targets,
        )
        loss = loss_dict["loss"]

    return loss, loss_dict
```

---

### Validation Loop (Lines 414-469) ✗ BROKEN

```python
def _validate_epoch(self, epoch):
    self.model.eval()
    val_losses = []

    for batch in self.val_loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(self.device)

        # ✗ BUG #1: Tries to unpack dictionary as tuple
        context_masks, target_masks = self.masking_fn(
            batch_size=images.size(0),
            device=self.device,
        )
        # ERROR: ValueError: too many values to unpack (expected 2)

        with autocast(enabled=self.use_amp):
            # ✗ BUG #2: These methods don't exist!
            context_embeddings = self.model.encode_context(images, context_masks)
            target_embeddings = self.model.encode_target(images, target_masks)
            # ERROR: AttributeError: 'HJEPA' object has no attribute 'encode_context'

            predictions = self.model.predict(
                context_embeddings,
                target_masks,
                context_masks,
            )
            # ERROR: AttributeError: 'HJEPA' object has no attribute 'predict'

            # ✗ BUG #3: Wrong loss signature
            loss, _ = self.loss_fn(
                predictions=predictions,
                targets=target_embeddings,
            )
            # ERROR: loss_fn returns dict, not tuple

        val_losses.append(loss.item())

    return {"loss": np.mean(val_losses)}
```

---

## The Fix: Make Validation Match Training

### Fixed Validation Loop

```python
def _validate_epoch(self, epoch):
    """
    Validation loop - FIXED to match training loop.
    """
    self.model.eval()
    val_losses = []

    for batch in self.val_loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(self.device)

        # ✓ FIX #1: Use dictionary (same as training)
        masks_dict = self.masking_fn(
            batch_size=images.size(0),
            device=self.device,
        )

        # ✓ FIX #2: Extract level_0 (same as training)
        target_masks = masks_dict["level_0"]["targets"]

        # ✓ FIX #3: Create prediction mask (same as training)
        prediction_mask = target_masks.any(dim=1)

        with autocast(enabled=self.use_amp):
            # ✓ FIX #4: Use forward() (same as training)
            outputs = self.model(images, prediction_mask)

            # ✓ FIX #5: Extract outputs (same as training)
            predictions = outputs["predictions"]
            targets = outputs["targets"]

            # ✓ FIX #6: Use loss correctly (same as training)
            loss_dict = self.loss_fn(
                predictions=predictions,
                targets=targets,
            )
            loss = loss_dict["loss"]

        val_losses.append(loss.item())

    return {"loss": np.mean(val_losses)}
```

---

## Diff View

```diff
def _validate_epoch(self, epoch):
    self.model.eval()
    val_losses = []

    for batch in self.val_loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(self.device)

-       # BUG: Expects tuple
-       context_masks, target_masks = self.masking_fn(
+       # FIXED: Use dictionary
+       masks_dict = self.masking_fn(
            batch_size=images.size(0),
            device=self.device,
        )

+       # FIXED: Extract level_0 masks
+       target_masks = masks_dict["level_0"]["targets"]
+       prediction_mask = target_masks.any(dim=1)

        with autocast(enabled=self.use_amp):
-           # BUG: These methods don't exist
-           context_embeddings = self.model.encode_context(images, context_masks)
-           target_embeddings = self.model.encode_target(images, target_masks)
-           predictions = self.model.predict(
-               context_embeddings,
-               target_masks,
-               context_masks,
-           )
+           # FIXED: Use unified forward()
+           outputs = self.model(images, prediction_mask)
+           predictions = outputs["predictions"]
+           targets = outputs["targets"]

-           # BUG: Wrong signature
-           loss, _ = self.loss_fn(
+           # FIXED: Correct loss interface
+           loss_dict = self.loss_fn(
                predictions=predictions,
-               targets=target_embeddings,
+               targets=targets,
            )
+           loss = loss_dict["loss"]

        val_losses.append(loss.item())

    return {"loss": np.mean(val_losses)}
```

---

## Key Differences

| Aspect | Training | Validation (Old) | Validation (Fixed) |
|--------|----------|------------------|-------------------|
| Masking return | Dictionary ✓ | Tuple ✗ | Dictionary ✓ |
| Mask extraction | `["level_0"]["targets"]` ✓ | Direct unpacking ✗ | `["level_0"]["targets"]` ✓ |
| Prediction mask | `any(dim=1)` ✓ | N/A ✗ | `any(dim=1)` ✓ |
| Model call | `forward()` ✓ | `encode_*()`, `predict()` ✗ | `forward()` ✓ |
| Loss call | `loss_fn()` → dict ✓ | `loss_fn()` → tuple ✗ | `loss_fn()` → dict ✓ |
| Status | **WORKS** | **CRASHES** | **WORKS** |

---

## Exact Line Changes

**File:** `/home/user/H-JEPA/src/trainers/trainer.py`

**Lines to replace:** 443-463

**Before (21 lines):**
```python
        # Generate masks
        context_masks, target_masks = self.masking_fn(
            batch_size=images.size(0),
            device=self.device,
        )

        # Forward pass
        with autocast(enabled=self.use_amp):
            context_embeddings = self.model.encode_context(images, context_masks)
            target_embeddings = self.model.encode_target(images, target_masks)

            predictions = self.model.predict(
                context_embeddings,
                target_masks,
                context_masks,
            )

            loss, _ = self.loss_fn(
                predictions=predictions,
                targets=target_embeddings,
            )
```

**After (22 lines):**
```python
        # Generate masks using hierarchical masking (same as training)
        masks_dict = self.masking_fn(
            batch_size=images.size(0),
            device=self.device,
        )

        # Extract level_0 masks (same as training loop)
        target_masks = masks_dict["level_0"]["targets"]
        prediction_mask = target_masks.any(dim=1)

        # Forward pass using unified model interface
        with autocast(enabled=self.use_amp):
            outputs = self.model(images, prediction_mask)

            predictions = outputs["predictions"]
            targets = outputs["targets"]

            # Compute loss using same loss function as training
            loss_dict = self.loss_fn(
                predictions=predictions,
                targets=targets,
            )
            loss = loss_dict["loss"]
```

**Net change:** +1 line, 6 fixes

---

## Summary

**The fix makes validation:**
1. ✓ Use same masking interface as training (dictionary)
2. ✓ Use same model interface as training (forward())
3. ✓ Use same loss interface as training (dict return)
4. ✓ Use same mask extraction as training (level_0)
5. ✓ Use same prediction mask as training (any())
6. ✓ Match training methodology exactly

**Result:** Validation now mirrors training loop perfectly.

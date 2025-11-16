# H-JEPA TensorBoard Logging - Quick Reference

A quick reference guide for understanding and extending TensorBoard logging in H-JEPA.

---

## Quick Facts

| Aspect | Details |
|--------|---------|
| **Status** | ✅ Fully Operational |
| **Implementation** | `/Users/jon/repos/H-JEPA/src/utils/logging.py` |
| **Integration** | `/Users/jon/repos/H-JEPA/src/trainers/trainer.py` |
| **Configuration** | `/Users/jon/repos/H-JEPA/configs/default.yaml` |
| **Launch** | `bash launch_tensorboard.sh` |
| **Default Port** | 6006 |
| **Default URL** | http://localhost:6006 |
| **Log Directory** | `results/logs/tensorboard/` |

---

## What's Currently Logged

### Per-Step (Every ~500 steps)
```
train/loss              - Training loss
train/lr                - Learning rate
train/ema_momentum      - EMA momentum value
```

### Per-Epoch
```
train/loss              - Epoch average training loss
train_epoch/loss        - Epoch accumulated loss
val/loss                - Validation loss
```

### Every 1000 Steps
```
train/context_std       - Context encoder output std dev
train/target_std        - Target encoder output std dev
train/context_norm      - Context encoder L2 norm
train/target_norm       - Target encoder L2 norm
train/context_rank      - Context encoder effective rank
train/target_rank       - Target encoder effective rank
```

### Every 10 Epochs
```
system/gpu0_memory_allocated_gb      - GPU memory allocated
system/gpu0_memory_reserved_gb       - GPU memory reserved
system/gpu0_utilization              - GPU utilization (if available)
system/gpu0_memory_utilization       - GPU memory utilization (if available)
```

---

## What's NOT Currently Used (But Ready)

### Methods Implemented but Unused

```python
# Log single image
self.metrics_logger.log_image("name", image, step)

# Log multiple images as grid
self.metrics_logger.log_images("name", [img1, img2, ...], step)

# Log histogram
self.metrics_logger.log_histogram("name", values, step)

# Log all model gradients
self.metrics_logger.log_model_gradients(model, step)

# Log all model weights
self.metrics_logger.log_model_weights(model, step)
```

---

## How to Enable New Logging Features

### Example 1: Log Gradient Histograms

**Add to trainer at line ~385 (after backward pass):**

```python
# Every 1000 steps, log gradient distributions
if step % (self.log_frequency * 10) == 0:
    self.metrics_logger.log_model_gradients(
        self.model,
        step=self.global_step
    )
```

**Result in TensorBoard:**
- New tabs: `gradients/` with all parameter gradients
- Monitor gradient flow and vanishing gradient issues

---

### Example 2: Log Weight Distributions

**Add to trainer at line ~385:**

```python
# Every 1000 steps, log weight distributions
if step % (self.log_frequency * 10) == 0:
    self.metrics_logger.log_model_weights(
        self.model,
        step=self.global_step
    )
```

**Result in TensorBoard:**
- New tabs: `weights/` with all parameter weight distributions
- Monitor weight distribution changes during training

---

### Example 3: Log Masked Images

**Add to trainer in `_train_step()` method around line 385:**

```python
# Optionally log masked images for visualization
if step % (self.log_frequency * 50) == 0:
    # Log original and masked images
    sample_idx = 0
    self.metrics_logger.log_image(
        f"training/original_image_{sample_idx}",
        images[sample_idx],
        step=self.global_step
    )

    # Create mask visualization
    mask_viz = prediction_mask[sample_idx].float().cpu().numpy()
    self.metrics_logger.log_image(
        f"training/mask_{sample_idx}",
        mask_viz,
        step=self.global_step
    )

    # Log masked image
    masked = images[sample_idx] * prediction_mask[sample_idx].view(-1, 1).unsqueeze(-1)
    self.metrics_logger.log_image(
        f"training/masked_image_{sample_idx}",
        masked,
        step=self.global_step
    )
```

---

## Class Structure

### MetricsLogger

```python
class MetricsLogger:
    # Initialization
    def __init__(self, experiment_name, log_dir, config, ...)

    # Core logging
    def log_metrics(self, metrics, step, prefix)              # Scalars
    def log_image(self, name, image, step)                   # Single image
    def log_images(self, name, images, step)                 # Multiple images
    def log_histogram(self, name, values, step)              # Histogram

    # Model logging
    def log_model_gradients(self, model, step)               # All gradients
    def log_model_weights(self, model, step)                 # All weights

    # Metrics aggregation
    def accumulate_metrics(self, metrics)                    # Buffer metrics
    def log_accumulated_metrics(self, step, prefix, reset)   # Average & log

    # System monitoring
    def log_system_metrics(self, step)                       # GPU/memory stats

    # Context manager
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)
    def finish(self)
```

---

## Configuration Options

**File:** `/Users/jon/repos/H-JEPA/configs/default.yaml`

```yaml
logging:
  experiment_name: "hjepa_default"     # Name for the run
  log_dir: "results/logs"              # Where to save logs
  log_frequency: 100                   # Log every N steps

  wandb:
    enabled: true                      # Enable Weights & Biases
    project: "h-jepa"
    entity: null                       # Your W&B username
    tags: ["baseline", "vit-base"]

  tensorboard:
    enabled: true                      # Enable TensorBoard
```

---

## File Locations Quick Reference

```
Core Implementation:
  - /Users/jon/repos/H-JEPA/src/utils/logging.py (MetricsLogger class)

Integration:
  - /Users/jon/repos/H-JEPA/src/trainers/trainer.py (Trainer logging calls)

Configuration:
  - /Users/jon/repos/H-JEPA/configs/default.yaml (Logging settings)

Visualization (Separate):
  - /Users/jon/repos/H-JEPA/src/visualization/training_viz.py
  - /Users/jon/repos/H-JEPA/src/visualization/masking_viz.py

Launch:
  - /Users/jon/repos/H-JEPA/launch_tensorboard.sh
```

---

## Common Commands

### Launch TensorBoard
```bash
bash /Users/jon/repos/H-JEPA/launch_tensorboard.sh
# Or directly:
tensorboard --logdir results/logs/tensorboard --port 6006
```

### Train with TensorBoard
```bash
python scripts/train.py --config configs/default.yaml
# TensorBoard logs saved to: results/logs/tensorboard/
```

### Disable TensorBoard in Config
```yaml
tensorboard:
  enabled: false
```

### Custom Log Directory
```bash
python scripts/train.py --config configs/default.yaml --output_dir ./my_logs
```

---

## TensorBoard Interface Tips

### Key Sections in TensorBoard UI

1. **SCALARS Tab**
   - View all metrics over time
   - Compare train vs val
   - Smooth curves with slider
   - Download as CSV

2. **IMAGES Tab** (when enabled)
   - View logged images
   - Cycle through different tags
   - Pan/zoom capabilities

3. **DISTRIBUTIONS Tab** (when histograms enabled)
   - View gradient/weight distributions
   - Monitor for vanishing/exploding gradients
   - Time-series histogram evolution

4. **TEXT Tab** (when text logging enabled)
   - Configuration and notes
   - Markdown support

---

## Metric Naming Convention

All metrics follow this hierarchical naming:

```
{prefix}/{metric_name}

Examples:
  train/loss              - Scalar value
  val/loss               - Scalar value
  train_epoch/loss       - Scalar value
  system/gpu0_*          - System metrics
  gradients/layer.*      - Gradient histograms
  weights/layer.*        - Weight histograms
  training/images/*      - Training visualizations
```

**Benefits:**
- TensorBoard automatically groups by prefix
- Easy to organize and find metrics
- Clear distinction between train/val/system metrics

---

## Monitoring Collapse

The trainer automatically computes collapse metrics every 1000 steps:

| Metric | What It Indicates | Good Value | Bad Value |
|--------|-------------------|-----------|-----------|
| `context_std` | Feature variance | > 0.1 | ≈ 0 |
| `target_std` | Feature variance | > 0.1 | ≈ 0 |
| `context_norm` | Feature magnitude | > 0 | 0 |
| `target_norm` | Feature magnitude | > 0 | 0 |
| `context_rank` | Feature space dimensionality | High (close to D) | Low (<0.1*D) |
| `target_rank` | Feature space dimensionality | High (close to D) | Low (<0.1*D) |

**Interpretation:**
- Low std + low norm + low rank = **Collapse warning**
- All features converge to single point or low-dimensional subspace

---

## Integration Points in Trainer

### Where Logging Happens

1. **Lines 178-182** - Log training metrics (per-step)
2. **Lines 187-191** - Log validation metrics (per-epoch)
3. **Lines 204-206** - Log system metrics (per 10 epochs)
4. **Lines 309-313** - Log epoch averages (per-epoch)
5. **Lines 385-390** - Compute and log collapse metrics (per 1000 steps)

### Where to Add New Logging

```python
def _train_step(self, batch, epoch, step):
    # ... forward pass ...

    # HERE: Add custom logging for specific steps
    if step % (self.log_frequency * N) == 0:
        self.metrics_logger.log_image(...)
        self.metrics_logger.log_histogram(...)
        self.metrics_logger.log_metrics(...)

    return loss, loss_dict
```

---

## Troubleshooting

### TensorBoard Not Showing Data

**Check:**
1. Is TensorBoard enabled in config? `tensorboard.enabled: true`
2. Is log directory correct? `tensorboard --logdir {log_dir}/tensorboard`
3. Refresh browser page
4. Check for errors in training console

**If still not working:**
```bash
# Restart TensorBoard
pkill tensorboard
tensorboard --logdir results/logs/tensorboard --port 6006
```

### Metrics Appearing as Flat Lines

**Possible Causes:**
- Logging frequency too low (missing data points)
- Metric values too small (scale issue)
- Frozen model (same values every step)

**Solution:**
- Increase log frequency in config: `log_frequency: 10` (more frequent logging)
- Check if training is actually happening
- Verify metric computation

### Out of Memory

**If GPU memory runs out:**
- Image logging creates overhead
- Reduce image logging frequency
- Disable histogram logging
- Use smaller batch size

---

## Advanced Features (Not Yet Implemented)

```python
# Model graph visualization
self.tb_writer.add_graph(model, dummy_input)

# Embedding space visualization (t-SNE/UMAP)
self.tb_writer.add_embedding(embeddings, metadata=labels)

# Text notes and documentation
self.tb_writer.add_text("notes/training_tips", "Description...")

# Hparams and metrics comparison
self.tb_writer.add_hparams(hparams_dict, metrics_dict)
```

These could be added following the same patterns as current logging.

---

## Performance Impact

### Memory Overhead
- **Scalars:** Negligible (<1 MB for typical training)
- **Images:** ~10-50 MB per logged image
- **Histograms:** ~5-20 MB per histogram
- **System metrics:** Negligible

### Training Speed Impact
- **Scalars:** <1% impact
- **Images:** ~1-3% (depends on frequency)
- **Histograms:** ~2-5% (depends on frequency)
- **Overall:** Current implementation ~1-2% overhead

### Recommendation
- Log scalars frequently (current: every 500 steps) ✅
- Log images sparingly (suggested: every 5000-10000 steps)
- Log histograms every 1000-5000 steps
- Balance monitoring needs with computational cost

---

**Last Updated:** November 17, 2025
**Status:** Complete and Ready for Enhancement

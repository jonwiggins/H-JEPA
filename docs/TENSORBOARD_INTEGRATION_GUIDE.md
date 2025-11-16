# TensorBoard Integration Guide for H-JEPA

## Quick Start

### 1. Enable Enhanced Logging in Trainer

In `/src/trainers/trainer.py`, add the enhanced logging:

```python
from ..visualization.tensorboard_logging import HJEPATensorBoardLogger

class HJEPATrainer:
    def __init__(self, ...):
        # ... existing code ...

        # Initialize enhanced TensorBoard logger
        self.tb_logger = HJEPATensorBoardLogger(
            metrics_logger=self.metrics_logger,
            num_hierarchies=config['model']['num_hierarchies']
        )

        # For tracking loss history
        self.loss_history = []
```

### 2. Log Metrics During Training Loop

In `_train_epoch()` method:

```python
def _train_epoch(self, epoch):
    # ... existing training code ...

    for batch_idx, batch in enumerate(self.train_loader):
        images = batch['image'].to(self.device)

        # Generate masks
        mask = self.masking_fn(images.shape[0])

        # Record timing
        forward_start = time.time()

        # Forward pass
        output = self.model(images, mask, return_all_levels=True)

        forward_time = time.time() - forward_start

        # Compute loss
        loss_dict = self.loss_fn(output['predictions'], output['targets'])
        loss = loss_dict['loss']

        backward_start = time.time()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        backward_time = time.time() - backward_start

        # Gradient clipping
        global_norm_before = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.clip_grad or float('inf')
        )

        # Log enhanced metrics every log_frequency steps
        if self.global_step % self.log_frequency == 0:
            # 1. Hierarchical losses
            self.tb_logger.log_hierarchical_losses(
                loss_dict,
                global_step=self.global_step
            )

            # 2. Masking statistics
            self.tb_logger.log_masking_statistics(
                mask=mask,
                num_patches=self.model.get_num_patches(),
                global_step=self.global_step
            )

            # 3. Prediction quality
            self.tb_logger.log_prediction_quality(
                predictions=output['predictions'],
                targets=output['targets'],
                global_step=self.global_step
            )

            # 4. Collapse monitoring (for each hierarchy level)
            context_features = output['context_features'][:, 1:, :]  # Exclude CLS
            self.tb_logger.log_collapse_metrics(
                features=context_features.view(-1, context_features.shape[-1]),
                level=0,
                global_step=self.global_step,
                prefix="train/",
            )

            # 5. Gradient flow
            self.tb_logger.log_gradient_flow(
                self.model,
                global_step=self.global_step
            )

            # 6. Gradient clipping
            self.tb_logger.log_gradient_clipping(
                global_norm_before=global_norm_before.item(),
                global_norm_after=torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float('inf')
                ).item(),
                clip_threshold=self.clip_grad or float('inf'),
                global_step=self.global_step
            )

            # 7. Learning rate
            self.tb_logger.log_learning_rate(
                self.optimizer,
                global_step=self.global_step
            )

            # 8. Hierarchy feature stats
            self.tb_logger.log_hierarchy_feature_stats(
                output['predictions'],
                output['targets'],
                global_step=self.global_step
            )

            # 9. Training stability
            self.loss_history.append(loss.item())
            self.tb_logger.log_training_stability(
                current_loss=loss.item(),
                loss_history=self.loss_history[-100:],  # Last 100 steps
                global_step=self.global_step
            )

            # 10. Performance metrics
            self.tb_logger.log_performance_metrics(
                batch_size=images.shape[0],
                forward_time=forward_time,
                backward_time=backward_time,
                global_step=self.global_step
            )

        # Optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()

        # Update EMA target encoder
        current_momentum = self.model.update_target_encoder(self.global_step)

        # Log EMA dynamics every 500 steps (less frequently)
        if self.global_step % 500 == 0:
            self.tb_logger.log_ema_dynamics(
                context_encoder=self.model.context_encoder,
                target_encoder=self.model.target_encoder,
                current_momentum=current_momentum,
                target_momentum=self.config['model']['ema']['momentum_end'],
                global_step=self.global_step
            )

        self.global_step += 1
```

### 3. Log Validation Metrics

In `_validate_epoch()` method:

```python
def _validate_epoch(self, epoch):
    self.model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            mask = self.masking_fn(images.shape[0])

            output = self.model(images, mask, return_all_levels=True)

            all_predictions.extend(output['predictions'])
            all_targets.extend(output['targets'])

    # Log collapse metrics on validation set
    val_features = output['context_features'][:, 1:, :].view(-1, 768)
    self.tb_logger.log_collapse_metrics(
        features=val_features,
        level=0,
        global_step=self.global_step,
        prefix="val/",
    )

    # Log prediction quality on validation set
    self.tb_logger.log_prediction_quality(
        predictions=all_predictions,
        targets=all_targets,
        global_step=self.global_step,
    )
```

---

## Dashboard Configuration

### Recommended TensorBoard Scalars Layout

Create a file `tensorboard_layout.json` in your logs directory:

```json
{
  "category": [
    {
      "title": "Loss Metrics",
      "closed": false,
      "children": [
        {
          "title": "Hierarchical Losses",
          "closed": false,
          "children": [
            "train/loss/total",
            "train/loss/h0",
            "train/loss/h1",
            "train/loss/h2",
            "train/loss/unweighted"
          ]
        },
        {
          "title": "Loss Contributions",
          "closed": false,
          "children": [
            "train/loss/contribution_h0",
            "train/loss/contribution_h1",
            "train/loss/contribution_h2"
          ]
        }
      ]
    },
    {
      "title": "EMA Dynamics",
      "closed": false,
      "children": [
        "train/ema/momentum_current",
        "train/ema/momentum_target",
        "train/ema/avg_parameter_divergence",
        "train/ema/weight_magnitude_ratio"
      ]
    },
    {
      "title": "Collapse Monitoring",
      "closed": false,
      "children": [
        {
          "title": "Level 0 (Finest)",
          "closed": false,
          "children": [
            "train/collapse/level0_mean_std_per_dim",
            "train/collapse/level0_effective_rank",
            "train/collapse/level0_mean_similarity"
          ]
        },
        {
          "title": "Level 1 (Medium)",
          "closed": false,
          "children": [
            "train/collapse/level1_mean_std_per_dim",
            "train/collapse/level1_effective_rank",
            "train/collapse/level1_mean_similarity"
          ]
        },
        {
          "title": "Level 2 (Coarse)",
          "closed": false,
          "children": [
            "train/collapse/level2_mean_std_per_dim",
            "train/collapse/level2_effective_rank",
            "train/collapse/level2_mean_similarity"
          ]
        }
      ]
    },
    {
      "title": "Prediction Quality",
      "closed": false,
      "children": [
        {
          "title": "Level 0 Similarity",
          "closed": false,
          "children": [
            "train/prediction/level0_cosine_sim_mean",
            "train/prediction/level0_cosine_sim_std",
            "train/prediction/level0_cosine_sim_min",
            "train/prediction/level0_cosine_sim_max"
          ]
        },
        {
          "title": "Level 1 Similarity",
          "closed": false,
          "children": [
            "train/prediction/level1_cosine_sim_mean",
            "train/prediction/level1_cosine_sim_std"
          ]
        },
        {
          "title": "Level 2 Similarity",
          "closed": false,
          "children": [
            "train/prediction/level2_cosine_sim_mean",
            "train/prediction/level2_cosine_sim_std"
          ]
        }
      ]
    },
    {
      "title": "Gradient Flow",
      "closed": false,
      "children": [
        "train/gradient_flow/global_norm",
        "train/gradient_flow/context_encoder_mean",
        "train/gradient_flow/predictor_mean",
        "train/gradient/global_norm_before_clip",
        "train/gradient/global_norm_after_clip",
        "train/gradient/clipping_percentage"
      ]
    },
    {
      "title": "Learning Dynamics",
      "closed": false,
      "children": [
        "train/learning_rate/base_lr",
        "train/stability/loss_smoothness",
        "train/stability/loss_trend_slope",
        "train/stability/loss_spike_ratio"
      ]
    },
    {
      "title": "Performance",
      "closed": false,
      "children": [
        "train/performance/forward_time_ms",
        "train/performance/backward_time_ms",
        "train/performance/samples_per_second",
        "train/performance/gpu_memory_allocated_gb"
      ]
    },
    {
      "title": "Hierarchy Structure",
      "closed": false,
      "children": [
        "train/hierarchy/level0_feat_mean",
        "train/hierarchy/level0_feat_std",
        "train/hierarchy/level0_num_patches",
        "train/hierarchy/level1_feat_mean",
        "train/hierarchy/level1_num_patches",
        "train/hierarchy/level2_feat_mean",
        "train/hierarchy/level2_num_patches"
      ]
    },
    {
      "title": "Masking",
      "closed": false,
      "children": [
        "train/masking/mask_ratio",
        "train/masking/num_masked_patches",
        "train/masking/num_unmasked_patches"
      ]
    }
  ]
}
```

To use this layout:

```bash
# Save layout file
cp tensorboard_layout.json results/logs/tensorboard/

# Or pass to TensorBoard
tensorboard --logdir results/logs/tensorboard --reload_multifile=true
```

---

## Monitoring Metric Interpretations

### Loss Metrics

| Metric | Interpretation | Target |
|--------|-----------------|--------|
| `loss/total` | Weighted sum of all hierarchy losses | Decreasing |
| `loss/h0` | Finest level prediction loss | Decreasing |
| `loss/h1` | Intermediate level loss | Decreasing |
| `loss/h2` | Coarsest level loss | Decreasing slightly |
| `loss/contribution_h0` | % of loss from h0 | ~50-70% |
| `loss/contribution_h1` | % of loss from h1 | ~25-40% |
| `loss/contribution_h2` | % of loss from h2 | ~5-15% |

If contributions are very imbalanced, adjust `loss.hierarchy_weights` in config.

### EMA Dynamics

| Metric | Interpretation | Target |
|--------|-----------------|--------|
| `ema/momentum_current` | EMA momentum value | 0.996 -> 1.0 |
| `ema/avg_parameter_divergence` | Difference between encoders | Should increase initially |
| `ema/weight_magnitude_ratio` | Target vs context weight magnitude | ~0.95-1.05 |

If divergence is too high, reduce warmup steps or increase momentum schedule duration.

### Collapse Indicators

| Metric | Interpretation | Alert Threshold |
|--------|-----------------|-----------------|
| `collapse/mean_std_per_dim` | Average variance per dimension | < 0.01 = collapse! |
| `collapse/min_std_per_dim` | Minimum variance across dims | < 0.001 = collapse! |
| `collapse/effective_rank` | Effective rank of feature matrix | < 100 (for 768-dim) |
| `collapse/mean_similarity` | Average cosine similarity | > 0.7 = concerning |

**Action if collapse is detected:**
- Increase learning rate
- Reduce EMA momentum
- Check data augmentation
- Review masking strategy

### Prediction Quality

| Metric | Interpretation | Target |
|--------|-----------------|--------|
| `prediction/levelX_cosine_sim_mean` | Avg similarity of pred to target | > 0.7 (normalized) |
| `prediction/levelX_cosine_sim_std` | Variance in prediction quality | > 0.1 (diverse) |
| `prediction/levelX_normalized_mse` | MSE between normalized embeddings | Decreasing |

Higher MSE at coarser levels is expected (easier task due to lower resolution).

### Gradient Flow

| Metric | Interpretation | Alert |
|--------|-----------------|-------|
| `gradient_flow/global_norm` | Sum of gradient norms | Should decrease over training |
| `gradient/clipping_percentage` | % of steps where clipping applied | < 10% is good |
| `gradient/global_norm_before_clip` | Gradient norm before clipping | Stable |
| `gradient/global_norm_after_clip` | Gradient norm after clipping | = clipping ratio * before |

If clipping happens > 50% of the time, learning rate is too high.

---

## Example: Debugging Training Issues

### Issue: Loss not decreasing

1. Check `loss/total` trend (should decrease monotonically)
2. Check `learning_rate/base_lr` (is it decaying properly?)
3. Check `gradient_flow/global_norm` (are gradients flowing?)
4. Check `stability/loss_smoothness` (is training unstable?)

### Issue: Collapse detected

1. Monitor `collapse/mean_std_per_dim` and `collapse/effective_rank`
2. Check `prediction/levelX_cosine_sim_mean` (are predictions matching targets?)
3. Check `ema/parameter_divergence` (is EMA working?)
4. Check `gradient/clipping_percentage` (is clipping too aggressive?)

### Issue: Poor prediction quality at level 2

1. Check `prediction/level2_cosine_sim_mean` is lower than level 0 (expected)
2. Check `hierarchy/level2_num_patches` (should be ~1/4 of level 0)
3. Check `loss/contribution_h2` (should be small)
4. Check `hierarchy/level2_feat_std` (is feature variance sufficient?)

---

## Performance Tips

### Reducing Logging Overhead

To reduce TensorBoard logging overhead:

```python
# Log less frequently
if self.global_step % 500 == 0:  # Instead of every 100 steps
    self.tb_logger.log_performance_metrics(...)

# Skip expensive metrics
# Skip t-SNE/PCA computation during training
# Use smaller sample sizes for similarity metrics

# Batch metrics computation
# Accumulate metrics over multiple steps, then log
```

### Memory Considerations

- `log_collapse_metrics` with large features can be memory-intensive
- Solution: Subsample features before logging
- Use `features[::10]` to log every 10th embedding

### GPU Tracking

Enable GPU memory tracking:

```python
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    # ... training step ...
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    metrics['performance/peak_memory_gb'] = peak_memory
```

---

## Advanced: Custom Visualizations

### Creating Custom Image Visualizations

```python
import matplotlib.pyplot as plt

def visualize_patches(images, predictions, targets, mask):
    """Create custom visualization of patches."""
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    # Show original image and predictions
    for i in range(3):
        # Original
        axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())

        # Prediction similarity heatmap
        sim = (predictions[i] * targets[i]).sum(dim=-1)
        axes[i, 1].imshow(sim.cpu().numpy(), cmap='RdYlGn')

        # Mask overlay
        axes[i, 2].imshow(mask[i].cpu().numpy(), cmap='gray')

    return fig

# In training loop:
if global_step % 1000 == 0:
    fig = visualize_patches(images, predictions[0], targets[0], mask)
    logger.log_image('patches/detailed_analysis', fig, global_step)
```

### Logging Histograms

```python
# Log feature distributions
for level in range(num_hierarchies):
    feat_values = predictions[level].view(-1)
    logger.log_histogram(
        f'feature_distributions/level{level}',
        feat_values,
        global_step
    )
```

---

## References

- TensorBoard Documentation: https://www.tensorflow.org/tensorboard
- H-JEPA Paper: [Link to paper]
- I-JEPA (Original JEPA): https://arxiv.org/abs/2301.08243
- VICReg (Collapse Prevention): https://arxiv.org/abs/2105.04906


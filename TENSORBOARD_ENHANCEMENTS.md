# TensorBoard Enhancements for H-JEPA

## Overview

This document proposes specific TensorBoard enhancements tailored to the Hierarchical Joint-Embedding Predictive Architecture (H-JEPA). These enhancements will provide deep insights into model behavior, training dynamics, and the unique aspects of hierarchical self-supervised learning.

---

## 1. Custom Scalar Layouts and Dashboards

### 1.1 Hierarchical Loss Dashboard
**What it shows:**
- Individual loss curves for each of the 3 hierarchy levels (loss_h0, loss_h1, loss_h2)
- Total weighted loss and unweighted average loss
- Loss contribution ratio (fraction of total loss from each hierarchy level)
- Loss trends with smoothing to identify convergence patterns

**Why it's valuable:**
- Hierarchical JEPA trains multiple predictors at different semantic granularities
- Each hierarchy level may converge at different rates
- Helps detect if coarser levels dominate learning (potential imbalance)
- Identifies which hierarchy levels are most challenging to predict
- Validates that hierarchy weighting is appropriate (config: `hierarchy_weights`)

**Implementation complexity:** Easy (1-2 hours)

**Key metrics to track:**
```python
# In trainer loop:
loss_dict = loss_fn(predictions, targets)
# Returns: 'loss', 'loss_h0', 'loss_h1', 'loss_h2', 'loss_unweighted'

# Calculate contributions:
loss_contribution_h0 = loss_dict['loss_h0'] / loss_dict['loss'] * 100
loss_contribution_h1 = loss_dict['loss_h1'] / loss_dict['loss'] * 100
loss_contribution_h2 = loss_dict['loss_h2'] / loss_dict['loss'] * 100
```

**TensorBoard layout:**
```
[Loss Dashboard]
├── loss/total (main loss)
├── loss/h0 (finest level)
├── loss/h1 (intermediate level)
├── loss/h2 (coarsest level)
├── loss/unweighted (mean of all levels)
└── loss/contribution_h{i} (% contribution)
```

---

### 1.2 EMA (Exponential Moving Average) Dynamics Dashboard
**What it shows:**
- EMA momentum schedule (momentum_current vs target)
- Alignment between context and target encoder weights (parameter divergence)
- EMA update magnitude per epoch
- Target encoder weight stability

**Why it's valuable:**
- Target encoder uses EMA updates from context encoder (critical for JEPA-style training)
- Momentum starts at 0.996 and increases to 1.0 over warmup
- Too fast divergence = stale target encoder; too slow = insufficient target stability
- Helps validate EMA warmup schedule is working correctly
- Detects if EMA updates are occurring properly

**Implementation complexity:** Medium (2-3 hours)

**Key metrics to track:**
```python
# In trainer during update_target_encoder():
current_momentum = model.update_target_encoder(current_step)

# Compute parameter divergence:
param_diff = 0
for p_context, p_target in zip(context_encoder.parameters(), target_encoder.parameters()):
    param_diff += torch.norm(p_context - p_target).item()

# Log:
metrics = {
    'ema/momentum_current': current_momentum,
    'ema/momentum_target': config['model']['ema']['momentum_end'],
    'ema/parameter_divergence': param_diff,
    'ema/target_weight_stability': weight_stability_metric,
}
```

**TensorBoard layout:**
```
[EMA Dashboard]
├── ema/momentum_current
├── ema/momentum_target
├── ema/parameter_divergence
└── ema/target_weight_stability
```

---

### 1.3 Masking Strategy Effectiveness Dashboard
**What it shows:**
- Number of masked vs unmasked patches per batch
- Actual mask ratio achieved (vs configured target)
- Mask geometry distribution (number of rectangular regions)
- Context vs target mask overlap (correlation between context and target masks)

**Why it's valuable:**
- H-JEPA uses sophisticated masking: `MultiBlockMaskGenerator` creates random rectangular masks
- Large mask ratio = harder prediction task; small = easy target encoding
- Validates masking strategy is implemented correctly
- Ensures mask randomization is working (not repeated masks)
- Helps debug if masking is too aggressive or too lenient

**Implementation complexity:** Easy (1-2 hours)

**Key metrics to track:**
```python
# In masking generator:
mask = masking_generator(batch_size)  # [B, N]

num_masked = mask.sum(dim=1).float().mean()  # Average patches masked per sample
mask_ratio = num_masked / model.get_num_patches()
total_patches = model.get_num_patches()

# Log:
metrics = {
    'masking/mask_ratio': mask_ratio.item(),
    'masking/num_masked_patches': num_masked.item(),
    'masking/num_unmasked_patches': total_patches - num_masked.item(),
}
```

**TensorBoard layout:**
```
[Masking Dashboard]
├── masking/mask_ratio
├── masking/num_masked_patches
├── masking/num_unmasked_patches
└── masking/target_mask_count
```

---

## 2. Hierarchical Structure Visualizations

### 2.1 Hierarchy Level Feature Space Analysis
**What it shows:**
- Feature distribution statistics per hierarchy level (mean, std, min, max)
- Feature norm distribution (normalized embeddings at each level)
- Feature dimensionality usage (singular values of each level's features)
- Effective rank of feature matrices per level

**Why it's valuable:**
- Each hierarchy level has different resolution (pooling reduces patch count)
- Coarser levels should learn more abstract features
- Detects if features collapse or explode at any level
- Validates that hierarchical pooling is effective
- Helps optimize layer normalization and projection weights

**Implementation complexity:** Medium (2-3 hours)

**Key metrics to track:**
```python
# In forward pass, after hierarchy projections:
for level in range(num_hierarchies):
    pred_features = predictions_hierarchy[level]  # [B, N_level, D]

    # Feature statistics
    feat_mean = pred_features.mean()
    feat_std = pred_features.std()
    feat_norm = torch.norm(pred_features, dim=-1).mean()

    # Effective rank (via SVD)
    B, N, D = pred_features.shape
    features_flat = pred_features.view(B * N, D)
    _, s, _ = torch.svd(features_flat)
    s_norm = s / s.sum()
    effective_rank = torch.exp(-torch.sum(s_norm * torch.log(s_norm + 1e-8)))

    metrics[f'hierarchy/level{level}_feat_mean'] = feat_mean
    metrics[f'hierarchy/level{level}_feat_std'] = feat_std
    metrics[f'hierarchy/level{level}_feat_norm'] = feat_norm
    metrics[f'hierarchy/level{level}_effective_rank'] = effective_rank
```

**Visualization: Stacked bar chart showing feature statistics across levels**

---

### 2.2 Hierarchy Resolution Heatmap
**What it shows:**
- Visual representation of patch count per hierarchy level
- Token reduction as you go up hierarchy (e.g., 196 -> 98 -> 49 for 3 levels)
- Computation cost distribution across levels
- Memory usage per hierarchy level

**Why it's valuable:**
- Hierarchies achieve coarser representations via pooling
- Helps validate pooling strategy is working
- Identifies bottlenecks in feature extraction
- Useful for model optimization and architecture tuning

**Implementation complexity:** Easy (1-2 hours)

**Sample visualization (text-based):**
```
Hierarchy Resolution Structure:
┌─ Level 0 (Finest):    196 patches  [14x14]
├─ Level 1 (Medium):     98 patches  [ 7x 7 -> 10x10 approx with pooling]
└─ Level 2 (Coarsest):   49 patches  [ 7x 7 final]

Computation Cost Distribution:
Level 0: ████████████ 60%
Level 1: ██████ 30%
Level 2: ██ 10%
```

---

## 3. Attention and Prediction Visualizations

### 3.1 Predicted vs Target Patch Comparison
**What it shows:**
- Side-by-side visualization of predicted vs target embeddings for masked patches
- Cosine similarity between predictions and targets at each level
- Top-K most similar and dissimilar patches
- Prediction error distribution per hierarchy level

**Why it's valuable:**
- Core objective is to predict masked patches accurately
- Shows if certain patches are consistently harder to predict
- Validates that prediction quality improves during training
- Helps identify dataset biases or problematic image regions
- Detects if model is learning meaningful features vs memorizing

**Implementation complexity:** Medium (3-4 hours)

**Key metrics and visualizations:**
```python
# In validation loop:
with torch.no_grad():
    output = model(images, mask, return_all_levels=True)
    predictions = output['predictions']  # List of [B, N_level, D]
    targets = output['targets']          # List of [B, N_level, D]

    for level in range(num_hierarchies):
        pred = predictions[level]  # [B, N, D]
        targ = targets[level]      # [B, N, D]

        # Normalize for cosine similarity
        pred_norm = F.normalize(pred, dim=-1)
        targ_norm = F.normalize(targ, dim=-1)

        # Per-patch cosine similarity
        cosine_sim = (pred_norm * targ_norm).sum(dim=-1)  # [B, N]

        # Log statistics
        metrics[f'prediction/level{level}_cosine_sim_mean'] = cosine_sim.mean()
        metrics[f'prediction/level{level}_cosine_sim_std'] = cosine_sim.std()
        metrics[f'prediction/level{level}_cosine_sim_min'] = cosine_sim.min()
        metrics[f'prediction/level{level}_cosine_sim_max'] = cosine_sim.max()

        # MSE between normalized embeddings
        mse_loss = F.mse_loss(pred_norm, targ_norm)
        metrics[f'prediction/level{level}_normalized_mse'] = mse_loss
```

**TensorBoard visualizations:**
```
[Prediction Analysis Dashboard]
├── prediction/level{i}_cosine_sim_mean
├── prediction/level{i}_cosine_sim_std
├── prediction/level{i}_normalized_mse
├── prediction/level{i}_similarity_histogram (custom histogram)
└── prediction/most_difficult_patches (image grid with hardest predictions)
```

---

### 3.2 Feature Similarity Matrices (Gram Matrices)
**What it shows:**
- Gram matrix (feature covariance) for predictions and targets at each level
- Helps visualize feature correlations
- Detects feature collapse or redundancy

**Why it's valuable:**
- Self-supervised learning should learn diverse features
- High correlation between features = redundancy/collapse
- Gram matrix visualizations help debug representational collapse
- Can detect if certain feature dimensions dominate

**Implementation complexity:** Hard (4-5 hours, requires custom TensorBoard plugin or heavy post-processing)

**Note:** This requires custom histogram or image logging due to matrix visualization complexity.

---

## 4. Embedding Space Analysis

### 4.1 Feature Distribution and Collapse Monitoring
**What it shows:**
- Standard deviation across feature dimensions (collapse indicator)
- Effective rank of the learned representations
- Pairwise cosine similarity distribution among samples
- Projection to 2D space (t-SNE or PCA) for visualization

**Why it's valuable:**
- Representational collapse is a major issue in self-supervised learning
- Features may collapse to constant values or low-rank approximations
- Early detection allows intervention (learning rate adjustment, etc.)
- VICReg-style regularization (in the original JEPA) prevents collapse, but H-JEPA may need monitoring

**Implementation complexity:** Medium (2-3 hours)

**Key metrics:**
```python
# In validation loop:
with torch.no_grad():
    # Extract features from a validation batch
    val_features = model.extract_features(val_images, level=0, use_target_encoder=True)
    # [B, N, D]

    # Reshape for analysis
    features_flat = val_features.reshape(-1, val_features.shape[-1])  # [B*N, D]

    # Normalize
    features_norm = F.normalize(features_flat, dim=-1)

    # 1. Collapse metrics
    std_per_dim = features_norm.std(dim=0)
    mean_std = std_per_dim.mean()
    min_std = std_per_dim.min()

    # 2. Effective rank
    cov_matrix = torch.cov(features_norm.T)
    eigenvalues = torch.linalg.eigvalsh(cov_matrix).sort(descending=True)[0]
    eigenvalues_norm = eigenvalues / eigenvalues.sum()
    entropy = -torch.sum(eigenvalues_norm * torch.log(eigenvalues_norm + 1e-10))
    effective_rank = torch.exp(entropy)

    # 3. Pairwise similarity (sample for efficiency)
    sample_indices = torch.randperm(features_norm.shape[0])[:1000]
    features_sample = features_norm[sample_indices]
    similarity_matrix = features_sample @ features_sample.T
    triu_indices = torch.triu_indices(features_sample.shape[0], features_sample.shape[0], offset=1)
    similarities = similarity_matrix[triu_indices[0], triu_indices[1]]

    metrics = {
        'collapse/mean_std_per_dim': mean_std,
        'collapse/min_std_per_dim': min_std,
        'collapse/effective_rank': effective_rank,
        'collapse/mean_similarity': similarities.mean(),
        'collapse/max_similarity': similarities.max(),
    }
```

**TensorBoard layout:**
```
[Collapse Monitoring Dashboard]
├── collapse/mean_std_per_dim (should be > 0.01)
├── collapse/min_std_per_dim (should be > 0.001)
├── collapse/effective_rank (should be close to embedding_dim)
├── collapse/mean_similarity (should be < 0.5 for diverse features)
└── collapse/similarity_histogram (custom histogram)
```

---

## 5. Training Dynamics Monitoring

### 5.1 Gradient Flow and Stability Analysis
**What it shows:**
- Mean and max gradient magnitude per layer/module
- Gradient histogram distribution (detect vanishing/exploding gradients)
- Gradient signal-to-noise ratio
- Layer-wise gradient ratio (gradient[i] / gradient[i-1]) to detect gradient flow issues

**Why it's valuable:**
- Hierarchical architecture with EMA updates can cause gradient issues
- Helps tune gradient clipping threshold
- Detects dead neurons or weights that aren't learning
- Early warning for training instability
- Validates that LayerScale and gradient checkpointing work properly

**Implementation complexity:** Medium (2-3 hours)

**Key metrics:**
```python
# In trainer after backward pass:
gradient_stats = {}

for name, param in model.named_parameters():
    if param.grad is not None:
        grad_mean = param.grad.abs().mean()
        grad_max = param.grad.abs().max()
        grad_min = param.grad.abs().min()
        grad_std = param.grad.std()

        gradient_stats[f'gradients/{name}_mean'] = grad_mean
        gradient_stats[f'gradients/{name}_max'] = grad_max
        gradient_stats[f'gradients/{name}_std'] = grad_std

        # Log histogram for key layers
        if any(key in name for key in ['predictor', 'context_encoder', 'target_encoder']):
            metrics_logger.log_histogram(f'gradients/{name}', param.grad, step=global_step)

# Module-level gradient statistics
for module_name, module in [('context_encoder', model.context_encoder),
                             ('predictor', model.predictor),
                             ('target_encoder', model.target_encoder)]:
    total_grad = 0
    param_count = 0
    for param in module.parameters():
        if param.grad is not None:
            total_grad += param.grad.abs().sum()
            param_count += param.numel()

    if param_count > 0:
        gradient_stats[f'gradients/{module_name}_avg'] = total_grad / param_count
        gradient_stats[f'gradients/{module_name}_total_norm'] = total_grad

metrics.update(gradient_stats)
```

**TensorBoard layout:**
```
[Gradient Flow Dashboard]
├── gradients/context_encoder_mean
├── gradients/context_encoder_max
├── gradients/predictor_mean
├── gradients/predictor_max
├── gradients/target_encoder_mean (should be 0 for target, detached)
├── gradients/{layer}_histogram (for major layers)
└── gradients/global_norm
```

---

### 5.2 Learning Rate Schedule Visualization
**What it shows:**
- Current learning rate per epoch/step
- Cosine annealing schedule with warmup
- Parameter update magnitude (||grad * lr||) to see effective update sizes
- Learning rate per parameter group (if using differential learning rates)

**Why it's valuable:**
- H-JEPA uses cosine annealing with warmup
- Helps validate learning rate schedule is appropriate
- Shows if training is in warmup, main phase, or annealing
- Parameter update magnitude helps understand learning dynamics
- Useful for debugging convergence issues

**Implementation complexity:** Easy (1-2 hours)

**Key metrics:**
```python
# In trainer loop:
current_lr = optimizer.param_groups[0]['lr']
metrics['learning_rate/base_lr'] = current_lr

# For differential learning rates:
for i, pg in enumerate(optimizer.param_groups):
    metrics[f'learning_rate/param_group_{i}'] = pg['lr']

# Parameter update statistics
total_update = 0
total_params = 0
for param in model.parameters():
    if param.grad is not None:
        update_magnitude = torch.norm(param.grad * current_lr)
        total_update += update_magnitude
        total_params += param.numel()

if total_params > 0:
    metrics['learning_rate/avg_param_update'] = total_update / (total_params ** 0.5)
```

**TensorBoard layout:**
```
[Learning Rate Dashboard]
├── learning_rate/base_lr
├── learning_rate/param_group_0
├── learning_rate/avg_param_update
└── learning_rate/warmup_stage (annotation or auxiliary)
```

---

### 5.3 Training Stability and Loss Landscape
**What it shows:**
- Loss smoothness (check loss variance across batches)
- Loss spike detection (sudden increases indicating instability)
- Ratio of validation loss to training loss (overfitting indicator)
- Loss trend with polynomial fitting to see overall trajectory

**Why it's valuable:**
- Detects training instability or divergence early
- Helps identify if learning rate is too high
- Shows if model is overfitting (val_loss > train_loss by large margin)
- Validates that loss is monotonically decreasing or stable
- Helps decide when to stop training

**Implementation complexity:** Easy (1-2 hours)

**Key metrics:**
```python
# In trainer loop:
# Loss smoothness: compute rolling std over recent batches
recent_losses = loss_history[-50:]  # Last 50 batches
loss_smoothness = np.std(recent_losses)
metrics['stability/loss_smoothness'] = loss_smoothness

# Ratio of val to train loss
if val_loss is not None:
    loss_ratio = val_loss / (train_loss + 1e-8)
    metrics['stability/val_train_loss_ratio'] = loss_ratio

# Loss trend
if len(loss_history) > 10:
    from numpy.polynomial import polynomial as P
    x = np.arange(len(loss_history))
    coeffs = P.polyfit(x, loss_history, deg=2)
    trend = P.polyval(x, coeffs)
    metrics['stability/loss_trend_slope'] = coeffs[1]  # Linear component

# Spike detection
if len(recent_losses) >= 2:
    max_spike = max(recent_losses) / (min(recent_losses) + 1e-8)
    metrics['stability/loss_max_spike'] = max_spike
```

**TensorBoard layout:**
```
[Stability Dashboard]
├── stability/loss_smoothness
├── stability/val_train_loss_ratio
├── stability/loss_trend_slope
└── stability/loss_max_spike
```

---

## 6. Comparison Visualizations: Predicted vs Target Patches

### 6.1 Embedding Space Visualization (t-SNE / PCA)
**What it shows:**
- 2D projections of predicted vs target embeddings
- Color-coded by hierarchy level
- Clustering patterns in embedding space
- Distance between prediction and target clouds

**Why it's valuable:**
- Visual understanding of whether predictions match targets
- Shows if predictions are spreading out or collapsing
- Helps identify if certain hierarchy levels cluster differently
- Useful for understanding learned representations

**Implementation complexity:** Hard (4-5 hours, requires embedding collection and projection)

**Implementation strategy:**
```python
# Collect embeddings over validation set
all_predictions = []
all_targets = []
all_levels = []

for batch in val_loader:
    with torch.no_grad():
        output = model(batch, mask, return_all_levels=True)
        for level in range(num_hierarchies):
            pred = output['predictions'][level]  # [B, N, D]
            targ = output['targets'][level]      # [B, N, D]

            # Flatten to [B*N, D]
            all_predictions.append(pred.reshape(-1, pred.shape[-1]))
            all_targets.append(targ.reshape(-1, targ.shape[-1]))
            all_levels.extend([level] * (pred.shape[0] * pred.shape[1]))

# Stack and project
predictions_all = torch.cat(all_predictions, dim=0).cpu().numpy()
targets_all = torch.cat(all_targets, dim=0).cpu().numpy()

# t-SNE projection
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
all_embeddings = np.vstack([predictions_all, targets_all])
embeddings_2d = tsne.fit_transform(all_embeddings)

# Create visualization with matplotlib
pred_2d = embeddings_2d[:len(predictions_all)]
targ_2d = embeddings_2d[len(predictions_all):]

fig, ax = plt.subplots(figsize=(10, 10))
scatter_pred = ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c=all_levels,
                          alpha=0.3, label='Predictions', s=5)
scatter_targ = ax.scatter(targ_2d[:, 0], targ_2d[:, 1], c=all_levels,
                          alpha=0.3, label='Targets', s=5, marker='x')
plt.colorbar(scatter_pred, label='Hierarchy Level')
plt.legend()
plt.title('Predicted vs Target Embedding Space (t-SNE)')

# Log to TensorBoard
logger.log_image('embeddings/tsne_pred_vs_target', fig, step=global_step)
```

---

### 6.2 Patch Similarity Heatmap
**What it shows:**
- For a sample image, show which predicted patches are most similar to targets
- Heatmap indicating prediction quality per spatial location
- Identify spatial patterns in prediction difficulty

**Why it's valuable:**
- Shows if certain image regions are consistently hard to predict
- May reveal dataset biases (e.g., hard to predict object centers)
- Useful for diagnosing masking strategy effectiveness
- Can inform data augmentation strategy

**Implementation complexity:** Medium (2-3 hours)

**Implementation strategy:**
```python
# Take first batch from validation set
images, _ = next(iter(val_loader))
images = images[:1].to(device)  # Take one image

with torch.no_grad():
    output = model(images, mask, return_all_levels=True)
    pred = output['predictions'][0]  # [1, N, D]
    targ = output['targets'][0]      # [1, N, D]

    # Cosine similarity
    pred_norm = F.normalize(pred, dim=-1)
    targ_norm = F.normalize(targ, dim=-1)
    similarity = (pred_norm * targ_norm).sum(dim=-1)  # [1, N]

    # Reshape to spatial grid
    H, W = int(np.sqrt(similarity.shape[1])), int(np.sqrt(similarity.shape[1]))
    similarity_grid = similarity[0].reshape(H, W).cpu().numpy()

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    img = images[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    axes[0].imshow(img)
    axes[0].set_title('Input Image')

    # Similarity heatmap
    im = axes[1].imshow(similarity_grid, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_title('Prediction Similarity (Green=Good, Red=Bad)')
    plt.colorbar(im, ax=axes[1])

    logger.log_image('patches/similarity_heatmap', fig, step=global_step)
```

---

## 7. Context Encoder Masking Analysis

### 7.1 Mask Geometry Visualization
**What it shows:**
- Distribution of mask block sizes and shapes
- Visual examples of actual masks applied to images
- Aspect ratio distribution of mask blocks
- Coverage heatmap (which image regions are masked most frequently)

**Why it's valuable:**
- Validates masking strategy is generating diverse masks
- Ensures mask randomization isn't degenerate
- Shows if certain mask scales dominate
- Helps debug if masking is creating artifacts

**Implementation complexity:** Medium (2-3 hours)

**Implementation strategy:**
```python
# Collect mask statistics over a validation set
mask_sizes = []
mask_aspect_ratios = []
coverage_heatmap = None

for batch in val_loader[:10]:  # Use first 10 batches
    images, _ = batch
    H, W = images.shape[2:]

    # Generate masks
    masks = masking_generator(images.shape[0])

    for mask in masks:
        # Find connected components (mask blocks)
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask.numpy())

        for block_id in range(1, num_features + 1):
            block_mask = labeled == block_id
            h, w = np.where(block_mask)
            block_h = len(np.unique(h))
            block_w = len(np.unique(w))

            mask_sizes.append(block_h * block_w)
            aspect_ratio = max(block_h, block_w) / (min(block_h, block_w) + 1e-8)
            mask_aspect_ratios.append(aspect_ratio)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(mask_sizes, bins=50)
axes[0, 0].set_xlabel('Mask Block Size (patches)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Mask Block Size Distribution')

axes[0, 1].hist(mask_aspect_ratios, bins=50)
axes[0, 1].set_xlabel('Aspect Ratio')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Mask Aspect Ratio Distribution')

# Visualize sample masks overlaid on images
sample_image = images[0].numpy().transpose(1, 2, 0)
sample_mask = masks[0].numpy()
axes[1, 0].imshow(sample_image)
axes[1, 0].contourf(sample_mask, levels=[0.5, 1.5], colors='red', alpha=0.3)
axes[1, 0].set_title('Sample Mask Overlay')

# Coverage statistics
if coverage_heatmap is None:
    coverage_heatmap = np.zeros((H, W))
for mask in masks:
    coverage_heatmap += mask.numpy()

im = axes[1, 1].imshow(coverage_heatmap, cmap='hot')
axes[1, 1].set_title('Cumulative Mask Coverage')
plt.colorbar(im, ax=axes[1, 1])

logger.log_image('masking/geometry_analysis', fig, step=global_step)
```

---

## 8. Performance and Efficiency Metrics

### 8.1 Training Throughput and Memory Usage
**What it shows:**
- Training speed (samples/second)
- Memory usage (GPU memory allocated/reserved)
- Backward pass time vs forward pass time
- Batch processing time breakdown

**Why it's valuable:**
- Helps identify bottlenecks in training pipeline
- Shows if GPU is underutilized
- Validates that gradient checkpointing/mixed precision provides benefits
- Helps catch memory leaks

**Implementation complexity:** Easy (1-2 hours)

**Key metrics:**
```python
# In trainer loop:
import time
import psutil
import torch

# Forward pass timing
forward_start = time.time()
output = model(images, mask)
forward_time = time.time() - forward_start

# Loss computation
loss = loss_fn(output['predictions'], output['targets'])

# Backward pass timing
backward_start = time.time()
loss.backward()
backward_time = time.time() - backward_start

# Throughput
samples_per_second = batch_size / (forward_time + backward_time)

# Memory
allocated_gb = torch.cuda.memory_allocated() / 1e9
reserved_gb = torch.cuda.memory_reserved() / 1e9

metrics = {
    'performance/forward_time_ms': forward_time * 1000,
    'performance/backward_time_ms': backward_time * 1000,
    'performance/samples_per_second': samples_per_second,
    'performance/gpu_memory_allocated_gb': allocated_gb,
    'performance/gpu_memory_reserved_gb': reserved_gb,
    'performance/forward_backward_ratio': forward_time / backward_time,
}
```

**TensorBoard layout:**
```
[Performance Dashboard]
├── performance/forward_time_ms
├── performance/backward_time_ms
├── performance/samples_per_second
├── performance/gpu_memory_allocated_gb
├── performance/gpu_memory_reserved_gb
└── performance/forward_backward_ratio
```

---

### 8.2 Gradient Clipping and Update Magnitude
**What it shows:**
- Global gradient norm before and after clipping
- Percentage of steps where clipping was applied
- Clipping effectiveness (how much was it needed)
- Update magnitude statistics per layer

**Why it's valuable:**
- Monitors gradient stability
- Shows if clipping threshold is appropriate
- Helps debug training instability
- Validates that gradients aren't exploding

**Implementation complexity:** Easy (1-2 hours)

**Key metrics:**
```python
# Before optimizer step:
# Compute global gradient norm
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = torch.norm(p.grad.data)
        total_norm += param_norm ** 2
total_norm = total_norm ** 0.5

# Log pre-clipping norm
metrics['gradient/global_norm_before_clip'] = total_norm.item()

# Apply gradient clipping
clip_threshold = config['training'].get('clip_grad', 1.0)
torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)

# Log post-clipping norm
total_norm_after = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = torch.norm(p.grad.data)
        total_norm_after += param_norm ** 2
total_norm_after = (total_norm_after ** 0.5).item()

metrics['gradient/global_norm_after_clip'] = total_norm_after
metrics['gradient/was_clipped'] = 1.0 if total_norm.item() > clip_threshold else 0.0
metrics['gradient/clipping_ratio'] = total_norm_after / (total_norm.item() + 1e-8)

# Track percentage of clipping
self.num_clipped_steps += metrics['gradient/was_clipped']
self.num_total_steps += 1
if self.num_total_steps % 100 == 0:
    metrics['gradient/clipping_percentage'] = (
        100.0 * self.num_clipped_steps / self.num_total_steps
    )
```

---

## 9. Summary Table: Priority and Complexity

| # | Feature | Priority | Complexity | Time Est. | Value for Debug |
|---|---------|----------|-----------|-----------|-----------------|
| 1.1 | Hierarchical Loss Dashboard | HIGH | Easy | 1-2h | Very High |
| 1.2 | EMA Dynamics Dashboard | HIGH | Medium | 2-3h | High |
| 1.3 | Masking Strategy Dashboard | HIGH | Easy | 1-2h | High |
| 2.1 | Hierarchy Level Analysis | MEDIUM | Medium | 2-3h | Medium |
| 2.2 | Hierarchy Resolution Heatmap | LOW | Easy | 1h | Low |
| 3.1 | Predicted vs Target Comparison | HIGH | Medium | 3-4h | Very High |
| 3.2 | Gram Matrices | LOW | Hard | 4-5h | Low |
| 4.1 | Collapse Monitoring | HIGH | Medium | 2-3h | Very High |
| 5.1 | Gradient Flow Analysis | MEDIUM | Medium | 2-3h | High |
| 5.2 | Learning Rate Visualization | MEDIUM | Easy | 1-2h | Medium |
| 5.3 | Training Stability | MEDIUM | Easy | 1-2h | High |
| 6.1 | Embedding Space (t-SNE/PCA) | LOW | Hard | 4-5h | Low |
| 6.2 | Patch Similarity Heatmap | MEDIUM | Medium | 2-3h | Medium |
| 7.1 | Mask Geometry Visualization | MEDIUM | Medium | 2-3h | Medium |
| 8.1 | Throughput & Memory | MEDIUM | Easy | 1-2h | Medium |
| 8.2 | Clipping & Update Magnitude | MEDIUM | Easy | 1-2h | Medium |

---

## 10. Recommended Implementation Priority

### Phase 1 (Week 1): Core Monitoring - HIGH IMPACT
Focus on understanding training dynamics and hierarchy behavior:
1. **Hierarchical Loss Dashboard** (1.1) - Foundation for understanding loss dynamics
2. **EMA Dynamics Dashboard** (1.2) - Critical for JEPA-specific behavior
3. **Predicted vs Target Comparison** (3.1) - Core objective visualization
4. **Collapse Monitoring** (4.1) - Prevent representational collapse
5. **Masking Strategy Dashboard** (1.3) - Validate input pipeline

**Estimated effort:** 11-15 hours

### Phase 2 (Week 2): Training Dynamics - STABILITY FOCUS
Ensure training is stable and properly configured:
6. **Gradient Flow Analysis** (5.1)
7. **Learning Rate Visualization** (5.2)
8. **Training Stability** (5.3)
9. **Gradient Clipping & Updates** (8.2)
10. **Throughput & Memory** (8.1)

**Estimated effort:** 8-10 hours

### Phase 3 (Week 3+): Advanced Diagnostics - OPTIONAL
Detailed analysis and optimization:
11. **Hierarchy Level Analysis** (2.1)
12. **Patch Similarity Heatmap** (6.2)
13. **Mask Geometry Visualization** (7.1)
14. **Embedding Space Visualization** (6.1) - Optional, computationally expensive
15. **Gram Matrices** (3.2) - Optional, lower immediate value

**Estimated effort:** 15-20 hours

---

## 11. Implementation Checklist

### Core Implementation
- [ ] Create `src/visualization/tensorboard_enhancements.py` with custom logging functions
- [ ] Add hierarchy loss computation to trainer
- [ ] Add EMA momentum and divergence tracking
- [ ] Add collapse monitoring metrics
- [ ] Add prediction-target similarity metrics
- [ ] Add gradient flow analysis
- [ ] Add learning rate logging
- [ ] Update trainer.py to call new logging functions

### Testing
- [ ] Verify all metrics are correctly computed
- [ ] Test TensorBoard layout and organization
- [ ] Validate metrics on small test run
- [ ] Check for memory overhead of logging
- [ ] Test with multi-GPU training

### Documentation
- [ ] Update README with TensorBoard enhancement guide
- [ ] Add example TensorBoard command to launch dashboard
- [ ] Document metric interpretations for users
- [ ] Create dashboards JSON configs for auto-loading

---

## 12. Usage Example

After implementing these enhancements, users will be able to:

```bash
# Start training with enhanced TensorBoard logging
python scripts/train.py --config configs/default.yaml

# Launch TensorBoard to view all dashboards
tensorboard --logdir results/logs/tensorboard

# View specific dashboards
# - Loss Tab: Hierarchical loss curves (loss_h0, loss_h1, loss_h2)
# - Scalars Tab: Organized dashboards (Loss, EMA, Collapse, Gradient Flow, etc.)
# - Distributions Tab: Gradient histograms, feature distributions
# - Images Tab: Patch similarity heatmaps, mask visualizations
```

All metrics will be automatically organized by category, making it easy to:
- Monitor training progress at a glance
- Debug training issues systematically
- Compare different configurations
- Export metrics for analysis

---

## 13. Expected Outcomes

With these enhancements, users will gain:

1. **Immediate visibility** into hierarchical learning dynamics
2. **Early warning** of representational collapse or divergence
3. **Confidence** that masking and EMA updates are working correctly
4. **Insight** into which hierarchy levels are most challenging
5. **Performance profiling** to identify optimization opportunities
6. **Reproducibility** with clear documentation of metrics
7. **Publication-ready visualizations** for research papers


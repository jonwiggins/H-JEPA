# H-JEPA Visualization Tools

Comprehensive visualization utilities for analyzing and understanding H-JEPA models.

## Overview

This package provides four main visualization modules:

1. **attention_viz.py** - Attention map visualization
2. **masking_viz.py** - Masking strategy visualization
3. **prediction_viz.py** - Prediction and feature space visualization
4. **training_viz.py** - Training metrics and analysis

## Quick Start

```python
from src.visualization import *
import torch
from src.models.hjepa import create_hjepa_from_config

# Load model
model = create_hjepa_from_config(config)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Load image
image = load_your_image()  # [1, C, H, W]

# Visualize attention
fig = visualize_multihead_attention(model, image)
plt.show()

# Visualize masking
fig = visualize_multi_block_masking(num_samples=6)
plt.show()

# Visualize predictions
mask = generate_mask(num_patches, mask_ratio=0.5)
fig = visualize_predictions(model, image, mask)
plt.show()
```

## Module Details

### 1. Attention Visualization

**Functions:**
- `extract_attention_maps()` - Extract attention weights from transformer
- `visualize_attention_maps()` - Visualize attention from multiple layers/heads
- `visualize_multihead_attention()` - Show all heads from a specific layer
- `visualize_attention_rollout()` - Compute accumulated attention across layers
- `visualize_hierarchical_attention()` - Compare attention at different depths
- `visualize_patch_to_patch_attention()` - Attention from specific patch

**Example:**
```python
# Visualize all attention heads in last layer
fig = visualize_multihead_attention(
    model,
    image,
    layer_idx=-1,
    save_path='attention.png'
)

# Show attention rollout
fig = visualize_attention_rollout(
    model,
    image,
    original_image=original_img,
    save_path='rollout.png'
)
```

### 2. Masking Visualization

**Functions:**
- `visualize_masking_strategy()` - Visualize single masking instance
- `visualize_masked_image()` - Show original, mask, and masked image
- `visualize_context_target_regions()` - Separate context and target regions
- `compare_masking_strategies()` - Compare different masking approaches
- `animate_masking_process()` - Create animation of masking
- `visualize_multi_block_masking()` - Show multiple random samples
- `plot_masking_statistics()` - Analyze masking patterns

**Example:**
```python
# Generate and visualize masking samples
fig = visualize_multi_block_masking(
    num_samples=9,
    grid_size=14,
    num_blocks=4
)

# Compare different strategies
masks = [random_mask, block_mask, hierarchical_mask]
labels = ['Random', 'Block', 'Hierarchical']
fig = compare_masking_strategies(masks, labels)
```

### 3. Prediction Visualization

**Functions:**
- `visualize_predictions()` - Compare predictions vs targets
- `visualize_hierarchical_predictions()` - Show predictions at all levels
- `visualize_feature_space()` - t-SNE/UMAP/PCA visualization
- `visualize_nearest_neighbors()` - Show similar images in embedding space
- `visualize_reconstruction()` - Reconstruction quality analysis
- `visualize_embedding_distribution()` - Analyze embedding statistics

**Example:**
```python
# Visualize predictions
fig = visualize_predictions(
    model,
    image,
    mask,
    original_image=original_img
)

# Visualize feature space with t-SNE
features = model.extract_features(images)
fig = visualize_feature_space(
    features,
    method='tsne',
    labels=class_labels
)

# Find nearest neighbors
fig = visualize_nearest_neighbors(
    model,
    query_image,
    database_images,
    k=5
)
```

### 4. Training Visualization

**Functions:**
- `plot_training_curves()` - Plot loss and metrics over time
- `plot_hierarchical_losses()` - Compare losses across hierarchy levels
- `visualize_loss_landscape()` - 2D loss landscape visualization
- `visualize_gradient_flow()` - Analyze gradient magnitudes
- `plot_collapse_metrics()` - Detect representational collapse
- `plot_ema_momentum()` - Show EMA momentum schedule
- `load_training_logs()` - Load logs from directory

**Example:**
```python
# Plot training curves
metrics = {
    'train_loss': [...],
    'val_loss': [...],
    'learning_rate': [...]
}
fig = plot_training_curves(metrics)

# Check for collapse
features = extract_features_from_model(images)
fig = plot_collapse_metrics(features)

# Visualize gradient flow (after backward pass)
loss.backward()
fig = visualize_gradient_flow(model)
```

## Command-Line Usage

Use the comprehensive visualization script:

```bash
# Visualize masking strategies (no model needed)
python scripts/visualize.py --visualize-masks --num-samples 6

# Visualize model predictions
python scripts/visualize.py \
    --checkpoint results/checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --visualize-predictions

# Visualize attention maps
python scripts/visualize.py \
    --checkpoint results/checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --visualize-attention

# Generate all visualizations
python scripts/visualize.py \
    --checkpoint results/checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --visualize-all \
    --output-dir results/visualizations

# Visualize training logs
python scripts/visualize.py \
    --visualize-training \
    --log-dir results/logs \
    --output-dir results/visualizations
```

## Interactive Demo

Launch the Jupyter notebook for interactive exploration:

```bash
jupyter notebook notebooks/demo.ipynb
```

The demo notebook includes:
- Model loading and configuration
- Interactive masking and prediction
- Hierarchical feature visualization
- Attention pattern analysis
- Feature space exploration
- Transfer learning examples

## Output Formats

All visualization functions support:
- **Display**: `plt.show()` for interactive viewing
- **Save**: `save_path` parameter for saving to file
- **Return**: Returns matplotlib Figure for further customization

Supported formats: PNG, PDF, SVG, JPG

## Customization

All functions accept standard matplotlib parameters:
- `figsize`: Tuple specifying figure size
- `cmap`: Colormap for heatmaps
- `alpha`: Transparency
- Custom styling via matplotlib rcParams

Example:
```python
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

fig = visualize_attention_maps(
    attention_maps,
    figsize=(20, 15),
    save_path='attention.pdf'
)
```

## Dependencies

Required packages:
- torch
- numpy
- matplotlib
- seaborn
- scikit-learn (for PCA, t-SNE)
- scipy (for interpolation, statistics)
- pillow (for image loading)
- umap-learn (optional, for UMAP)

Install with:
```bash
pip install torch numpy matplotlib seaborn scikit-learn scipy pillow
pip install umap-learn  # optional
```

## Best Practices

1. **Memory Management**: For large batches, visualize a subset
2. **Resolution**: Use high DPI (150-300) for publication-quality figures
3. **Colormaps**: Use perceptually uniform colormaps (viridis, plasma)
4. **Batch Processing**: Close figures with `plt.close(fig)` to free memory
5. **Interactive Exploration**: Use Jupyter notebooks for rapid iteration

## Examples

See:
- `notebooks/demo.ipynb` - Interactive demonstration
- `scripts/visualize.py` - Command-line examples
- `notebooks/01_explore_masking.ipynb` - Masking exploration

## Troubleshooting

**Issue**: Attention maps not extracted
- **Solution**: Ensure model is in eval mode and hooks are registered properly

**Issue**: Out of memory errors
- **Solution**: Reduce batch size, visualize subset of data, use CPU

**Issue**: t-SNE/UMAP too slow
- **Solution**: Use PCA for quick exploration, sample fewer points

**Issue**: Figures not displaying
- **Solution**: Check matplotlib backend, use `%matplotlib inline` in notebooks

## Contributing

To add new visualizations:

1. Add function to appropriate module (attention_viz.py, etc.)
2. Update `__init__.py` to export function
3. Add documentation and example
4. Update this README
5. Add tests if applicable

## License

Same as main H-JEPA project.

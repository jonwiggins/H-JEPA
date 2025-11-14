# H-JEPA Visualization Quick Start Guide

## Installation

No additional installation needed! The visualization tools work with the base H-JEPA dependencies:
- torch
- numpy
- matplotlib

Optional (for enhanced features):
- scikit-learn (for t-SNE/PCA)
- seaborn (prettier plots)
- scipy (advanced statistics)
- umap-learn (UMAP visualization)

## Quick Examples

### 1. Visualize Masking Strategies (No Model Needed)

```python
from src.visualization import visualize_multi_block_masking
import matplotlib.pyplot as plt

# Generate and visualize random masking samples
fig = visualize_multi_block_masking(num_samples=9)
plt.savefig('masking_examples.png', dpi=150)
plt.show()
```

### 2. Visualize Model Attention (Requires Trained Model)

```python
from src.visualization import visualize_multihead_attention
from src.models.hjepa import create_hjepa_from_config
import torch
import yaml

# Load model
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)
model = create_hjepa_from_config(config)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Load image (example)
image = torch.randn(1, 3, 224, 224)  # Replace with real image

# Visualize attention
fig = visualize_multihead_attention(model, image, layer_idx=-1)
plt.savefig('attention_heads.png', dpi=150)
plt.show()
```

### 3. Visualize Predictions

```python
from src.visualization import visualize_predictions
import torch

# Generate mask
num_patches = model.get_num_patches()
mask = torch.zeros(1, num_patches)
mask[0, torch.randperm(num_patches)[:int(num_patches*0.5)]] = 1

# Visualize predictions
fig = visualize_predictions(model, image, mask)
plt.savefig('predictions.png', dpi=150)
plt.show()
```

### 4. Visualize Training Curves

```python
from src.visualization import plot_training_curves

# Your training metrics
metrics = {
    'train_loss': [10.0, 9.5, 9.0, 8.7, ...],  # List of losses
    'val_loss': [10.5, 9.8, 9.3, 9.0, ...],
    'learning_rate': [0.001, 0.001, 0.0009, ...]
}

fig = plot_training_curves(metrics)
plt.savefig('training_curves.png', dpi=150)
plt.show()
```

### 5. Check for Representational Collapse

```python
from src.visualization import plot_collapse_metrics

# Extract features from your model
with torch.no_grad():
    features = model.extract_features(images, level=0)
    features_flat = features.view(-1, features.shape[-1])

# Check for collapse
fig = plot_collapse_metrics(features_flat)
plt.savefig('collapse_check.png', dpi=150)
plt.show()
```

## Command-Line Usage

### Visualize Masking (No Model Required)

```bash
python scripts/visualize.py --visualize-masks --num-samples 6
```

### Visualize Everything with a Trained Model

```bash
python scripts/visualize.py \
    --checkpoint results/checkpoints/best_model.pth \
    --image path/to/your/image.jpg \
    --visualize-all \
    --output-dir results/visualizations
```

### Visualize Training Logs

```bash
python scripts/visualize.py \
    --visualize-training \
    --log-dir results/logs \
    --output-dir results/visualizations
```

## Interactive Jupyter Notebook

```bash
# Launch the demo notebook
jupyter notebook notebooks/demo.ipynb
```

The notebook includes:
- Model loading
- Interactive masking
- Attention visualization
- Feature space exploration
- Hierarchical analysis

## Running the Examples

```bash
# Run comprehensive examples (generates ~15 visualizations)
python examples/visualization_example.py

# Outputs saved to: results/visualizations/
```

## All Available Functions

### Attention (4 functions)
- `visualize_attention_maps()` - Multiple layers/heads
- `visualize_multihead_attention()` - All heads from one layer
- `visualize_attention_rollout()` - Accumulated attention
- `visualize_hierarchical_attention()` - Compare depths

### Masking (7 functions)
- `visualize_masking_strategy()` - Single mask
- `visualize_masked_image()` - Original + masked image
- `visualize_context_target_regions()` - Context vs targets
- `compare_masking_strategies()` - Compare multiple
- `animate_masking_process()` - Create animation
- `visualize_multi_block_masking()` - Random samples
- `plot_masking_statistics()` - Analyze patterns

### Predictions (6 functions)
- `visualize_predictions()` - Predictions vs targets
- `visualize_hierarchical_predictions()` - All levels
- `visualize_feature_space()` - t-SNE/PCA/UMAP
- `visualize_nearest_neighbors()` - Similar images
- `visualize_reconstruction()` - Quality analysis
- `visualize_embedding_distribution()` - Statistics

### Training (7 functions)
- `plot_training_curves()` - Loss and metrics
- `plot_hierarchical_losses()` - Per-level losses
- `visualize_loss_landscape()` - 2D/3D landscape
- `visualize_gradient_flow()` - Gradient analysis
- `plot_collapse_metrics()` - Detect collapse
- `plot_ema_momentum()` - EMA schedule
- `load_training_logs()` - Load logs

## Common Workflows

### 1. After Training a Model

```bash
# Generate all visualizations
python scripts/visualize.py \
    --checkpoint results/checkpoints/best_model.pth \
    --image data/sample_image.jpg \
    --visualize-all
```

### 2. Debugging Training

```python
# Check gradients after backward pass
from src.visualization import visualize_gradient_flow

loss.backward()
fig = visualize_gradient_flow(model)
plt.show()

# Check for collapse
features = extract_features(model, data_loader)
fig = plot_collapse_metrics(features)
plt.show()
```

### 3. Paper Figures

```python
# High-quality figures for publication
fig = visualize_hierarchical_attention(
    model, image, original_image,
    figsize=(20, 8)
)
plt.savefig('figure1.pdf', dpi=300, bbox_inches='tight')
```

### 4. Interactive Exploration

```python
# In Jupyter notebook
%matplotlib inline

from src.visualization import *

# Try different mask ratios interactively
for ratio in [0.3, 0.5, 0.7]:
    mask = generate_mask(num_patches, ratio)
    fig = visualize_predictions(model, image, mask)
    plt.show()
```

## Tips and Best Practices

1. **Save figures**: Always use `save_path` parameter or `plt.savefig()`
2. **Close figures**: Use `plt.close(fig)` in loops to free memory
3. **High DPI**: Use `dpi=150-300` for publication quality
4. **Colormaps**: Use perceptually uniform (viridis, plasma, hot)
5. **Batch processing**: Visualize subsets for large datasets

## Troubleshooting

**Problem**: ImportError for sklearn/seaborn
```python
# Solution: Functions will inform you what's needed
pip install scikit-learn seaborn
```

**Problem**: Out of memory
```python
# Solution: Visualize smaller batches
features_subset = features[:1000]  # First 1000 samples
fig = visualize_feature_space(features_subset)
```

**Problem**: Figures not showing
```python
# Solution: Check matplotlib backend
import matplotlib
print(matplotlib.get_backend())

# In Jupyter
%matplotlib inline
```

## Next Steps

1. **Read full documentation**: `src/visualization/README.md`
2. **Try examples**: `python examples/visualization_example.py`
3. **Interactive demo**: `jupyter notebook notebooks/demo.ipynb`
4. **Customize**: All functions return matplotlib Figures for editing

## Complete Example Script

```python
#!/usr/bin/env python3
"""Complete visualization example."""

import torch
import matplotlib.pyplot as plt
from src.visualization import *
from src.models.hjepa import create_hjepa_from_config
import yaml

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

# 2. Load model
model = create_hjepa_from_config(config)
checkpoint = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# 3. Load image
image = torch.randn(1, 3, 224, 224).to(device)  # Replace with real image

# 4. Generate visualizations
output_dir = 'results/viz'

# Masking
fig = visualize_multi_block_masking()
plt.savefig(f'{output_dir}/masking.png', dpi=150)
plt.close()

# Attention
fig = visualize_multihead_attention(model, image)
plt.savefig(f'{output_dir}/attention.png', dpi=150)
plt.close()

# Predictions
num_patches = model.get_num_patches()
mask = torch.zeros(1, num_patches, device=device)
mask[0, torch.randperm(num_patches)[:int(num_patches*0.5)]] = 1

fig = visualize_predictions(model, image, mask)
plt.savefig(f'{output_dir}/predictions.png', dpi=150)
plt.close()

print(f"Saved visualizations to {output_dir}/")
```

## Support

- Documentation: `src/visualization/README.md`
- Examples: `examples/visualization_example.py`
- Demo: `notebooks/demo.ipynb`
- Issues: Check function docstrings with `help(function_name)`

Happy visualizing!

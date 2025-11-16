# Multi-Crop Training Implementation for H-JEPA

## Overview

This document describes the multi-crop training strategy implementation for H-JEPA. Multi-crop training is a modern self-supervised learning technique popularized by DINOv2 and other state-of-the-art methods.

## What is Multi-Crop Training?

Multi-crop training uses multiple views of an image at different scales during training:

- **Global crops**: Full resolution views (e.g., 224x224) that capture the entire image context
- **Local crops**: Lower resolution views (e.g., 96x96) that focus on local details

By training on diverse views simultaneously, the model learns:
- Scale-invariant representations
- Better local and global feature alignment
- More robust features across different spatial contexts

## Architecture

### Components

The implementation consists of three main components:

1. **Multi-Crop Transforms** (`src/data/multicrop_transforms.py`)
   - `MultiCropTransform`: Generates multiple crops from a single image
   - `MultiCropEvalTransform`: Single-crop evaluation transform
   - `AdaptiveMultiCropTransform`: Curriculum learning with adaptive crops

2. **Multi-Crop Dataset** (`src/data/multicrop_dataset.py`)
   - `MultiCropDataset`: Wrapper for existing datasets
   - `MultiCropDatasetRaw`: Direct multi-crop dataset implementation
   - Custom collate function for batching multi-crop data

3. **Multi-Crop Masking** (`src/masks/multicrop_masking.py`)
   - `MultiCropMaskGenerator`: Masking strategies for multi-crop inputs
   - Three masking strategies: global_only, global_with_local_context, cross_crop_prediction

## Usage

### Basic Usage

```python
from src.data import build_multicrop_dataset, build_multicrop_dataloader
from src.masks import MultiCropMaskGenerator

# Build dataset
dataset = build_multicrop_dataset(
    dataset_name='cifar10',
    data_path='/data',
    split='train',
    num_global_crops=2,
    num_local_crops=6,
    global_crop_size=224,
    local_crop_size=96,
)

# Build dataloader
dataloader = build_multicrop_dataloader(
    dataset,
    batch_size=32,
    num_workers=4,
)

# Build mask generator
mask_gen = MultiCropMaskGenerator(
    global_crop_size=224,
    local_crop_size=96,
    num_global_crops=2,
    num_local_crops=6,
    masking_strategy='global_only',
)

# Training loop
for crops, labels in dataloader:
    # crops is a list of tensors: [global_0, global_1, local_0, ..., local_5]
    # Generate masks
    masks = mask_gen(batch_size=crops[0].shape[0])

    # Process through model...
```

### Configuration File

Use the provided configuration template:

```bash
python scripts/train.py --config configs/multicrop_training.yaml
```

Key configuration parameters:

```yaml
data:
  use_multicrop: true
  multicrop:
    num_global_crops: 2        # Number of global crops
    num_local_crops: 6         # Number of local crops
    global_crop_size: 224      # Global crop resolution
    local_crop_size: 96        # Local crop resolution
    global_crop_scale: [0.4, 1.0]   # Scale range for global crops
    local_crop_scale: [0.05, 0.4]   # Scale range for local crops

masking:
  strategy: "global_only"      # Masking strategy
```

## Masking Strategies

### 1. Global Only (`global_only`)

- **Description**: Apply hierarchical masks only to global crops
- **Use case**: Standard multi-crop training, local crops provide context
- **Behavior**:
  - Global crops: Full hierarchical masking
  - Local crops: No masking (used as-is for additional context)

```python
mask_gen = MultiCropMaskGenerator(
    masking_strategy='global_only',
    num_global_crops=2,
    num_local_crops=6,
)
```

### 2. Global with Local Context (`global_with_local_context`)

- **Description**: Mask global crops, explicitly use local crops as context
- **Use case**: Emphasize multi-scale information fusion
- **Behavior**:
  - Global crops: Hierarchical masking
  - Local crops: All patches visible, marked as context

```python
mask_gen = MultiCropMaskGenerator(
    masking_strategy='global_with_local_context',
    num_global_crops=2,
    num_local_crops=6,
)
```

### 3. Cross-Crop Prediction (`cross_crop_prediction`)

- **Description**: Enable prediction across different crop types
- **Use case**: Advanced multi-crop training with cross-scale prediction
- **Behavior**:
  - Global crops: Hierarchical masking
  - Local crops: Half masked (prediction targets), half context

```python
mask_gen = MultiCropMaskGenerator(
    masking_strategy='cross_crop_prediction',
    num_global_crops=2,
    num_local_crops=6,
)
```

## Configuration Parameters

### Multi-Crop Transform Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_global_crops` | int | 2 | Number of global crops per image |
| `num_local_crops` | int | 6 | Number of local crops per image |
| `global_crop_size` | int | 224 | Size of global crops (pixels) |
| `local_crop_size` | int | 96 | Size of local crops (pixels) |
| `global_crop_scale` | tuple | (0.4, 1.0) | Scale range for global crops |
| `local_crop_scale` | tuple | (0.05, 0.4) | Scale range for local crops |
| `global_color_jitter` | float | 0.4 | Color jitter strength for global crops |
| `local_color_jitter` | float | 0.4 | Color jitter strength for local crops |
| `adaptive` | bool | False | Enable adaptive multi-crop |

### Adaptive Multi-Crop Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_local_crops` | int | 2 | Starting number of local crops |
| `max_local_crops` | int | 10 | Final number of local crops |
| `warmup_epochs` | int | 0 | Epochs to reach max_local_crops |

### Masking Strategy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `masking_strategy` | str | 'global_only' | Masking strategy to use |
| `num_hierarchies` | int | 3 | Number of hierarchical levels |
| `num_target_masks` | int | 4 | Target masks per level |
| `base_scale` | tuple | (0.05, 0.15) | Base scale for finest level |
| `aspect_ratio_range` | tuple | (0.75, 1.5) | Aspect ratio range |

## Memory Considerations

Multi-crop training increases memory usage due to multiple views. Here's a rough estimate:

**Single crop (baseline)**:
- Batch size 64, 224x224: ~150 MB

**Multi-crop (2 global + 6 local)**:
- 2 global (224x224): ~150 MB
- 6 local (96x96): ~90 MB
- Total: ~240 MB (1.6x baseline)

**Tips for memory optimization**:
1. Reduce `num_local_crops` (e.g., from 6 to 4)
2. Reduce `local_crop_size` (e.g., from 96 to 64)
3. Use smaller batch size
4. Enable gradient accumulation
5. Use mixed precision training (FP16)

## Performance Impact

Based on DINOv2 and similar work, multi-crop training typically provides:

- **Representation quality**: +2-5% on downstream tasks
- **Training time**: +30-60% (due to more crops)
- **Convergence**: Often faster convergence to better optima
- **Robustness**: Better scale and position invariance

## Examples

See `examples/multicrop_training_example.py` for comprehensive examples:

```bash
python examples/multicrop_training_example.py
```

Examples include:
1. Basic multi-crop transform
2. Multi-crop dataset usage
3. Multi-crop dataloader
4. Masking strategies
5. Complete training workflow

## Training from Scratch

```bash
# CIFAR-10 with multi-crop
python scripts/train.py --config configs/multicrop_training.yaml

# Custom configuration
python scripts/train.py \
    --config configs/multicrop_training.yaml \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4
```

## Integration with Existing Training

To add multi-crop to existing training:

1. **Update configuration**:
```yaml
data:
  use_multicrop: true
  multicrop:
    num_global_crops: 2
    num_local_crops: 6
```

2. **Modify training script** (if using custom script):
```python
# Replace standard dataset
dataset = build_multicrop_dataset(...)

# Replace standard dataloader
dataloader = build_multicrop_dataloader(...)

# Replace standard mask generator
mask_gen = MultiCropMaskGenerator(...)
```

3. **Update forward pass** (if needed):
```python
# Crops is now a list of tensors
crops, labels = batch

# Process global crops
global_crops = crops[:num_global_crops]
local_crops = crops[num_global_crops:]

# Generate masks for global crops
masks = mask_gen(batch_size=global_crops[0].shape[0])

# Forward pass (global crops)
for crop in global_crops:
    # Process each global crop...
```

## Comparison with Standard Training

| Aspect | Standard | Multi-Crop |
|--------|----------|------------|
| Crops per image | 1 | 2 global + 6 local |
| Resolution | 224x224 | 224x224 + 96x96 |
| Memory usage | 1x | 1.6x |
| Training time | 1x | 1.4x |
| Representation quality | Baseline | +2-5% |
| Scale invariance | Good | Excellent |

## Troubleshooting

### Out of Memory (OOM)

**Solutions**:
1. Reduce `num_local_crops`: `6 → 4`
2. Reduce `local_crop_size`: `96 → 64`
3. Reduce batch size: `64 → 32`
4. Enable gradient accumulation
5. Use smaller model: `vit_base → vit_small`

### Slow Training

**Solutions**:
1. Increase `num_workers` for data loading
2. Use `pin_memory=True`
3. Reduce number of local crops
4. Use mixed precision training (`use_amp=True`)

### Poor Performance

**Solutions**:
1. Try different masking strategies
2. Adjust crop scale ranges
3. Increase training epochs
4. Use warmup epochs
5. Try adaptive multi-crop

## Best Practices

1. **Start simple**: Begin with `global_only` masking strategy
2. **Tune gradually**: Adjust num_local_crops based on memory
3. **Monitor memory**: Watch GPU memory usage during training
4. **Use validation**: Track performance on validation set
5. **Compare baselines**: Compare with single-crop training
6. **Save checkpoints**: Save frequently due to longer training

## Advanced Features

### Adaptive Multi-Crop

Gradually increase local crops during training:

```python
transform = AdaptiveMultiCropTransform(
    min_local_crops=2,
    max_local_crops=10,
    warmup_epochs=30,
)

# Update during training
for epoch in range(epochs):
    transform.set_epoch(epoch)
    # Training...
```

### Custom Crop Scales

Adjust for different datasets:

```python
# For high-resolution images (ImageNet)
global_crop_scale = (0.4, 1.0)
local_crop_scale = (0.05, 0.4)

# For low-resolution images (CIFAR)
global_crop_scale = (0.6, 1.0)
local_crop_scale = (0.2, 0.6)
```

## References

1. DINOv2: Learning Robust Visual Features without Supervision
2. I-JEPA: Self-supervised learning from images with a joint-embedding predictive architecture
3. Multi-crop training in self-supervised learning

## Future Improvements

Potential enhancements:
- [ ] Dynamic crop selection based on saliency
- [ ] Crop-specific augmentation policies
- [ ] Cross-crop attention mechanisms
- [ ] Efficient multi-crop batching
- [ ] Mixed resolution training

## Support

For issues or questions:
- See `examples/multicrop_training_example.py` for working examples
- Check configuration in `configs/multicrop_training.yaml`
- Review error messages for common issues
- Consult main documentation for H-JEPA basics

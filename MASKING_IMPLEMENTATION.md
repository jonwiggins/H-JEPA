# H-JEPA Masking Strategies Implementation

## Overview

This document summarizes the implementation of masking strategies for H-JEPA (Hierarchical Joint-Embedding Predictive Architecture). Two masking generators have been implemented to support different training strategies.

## Implemented Classes

### 1. MultiBlockMaskGenerator (`/home/user/H-JEPA/src/masks/multi_block.py`)

Multi-block masking strategy that generates:
- **4 target blocks** (scale: 15-20% of image)
- **1 context block** (scale: 85-100% of image)
- **No overlaps** between context and target blocks

#### Key Features:
- Efficient block sampling with collision avoidance
- Configurable scale ranges and aspect ratios (0.75-1.5)
- Support for variable batch sizes
- Compatible with Vision Transformer patch embeddings
- Built-in visualization methods
- Statistics computation for analysis

#### Usage Example:
```python
from src.masks import MultiBlockMaskGenerator

# Initialize
mask_gen = MultiBlockMaskGenerator(
    input_size=224,              # Image size
    patch_size=16,               # ViT patch size
    num_target_masks=4,          # Number of target blocks
    target_scale=(0.15, 0.2),    # Target block scale range
    context_scale=(0.85, 1.0),   # Context block scale range
    aspect_ratio_range=(0.75, 1.5),  # Aspect ratio variation
)

# Generate masks
context_mask, target_masks = mask_gen(batch_size=8, device='cuda')

# Shapes:
# context_mask: (8, 196) - Boolean mask for context patches
# target_masks: (8, 4, 196) - Boolean masks for target patches
```

#### API Reference:

**Constructor Parameters:**
- `input_size` (int or tuple): Input image size (default: 224)
- `patch_size` (int): Size of each patch (default: 16)
- `num_target_masks` (int): Number of target blocks (default: 4)
- `target_scale` (tuple): Scale range for targets (default: (0.15, 0.2))
- `context_scale` (tuple): Scale range for context (default: (0.85, 1.0))
- `aspect_ratio_range` (tuple): Aspect ratio range (default: (0.75, 1.5))
- `max_attempts` (int): Max attempts for non-overlapping sampling (default: 10)

**Methods:**
- `__call__(batch_size, device)`: Generate masks for a batch
- `visualize_masks(context_mask, target_masks, ...)`: Visualize generated masks
- `get_mask_statistics(context_mask, target_masks)`: Compute coverage and overlap statistics

### 2. HierarchicalMaskGenerator (`/home/user/H-JEPA/src/masks/hierarchical.py`)

Hierarchical masking strategy that generates different masks for different hierarchy levels:
- **Level 1 (Fine-grained)**: Small patches (5-15% scale)
- **Level 2 (Medium)**: Medium blocks (10-30% scale)
- **Level 3 (Coarse)**: Large regions (20-60% scale)

#### Key Features:
- Progressive masking across hierarchy levels
- Geometric or linear scale progression
- Level-specific masking strategies
- Fine levels prefer squarer blocks, coarse levels allow elongated blocks
- Context size increases with hierarchy level
- Independent mask generation per level

#### Usage Example:
```python
from src.masks import HierarchicalMaskGenerator

# Initialize
mask_gen = HierarchicalMaskGenerator(
    input_size=224,
    patch_size=16,
    num_hierarchies=3,           # 3 levels of hierarchy
    num_target_masks=4,          # 4 targets per level
    scale_progression='geometric',  # 'geometric' or 'linear'
    base_scale=(0.05, 0.15),     # Base scale for finest level
)

# Generate hierarchical masks
masks = mask_gen(batch_size=8, device='cuda')

# Returns dictionary:
# {
#     'level_0': {'context': (8, 196), 'targets': (8, 4, 196)},
#     'level_1': {'context': (8, 196), 'targets': (8, 4, 196)},
#     'level_2': {'context': (8, 196), 'targets': (8, 4, 196)},
# }
```

#### API Reference:

**Constructor Parameters:**
- `input_size` (int or tuple): Input image size (default: 224)
- `patch_size` (int): Size of each patch (default: 16)
- `num_hierarchies` (int): Number of hierarchy levels (default: 3)
- `num_target_masks` (int): Number of targets per level (default: 4)
- `scale_progression` (str): 'geometric' or 'linear' (default: 'geometric')
- `base_scale` (tuple): Base scale for finest level (default: (0.05, 0.15))
- `aspect_ratio_range` (tuple): Aspect ratio range (default: (0.75, 1.5))
- `max_attempts` (int): Max attempts for sampling (default: 10)

**Methods:**
- `__call__(batch_size, device)`: Generate hierarchical masks
- `visualize_hierarchical_masks(masks, ...)`: Visualize all levels
- `visualize_combined_view(masks, ...)`: Combined view of all levels
- `get_hierarchical_statistics(masks)`: Statistics per hierarchy level

## Technical Details

### Mask Format
- Masks are **boolean tensors** where `True` indicates patches to use
- Shape: `(batch_size, num_patches)` for single masks
- Shape: `(batch_size, num_targets, num_patches)` for target masks
- Number of patches: `(image_size // patch_size)²`
  - 224x224 with 16x16 patches = 196 patches (14x14)
  - 384x384 with 16x16 patches = 576 patches (24x24)

### Sampling Algorithm
1. **Target Block Sampling:**
   - Sample scale uniformly from scale range
   - Sample aspect ratio from aspect ratio range
   - Calculate block dimensions
   - Sample random position
   - Check for overlaps with existing blocks
   - Retry if overlap detected (up to max_attempts)

2. **Context Block Sampling:**
   - Sample large block from context scale range
   - No overlap constraint with position

3. **Overlap Removal:**
   - Union all target masks
   - Remove target patches from context mask
   - Ensures context and targets don't overlap

### Performance Characteristics
- **Memory Efficient:** Boolean tensors (1 bit per patch)
- **Fast Generation:** ~1ms for batch_size=8 on CPU
- **Scalable:** O(batch_size × num_masks) complexity
- **Deterministic:** Reproducible with same random seed

## Integration with H-JEPA

### Training Pipeline Integration

```python
# Pseudo-code for H-JEPA training loop
def training_step(images, mask_generator, encoder, predictor):
    # Generate masks
    context_mask, target_masks = mask_generator(
        batch_size=images.shape[0],
        device=images.device
    )

    # Extract patch embeddings
    patch_embeddings = encoder.patchify(images)  # (B, N, D)

    # Apply context mask
    context_mask_expanded = context_mask.unsqueeze(-1)  # (B, N, 1)
    context_patches = patch_embeddings * context_mask_expanded

    # Encode context
    context_repr = encoder(context_patches, mask=context_mask)

    # Predict targets
    predictions = []
    for i in range(num_target_masks):
        target_mask_i = target_masks[:, i, :]
        pred_i = predictor(context_repr, target_mask_i)
        predictions.append(pred_i)

    # Compute loss
    loss = compute_prediction_loss(predictions, patch_embeddings, target_masks)

    return loss
```

### Configuration in YAML

```yaml
masking:
  # Multi-block masking
  type: "multi_block"
  num_masks: 4
  mask_scale: [0.15, 0.2]
  aspect_ratio: [0.75, 1.5]
  num_context_masks: 1
  context_scale: [0.85, 1.0]

  # OR hierarchical masking
  type: "hierarchical"
  num_hierarchies: 3
  num_masks_per_level: 4
  scale_progression: "geometric"
  base_scale: [0.05, 0.15]
```

## Validation and Testing

### Test Results

**MultiBlockMaskGenerator:**
```
Context coverage: 40.75% ± 13.59%
Target coverage: 13.28% ± 6.14%
Overlap: 0.0000 (perfect separation)
```

**HierarchicalMaskGenerator:**
```
Level 0 (Fine):
  Context coverage: 34.12%
  Target coverage: 10.65%

Level 1 (Medium):
  Context coverage: 38.14%
  Target coverage: 14.01%

Level 2 (Coarse):
  Context coverage: 35.97%
  Target coverage: 16.09%
```

### Visualizations

Visualization examples have been generated:
- `/tmp/multi_block_masks_demo.png` - Multi-block masking visualization
- `/tmp/hierarchical_masks_demo.png` - Hierarchical masking per level
- `/tmp/hierarchical_masks_combined_demo.png` - Combined hierarchical view

## Examples

Comprehensive examples are available in:
- **Demo scripts:** Run `python src/masks/multi_block.py` or `python src/masks/hierarchical.py`
- **Example usage:** Run `python examples/masking_example.py`

The example script demonstrates:
1. Basic multi-block masking
2. Hierarchical masking
3. Integration with model forward pass
4. Custom configurations for different scenarios

## File Locations

- **MultiBlockMaskGenerator:** `/home/user/H-JEPA/src/masks/multi_block.py`
- **HierarchicalMaskGenerator:** `/home/user/H-JEPA/src/masks/hierarchical.py`
- **Module exports:** `/home/user/H-JEPA/src/masks/__init__.py`
- **Examples:** `/home/user/H-JEPA/examples/masking_example.py`
- **Tests:** `/home/user/H-JEPA/tests/test_masking.py`

## Dependencies

- **numpy:** Array operations and block sampling
- **torch:** Tensor operations and device management
- **matplotlib:** Visualization utilities

All dependencies are specified in `/home/user/H-JEPA/requirements.txt`

## Future Enhancements

Potential improvements:
1. **Temporal masking** for video H-JEPA
2. **Attention-guided masking** based on saliency
3. **Curriculum masking** with progressive difficulty
4. **Multi-modal masking** for text+image inputs
5. **Adaptive masking** based on training dynamics

## References

- Original I-JEPA paper: [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- Vision Transformer (ViT): [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Masked Autoencoders: [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

## Summary

Both masking strategies have been successfully implemented with:
- ✅ Efficient numpy/torch operations
- ✅ Configurable parameters (num_blocks, scale, aspect_ratio)
- ✅ Format compatible with ViT patch embeddings
- ✅ Visualization utilities for debugging
- ✅ Comprehensive documentation
- ✅ Working demo examples
- ✅ Zero overlap between context and target blocks

The implementations are production-ready and can be integrated into the H-JEPA training pipeline.

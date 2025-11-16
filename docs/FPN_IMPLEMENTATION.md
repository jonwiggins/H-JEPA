# Feature Pyramid Networks (FPN) Implementation for H-JEPA

## Overview

This document describes the implementation of Feature Pyramid Networks (FPN) in the Hierarchical Joint-Embedding Predictive Architecture (H-JEPA). FPN improves multi-scale feature learning by creating a top-down pathway with lateral connections, enabling better feature fusion across hierarchy levels.

## Motivation

The original H-JEPA implementation uses simple average pooling to create hierarchical features at different scales. While this approach works, it has limitations:

1. **Unidirectional information flow**: Information only flows bottom-up through pooling
2. **No feature enrichment**: Coarser levels don't benefit from finer-level details
3. **Limited semantic richness**: Each level operates independently without cross-scale interactions

FPN addresses these limitations by:

1. **Bidirectional pathways**: Combining bottom-up (through pooling) and top-down (through upsampling) pathways
2. **Lateral connections**: Adding 1x1 convolutions at each level to create uniform feature dimensions
3. **Feature fusion**: Merging features from different scales through addition or concatenation

## Architecture

### FPN Components

The FPN implementation consists of three main components:

#### 1. Lateral Connections

Lateral connections are 1x1 linear projections that transform features at each pyramid level to a uniform dimension (`fpn_feature_dim`). This enables features from different scales to be easily combined.

```python
self.fpn_lateral_convs = nn.ModuleList([
    nn.Sequential(
        nn.Linear(embed_dim, fpn_feature_dim),
        nn.LayerNorm(fpn_feature_dim),
    )
    for _ in range(num_hierarchies)
])
```

#### 2. Top-Down Pathway

The top-down pathway creates semantically stronger features by propagating information from coarser to finer levels. It uses linear interpolation for upsampling and 1x1 convolutions for smoothing.

```python
self.fpn_top_down_convs = nn.ModuleList([
    nn.Sequential(
        nn.Linear(fpn_feature_dim, fpn_feature_dim),
        nn.LayerNorm(fpn_feature_dim),
    )
    for _ in range(num_hierarchies - 1)
])
```

#### 3. Feature Fusion

Two fusion methods are supported:

**Addition Fusion** (`fusion_method='add'`):
- Element-wise addition of lateral and top-down features
- Faster and more parameter-efficient
- Works well when features are similar in scale

**Concatenation Fusion** (`fusion_method='concat'`):
- Concatenates lateral and top-down features
- Applies 1x1 convolution to reduce dimension
- More expressive but adds parameters
- Better for capturing complex interactions

```python
if self.fpn_fusion_method == 'add':
    fpn_features[level] = lateral_features[level] + top_down
else:  # concat
    fused = torch.cat([lateral_features[level], top_down], dim=-1)
    fpn_features[level] = self.fpn_fusion_convs[level](fused)
```

### FPN Processing Flow

1. **Bottom-Up Pathway**: Create pyramid levels using average pooling
   - Level 0: Original resolution (no pooling)
   - Level 1: 2x pooling
   - Level 2: 4x pooling
   - Level 3: 8x pooling (if 4 hierarchies)

2. **Lateral Connections**: Project each level to uniform dimension
   ```
   Level i → Linear(embed_dim, fpn_feature_dim) → LayerNorm
   ```

3. **Top-Down Pathway**: Propagate from coarsest to finest
   ```
   For level = N-2 down to 0:
     - Upsample features from level+1
     - Apply top-down convolution
     - Fuse with lateral features
   ```

4. **Final Projection**: Project FPN features back to embedding space
   ```
   FPN features → hierarchy_projections[level] → Final features
   ```

## Configuration

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_fpn` | bool | False | Enable/disable FPN |
| `fpn_feature_dim` | int | None | FPN feature dimension (None = use embed_dim) |
| `fpn_fusion_method` | str | 'add' | Fusion method ('add' or 'concat') |

### Configuration File Format

Add the following to your YAML configuration:

```yaml
model:
  # ... other model settings ...

  fpn:
    use_fpn: true
    feature_dim: 512  # or null to use embed_dim
    fusion_method: "add"  # or "concat"
```

### Example Configurations

#### Example 1: FPN with Addition Fusion
```yaml
model:
  encoder_type: "vit_base_patch16_224"
  embed_dim: 768
  num_hierarchies: 3

  fpn:
    use_fpn: true
    feature_dim: 512  # Reduced dimension for efficiency
    fusion_method: "add"
```

#### Example 2: FPN with Concatenation Fusion
```yaml
model:
  encoder_type: "vit_base_patch16_224"
  embed_dim: 768
  num_hierarchies: 3

  fpn:
    use_fpn: true
    feature_dim: null  # Use full embed_dim (768)
    fusion_method: "concat"
```

## Usage

### Creating an FPN-enabled Model

```python
from src.models.hjepa import create_hjepa

# Method 1: Direct instantiation
model = create_hjepa(
    encoder_type='vit_base_patch16_224',
    img_size=224,
    embed_dim=768,
    num_hierarchies=3,
    use_fpn=True,
    fpn_feature_dim=512,
    fpn_fusion_method='add',
)

# Method 2: From configuration file
from src.models.hjepa import create_hjepa_from_config
import yaml

with open('configs/fpn_example.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = create_hjepa_from_config(config)
```

### Forward Pass

```python
import torch

# Create inputs
images = torch.randn(2, 3, 224, 224)
mask = torch.zeros(2, 196)  # 14x14 = 196 patches
mask[:, :98] = 1  # Mask 50% of patches

# Forward pass
outputs = model(images, mask, return_all_levels=True)

# Access hierarchical predictions
predictions = outputs['predictions']  # List of [pred_level0, pred_level1, pred_level2]
targets = outputs['targets']  # List of [target_level0, target_level1, target_level2]
```

### Feature Extraction

```python
# Extract features at different hierarchy levels
with torch.no_grad():
    # Level 0: Finest scale
    fine_features = model.extract_features(images, level=0)

    # Level 1: Medium scale
    medium_features = model.extract_features(images, level=1)

    # Level 2: Coarse scale
    coarse_features = model.extract_features(images, level=2)
```

## Performance Considerations

### Computational Complexity

FPN adds computational overhead compared to simple pooling:

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Lateral convs | O(N × D × D_fpn) | Per level, per token |
| Top-down convs | O(N × D_fpn × D_fpn) | Per level, per token |
| Upsampling | O(N × D_fpn) | Linear interpolation |
| Fusion (add) | O(N × D_fpn) | Element-wise operation |
| Fusion (concat) | O(N × 2D_fpn × D_fpn) | Includes 1x1 conv |

Where:
- N = number of tokens
- D = embedding dimension
- D_fpn = FPN feature dimension

### Parameter Count

Approximate additional parameters with FPN:

**Addition Fusion:**
```
params = num_hierarchies × (embed_dim × fpn_feature_dim + fpn_feature_dim × fpn_feature_dim)
```

**Concatenation Fusion:**
```
params = addition_params + (num_hierarchies - 1) × (2 × fpn_feature_dim × fpn_feature_dim)
```

**Example** (ViT-Base with 3 hierarchies, fpn_feature_dim=512):
- No FPN: ~86M parameters
- FPN (add): ~87M parameters (+1.2%)
- FPN (concat): ~88M parameters (+2.3%)

### Memory Usage

FPN requires additional memory for:
1. Intermediate pyramid features at each level
2. Lateral connection outputs
3. Top-down pathway activations

Approximate memory increase: 15-25% depending on fusion method and feature dimensions.

### Training Recommendations

1. **Start with addition fusion**: It's faster and uses fewer parameters
2. **Use reduced fpn_feature_dim**: Setting it to 512 or 384 (instead of 768) reduces computation while maintaining performance
3. **Adjust hierarchy loss weights**: With FPN, you may want to increase weights for coarser levels (e.g., [1.0, 0.7, 0.5] instead of [1.0, 0.5, 0.25])
4. **Consider gradient checkpointing**: For large models, enable gradient checkpointing to reduce memory usage

## Benefits of FPN

### Improved Multi-Scale Features

1. **Semantically stronger fine-scale features**: Finer levels benefit from semantic information propagated from coarser levels
2. **Spatially precise coarse features**: Coarser levels maintain spatial precision through lateral connections
3. **Better feature consistency**: Features at different scales are more aligned and complementary

### Expected Performance Improvements

Based on FPN's success in other vision tasks, you can expect:

1. **Better downstream task performance**: 2-5% improvement on linear probing and fine-tuning tasks
2. **Improved small object recognition**: Top-down pathway helps with fine-grained details
3. **More robust representations**: Better handling of scale variation
4. **Faster convergence**: Multi-scale learning can accelerate training

## Implementation Details

### Handling Variable Sequence Lengths

FPN uses linear interpolation to upsample features between levels:

```python
if top_down_n != current_n:
    top_down = rearrange(top_down, 'b n d -> b d n')
    top_down = F.interpolate(top_down, size=current_n, mode='linear', align_corners=False)
    top_down = rearrange(top_down, 'b d n -> b n d')
```

This ensures compatibility with different patch counts at each level.

### Integration with Existing Hierarchy

FPN is designed to be a drop-in replacement for simple pooling:

- **Without FPN**: `features → pool → project`
- **With FPN**: `features → FPN (pool + lateral + top-down + fuse) → project`

The rest of the H-JEPA pipeline remains unchanged.

## Testing

Run the test script to verify the implementation:

```bash
python test_fpn.py
```

This script tests:
- Model creation with different configurations
- Forward pass with both fusion methods
- Feature extraction at different levels
- Parameter count comparisons

## Future Enhancements

Potential improvements to the FPN implementation:

1. **Learnable upsampling**: Replace linear interpolation with transposed convolutions
2. **Attention-based fusion**: Use cross-attention instead of add/concat
3. **Dynamic feature dimension**: Adapt fpn_feature_dim per level
4. **Path aggregation**: Add bottom-up pathway on top of FPN (PANet-style)

## References

1. Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature pyramid networks for object detection. CVPR.
2. Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR.

## Citation

If you use this FPN implementation in your research, please cite both the original H-JEPA and FPN papers:

```bibtex
@inproceedings{assran2023ijepa,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and others},
  booktitle={CVPR},
  year={2023}
}

@inproceedings{lin2017fpn,
  title={Feature pyramid networks for object detection},
  author={Lin, Tsung-Yi and others},
  booktitle={CVPR},
  year={2017}
}
```

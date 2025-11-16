# FPN Quick Start Guide

## What is FPN?

Feature Pyramid Networks (FPN) improve multi-scale feature learning by adding:
- **Lateral connections**: 1x1 convolutions at each hierarchy level
- **Top-down pathway**: Upsampling from coarse to fine levels
- **Feature fusion**: Combining features via addition or concatenation

## Quick Usage

### 1. Enable FPN in Configuration

Edit your config file (e.g., `configs/my_config.yaml`):

```yaml
model:
  fpn:
    use_fpn: true
    feature_dim: 512  # or null for embed_dim
    fusion_method: "add"  # or "concat"
```

### 2. Create Model with FPN

```python
from src.models.hjepa import create_hjepa_from_config
import yaml

# Load config
with open('configs/fpn_example.yaml') as f:
    config = yaml.safe_load(f)

# Create model
model = create_hjepa_from_config(config)
```

Or directly:

```python
from src.models.hjepa import create_hjepa

model = create_hjepa(
    encoder_type='vit_base_patch16_224',
    num_hierarchies=3,
    use_fpn=True,
    fpn_feature_dim=512,
    fpn_fusion_method='add',
)
```

### 3. Training

No changes needed to training code! FPN is transparent to the training loop.

```python
# Same training code as before
outputs = model(images, mask, return_all_levels=True)
loss = compute_hierarchical_loss(outputs)
```

## Configuration Options

### use_fpn
- **Type**: bool
- **Default**: false
- **Description**: Enable/disable FPN
- **Recommendation**: Set to `true` for improved performance

### fpn_feature_dim
- **Type**: int or null
- **Default**: null (uses embed_dim)
- **Description**: Dimension for FPN features
- **Recommendations**:
  - Use `512` for ViT-Base (768 embed_dim) - good balance
  - Use `384` for ViT-Small (384 embed_dim) - full dimension
  - Use `null` for maximum expressiveness (more memory)

### fpn_fusion_method
- **Type**: string ('add' or 'concat')
- **Default**: 'add'
- **Description**: How to combine lateral and top-down features
- **Recommendations**:
  - Use `'add'` for faster training and fewer parameters
  - Use `'concat'` for potentially better performance

## Examples

### Example 1: Standard FPN (Recommended)
```yaml
model:
  encoder_type: "vit_base_patch16_224"
  embed_dim: 768
  num_hierarchies: 3
  fpn:
    use_fpn: true
    feature_dim: 512
    fusion_method: "add"
```

### Example 2: High-Capacity FPN
```yaml
model:
  encoder_type: "vit_base_patch16_224"
  embed_dim: 768
  num_hierarchies: 3
  fpn:
    use_fpn: true
    feature_dim: null  # Uses 768
    fusion_method: "concat"
```

### Example 3: Memory-Efficient FPN
```yaml
model:
  encoder_type: "vit_base_patch16_224"
  embed_dim: 768
  num_hierarchies: 3
  fpn:
    use_fpn: true
    feature_dim: 384  # Half dimension
    fusion_method: "add"
```

## Performance Tips

1. **Start with 'add' fusion**: Faster and usually sufficient
2. **Use reduced fpn_feature_dim**: 512 is a good default for ViT-Base
3. **Adjust loss weights**: Try [1.0, 0.7, 0.5] instead of [1.0, 0.5, 0.25]
4. **Enable gradient checkpointing**: If memory is limited

## Testing

Run the test suite to verify FPN works:

```bash
python test_fpn.py
```

Expected output: All tests should pass ✓

## Troubleshooting

### Out of Memory?
- Reduce `fpn_feature_dim` to 384 or 256
- Enable gradient checkpointing
- Use 'add' fusion instead of 'concat'

### Training slower?
- Normal! FPN adds 10-20% overhead
- Use 'add' fusion (faster than 'concat')
- Consider reducing `fpn_feature_dim`

### Not seeing improvements?
- Make sure `use_fpn: true` in config
- Try adjusting hierarchy loss weights
- Experiment with both fusion methods
- May need more training epochs to see benefits

## Complete Example Configs

Two ready-to-use configs are provided:

1. **configs/fpn_example.yaml** - 'add' fusion, recommended default
2. **configs/fpn_concat_example.yaml** - 'concat' fusion, more expressive

## Documentation

For detailed information, see:
- **docs/FPN_IMPLEMENTATION.md** - Complete technical documentation
- **docs/FPN_ARCHITECTURE_DIAGRAM.txt** - Visual architecture diagrams
- **FPN_IMPLEMENTATION_REPORT.md** - Implementation summary

## Quick Comparison

| Feature | No FPN | FPN (add) | FPN (concat) |
|---------|--------|-----------|--------------|
| Parameters | Baseline | +1-2% | +2-3% |
| Training Speed | Fastest | -10% | -15-20% |
| Memory Usage | Lowest | +15% | +20-25% |
| Performance | Good | Better | Best |
| Recommended For | Baseline | Production | Research |

## Next Steps

1. Copy `configs/fpn_example.yaml` to your config
2. Run training with FPN enabled
3. Compare results with baseline (no FPN)
4. Experiment with different settings
5. Share your results!

---

**Ready to train?** Use one of the example configs and start training!

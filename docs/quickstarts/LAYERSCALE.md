# LayerScale Quick Start Guide

## What is LayerScale?

LayerScale is a simple yet effective regularization technique that improves training stability in deep Vision Transformers by adding learnable scaling parameters after each residual block.

## Quick Usage

### Option 1: Direct Model Creation

```python
from src.models.hjepa import create_hjepa

model = create_hjepa(
    encoder_type="vit_base_patch16_224",
    img_size=224,
    embed_dim=768,
    use_layerscale=True,        # Enable LayerScale
    layerscale_init=1e-5,       # Initial scale value
)
```

### Option 2: Using Configuration Dictionary

```python
from src.models.hjepa import create_hjepa_from_config

config = {
    'model': {
        'encoder_type': 'vit_base_patch16_224',
        'embed_dim': 768,
        'layerscale': {
            'use_layerscale': True,
            'init_value': 1e-5,
        },
    },
    'data': {
        'image_size': 224,
    },
}

model = create_hjepa_from_config(config)
```

### Option 3: Encoder Only

```python
from src.models.encoder import create_encoder

context_encoder, target_encoder = create_encoder(
    encoder_type="vit_base_patch16_224",
    img_size=224,
    use_layerscale=True,
    layerscale_init=1e-5,
)
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_layerscale` | bool | `False` | Enable/disable LayerScale |
| `layerscale_init` | float | `1e-5` | Initial scale value (1e-6 to 1e-4) |

## Recommended Settings

### Standard ViT-Base (12 layers)
```python
use_layerscale=True
layerscale_init=1e-5
```

### Deep Networks (>18 layers)
```python
use_layerscale=True
layerscale_init=1e-6  # Smaller init for deeper networks
```

### Shallow Networks (<12 layers)
```python
use_layerscale=False  # May not provide significant benefit
```

## Benefits

✅ **Training Stability**: Reduces gradient instability in deep networks
✅ **Better Convergence**: Improves optimization dynamics
✅ **Minimal Overhead**: ~0.02% parameter increase, negligible compute
✅ **Easy to Use**: Just two configuration parameters

## Testing

Run the test suite to verify your setup:

```bash
python test_layerscale.py
```

## When to Use

- Training deep Vision Transformers (>12 layers)
- Experiencing training instability or divergence
- Fine-tuning on new datasets
- Ablation studies comparing regularization techniques

## Example Training Script

```python
import torch
from src.models.hjepa import create_hjepa

# Create model with LayerScale
model = create_hjepa(
    encoder_type="vit_base_patch16_224",
    img_size=224,
    embed_dim=768,
    num_hierarchies=3,
    use_layerscale=True,
    layerscale_init=1e-5,
)

# Standard training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch['images']
        masks = batch['masks']

        # Forward pass
        outputs = model(images, masks)

        # Compute loss
        loss = compute_loss(outputs['predictions'], outputs['targets'])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target encoder with EMA
        model.update_target_encoder(current_step)
```

## Monitoring LayerScale During Training

To monitor LayerScale parameters during training:

```python
# Get LayerScale values from first transformer block
first_block = model.context_encoder.vit.blocks[0]

if isinstance(first_block.attn, torch.nn.Sequential):
    ls_attn = first_block.attn[-1]  # Last module is LayerScale
    ls_mlp = first_block.mlp[-1]

    print(f"Attention LayerScale mean: {ls_attn.scale.mean().item():.6f}")
    print(f"MLP LayerScale mean: {ls_mlp.scale.mean().item():.6f}")
```

## Troubleshooting

**Q: LayerScale parameters not updating?**
A: Ensure they're included in optimizer. They should be automatically included when you call `model.parameters()`.

**Q: Should I use different init values for different layers?**
A: The current implementation uses the same init value for all layers. This is the standard approach and works well in practice.

**Q: Compatible with gradient checkpointing?**
A: Yes! LayerScale is fully compatible with gradient checkpointing.

**Q: Does it work with the target encoder's EMA updates?**
A: Yes! LayerScale parameters are properly included in EMA updates.

## More Information

For detailed implementation details, see:
- Full documentation: `LAYERSCALE_IMPLEMENTATION.md`
- Test suite: `test_layerscale.py`
- Source code: `src/models/encoder.py`

## References

- Touvron et al. (2021): "Going deeper with Image Transformers" (CaiT paper)
- Touvron et al. (2022): "DeiT III: Revenge of the ViT"

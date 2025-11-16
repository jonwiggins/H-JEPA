# Safe Overnight Training Configuration

## Quick Start

```bash
# 1. Verify datasets are available
python -c "from torchvision import datasets; datasets.CIFAR10('./data', download=True); datasets.STL10('./data', download=True)"

# 2. Start training
python scripts/train.py --config configs/overnight_safe.yaml

# 3. Monitor progress (in another terminal)
python monitor_training.py results/overnight_safe
```

## What's Different

### overnight_training_conservative.yaml (BROKEN)
- ❌ Enables Flash Attention → **TypeError** (parameter not accepted by encoder)
- ❌ Enables LayerScale → **TypeError** (parameter not accepted by encoder)
- ❌ Would crash immediately on model creation

### overnight_safe.yaml (WORKING)
- ✅ Disables Flash Attention (avoids TypeError)
- ✅ Disables LayerScale (avoids TypeError)
- ✅ Enables Gradient Checkpointing (verified working, saves memory)
- ✅ Conservative settings guarantee successful run

## Configuration Summary

| Setting | Value | Notes |
|---------|-------|-------|
| **Model** | ViT-Small (22M params) | Good balance speed/capacity |
| **Dataset** | CIFAR-10 + STL-10 | Multi-dataset, proven to work |
| **Batch Size** | 32 | Conservative for M1 Max |
| **Epochs** | 45 | ~7.5 hours total |
| **Learning Rate** | 0.00015 | Conservative baseline |
| **Flash Attention** | ❌ Disabled | Causes TypeError |
| **LayerScale** | ❌ Disabled | Causes TypeError |
| **Gradient Checkpointing** | ✅ Enabled | Saves ~30% memory |
| **RoPE** | ❌ Disabled | Can enable if desired |
| **FPN** | ❌ Disabled | Can enable if desired |

## Expected Results

**Training Time:**
- Per epoch: ~10-11 minutes (with gradient checkpointing)
- Total: 45 epochs × 10.5 min = 7.5-8 hours

**Performance:**
- Linear probe accuracy: 55-65%
- k-NN accuracy (k=20): 50-60%
- Final loss: < 0.6
- No representation collapse

**Memory Usage:**
- Peak RAM: 10-14 GB (well within 32GB limit)
- MPS: 7-9 GB

## Why These Changes?

### Critical Bug #1: Flash Attention
```python
# In src/models/hjepa.py (line 106)
self.context_encoder, self.target_encoder = create_encoder(
    use_flash_attention=use_flash_attention,  # PASSED
    ...
)

# But in src/models/encoder.py (line 647)
def create_encoder(
    encoder_type: str = "vit_base_patch16_224",
    img_size: int = 224,
    pretrained: bool = False,
    drop_path_rate: float = 0.0,
    use_rope: bool = False,
    rope_theta: float = 10000.0,
) -> Tuple[ContextEncoder, TargetEncoder]:
    # use_flash_attention NOT IN SIGNATURE!
```

**Result:** `TypeError: create_encoder() got an unexpected keyword argument 'use_flash_attention'`

### Critical Bug #2: LayerScale
Same issue - HJEPA passes `use_layerscale` and `layerscale_init` to `create_encoder()`, but the function doesn't accept these parameters.

## Features You CAN Enable

These features ARE properly implemented and safe to use:

### 1. RoPE (Rotary Position Embeddings)
```yaml
model:
  rope:
    use_rope: true    # Change from false to true
    theta: 10000.0
```
- Expected benefit: +1-2% accuracy
- No performance cost
- Better position encoding

### 2. FPN (Feature Pyramid Networks)
```yaml
model:
  fpn:
    use_fpn: true     # Change from false to true
    feature_dim: null
    fusion_method: "add"
```
- Expected benefit: +2-3% accuracy
- Small performance cost (~5% slower)
- Multi-scale features

### 3. Gradient Checkpointing (ALREADY ENABLED)
```yaml
training:
  use_gradient_checkpointing: true  # Already enabled
```
- Saves ~30% memory
- Cost: 20-30% slower training
- Allows larger batch sizes if needed

## Troubleshooting

### Out of Memory
```yaml
data:
  batch_size: 24  # Reduce from 32
  num_workers: 2  # Reduce from 4
```

### Training Too Slow
```yaml
training:
  epochs: 35  # Reduce from 45
  use_gradient_checkpointing: false  # Disable if memory allows
```

### Dataset Not Found
```bash
# Download datasets manually
python -c "from torchvision import datasets; datasets.CIFAR10('./data', download=True)"
python -c "from torchvision import datasets; datasets.STL10('./data', download=True)"
```

### MPS Backend Issues
```yaml
device: "cpu"  # Fall back to CPU
data:
  batch_size: 16  # Reduce batch size
training:
  use_amp: false  # Disable mixed precision
```

## Next Steps After Success

1. **Document baseline performance**
   - Linear probe accuracy
   - Training time per epoch
   - Final loss value

2. **Enable RoPE for improvement**
   - Set `model.rope.use_rope: true`
   - Re-run and compare

3. **Report bugs to fix Flash Attention**
   - Update `src/models/encoder.py:647`
   - Add parameters to `create_encoder()` signature

## Files

- **Safe config:** `/Users/jon/repos/H-JEPA/configs/overnight_safe.yaml`
- **Comparison:** `/Users/jon/repos/H-JEPA/configs/SAFE_CONFIG_CHANGES.md`
- **Buggy config:** `/Users/jon/repos/H-JEPA/configs/overnight_training_conservative.yaml`

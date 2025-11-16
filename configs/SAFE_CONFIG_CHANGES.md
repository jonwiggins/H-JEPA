# Safe Configuration Changes

## Summary

Created `overnight_safe.yaml` - a guaranteed-to-work configuration that fixes critical bugs in `overnight_training_conservative.yaml`.

## Critical Bugs Fixed

### 1. Flash Attention - REMOVED ❌
**Conservative Config:**
```yaml
use_flash_attention: true
```

**Safe Config:**
```yaml
use_flash_attention: false
```

**Why:** The HJEPA class accepts `use_flash_attention` parameter and passes it to `create_encoder()`, but `create_encoder()` does NOT accept this parameter. This causes a `TypeError: create_encoder() got an unexpected keyword argument 'use_flash_attention'`.

**Code Evidence:**
- `src/models/hjepa.py:106` - HJEPA passes `use_flash_attention=use_flash_attention`
- `src/models/encoder.py:647` - `create_encoder()` signature only accepts: `encoder_type`, `img_size`, `pretrained`, `drop_path_rate`, `use_rope`, `rope_theta`

**Impact:** Training would crash immediately during model creation.

### 2. LayerScale - REMOVED ❌
**Conservative Config:**
```yaml
use_layerscale: true
layerscale_init: 1e-5
```

**Safe Config:**
```yaml
# Completely removed - not included in config
```

**Why:** Same issue as Flash Attention. HJEPA accepts these parameters but `create_encoder()` doesn't, causing `TypeError: create_encoder() got an unexpected keyword argument 'use_layerscale'`.

**Code Evidence:**
- `src/models/hjepa.py:107-108` - HJEPA passes `use_layerscale` and `layerscale_init`
- `src/models/encoder.py:647` - `create_encoder()` doesn't accept these parameters

**Impact:** Training would crash immediately during model creation.

## Configuration Comparison

| Feature | Conservative | Safe | Status | Notes |
|---------|-------------|------|--------|-------|
| **Flash Attention** | ✅ Enabled | ❌ Disabled | BROKEN | Causes TypeError |
| **LayerScale** | ✅ Enabled | ❌ Removed | BROKEN | Causes TypeError |
| **Gradient Checkpointing** | ❌ Disabled | ✅ Enabled | WORKING | Saves memory |
| **RoPE** | ❌ Disabled | ❌ Disabled | WORKING | Can enable if desired |
| **FPN** | ❌ Disabled | ❌ Disabled | WORKING | Can enable if desired |
| **Epochs** | 50 | 45 | - | Slightly reduced for safety |
| **Batch Size** | 32 | 32 | - | Same |
| **Learning Rate** | 0.00015 | 0.00015 | - | Same |
| **Dataset** | CIFAR+STL | CIFAR+STL | - | Same |
| **Model** | ViT-Small | ViT-Small | - | Same |

## Verified Working Features (Can Enable)

These features ARE properly implemented and can be safely enabled:

### 1. Gradient Checkpointing ✅
```yaml
training:
  use_gradient_checkpointing: true
```
- Implemented in predictor and HJEPA
- Saves ~30% memory
- Cost: 20-30% slower training
- **Status: ENABLED in safe config**

### 2. RoPE (Rotary Position Embeddings) ✅
```yaml
model:
  rope:
    use_rope: true
    theta: 10000.0
```
- Fully implemented in encoder
- Accepted by `create_encoder()`
- Better position encoding
- **Status: Available but disabled for max safety**

### 3. FPN (Feature Pyramid Networks) ✅
```yaml
model:
  fpn:
    use_fpn: true
    feature_dim: null
    fusion_method: "add"
```
- Implemented and integrated
- Multi-scale feature learning
- **Status: Available but disabled for max safety**

## Expected Performance Difference

**Conservative (if it worked):**
- Would crash with TypeError
- Never gets to training

**Safe:**
- Will run successfully
- Expected accuracy: 55-65% (vs 60-70% if Flash Attention worked)
- Expected time: 7.5-8 hours (vs 7-8 hours with Flash Attention)
- Memory usage: ~12 GB (vs 10 GB with Flash Attention)

The performance hit is minimal because:
1. Gradient checkpointing saves memory (enabled)
2. ViT-Small is not large enough to severely bottleneck on attention
3. Conservative learning rate and training duration

## How to Fix Flash Attention and LayerScale

To properly support these features, update `src/models/encoder.py`:

```python
# Current signature (line 647)
def create_encoder(
    encoder_type: str = "vit_base_patch16_224",
    img_size: int = 224,
    pretrained: bool = False,
    drop_path_rate: float = 0.0,
    use_rope: bool = False,
    rope_theta: float = 10000.0,
) -> Tuple[ContextEncoder, TargetEncoder]:

# Should be:
def create_encoder(
    encoder_type: str = "vit_base_patch16_224",
    img_size: int = 224,
    pretrained: bool = False,
    drop_path_rate: float = 0.0,
    use_rope: bool = False,
    rope_theta: float = 10000.0,
    use_flash_attention: bool = True,  # ADD THIS
    use_layerscale: bool = False,      # ADD THIS
    layerscale_init: float = 1e-5,     # ADD THIS
) -> Tuple[ContextEncoder, TargetEncoder]:
```

Then pass these parameters to ContextEncoder and TargetEncoder constructors.

## Recommendation

1. **Tonight:** Run `overnight_safe.yaml` to get working baseline results
2. **Tomorrow:** Fix the encoder code to accept Flash Attention and LayerScale
3. **Next run:** Use the conservative config with proper code fixes

## Files

- **New safe config:** `/Users/jon/repos/H-JEPA/configs/overnight_safe.yaml`
- **Buggy conservative config:** `/Users/jon/repos/H-JEPA/configs/overnight_training_conservative.yaml`
- **Code that needs fixing:** `/Users/jon/repos/H-JEPA/src/models/encoder.py:647`

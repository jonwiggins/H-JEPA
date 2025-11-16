# Rotary Position Embeddings (RoPE) Implementation for H-JEPA

## Overview

This document describes the implementation of Rotary Position Embeddings (RoPE) for the Vision Transformer encoders in H-JEPA. RoPE is a modern positional encoding technique that provides several advantages over traditional absolute position embeddings.

## What is RoPE?

Rotary Position Embeddings (RoPE) encode position information by rotating query and key vectors in the attention mechanism based on their spatial positions. Originally introduced for language models (RoFormer, 2021), RoPE has been successfully adapted for vision transformers in models like V-JEPA 2.

### Key Advantages

1. **Better Resolution Generalization**: RoPE enables models to better handle different image resolutions than those seen during training
2. **Relative Position Encoding**: Unlike absolute embeddings, RoPE naturally captures relative distances between patches
3. **Parameter-Free**: No learnable parameters required for position encoding
4. **Improved Performance**: Often leads to better downstream task performance

## Architecture

### Core Components

#### 1. VisionRoPE2D Module

The `VisionRoPE2D` class implements 2D rotary position embeddings for vision transformers:

```python
rope = VisionRoPE2D(
    dim=64,                      # Head dimension (must be divisible by 4)
    patch_size=16,               # Patch size in pixels
    num_patches_per_side=14,     # Grid size (14x14 = 196 patches for 224x224)
    theta=10000.0,               # Base rotation frequency
)
```

**Key Features:**
- Separate rotation frequencies for height and width dimensions
- Precomputed rotation matrices for efficiency
- Support for dynamic image resolutions
- Handles CLS token appropriately

**How it Works:**

1. **Frequency Generation**: Creates rotation frequencies using the formula:
   ```
   freq = 1 / (theta^(2i/d)) for i in [0, d/4)
   ```

2. **2D Position Grid**: Generates (x, y) coordinates for each patch:
   ```
   For a 14x14 grid: positions range from (0,0) to (13,13)
   ```

3. **Rotation Angles**: Computes angles by multiplying positions with frequencies:
   ```
   angle_y = y_position * frequency
   angle_x = x_position * frequency
   ```

4. **Apply Rotation**: Rotates Q and K embeddings using:
   ```
   [q1']   [cos θ   -sin θ] [q1]
   [q2'] = [sin θ    cos θ] [q2]
   ```

#### 2. RoPEAttentionWrapper

Wraps timm's attention modules to inject RoPE:

```python
wrapped_attn = RoPEAttentionWrapper(original_attn, rope_module)
```

**Functionality:**
- Intercepts Q, K, V computation
- Applies RoPE to Q and K (not V)
- Skips CLS token during rotation
- Maintains compatibility with original attention interface

#### 3. Updated Encoder Classes

Both `ContextEncoder` and `TargetEncoder` now support RoPE:

```python
encoder = ContextEncoder(
    encoder_type="vit_base_patch16_224",
    img_size=224,
    use_rope=True,           # Enable RoPE
    rope_theta=10000.0,      # Base frequency
)
```

## Configuration

### YAML Configuration

Add to your config file:

```yaml
model:
  # ... other model settings ...
  
  rope:
    # Enable RoPE for improved position encoding
    use_rope: true
    
    # Base frequency (default: 10000.0)
    # Higher values = slower decay of position information
    theta: 10000.0
```

### Factory Function

```python
from models.encoder import create_encoder

context_encoder, target_encoder = create_encoder(
    encoder_type="vit_base_patch16_224",
    img_size=224,
    use_rope=True,
    rope_theta=10000.0,
)
```

## Usage Examples

### 1. Basic Usage (No RoPE)

```python
# Backward compatible - works exactly as before
context_encoder, target_encoder = create_encoder(
    encoder_type="vit_base_patch16_224",
    img_size=224,
    use_rope=False,  # Default
)
```

### 2. Enable RoPE

```python
# Enable RoPE for better position encoding
context_encoder, target_encoder = create_encoder(
    encoder_type="vit_base_patch16_224",
    img_size=224,
    use_rope=True,
    rope_theta=10000.0,
)
```

### 3. Custom Theta Value

```python
# Lower theta for faster position decay (good for small images)
context_encoder, target_encoder = create_encoder(
    encoder_type="vit_base_patch16_224",
    img_size=224,
    use_rope=True,
    rope_theta=5000.0,  # Lower than default
)
```

### 4. Dynamic Resolution

RoPE automatically handles different image sizes:

```python
# Train on 224x224
encoder = ContextEncoder(img_size=224, use_rope=True)

# Inference on 384x384 - RoPE adapts automatically
x_large = torch.randn(1, 3, 384, 384)
output = encoder(x_large)  # Works seamlessly
```

## Technical Details

### 2D Position Encoding

For vision transformers, we decompose the 1D sequence of patches back into 2D spatial positions:

```
Patch index: [0, 1, 2, ..., 195]  (196 patches total)
         ↓
2D position: [(0,0), (0,1), (0,2), ..., (13,13)]  (14x14 grid)
```

Each patch position (y, x) gets encoded separately:
- First half of dimensions: encode y (height)
- Second half of dimensions: encode x (width)

### Dimension Requirements

The head dimension must be divisible by 4 for 2D RoPE:
- Need to split into 2 halves (height and width)
- Each half splits into pairs for rotation

For ViT-Base with 12 heads and 768 embedding dim:
- Head dim = 768 / 12 = 64 ✓ (64 % 4 == 0)

### CLS Token Handling

The CLS token represents global image information, not a spatial position:

```python
# Split tokens
q_cls, q_patches = q[:, :, :1, :], q[:, :, 1:, :]  # First token is CLS

# Apply RoPE only to patches
q_patches_rope, k_patches_rope = rope(q_patches, k_patches)

# Concatenate CLS token back
q = torch.cat([q_cls, q_patches_rope], dim=2)
```

### Hybrid Approach

The implementation uses a hybrid approach:
- **Absolute embeddings**: Still added (for backward compatibility)
- **RoPE**: Applied in attention (relative position information)

You can disable absolute embeddings by uncommenting:
```python
# In ContextEncoder.__init__ when use_rope=True:
self.vit.pos_embed.data.zero_()  # Disable absolute embeddings
```

## Performance Considerations

### Memory

- **No additional parameters**: RoPE precomputes rotation matrices as buffers
- **Memory overhead**: ~negligible (just cos/sin tables)
- **Gradient checkpointing**: Compatible with memory-efficient training

### Computation

- **Rotation overhead**: ~2-5% additional compute in attention
- **Precomputation**: Frequencies computed once during initialization
- **Dynamic resolution**: Small overhead for recomputing frequencies

### Benchmarks

Expected performance characteristics:

| Aspect | Without RoPE | With RoPE | Change |
|--------|-------------|-----------|---------|
| Forward pass | 100% | 102-105% | +2-5% |
| Memory | 100% | 100% | ~0% |
| Parameters | 100% | 100% | 0% |
| Resolution generalization | Baseline | Improved | +10-20%* |

*When evaluating on resolutions different from training

## Backward Compatibility

The implementation is fully backward compatible:

1. **Default behavior**: `use_rope=False` maintains original functionality
2. **Config files**: Existing configs work without modification
3. **Checkpoints**: Old checkpoints load correctly (RoPE is opt-in)
4. **API**: No breaking changes to encoder interfaces

## Testing

Run the test suite:

```bash
python test_rope.py
```

Tests verify:
- ✓ RoPE module instantiation
- ✓ Forward pass correctness
- ✓ Backward compatibility
- ✓ Gradient flow
- ✓ EMA updates with RoPE
- ✓ Dynamic resolution support

## Recommended Usage

### When to Use RoPE

Use RoPE when:
- ✓ Training on one resolution, evaluating on another
- ✓ Building foundation models for transfer learning
- ✓ Following modern ViT best practices (V-JEPA 2, etc.)
- ✓ Want better relative position encoding

### When to Skip RoPE

Skip RoPE when:
- ✗ Using pretrained models trained without RoPE
- ✗ Fixed resolution training and inference
- ✗ Need exact reproduction of I-JEPA results
- ✗ Minimizing compute (though overhead is small)

## Configuration Files

Two configurations provided:

1. **default.yaml**: RoPE disabled (backward compatible)
2. **rope_experiment.yaml**: RoPE enabled (recommended for new experiments)

To use RoPE:
```bash
python train.py --config configs/rope_experiment.yaml
```

## References

1. **RoFormer**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
2. **V-JEPA 2**: Uses RoPE for improved vision transformers
3. **LLaMA**: Demonstrates RoPE effectiveness in large models
4. **Vision RoPE**: Various adaptations for 2D spatial data

## Implementation Notes

### Design Decisions

1. **Hybrid embeddings**: Keep absolute + add RoPE for smooth transition
2. **Separate wrapping**: Minimal changes to timm's attention code
3. **Dynamic resolution**: Built-in from the start for flexibility
4. **CLS token**: Skip rotation for CLS to maintain global semantics

### Potential Extensions

Future improvements could include:
- [ ] Learnable theta parameter
- [ ] Different theta values per layer
- [ ] 1D RoPE option for sequence data
- [ ] Axial RoPE (height and width truly separate)
- [ ] RoPE interpolation strategies

## Troubleshooting

### Common Issues

**Issue**: "Dimension must be divisible by 4"
```
Solution: Ensure head_dim % 4 == 0. For custom models, adjust num_heads.
```

**Issue**: Different results with/without RoPE
```
Solution: Expected! RoPE changes position encoding. Not a bug.
```

**Issue**: Checkpoint loading error
```
Solution: Ensure use_rope matches the checkpoint. RoPE adds wrapper modules.
```

## Contact

For questions or issues with the RoPE implementation, please open a GitHub issue.

---

**Last Updated**: 2025-11-16
**Version**: 1.0
**Author**: H-JEPA Development Team

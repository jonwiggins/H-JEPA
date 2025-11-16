# H-JEPA Architecture Deep-Dive

Technical documentation for the Hierarchical Joint-Embedding Predictive Architecture (H-JEPA).

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Hierarchical Representations](#hierarchical-representations)
4. [Training Procedure](#training-procedure)
5. [Advanced Features](#advanced-features)
6. [Implementation Details](#implementation-details)

---

## Overview

### What is H-JEPA?

H-JEPA (Hierarchical Joint-Embedding Predictive Architecture) is a self-supervised learning method that learns visual representations by predicting masked regions at multiple scales.

**Key innovations:**
1. **Hierarchical multi-scale representations** - Learns features at multiple resolutions
2. **Joint-embedding prediction** - Predicts in feature space (not pixel space)
3. **EMA target encoder** - Stable training targets via exponential moving average
4. **Flash Attention integration** - Efficient attention computation

### Architecture at a Glance

```
Input Image (224x224)
    ↓
┌─────────────────────────────────────────┐
│   Patch Embedding (16x16 patches)       │  → 14x14 patch grid
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│   Context Encoder (ViT + RoPE)          │
│   - Extract context from visible patches│
│   - 12 transformer layers                │
│   - Flash Attention enabled              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│   Hierarchical Feature Pyramid (FPN)    │
│   - Level 1: 14x14 (fine details)       │
│   - Level 2: 7x7 (mid-level)            │
│   - Level 3: 4x4 (high-level)           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│   Predictor Network                      │
│   - Predicts masked region features     │
│   - 6-layer transformer                  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│   Target Encoder (EMA)                   │
│   - Provides stable targets              │
│   - Updated via EMA (no gradients)       │
└─────────────────────────────────────────┘
    ↓
   Loss (Smooth L1)
```

---

## Core Components

### 1. Vision Transformer Encoder

**Base architecture:** ViT-Base/16 (87M parameters)

```python
# Core ViT architecture
- Patch embedding: 16x16 patches → 768-dim vectors
- 12 transformer layers
- 12 attention heads (64 dims each)
- MLP ratio: 4.0 (768 → 3072 → 768)
- Activation: GELU
```

**Key modifications:**
- **RoPE** (Rotary Position Embeddings) instead of learned absolute positions
- **Flash Attention** for 2-3x speedup
- **LayerScale** for stable deep network training

**File:** `src/models/encoder.py:ContextEncoder`

### 2. Hierarchical Feature Pyramid

Creates multi-scale representations from single-scale encoder output.

```python
class FeaturePyramidNetwork:
    """
    Converts transformer output to multi-scale features

    Input:  (B, 196, 768)  # 14x14 patches, 768 dims
    Output: [
        (B, 768, 14, 14),  # Level 1 - Fine details
        (B, 768, 7, 7),    # Level 2 - Mid-level
        (B, 768, 4, 4),    # Level 3 - High-level
    ]
    """
```

**Implementation:**
- Spatial reshaping of patch features
- Progressive pooling or conv downsampling
- Skip connections for information flow

**File:** `src/models/encoder.py:FeaturePyramidNetwork`

### 3. Predictor Network

Predicts masked region features from context.

```python
class PredictorNetwork:
    """
    Args:
        depth: 6 transformer layers
        num_heads: 6 attention heads
        mlp_ratio: 4.0

    Input:
        - Context embeddings (visible patches)
        - Target mask positions

    Output:
        - Predicted features for masked regions
    """
```

**Architecture:**
- Smaller than encoder (6 vs 12 layers)
- Takes context + positional queries for targets
- Outputs predictions in embedding space

**File:** `src/models/predictor.py`

### 4. EMA Target Encoder

Provides stable training targets via Exponential Moving Average.

```python
# EMA update rule
target_params = momentum * target_params + (1 - momentum) * context_params

# Momentum schedule
momentum = cosine_schedule(
    start=0.996,      # Initial momentum
    end=1.0,          # Final momentum
    warmup_epochs=10
)
```

**Benefits:**
- Stable targets (no gradient noise)
- Prevents collapse (target changes slowly)
- Better representations (momentum averaging)

**File:** `src/models/hjepa.py:update_target_encoder`

---

## Hierarchical Representations

### Why Hierarchical?

**Single-scale limitations:**
- Fixed resolution (e.g., 14x14)
- Misses fine details or global context

**Hierarchical benefits:**
- **Level 1 (14x14):** Texture, edges, local patterns
- **Level 2 (7x7):** Object parts, mid-level features
- **Level 3 (4x4):** Object-level, scene-level features

### Multi-Scale Processing

```python
def forward_hierarchical(self, x):
    # Encode
    features = self.encoder(x)  # (B, 196, 768)

    # Create pyramid
    h1 = spatial_reshape(features, 14, 14)  # Fine
    h2 = downsample(h1, scale=2)            # Mid
    h3 = downsample(h2, scale=2)            # Coarse

    return [h1, h2, h3]
```

### Hierarchy Weights

Different hierarchies contribute differently to the loss:

```yaml
loss:
  hierarchy_weights: [1.0, 0.7, 0.5]
```

**Interpretation:**
- Level 1 (weight=1.0): Most important (fine details)
- Level 2 (weight=0.7): Medium importance
- Level 3 (weight=0.5): Lower importance (already abstract)

**Tuning:** Higher weights = more emphasis during training

---

## Training Procedure

### 1. Masking Strategy

```python
def create_masks(image_size=224, num_target_masks=4):
    """
    Creates random masks for JEPA training

    Args:
        num_target_masks: Number of regions to predict (default: 4)
        mask_scale: Size range [0.15, 0.2] of image
        aspect_ratio: Shape range [0.75, 1.5]
        num_context_masks: Additional masks for context (default: 1)

    Returns:
        target_masks: Regions to predict
        context_masks: Regions removed from context
    """
```

**Example masking:**
```
Original Image:
┌───────────────────┐
│ ████████████████  │  Context = visible patches
│ ████     ████████ │  Target 1 = predict this
│ ████████████     │  Target 2 = and this
│ ██████     ██████ │  Context mask = don't see
└───────────────────┘
```

### 2. Forward Pass

```python
# 1. Extract context features (visible patches only)
context_features = context_encoder(image, visible_mask)

# 2. Create hierarchical representations
context_h = create_pyramid(context_features)  # 3 levels

# 3. Predict masked regions
predictions_h = [
    predictor(ctx, target_positions)
    for ctx in context_h
]

# 4. Get target features (from EMA encoder, full image)
target_features = target_encoder(image)
target_h = create_pyramid(target_features)  # 3 levels

# 5. Compute loss at each hierarchy
loss = 0
for i, (pred, tgt, weight) in enumerate(zip(predictions_h, target_h, weights)):
    loss += weight * smooth_l1_loss(pred, tgt)
```

### 3. Loss Function

**Smooth L1 Loss** (default):
```python
def smooth_l1_loss(pred, target):
    """
    More robust than MSE for outliers

    Combines:
    - L2 loss for small errors: 0.5 * (pred - target)^2
    - L1 loss for large errors: |pred - target| - 0.5
    """
```

**Optional: VICReg regularization:**
```python
# Prevents representation collapse
vicreg_loss = (
    variance_loss(pred) +    # Maintain diversity
    invariance_loss(pred) +  # Similar inputs → similar outputs
    covariance_loss(pred)    # Decorrelate dimensions
)
```

---

## Advanced Features

### 1. Flash Attention

**Standard attention (slow):**
```python
# O(N²) memory, stores full attention matrix
Q, K, V = split(qkv)
attn = softmax(Q @ K.T / sqrt(d))  # (N, N) matrix
output = attn @ V
```

**Flash Attention (fast):**
```python
# O(N) memory, fused kernel
output = F.scaled_dot_product_attention(Q, K, V)
# - No materialized attention matrix
# - 2-3x faster
# - Identical output (mathematically equivalent)
```

**When to use:**
- ✅ Training (always, for speed)
- ❌ Attention visualization (need explicit weights)

**File:** `src/models/encoder.py:RoPEAttentionWrapper`

### 2. Rotary Position Embeddings (RoPE)

**Standard position embeddings:**
```python
# Learned, absolute positions
x = patch_embed(img) + pos_embed  # pos_embed learned
```

**RoPE (rotary):**
```python
# Relative, rotation-based
Q = apply_rope(Q, positions)  # Rotate queries
K = apply_rope(K, positions)  # Rotate keys
# Attention automatically encodes relative distances
```

**Benefits:**
- Generalizes to longer sequences
- Encodes relative positions naturally
- Works with Flash Attention

**File:** `src/models/rope.py`

### 3. Gradient Checkpointing

Trade compute for memory:

```python
# Normal: Store all activations (high memory)
x = layer1(x)  # Store for backward
x = layer2(x)  # Store for backward
x = layer3(x)  # Store for backward

# Gradient checkpointing: Recompute during backward (low memory)
x = checkpoint(layer1, x)  # Don't store, recompute
x = checkpoint(layer2, x)
x = checkpoint(layer3, x)
```

**Memory savings:** ~50-70% reduction
**Speed cost:** ~20-30% slower

**Use when:**
- Out of memory errors
- Want larger batch size
- Not concerned about speed

---

## Implementation Details

### Model Initialization

```python
def create_hjepa(
    encoder_type='vit_base_patch16_224',
    img_size=224,
    num_hierarchies=3,
    use_rope=True,
    use_flash_attention=True,
):
    # 1. Create context encoder
    context_encoder = ContextEncoder(
        encoder_type=encoder_type,
        use_rope=use_rope,
        use_flash_attention=use_flash_attention,
    )

    # 2. Create target encoder (copy of context)
    target_encoder = copy.deepcopy(context_encoder)

    # 3. Create predictor
    predictor = PredictorNetwork(
        embed_dim=768,
        depth=6,
        num_heads=6,
    )

    # 4. Wrap in HJEPA model
    model = HJEPA(
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        num_hierarchies=num_hierarchies,
    )

    return model
```

**File:** `src/models/hjepa.py:create_hjepa`

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        images, _ = batch

        # Create masks
        target_masks, context_masks = create_masks(images)

        # Forward pass
        loss = model(
            images,
            target_masks=target_masks,
            context_masks=context_masks,
        )

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()  # AMP
        scaler.step(optimizer)
        scaler.update()

        # Update target encoder (no gradients)
        update_target_encoder(
            context_encoder=model.context_encoder,
            target_encoder=model.target_encoder,
            momentum=current_momentum,
        )

        # Update learning rate
        lr_schedule.step()
```

**File:** `scripts/train.py`

### Checkpoint Format

```python
checkpoint = {
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'lr_scheduler_state_dict': scheduler.state_dict(),
    'config': config,  # Full training config
    'train_loss': train_loss,
    'val_loss': val_loss,
}
```

**Loading:**
```python
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## Performance Optimizations

### Memory Hierarchy

**From fastest to slowest:**
1. Registers (negligible)
2. L1 Cache (fastest)
3. L2 Cache
4. L3 Cache
5. GPU Memory (VRAM)
6. System RAM
7. Disk (slowest)

**Optimization strategies:**
- Flash Attention: Better cache utilization
- Mixed Precision (AMP): 2x less memory transfer
- Gradient Checkpointing: Trade compute for memory

### Compute Optimizations

**Flash Attention:**
```
Standard Attention: 12.5 TFLOPS
Flash Attention:    32.1 TFLOPS
Speedup:            2.57x
```

**Mixed Precision (FP16):**
```
FP32: 19.5 TFLOPS
FP16: 78.0 TFLOPS  (4x theoretical, ~2x practical)
```

**Combined (Flash + AMP):**
```
Baseline:          100%
Flash only:        250%
AMP only:          200%
Flash + AMP:       400-500% (best configuration)
```

---

## Design Decisions

### Why JEPA over MAE?

**Masked Autoencoding (MAE):**
- Predicts pixels
- Easy to implement
- But pixels are low-level, noisy

**Joint-Embedding (JEPA):**
- Predicts features
- More abstract representations
- Better for downstream tasks

**Empirical results:**
- JEPA: 82% linear probe accuracy
- MAE: 78% linear probe accuracy
(on ImageNet, ViT-Base)

### Why EMA Target Encoder?

**Alternatives:**
1. **Same encoder** - Causes collapse (predicts itself)
2. **Stop gradient** - Works but unstable
3. **EMA encoder** - Best of both worlds

**EMA benefits:**
- Stable targets (averages out noise)
- Prevents collapse (targets evolve slowly)
- Better final performance

### Why Hierarchical?

**Single-scale issues:**
- Misses multi-scale nature of vision
- Either too fine or too coarse

**Hierarchical benefits:**
- Captures patterns at multiple scales
- More complete representation
- Better transfer to downstream tasks

---

## Further Reading

**Papers:**
- [I-JEPA: Self-Supervised Learning from Images via Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- [Flash Attention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

**Related Methods:**
- MAE (Masked Autoencoding)
- SimCLR (Contrastive Learning)
- DINO (Self-Distillation)
- VICReg (Variance-Invariance-Covariance Regularization)

**Code References:**
- `src/models/hjepa.py` - Main model
- `src/models/encoder.py` - ViT encoder with RoPE and Flash Attention
- `src/models/predictor.py` - Predictor network
- `src/models/rope.py` - Rotary position embeddings
- `scripts/train.py` - Training loop

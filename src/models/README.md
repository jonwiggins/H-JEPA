# H-JEPA Model Implementation

This directory contains the core model architectures for Hierarchical Joint-Embedding Predictive Architecture (H-JEPA).

## Architecture Overview

H-JEPA consists of three main components:

1. **Context Encoder** - Encodes visible/context patches
2. **Target Encoder** - Encodes full image with EMA updates
3. **Predictor** - Predicts target representations for masked regions

## Files

### encoder.py

Implements the Vision Transformer-based encoders:

- **ContextEncoder**: Processes visible patches with optional masking
  - Built on timm's Vision Transformer models
  - Supports configurable sizes (small/base/large)
  - Handles patch embedding and positional encoding
  - Forward pass allows masking of specific patches

- **TargetEncoder**: Processes full images for target representations
  - Identical architecture to ContextEncoder
  - Updated via Exponential Moving Average (EMA) from context encoder
  - Implements cosine schedule for EMA momentum (tau: 0.996 → 1.0)
  - All parameters set to `requires_grad=False` (no direct gradient updates)

**Key Design Decisions**:
- Used timm library for robust ViT implementations
- EMA updates use cosine schedule for smooth convergence
- Both encoders share architecture but different update mechanisms
- CLS token included in encodings for potential downstream tasks

### predictor.py

Implements the lightweight predictor network:

- **Predictor**: Transformer-based prediction network
  - Fewer layers than encoders (configurable depth, default 6)
  - Learnable mask tokens for masked positions
  - Predicts target representations from context features
  - Uses standard transformer blocks with self-attention

- **PredictorBlock**: Individual transformer layer
  - Multi-head self-attention
  - MLP with GELU activation
  - Residual connections and layer normalization
  - Stochastic depth (DropPath) for regularization

**Key Design Decisions**:
- Lightweight architecture (fewer parameters than encoder)
- Learnable mask tokens initialized with small normal distribution
- Supports both indexed masking and full sequence with binary mask
- Positional embeddings added to mask tokens for spatial awareness

### hjepa.py

Implements the main H-JEPA model:

- **HJEPA**: Complete hierarchical architecture
  - Combines context encoder, target encoder, and predictor
  - Supports 2-4 hierarchical levels
  - Each level uses different pooling for multi-scale representations
  - Hierarchy-specific projection heads

**Hierarchical Levels**:
- Level 0: Finest granularity (no pooling)
- Level 1+: Progressively coarser (average pooling with kernel 2^level)
- Each level has dedicated projection and normalization

**Key Methods**:
- `forward()`: Main training forward pass with hierarchical predictions
- `extract_features()`: Extract features at specific hierarchy level
- `update_target_encoder()`: EMA update with step-based scheduling

**Key Design Decisions**:
- Hierarchical pooling uses powers of 2 for clean downsampling
- Separate projection heads per level allow different semantic spaces
- Factory functions (`create_hjepa`, `create_hjepa_from_config`) for easy instantiation
- Validation for hierarchy count (2-4 levels) to ensure reasonable complexity

## Usage

### Basic Usage

```python
from src.models import create_hjepa

# Create model
model = create_hjepa(
    encoder_type="vit_base_patch16_224",
    img_size=224,
    embed_dim=768,
    predictor_depth=6,
    num_hierarchies=3,
)

# Forward pass
images = torch.randn(2, 3, 224, 224)
mask = torch.zeros(2, 196)  # 196 = (224/16)^2 patches
mask[:, :98] = 1  # Mask 50% of patches

outputs = model(images, mask, return_all_levels=True)
predictions = outputs['predictions']  # List of 3 tensors (one per level)
targets = outputs['targets']  # List of 3 tensors (one per level)

# Update target encoder (during training)
momentum = model.update_target_encoder(current_step=100)
```

### Config-based Usage

```python
from src.models import create_hjepa_from_config
import yaml

with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = create_hjepa_from_config(config)
```

### Feature Extraction

```python
# Extract features at different hierarchy levels
with torch.no_grad():
    fine_features = model.extract_features(images, level=0)  # Finest
    coarse_features = model.extract_features(images, level=2)  # Coarser
```

## Model Components

### Parameter Counts (approximate)

For ViT-Base configuration:
- **Context Encoder**: ~86M parameters (trainable)
- **Target Encoder**: ~86M parameters (EMA, non-trainable)
- **Predictor**: ~20M parameters (trainable)
- **Hierarchy Projections**: ~2M parameters (trainable)
- **Total**: ~194M parameters (~108M trainable)

### Memory Requirements

- Model: ~800MB (fp32) or ~400MB (fp16)
- Activations depend on batch size and image resolution
- Recommended: 16GB+ GPU for batch size 128

## Implementation Details

### EMA Update

The target encoder is updated using a cosine schedule:

```
tau(t) = tau_base + (tau_end - tau_base) * (1 + cos(pi * t / T)) / 2
```

Where:
- `tau_base` = 0.996 (initial momentum)
- `tau_end` = 1.0 (final momentum)
- `T` = warmup steps (default: 30 epochs)

Update rule:
```
θ_target = tau * θ_target + (1 - tau) * θ_context
```

### Masking Strategy

- Masks are binary tensors [B, N] where N is number of patches
- 1 indicates masked position, 0 indicates visible
- Context encoder receives masked input (masked patches zeroed)
- Target encoder receives full image
- Predictor predicts representations for masked positions only

### Hierarchical Representations

Each hierarchy level represents different semantic granularity:
- **Level 0**: Patch-level representations (finest details)
- **Level 1**: 2x2 patch groups (local patterns)
- **Level 2**: 4x4 patch groups (medium-scale structures)
- **Level 3**: 8x8 patch groups (global context)

## Configuration

See `configs/default.yaml` for all configuration options:

```yaml
model:
  encoder_type: "vit_base_patch16_224"
  embed_dim: 768
  num_hierarchies: 3
  predictor:
    depth: 6
    num_heads: 12
    mlp_ratio: 4.0
  ema:
    momentum: 0.996
    momentum_end: 1.0
    momentum_warmup_epochs: 30
```

## Testing

Run the test script to verify installation:

```bash
python test_models.py
```

This will test:
- Model creation
- Forward pass
- Feature extraction
- EMA updates
- Config loading
- Parameter counts

## Dependencies

- PyTorch >= 2.0.0
- timm >= 0.9.0 (Vision Transformers)
- einops >= 0.7.0 (tensor operations)

## References

- JEPA: Joint-Embedding Predictive Architecture
- Vision Transformer (ViT): "An Image is Worth 16x16 Words"
- timm: PyTorch Image Models library

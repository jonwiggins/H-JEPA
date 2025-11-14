# H-JEPA Loss Functions - Implementation Summary

## Overview

This document summarizes the loss function implementations for the H-JEPA (Hierarchical Joint-Embedding Predictive Architecture) project. Three main loss modules have been implemented in `src/losses/`:

1. **HJEPALoss** (`hjepa_loss.py`) - Hierarchical prediction loss
2. **VICRegLoss** (`vicreg.py`) - Variance-Invariance-Covariance regularization
3. **CombinedLoss** (`combined.py`) - Integration of JEPA and VICReg losses

---

## 1. HJEPALoss - Hierarchical Prediction Loss

**File:** `/home/user/H-JEPA/src/losses/hjepa_loss.py`

### Mathematical Formulation

```
L_HJEPA = Σ_h w_h × L(pred_h, target_h)
```

Where:
- `h` ∈ {0, 1, ..., H-1} is the hierarchy level index
- `w_h` is the weight for hierarchy level h
- `L(·,·)` is the base loss function (MSE, SmoothL1, or Huber)
- `pred_h` is the predicted representation at level h (shape: [B, N, D])
- `target_h` is the target representation at level h (shape: [B, N, D])

### Base Loss Functions

1. **MSE (Mean Squared Error):**
   ```
   L_MSE(pred, target) = (1/N) Σ (pred - target)²
   ```

2. **Smooth L1 (Huber with β=1):**
   ```
   L_SmoothL1(x) = {
       0.5 × x²     if |x| < 1
       |x| - 0.5    otherwise
   }
   where x = pred - target
   ```

3. **Huber:**
   ```
   L_Huber(x, δ) = {
       0.5 × x²           if |x| ≤ δ
       δ × (|x| - 0.5δ)   otherwise
   }
   ```

### Key Features

- **Hierarchical Weighting:** Different weights for each hierarchy level
- **Normalization:** Optional L2 normalization of embeddings before loss computation
- **Masking Support:** Can compute loss only on selected patches
- **Flexible Reduction:** Supports mean, sum, or no reduction
- **Gradient Stopping:** Automatically detaches target representations

### Input/Output Specification

**Input:**
- `predictions`: List[Tensor] of shape [B, N, D] for each hierarchy level
- `targets`: List[Tensor] of shape [B, N, D] for each hierarchy level
- `masks`: Optional List[Tensor] of shape [B, N] (binary masks)

**Output:** Dictionary containing:
- `'loss'`: Total weighted hierarchical loss (scalar)
- `'loss_h{i}'`: Loss at hierarchy level i (scalar)
- `'loss_unweighted'`: Average loss across all levels (scalar)

### Usage Example

```python
from src.losses import HJEPALoss

loss_fn = HJEPALoss(
    loss_type='smoothl1',
    hierarchy_weights=[1.0, 0.5, 0.25],
    num_hierarchies=3,
    normalize_embeddings=True
)

# predictions and targets: List of 3 tensors [32, 196, 768]
loss_dict = loss_fn(predictions, targets)
total_loss = loss_dict['loss']
```

---

## 2. VICRegLoss - Variance-Invariance-Covariance Regularization

**File:** `/home/user/H-JEPA/src/losses/vicreg.py`

### Mathematical Formulation

```
L_VICReg = λ × L_inv + μ × L_var + ν × L_cov
```

#### 2.1 Invariance Loss (L_inv)

Encourages consistency between different views:

```
L_inv = MSE(Z_a, Z_b) = (1/N) Σ ||z_a^i - z_b^i||²
```

Where:
- `Z_a, Z_b` are representations from two views (shape: [N, D])
- `N` is the number of samples (batch × patches if flattened)
- `D` is the embedding dimension

#### 2.2 Variance Loss (L_var)

Maintains variance above threshold γ for each dimension:

```
L_var = (1/D) Σ_d max(0, γ - √(Var(Z_d) + ε))
```

Where:
- `Var(Z_d) = (1/N) Σ_i (z_i^d - μ_d)²` is variance of dimension d
- `γ` is the variance threshold (default: 1.0)
- `ε` is numerical stability constant (default: 1e-4)

This prevents dimension collapse by penalizing dimensions with variance below γ.

#### 2.3 Covariance Loss (L_cov)

Decorrelates different dimensions:

```
L_cov = (1/D) Σ_{i≠j} Cov(Z_i, Z_j)²
```

Where:
- Covariance matrix: `Cov(Z) = (1/(N-1)) × Z̃ᵀZ̃`
- `Z̃ = Z - mean(Z)` (centered features)
- Only off-diagonal elements are penalized

This prevents feature redundancy by encouraging orthogonal representations.

### Key Features

- **Three-Component Regularization:** Prevents collapse through complementary mechanisms
- **Configurable Weights:** Independent control over each loss component
- **Patch Flattening:** Handles patch-based representations (ViT-style)
- **Adaptive Variant:** `AdaptiveVICRegLoss` with learnable weights

### Input/Output Specification

**Input:**
- `z_a`: First view tensor [B, N, D] or [B, D]
- `z_b`: Optional second view tensor (if None, z_a is split)

**Output:** Dictionary containing:
- `'loss'`: Total weighted VICReg loss
- `'invariance_loss'`: Invariance component
- `'variance_loss'`: Average variance loss (both views)
- `'covariance_loss'`: Average covariance loss (both views)
- `'variance_loss_a'`, `'variance_loss_b'`: Per-view variance
- `'covariance_loss_a'`, `'covariance_loss_b'`: Per-view covariance

### Default Hyperparameters

Based on the VICReg paper:
- λ (invariance_weight) = 25.0
- μ (variance_weight) = 25.0
- ν (covariance_weight) = 1.0
- γ (variance_threshold) = 1.0

### Usage Example

```python
from src.losses import VICRegLoss

loss_fn = VICRegLoss(
    invariance_weight=25.0,
    variance_weight=25.0,
    covariance_weight=1.0,
    variance_threshold=1.0
)

# Two views of same data [32, 196, 768]
loss_dict = loss_fn(view_a, view_b)
```

---

## 3. CombinedLoss - H-JEPA + VICReg Integration

**File:** `/home/user/H-JEPA/src/losses/combined.py`

### Mathematical Formulation

```
L_total = L_JEPA + Σ_h α_h × L_VICReg(h)
```

Expanded:
```
L_total = Σ_h w_h × L(pred_h, target_h) + Σ_h α_h × [λ × L_inv(h) + μ × L_var(h) + ν × L_cov(h)]
```

Where:
- `L_JEPA`: Hierarchical prediction loss (from HJEPALoss)
- `α_h`: VICReg weight for hierarchy level h
- All other symbols as defined previously

### Key Features

- **Dual Objective:** Combines prediction accuracy with collapse prevention
- **Flexible Architecture:** Can apply VICReg per-level or only at finest level
- **Configurable Weighting:** Independent control of JEPA and VICReg contributions
- **Hierarchical Variant:** `HierarchicalCombinedLoss` with per-level VICReg configs
- **Factory Function:** `create_loss_from_config()` for easy initialization

### Loss Components

1. **JEPA Component:** Ensures accurate prediction of target representations
2. **VICReg Component:** Prevents representation collapse through:
   - Invariance: Consistency across views
   - Variance: Maintains feature diversity
   - Covariance: Reduces redundancy

### Design Choices

#### Option 1: Per-Level VICReg (Default)
```python
apply_vicreg_per_level=True
```
- Applies VICReg at each hierarchy level independently
- Better multi-scale regularization
- More computational cost

#### Option 2: Single-Level VICReg
```python
apply_vicreg_per_level=False
```
- Applies VICReg only at finest hierarchy level
- Lower computational cost
- Simpler optimization landscape

### Input/Output Specification

**Input:**
- `predictions`: List[Tensor] of shape [B, N, D]
- `targets`: List[Tensor] of shape [B, N, D]
- `masks`: Optional List[Tensor] of shape [B, N]

**Output:** Dictionary containing:
- `'loss'`: Total combined loss
- `'jepa_loss'`: Total JEPA component
- `'vicreg_loss'`: Total VICReg component
- `'loss_h{i}'`: JEPA loss at level i
- `'vicreg_h{i}'`: VICReg loss at level i
- Plus all sub-components from both losses

### Usage Examples

#### Basic Combined Loss

```python
from src.losses import CombinedLoss

loss_fn = CombinedLoss(
    jepa_loss_type='smoothl1',
    jepa_hierarchy_weights=[1.0, 0.5, 0.25],
    num_hierarchies=3,
    vicreg_weight=0.1,
    apply_vicreg_per_level=True
)

loss_dict = loss_fn(predictions, targets)
```

#### Hierarchical Combined Loss (Advanced)

```python
from src.losses import HierarchicalCombinedLoss

# Different VICReg configs per level
vicreg_configs = [
    {'invariance_weight': 25.0, 'variance_weight': 25.0},
    {'invariance_weight': 15.0, 'variance_weight': 15.0},
    {'invariance_weight': 10.0, 'variance_weight': 10.0},
]

loss_fn = HierarchicalCombinedLoss(
    jepa_hierarchy_weights=[1.0, 0.5, 0.25],
    vicreg_weight=[0.1, 0.05, 0.025],
    vicreg_configs=vicreg_configs,
    num_hierarchies=3
)
```

#### From Configuration

```python
from src.losses import create_loss_from_config

config = {
    'type': 'combined',
    'jepa_loss_type': 'smoothl1',
    'hierarchy_weights': [1.0, 0.5, 0.25],
    'num_hierarchies': 3,
    'vicreg_weight': 0.1,
}

loss_fn = create_loss_from_config(config)
```

---

## Implementation Quality Features

### 1. Efficient PyTorch Operations

All implementations use:
- Vectorized operations (no explicit loops)
- In-place operations where beneficial
- Efficient tensor broadcasting
- GPU-friendly operations

Example from variance loss:
```python
# Vectorized variance computation across all dimensions
std = torch.sqrt(z.var(dim=0) + self.eps)  # [D]
variance_loss = torch.mean(F.relu(self.variance_threshold - std))
```

### 2. Numerical Stability

- Small epsilon values for division and square roots
- Gradient clipping recommendations
- Normalized covariance computation
- Stable variance calculation

### 3. Input Validation

Comprehensive assertions for:
- Shape compatibility
- Correct number of hierarchy levels
- Valid hyperparameter ranges
- Mask dimension matching

Example:
```python
assert pred.shape == target.shape, (
    f"Prediction and target shapes must match at level {i}. "
    f"Got pred: {pred.shape}, target: {target.shape}"
)
```

### 4. Loss Monitoring

All loss functions return detailed dictionaries:
- Total loss
- Component-wise breakdown
- Per-hierarchy-level losses
- Sub-component losses (for VICReg)

This enables:
- Detailed TensorBoard/WandB logging
- Loss balancing analysis
- Training diagnostics

### 5. Flexible Configuration

- Multiple loss types (MSE, SmoothL1, Huber)
- Configurable weights per hierarchy
- Optional normalization
- Multiple reduction modes
- Factory pattern for config-based creation

---

## File Statistics

```
Total lines of code: 1,141
├── hjepa_loss.py:    285 lines
├── vicreg.py:        349 lines
├── combined.py:      470 lines
└── __init__.py:       37 lines
```

---

## Integration with Training Pipeline

### Typical Training Loop

```python
# 1. Initialize loss
loss_fn = CombinedLoss(
    jepa_loss_type='smoothl1',
    jepa_hierarchy_weights=[1.0, 0.5, 0.25],
    num_hierarchies=3,
    vicreg_weight=0.1,
)

# 2. Training step
for batch in dataloader:
    # Forward pass through model
    predictions, targets = model(batch)

    # Compute loss
    loss_dict = loss_fn(predictions, targets)
    total_loss = loss_dict['loss']

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Logging
    if step % log_frequency == 0:
        logger.log({
            'loss/total': loss_dict['loss'].item(),
            'loss/jepa': loss_dict['jepa_loss'].item(),
            'loss/vicreg': loss_dict['vicreg_loss'].item(),
        })
```

### Configuration File Integration

The losses integrate seamlessly with YAML configs:

```yaml
# config.yaml
loss:
  type: "combined"
  jepa_loss_type: "smoothl1"
  hierarchy_weights: [1.0, 0.5, 0.25]
  num_hierarchies: 3
  normalize_embeddings: true
  vicreg_weight: 0.1
  vicreg_invariance_weight: 25.0
  vicreg_variance_weight: 25.0
  vicreg_covariance_weight: 1.0
```

Usage:
```python
import yaml
from src.losses import create_loss_from_config

with open('config.yaml') as f:
    config = yaml.safe_load(f)

loss_fn = create_loss_from_config(config['loss'])
```

---

## Testing and Validation

### Syntax Validation

All files have been validated for Python syntax:
```bash
python -m py_compile src/losses/*.py
# All files compiled successfully!
```

### Recommended Tests

Unit tests should cover:

1. **Shape Handling:**
   - Various batch sizes
   - Different patch counts
   - Multiple embedding dimensions

2. **Numerical Properties:**
   - Loss is always non-negative
   - Gradient flow verification
   - Numerical stability with extreme values

3. **Component Interaction:**
   - JEPA + VICReg weight balance
   - Hierarchy level weighting
   - Mask application correctness

4. **Edge Cases:**
   - Single sample batches
   - Zero gradients
   - Identical predictions/targets

---

## Hyperparameter Recommendations

### For Small-Scale Experiments

```python
loss_fn = CombinedLoss(
    jepa_loss_type='smoothl1',
    jepa_hierarchy_weights=[1.0, 0.5, 0.25],
    num_hierarchies=3,
    vicreg_weight=0.05,  # Lower for small datasets
    vicreg_invariance_weight=10.0,
    vicreg_variance_weight=10.0,
    vicreg_covariance_weight=0.5,
)
```

### For ImageNet-Scale Training

```python
loss_fn = CombinedLoss(
    jepa_loss_type='smoothl1',
    jepa_hierarchy_weights=[1.0, 0.5, 0.25],
    num_hierarchies=3,
    vicreg_weight=0.1,  # Standard weight
    vicreg_invariance_weight=25.0,
    vicreg_variance_weight=25.0,
    vicreg_covariance_weight=1.0,
)
```

### For Fine-Tuning / Transfer Learning

```python
loss_fn = HJEPALoss(
    loss_type='mse',  # Simpler loss for fine-tuning
    hierarchy_weights=[1.0, 0.3, 0.1],  # Focus on finest level
    num_hierarchies=3,
    normalize_embeddings=True,
)
```

---

## References

1. **I-JEPA:** Joint-Embedding Predictive Architecture
   - Paper: https://arxiv.org/abs/2301.08243
   - Introduces self-supervised learning via masked prediction in embedding space

2. **VICReg:** Variance-Invariance-Covariance Regularization
   - Paper: https://arxiv.org/abs/2105.04906
   - Proposes three-component regularization to prevent collapse

3. **Vision Transformers (ViT)**
   - Paper: https://arxiv.org/abs/2010.11929
   - Foundation for patch-based image representations

---

## Future Extensions

Potential improvements and extensions:

1. **Dynamic Weight Scheduling:**
   - Curriculum learning for hierarchy weights
   - Cosine annealing for VICReg weights

2. **Memory-Efficient Variants:**
   - Gradient checkpointing
   - Chunked covariance computation

3. **Additional Regularizers:**
   - Barlow Twins-style redundancy reduction
   - SimCLR-style contrastive loss

4. **Multi-Modal Extensions:**
   - Cross-modal VICReg
   - Text-image alignment losses

---

## Quick Reference

### Import Statements
```python
from src.losses import (
    HJEPALoss,
    VICRegLoss,
    AdaptiveVICRegLoss,
    CombinedLoss,
    HierarchicalCombinedLoss,
    create_loss_from_config,
)
```

### File Locations
- Main implementations: `/home/user/H-JEPA/src/losses/`
- Usage examples: `/home/user/H-JEPA/examples/loss_usage_examples.py`
- Unit tests: `/home/user/H-JEPA/tests/test_losses.py`

### Key Classes
- `HJEPALoss`: Hierarchical prediction loss
- `VICRegLoss`: VICReg regularization
- `CombinedLoss`: JEPA + VICReg
- `HierarchicalCombinedLoss`: Advanced combined loss

### Factory Function
```python
loss_fn = create_loss_from_config(config_dict)
```

---

*Document generated: 2025-11-14*
*H-JEPA Implementation v1.0*

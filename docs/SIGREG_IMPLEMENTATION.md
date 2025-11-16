# SIGReg Implementation Guide

## Overview

This document describes the implementation of **SIGReg (Sketched Isotropic Gaussian Regularization)** from the LeJEPA paper, which provides improved training stability over standard VICReg for self-supervised learning.

## What is SIGReg?

SIGReg is a theoretically grounded regularization technique that prevents representation collapse by constraining embeddings to follow an **isotropic Gaussian distribution**. Unlike VICReg which separately handles variance, invariance, and covariance with multiple hyperparameters, SIGReg uses a unified approach based on statistical testing.

### Key Innovation

The core insight from the LeJEPA paper is that **isotropic Gaussian distributions minimize both bias and variance** in downstream prediction tasks. SIGReg enforces this optimal distribution using:

1. **Random Slicing**: Projects embeddings onto random 1D directions
2. **Statistical Testing**: Uses Epps-Pulley test to measure distance from standard Gaussian
3. **Efficient Computation**: O(K) complexity vs O(K²) for VICReg's covariance matrix

## Mathematical Formulation

### SIGReg Loss

For M random unit vectors (slices) sampled from the sphere S^(K-1):

```
L_SIGReg = (1/M) * Σ_{i=1}^M T({a_i^T z_n}_{n=1}^N)
```

where:
- `a_i`: Random unit vector (slice direction)
- `z_n`: Embedding vectors
- `T`: Univariate statistical test (Epps-Pulley)
- `M`: Number of slices (typically 1024)

### Epps-Pulley Test

The Epps-Pulley test compares a distribution to a standard Gaussian using characteristic functions:

```
EP(y) = (1/N²) Σ_{i,j} ψ(y_i - y_j) - 2/(N*K) Σ_{i,k} ψ(y_i - g_k) + (1/K²) Σ_{k,l} ψ(g_k - g_l)
```

where:
- `y`: 1D projected samples (a^T @ z)
- `g`: Reference Gaussian samples (quantiles)
- `ψ`: Smooth kernel function: ψ(x,y) = exp(-0.5 * (x-y)²)
- `N`: Number of samples
- `K`: Number of reference points (typically 17)

The test statistic is **minimized** when the distribution of `y` matches a standard Gaussian.

### Total Loss

```
L_total = λ_inv * L_invariance + λ_sig * L_SIGReg
```

where:
- `L_invariance`: MSE between different views (encourages consistency)
- `L_SIGReg`: SIGReg regularization (prevents collapse)
- `λ_inv`, `λ_sig`: Loss weights (default: 25.0 each)

## Comparison: SIGReg vs VICReg

| Aspect | VICReg | SIGReg |
|--------|--------|--------|
| **Complexity** | O(K²) for covariance | O(K) for slicing |
| **Hyperparameters** | 3 weights (var, inv, cov) | 1 weight (num_slices) |
| **Theoretical Foundation** | Heuristic regularization | Optimal Gaussian theory |
| **Covariance Handling** | Explicit off-diagonal penalty | Implicit via isotropy test |
| **Variance Handling** | Hinge loss threshold | Statistical test |
| **Sign Consistency** | Not explicitly addressed | Built into statistical test |
| **Scalability** | Poor (quadratic) | Excellent (linear) |
| **Training Stability** | Good | **Superior** |

### Key Advantages of SIGReg

1. **Single Hyperparameter**: Only `num_slices` needs tuning (typically 1024 works well)
2. **Better Scaling**: Linear complexity allows efficient training of large models
3. **Theoretical Grounding**: Based on Cramér-Wold theorem and optimal distribution theory
4. **Improved Stability**: Sign consistency and better variance handling prevent training instabilities
5. **No Covariance Matrix**: Avoids memory-intensive covariance computation

## Implementation Details

### Core Components

#### 1. EppsPulleyTest

```python
from src.losses import EppsPulleyTest

# Initialize test
test = EppsPulleyTest(num_points=17)

# Test 1D samples
x = torch.randn(1000)  # Should be close to 0 for standard Gaussian
test_statistic = test(x)
```

**Parameters:**
- `num_points`: Number of reference Gaussian quantiles (default: 17)
  - Higher values = more accurate but slower
  - 17 provides good balance (from LeJEPA paper)
- `eps`: Numerical stability constant (default: 1e-6)

#### 2. SIGRegLoss

```python
from src.losses import SIGRegLoss

# Initialize loss
loss_fn = SIGRegLoss(
    num_slices=1024,              # Number of random projections
    num_test_points=17,            # Points for Epps-Pulley test
    invariance_weight=25.0,        # Weight for invariance term
    sigreg_weight=25.0,            # Weight for SIGReg term
    flatten_patches=True,          # Flatten [B,N,D] -> [B*N,D]
    fixed_slices=False,            # Use same slices each time
)

# Compute loss
z_a = torch.randn(32, 196, 768)  # View 1
z_b = torch.randn(32, 196, 768)  # View 2
loss_dict = loss_fn(z_a, z_b)

print(loss_dict['loss'])           # Total loss
print(loss_dict['invariance_loss'])  # MSE term
print(loss_dict['sigreg_loss'])      # SIGReg term
```

**Parameters:**
- `num_slices`: Number of random 1D projections (default: 1024)
  - Higher = more thorough testing but slower
  - LeJEPA uses 1024 as sweet spot
- `num_test_points`: Reference points for EP test (default: 17)
- `invariance_weight`: Weight for invariance (MSE) term (default: 25.0)
- `sigreg_weight`: Weight for SIGReg regularization (default: 25.0)
- `eps`: Numerical stability (default: 1e-6)
- `flatten_patches`: Whether to flatten patch dimension (default: True)
- `fixed_slices`: Use fixed random slices for reproducibility (default: False)

#### 3. HybridVICRegSIGRegLoss

For gradual transition or ablation studies:

```python
from src.losses import HybridVICRegSIGRegLoss

# Start with VICReg, gradually add SIGReg
loss_fn = HybridVICRegSIGRegLoss(
    vicreg_weight=1.0,      # Start high
    sigreg_weight=0.0,      # Start low, increase over training
    invariance_weight=25.0,
    num_slices=1024,
)

# During training, can adjust weights:
# loss_fn.vicreg_weight = 0.5
# loss_fn.sigreg_weight = 0.5
```

### Usage in Training

#### Configuration File

Add to your YAML config:

```yaml
loss:
  type: 'sigreg'  # Use SIGReg instead of VICReg

  # SIGReg-specific parameters
  sigreg_num_slices: 1024          # Number of random projections
  sigreg_num_test_points: 17       # Reference Gaussian points
  sigreg_invariance_weight: 25.0   # MSE weight
  sigreg_weight: 25.0              # SIGReg regularization weight
  sigreg_fixed_slices: false       # Randomize slices each iteration

  # General parameters
  flatten_patches: true
  eps: 1.0e-6
```

#### From Config

```python
from src.losses import create_loss_from_config

config = {
    'loss': {
        'type': 'sigreg',
        'sigreg_num_slices': 1024,
        'sigreg_weight': 25.0,
    }
}

loss_fn = create_loss_from_config(config)
```

#### Direct Instantiation

```python
from src.losses import SIGRegLoss

loss_fn = SIGRegLoss(
    num_slices=1024,
    invariance_weight=25.0,
    sigreg_weight=25.0,
)
```

## Hyperparameter Tuning

### Recommended Settings

Based on the LeJEPA paper:

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| `num_slices` | 1024 | Good balance of accuracy/speed |
| `num_test_points` | 17 | Sufficient for Gaussian testing |
| `invariance_weight` | 25.0 | Same as VICReg default |
| `sigreg_weight` | 25.0 | Equal to invariance weight |
| `fixed_slices` | False | Random slices reduce variance |

### Tuning Guidelines

1. **Start with defaults**: The default values work well for most cases

2. **Adjust num_slices**:
   - Fewer slices (256-512): Faster but less accurate
   - More slices (2048-4096): More thorough but slower
   - Trade-off: Computational cost vs statistical accuracy

3. **Adjust weights**:
   - If representations collapse: Increase `sigreg_weight`
   - If views diverge too much: Increase `invariance_weight`
   - Keep ratio around 1:1 for balanced training

4. **Memory constraints**:
   - If OOM: Reduce `num_slices` or use `fixed_slices=True`
   - Fixed slices use less memory but may increase variance

## Performance Characteristics

### Computational Cost

**Per forward pass:**
- Random slice generation: O(M × K)
- Projection: O(N × M × K)
- EP test per slice: O(N² + N×K + K²) ≈ O(N²) for each slice
- Total: O(M × N²)

**Comparison:**
- VICReg covariance: O(N × K²)
- SIGReg: O(M × N²)

**When SIGReg is faster:**
- Large embedding dimension (K > 1024)
- Moderate batch size (N < 1024)
- Patch-based models (N = B × num_patches)

### Memory Usage

- **VICReg**: Stores K × K covariance matrix
- **SIGReg**: Stores M × K random slices + intermediate projections
- **Advantage**: Linear in K vs quadratic for VICReg

### Training Stability

From LeJEPA experiments:
- **Stable training** up to 1.8B parameters (ViT-Giant)
- **No heuristics needed**: No stop-gradient, EMA teacher, etc.
- **Robust to hyperparameters**: Wide range of settings work well
- **Better convergence**: Smoother loss curves, fewer spikes

## Advanced Usage

### Combining with H-JEPA

SIGReg can be used as drop-in replacement for VICReg in combined losses:

```python
# Currently, CombinedLoss uses VICReg
# To use SIGReg, instantiate directly:

from src.losses import HJEPALoss, SIGRegLoss

jepa_loss = HJEPALoss(
    loss_type='smoothl1',
    num_hierarchies=3,
    hierarchy_weights=[1.0, 0.5, 0.25],
)

sigreg_loss = SIGRegLoss(
    num_slices=1024,
    sigreg_weight=25.0,
)

# In training loop:
def compute_loss(predictions, targets):
    jepa_dict = jepa_loss(predictions, targets)

    # Apply SIGReg to last level
    last_pred = predictions[-1]
    last_target = targets[-1]
    sigreg_dict = sigreg_loss(last_pred, last_target)

    total_loss = jepa_dict['loss'] + 0.1 * sigreg_dict['loss']
    return total_loss
```

### Monitoring Training

Key metrics to track:

```python
loss_dict = loss_fn(z_a, z_b)

# Log these values
print(f"Total Loss: {loss_dict['loss'].item():.4f}")
print(f"Invariance: {loss_dict['invariance_loss'].item():.4f}")
print(f"SIGReg:     {loss_dict['sigreg_loss'].item():.4f}")
print(f"SIGReg A:   {loss_dict['sigreg_loss_a'].item():.4f}")
print(f"SIGReg B:   {loss_dict['sigreg_loss_b'].item():.4f}")
```

**What to watch:**
- **Invariance** should decrease over time (views becoming similar)
- **SIGReg** should stay relatively stable and low
- **Ratio** of invariance to SIGReg should be balanced

### Fixed Slices for Reproducibility

For deterministic training or debugging:

```python
loss_fn = SIGRegLoss(
    num_slices=1024,
    fixed_slices=True,  # Use same slices every iteration
)

# Slices are generated once and cached
# Ensures exact reproducibility with same random seed
```

## Troubleshooting

### Issue: SIGReg loss increasing

**Cause**: Embeddings diverging from Gaussian distribution

**Solutions:**
1. Increase `sigreg_weight`
2. Check if embeddings are properly normalized
3. Verify data augmentation isn't too aggressive
4. Reduce learning rate

### Issue: Out of memory

**Cause**: Too many slices or large batch size

**Solutions:**
1. Reduce `num_slices` (try 512 or 256)
2. Use `fixed_slices=True` to cache slices
3. Reduce batch size
4. Use gradient checkpointing

### Issue: Training unstable

**Cause**: Imbalanced loss terms

**Solutions:**
1. Adjust weight ratio (try 1:1 for invariance:sigreg)
2. Use gradient clipping
3. Reduce learning rate during warmup
4. Check for NaN values in projections

### Issue: Slower than VICReg

**Cause**: Large batch size, small embedding dimension

**Solutions:**
1. Reduce `num_slices`
2. Use `fixed_slices=True`
3. Consider VICReg for small K (< 512)
4. Enable mixed precision training

## References

1. **LeJEPA Paper**: [Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544)
   - Introduces SIGReg and theoretical foundation
   - Demonstrates scaling to 1.8B parameters

2. **VICReg Paper**: [Variance-Invariance-Covariance Regularization](https://arxiv.org/abs/2105.04906)
   - Original regularization approach
   - SIGReg recovers VICReg in the limit

3. **Epps-Pulley Test**: Classical statistical test for comparing distributions
   - Smooth, differentiable characteristic function test
   - Well-suited for gradient-based optimization

## Example Configurations

### Small Model (ViT-Small)

```yaml
loss:
  type: 'sigreg'
  sigreg_num_slices: 512        # Fewer slices for speed
  sigreg_weight: 25.0
  sigreg_invariance_weight: 25.0
```

### Large Model (ViT-Base/Large)

```yaml
loss:
  type: 'sigreg'
  sigreg_num_slices: 1024       # Standard setting
  sigreg_weight: 25.0
  sigreg_invariance_weight: 25.0
```

### Giant Model (ViT-Huge/Giant)

```yaml
loss:
  type: 'sigreg'
  sigreg_num_slices: 2048       # More thorough for large models
  sigreg_weight: 30.0           # Slightly higher regularization
  sigreg_invariance_weight: 25.0
  fixed_slices: true            # Save memory
```

### Hybrid VICReg → SIGReg Transition

```yaml
# Early training (epochs 0-100)
loss:
  type: 'hybrid_vicreg_sigreg'
  vicreg_weight: 1.0
  sigreg_weight: 0.0

# Mid training (epochs 100-200)
# vicreg_weight: 0.5
# sigreg_weight: 0.5

# Late training (epochs 200+)
# vicreg_weight: 0.0
# sigreg_weight: 1.0
```

## Citation

If you use SIGReg in your research, please cite:

```bibtex
@article{lejepa2024,
  title={LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author={Authors from Meta AI and collaborators},
  journal={arXiv preprint arXiv:2511.08544},
  year={2024}
}
```

## Implementation Notes

This implementation in H-JEPA includes:

- ✅ Complete SIGReg loss with Epps-Pulley test
- ✅ Efficient random slicing on GPU
- ✅ Hybrid VICReg/SIGReg for ablation studies
- ✅ Integration with loss factory
- ✅ Support for patch-based Vision Transformers
- ✅ Comprehensive documentation and examples
- ✅ Configurable hyperparameters
- ✅ Memory-efficient fixed slices option

**File Location**: `/Users/jon/repos/H-JEPA/src/losses/sigreg.py`

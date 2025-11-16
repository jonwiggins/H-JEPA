# SIGReg Quick Start Guide

**Get started with SIGReg in 5 minutes**

---

## What is SIGReg?

SIGReg (Sketched Isotropic Gaussian Regularization) is an **improved alternative to VICReg** from the LeJEPA paper that provides:

- ✅ **Better training stability**
- ✅ **O(K) complexity vs O(K²) for VICReg**
- ✅ **Single hyperparameter** vs 3 weights
- ✅ **Scales to 1.8B+ parameters**

---

## Quick Start

### 1. Basic Usage (30 seconds)

```python
from src.losses import SIGRegLoss

# Create loss function
loss_fn = SIGRegLoss(
    num_slices=1024,          # Number of random projections
    sigreg_weight=25.0,       # Regularization strength
)

# Compute loss on two views
import torch
view_a = torch.randn(32, 196, 768)  # [Batch, Patches, Dim]
view_b = torch.randn(32, 196, 768)

loss_dict = loss_fn(view_a, view_b)
loss = loss_dict['loss']
loss.backward()
```

### 2. Configuration File (1 minute)

```yaml
# config.yaml
loss:
  type: 'sigreg'
  sigreg_num_slices: 1024
  sigreg_weight: 25.0
```

```python
from src.losses import create_loss_from_config
import yaml

config = yaml.safe_load(open('config.yaml'))
loss_fn = create_loss_from_config(config)
```

### 3. Training Integration (2 minutes)

```python
from src.losses import SIGRegLoss
import torch.optim as optim

# Setup
model = MyEncoder()
loss_fn = SIGRegLoss(num_slices=1024)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    # Create two augmented views
    view_a, view_b = augment(batch)

    # Encode
    z_a = model(view_a)
    z_b = model(view_b)

    # Loss
    loss_dict = loss_fn(z_a, z_b)

    # Optimize
    optimizer.zero_grad()
    loss_dict['loss'].backward()
    optimizer.step()

    # Monitor
    print(f"Loss: {loss_dict['loss'].item():.4f}")
```

---

## Key Parameters

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `num_slices` | 1024 | Random projections | 512 (faster), 2048 (more accurate) |
| `sigreg_weight` | 25.0 | Regularization | Increase if collapse, decrease if constrained |
| `invariance_weight` | 25.0 | MSE weight | Keep 1:1 ratio with sigreg_weight |
| `fixed_slices` | False | Cache slices | True for debugging/reproducibility |

---

## Model-Specific Configs

### Small Model (ViT-Tiny)
```python
loss_fn = SIGRegLoss(num_slices=512, sigreg_weight=25.0)
```

### Standard Model (ViT-Small/Base)
```python
loss_fn = SIGRegLoss(num_slices=1024, sigreg_weight=25.0)
```

### Large Model (ViT-Large/Huge)
```python
loss_fn = SIGRegLoss(num_slices=2048, sigreg_weight=30.0, fixed_slices=True)
```

---

## Monitoring Training

```python
loss_dict = loss_fn(z_a, z_b)

# Log these metrics
print(f"Total:      {loss_dict['loss'].item():.4f}")
print(f"Invariance: {loss_dict['invariance_loss'].item():.4f}")
print(f"SIGReg:     {loss_dict['sigreg_loss'].item():.4f}")
```

**What to watch:**
- **Invariance**: Should decrease (views aligning)
- **SIGReg**: Should stay stable and low (embeddings Gaussian)
- **Ratio**: Keep around 1:1 for balanced training

---

## Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce `num_slices` to 512 or 256

### Issue: SIGReg Loss Increasing
**Solution:** Increase `sigreg_weight` to 30.0 or 35.0

### Issue: Training Unstable
**Solution:** Keep inv:sig ratio at 1:1, use gradient clipping

### Issue: Slower than VICReg
**Solution:** Reduce `num_slices` or use `fixed_slices=True`

---

## Examples

### Run Examples
```bash
# Comprehensive examples
python examples/sigreg_usage_example.py

# Compare with VICReg
python examples/loss_usage_examples.py
```

### Example Files
- `examples/sigreg_usage_example.py` - 10 different use cases
- `configs/sigreg_example.yaml` - Production configuration
- `docs/SIGREG_IMPLEMENTATION.md` - Full documentation

---

## SIGReg vs VICReg

| Aspect | VICReg | SIGReg |
|--------|--------|--------|
| Complexity | O(K²) | **O(K)** ✅ |
| Memory | High | **Low** ✅ |
| Hyperparameters | 3 weights | **1 weight** ✅ |
| Stability | Good | **Superior** ✅ |
| Scalability | Poor (>1B) | **Excellent** ✅ |

---

## Next Steps

1. **Read Full Documentation**: `docs/SIGREG_IMPLEMENTATION.md`
2. **Try Examples**: `examples/sigreg_usage_example.py`
3. **Run Tests**: `pytest tests/test_sigreg.py -v`
4. **Read Paper**: [LeJEPA](https://arxiv.org/abs/2511.08544)

---

## Citation

If you use SIGReg, please cite:

```bibtex
@article{lejepa2024,
  title={LeJEPA: Provable and Scalable Self-Supervised Learning},
  journal={arXiv preprint arXiv:2511.08544},
  year={2024}
}
```

---

**Happy Training! 🚀**

# ImageNet-100 Quick Start Guide

Get started with ImageNet-100 training in 5 minutes.

## Prerequisites

- ImageNet dataset downloaded and extracted
- PyTorch and torchvision installed
- H-JEPA repository cloned

## Quick Start (3 Steps)

### Step 1: Verify Dataset

```bash
# Verify ImageNet-100 dataset structure
python examples/imagenet100_example.py --data-path ./data/imagenet --verify-only
```

Expected output:
```
✓ Dataset verification passed!
```

### Step 2: Test Dataset Loading

```bash
# Run all examples (verify everything works)
python examples/imagenet100_example.py --data-path ./data/imagenet
```

### Step 3: Start Training

```bash
# Option A: Single dataset (ImageNet-100 only)
python scripts/train.py --config configs/m1_max_imagenet100_100epoch.yaml

# Option B: Multi-dataset foundation model
python scripts/train.py --config configs/imagenet100_multi_dataset.yaml

# Option C: Quick validation (1 epoch test)
python scripts/train.py --config configs/m1_max_imagenet100_100epoch.yaml --epochs 1
```

## Common Configurations

### Configuration 1: ImageNet-100 Only (Fastest)

**File**: `configs/m1_max_imagenet100_100epoch.yaml`

```yaml
data:
  dataset: "imagenet100"
  data_path: "./data"
  batch_size: 32
training:
  epochs: 100
```

**Expected**:
- Training time: ~12 hours (M1 Max)
- Linear probe: 60-70%
- Memory: ~8-10GB

### Configuration 2: Multi-Dataset Foundation Model (Best Performance)

**File**: `configs/imagenet100_multi_dataset.yaml`

```yaml
data:
  use_multi_dataset: true
  datasets:
    - name: imagenet100
      weight: 0.60
    - name: stl10
      weight: 0.25
    - name: cifar100
      weight: 0.15
training:
  epochs: 100
```

**Expected**:
- Training time: ~18 hours (M1 Max)
- Linear probe: 65-75%
- Memory: ~8-10GB

### Configuration 3: Custom Configuration

Create your own config:

```yaml
# my_config.yaml
experiment:
  name: "my_imagenet100_experiment"

data:
  dataset: "imagenet100"
  data_path: "./data/imagenet"
  image_size: 224
  batch_size: 32  # Adjust based on GPU memory

training:
  epochs: 100  # 100 for quick, 300 for best results
  lr: 0.0001
  use_amp: true  # Mixed precision for speed

device: "mps"  # or "cuda" or "cpu"
```

Run with:
```bash
python scripts/train.py --config my_config.yaml
```

## Memory and Batch Size Guide

| GPU Memory | Batch Size | Training Time (100 epochs) |
|------------|------------|----------------------------|
| 8GB | 16 | ~15 hours |
| 16GB | 32 | ~12 hours |
| 24GB+ | 64 | ~10 hours |

## Python API Usage

### Simple Training

```python
from src.data import build_dataset, build_dataloader

# Build dataset
train_dataset = build_dataset(
    dataset_name='imagenet100',
    data_path='./data/imagenet',
    split='train',
    image_size=224,
)

# Build dataloader
train_loader = build_dataloader(
    train_dataset,
    batch_size=32,
    num_workers=6,
)

# Training loop
for images, labels in train_loader:
    # Your training code here
    pass
```

### Multi-Dataset Training

```python
from src.data import build_multi_dataset

# Configure datasets
configs = [
    {'name': 'imagenet100', 'weight': 0.6},
    {'name': 'stl10', 'weight': 0.3},
    {'name': 'cifar100', 'weight': 0.1},
]

# Build multi-dataset
train_dataset = build_multi_dataset(
    dataset_configs=configs,
    data_path='./data',
    split='train',
    sampling_strategy='weighted',
    image_size=224,
)
```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir results/checkpoints/logs

# Open browser to: http://localhost:6006
```

### Key Metrics to Watch

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease without overfitting
3. **Linear Probe Accuracy**: Main evaluation metric
4. **Dataset Distribution**: Should match configured weights (multi-dataset only)

## Expected Results

### After 100 Epochs

| Configuration | Linear Probe | k-NN | Time (M1 Max) |
|---------------|--------------|------|---------------|
| ImageNet-100 only | 60-70% | 55-65% | ~12 hours |
| Multi-dataset | 65-75% | 58-68% | ~18 hours |

### Performance Improvement

- **vs CIFAR-10**: +10-15% linear probe accuracy
- **vs CIFAR-100**: +15-20% linear probe accuracy
- **Higher resolution**: 224x224 vs 32x32
- **Better features**: More transferable to downstream tasks

## Troubleshooting

### Problem: Dataset not found

```bash
# Check directory structure
ls data/imagenet/
# Should show: train/ and val/

ls data/imagenet/train/ | head
# Should show: n01440764 n01443537 ... (synset IDs)
```

**Solution**: Download ImageNet from https://image-net.org/download.php

### Problem: Out of memory

```yaml
# Reduce batch size in config
data:
  batch_size: 16  # or 8

# Enable gradient accumulation
training:
  gradient_accumulation_steps: 4  # Effective batch = 16 * 4 = 64
```

### Problem: Training too slow

```yaml
# Optimize settings
data:
  num_workers: 6  # Parallel data loading
  batch_size: 32  # Larger batches if memory allows

training:
  use_amp: true  # Mixed precision (~2x speedup)
```

### Problem: Multi-dataset not working

```bash
# Install missing datasets
python scripts/download_data.py --dataset stl10
python scripts/download_data.py --dataset cifar100

# Or skip multi-dataset
python scripts/train.py --config configs/m1_max_imagenet100_100epoch.yaml
```

## Next Steps

1. **Quick Validation**: Run 1 epoch to test setup
   ```bash
   python scripts/train.py --config configs/m1_max_imagenet100_100epoch.yaml --epochs 1
   ```

2. **Full Training**: Run 100 epochs
   ```bash
   python scripts/train.py --config configs/imagenet100_multi_dataset.yaml
   ```

3. **Evaluation**: Evaluate trained model
   ```bash
   python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pth
   ```

4. **Experiment**: Try different configurations
   - Adjust learning rate
   - Change batch size
   - Modify dataset weights
   - Extend to more epochs

## More Information

- **Complete Documentation**: `docs/IMAGENET100_INTEGRATION.md`
- **Implementation Report**: `IMAGENET100_IMPLEMENTATION_REPORT.md`
- **Example Code**: `examples/imagenet100_example.py`
- **Dataset Code**: `src/data/datasets.py`
- **Multi-Dataset Code**: `src/data/multi_dataset.py`

## Support

For issues or questions:
1. Check documentation in `docs/`
2. Run example script: `examples/imagenet100_example.py`
3. Verify dataset: `--verify-only` flag
4. Review configuration files in `configs/`

---

**Ready to train!** 🚀

Start with:
```bash
python scripts/train.py --config configs/imagenet100_multi_dataset.yaml
```

# H-JEPA Dataset Guide

This guide provides comprehensive information about downloading, preparing, and using datasets for H-JEPA training.

## Table of Contents

- [Quick Start](#quick-start)
- [Supported Datasets](#supported-datasets)
- [Dataset Download](#dataset-download)
- [Storage Requirements](#storage-requirements)
- [Usage Examples](#usage-examples)
- [Dataset Details](#dataset-details)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Download CIFAR-10 (Recommended for Testing)

```bash
# Using bash script (recommended)
./scripts/download_data.sh cifar10

# Or using Python module
python -m src.data.download cifar10 --data-path ./data
```

### Download All Auto-Downloadable Datasets

```bash
./scripts/download_data.sh --all-auto
```

### View Dataset Summary

```bash
./scripts/download_data.sh
```

## Supported Datasets

| Dataset | Auto-Download | Size | Images | Classes | Resolution | Description |
|---------|--------------|------|---------|---------|------------|-------------|
| **CIFAR-10** | ✓ | 170 MB | 60,000 | 10 | 32×32 | General object recognition |
| **CIFAR-100** | ✓ | 170 MB | 60,000 | 100 | 32×32 | Fine-grained objects |
| **STL-10** | ✓ | 2.5 GB | 113,000 | 10 | 96×96 | Includes unlabeled data |
| **ImageNet-100** | ✗ | ~15 GB | ~130,000 | 100 | Varies | ImageNet subset |
| **ImageNet** | ✗ | ~150 GB | 1,331,167 | 1000 | Varies | Full ILSVRC2012 |

### Dataset Selection Guide

- **Quick Testing**: Use **CIFAR-10** or **CIFAR-100** (small, fast to download)
- **Serious Prototyping**: Use **STL-10** (larger images, includes unlabeled data)
- **Research/Production**: Use **ImageNet-100** or **ImageNet** (industry standard)

## Dataset Download

### Method 1: Using Bash Script (Recommended)

The bash script provides a user-friendly interface with progress bars and verification.

```bash
# Show help and options
./scripts/download_data.sh --help

# Download single dataset
./scripts/download_data.sh cifar10

# Download multiple datasets
./scripts/download_data.sh cifar10 cifar100 stl10

# Download to custom location
./scripts/download_data.sh --data-path /mnt/datasets cifar10

# Force re-download
./scripts/download_data.sh --force cifar10

# Verify existing datasets
./scripts/download_data.sh --verify cifar10 cifar100
```

### Method 2: Using Python Module

```bash
# Download datasets
python -m src.data.download cifar10 cifar100 --data-path ./data

# Verify only (no download)
python -m src.data.download cifar10 --verify-only --data-path ./data

# Force re-download
python -m src.data.download cifar10 --force --data-path ./data
```

### Method 3: In Python Code

```python
from src.data import download_dataset, verify_dataset

# Download CIFAR-10
success = download_dataset('cifar10', data_path='./data')

# Verify dataset
is_valid = verify_dataset('cifar10', data_path='./data')
```

## Storage Requirements

### Minimum Requirements

- **Quick Testing**: 1 GB (CIFAR-10 + CIFAR-100)
- **Serious Prototyping**: 5 GB (CIFAR + STL-10)
- **Full Research**: 160+ GB (All datasets including ImageNet)

### Recommended Storage

- Add 5-10 GB buffer for temporary files and caching
- Use SSD for faster data loading during training
- Consider using external storage or NAS for large datasets

### Disk Space Check

The scripts automatically check available disk space before downloading:

```bash
./scripts/download_data.sh cifar10
# Output:
# ✓ Disk space check passed: 150.5 GB available
```

## Usage Examples

### Basic Dataset Loading

```python
from src.data import build_dataset, build_dataloader

# Build dataset
train_dataset = build_dataset(
    dataset_name='cifar10',
    data_path='./data/cifar10',
    split='train',
    image_size=224,
    download=True
)

val_dataset = build_dataset(
    dataset_name='cifar10',
    data_path='./data/cifar10',
    split='val',
    image_size=224,
    download=True
)

# Build dataloader
train_loader = build_dataloader(
    train_dataset,
    batch_size=128,
    num_workers=4,
    shuffle=True,
    pin_memory=True
)

# Iterate over data
for images, labels in train_loader:
    # images: (batch_size, 3, 224, 224)
    # labels: (batch_size,)
    pass
```

### Using Different Datasets

```python
# ImageNet
imagenet_dataset = build_dataset(
    dataset_name='imagenet',
    data_path='/path/to/imagenet',
    split='train',
    image_size=224,
)

# ImageNet-100 (automatically filters 100 classes)
imagenet100_dataset = build_dataset(
    dataset_name='imagenet100',
    data_path='/path/to/imagenet',
    split='train',
    image_size=224,
)

# STL-10 with unlabeled data
stl10_unlabeled = build_dataset(
    dataset_name='stl10',
    data_path='./data/stl10',
    split='unlabeled',
    image_size=224,
)
```

### Custom Transforms

```python
from src.data import JEPATransform, CIFAR10Dataset

# Custom transform
custom_transform = JEPATransform(
    image_size=224,
    crop_scale=(0.8, 1.0),
    color_jitter=0.4,  # Mild color jitter
    horizontal_flip=True,
)

# Use with dataset
dataset = CIFAR10Dataset(
    data_path='./data/cifar10',
    split='train',
    transform=custom_transform,
)
```

### Training Configuration

Update your config file (e.g., `configs/default.yaml`):

```yaml
data:
  dataset: "cifar10"  # or imagenet, cifar100, stl10, imagenet100
  data_path: "./data/cifar10"
  image_size: 224
  batch_size: 128
  num_workers: 8
  pin_memory: true

  augmentation:
    color_jitter: 0.4
    horizontal_flip: true
    random_crop: true
```

## Dataset Details

### CIFAR-10 / CIFAR-100

- **Auto-download**: Yes
- **Source**: `torchvision.datasets`
- **Format**: Automatically handled
- **Classes**:
  - CIFAR-10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
  - CIFAR-100: 100 fine-grained classes
- **Split**: 50k train, 10k test
- **Best for**: Quick prototyping and algorithm validation

### STL-10

- **Auto-download**: Yes
- **Source**: `torchvision.datasets`
- **Format**: Automatically handled
- **Special feature**: 100k unlabeled images for self-supervised learning
- **Split**: 5k labeled train, 8k test, 100k unlabeled
- **Best for**: Semi-supervised and self-supervised learning experiments

### ImageNet (ILSVRC2012)

- **Auto-download**: No (requires manual download)
- **Source**: https://image-net.org/download.php
- **Registration**: Required
- **Size**: ~144 GB training + ~6.3 GB validation
- **Format**: JPEG images organized in class folders

#### Directory Structure Required:

```
imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_18.JPEG
│   │   ├── n01440764_36.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ... (1000 classes)
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ... (1000 classes)
```

#### Download Instructions:

```bash
# Get manual download instructions
./scripts/download_data.sh imagenet

# Or in Python
from src.data import print_manual_download_instructions
print_manual_download_instructions('imagenet')
```

### ImageNet-100

- **Auto-download**: No (requires ImageNet first)
- **Source**: Subset of ImageNet
- **Format**: Same as ImageNet
- **Dataset automatically filters** to 100 standard classes at runtime
- **Best for**: Faster experiments while maintaining ImageNet-scale quality

## Troubleshooting

### Issue: Download Failed

```bash
✗ Download failed due to network error
```

**Solutions**:
- Check internet connection
- Retry download (it may resume automatically)
- Try downloading at a different time
- For manual downloads, use a download manager that supports resume

### Issue: Verification Failed

```bash
✗ Verification failed
```

**Solutions**:
- Delete corrupted data and re-download:
  ```bash
  rm -rf ./data/cifar10
  ./scripts/download_data.sh cifar10
  ```
- Force re-download:
  ```bash
  ./scripts/download_data.sh --force cifar10
  ```

### Issue: Insufficient Disk Space

```bash
⚠️  WARNING: Insufficient disk space!
```

**Solutions**:
- Free up disk space
- Use external storage:
  ```bash
  ./scripts/download_data.sh --data-path /mnt/external/datasets cifar10
  ```
- Download smaller datasets first (CIFAR before ImageNet)

### Issue: ImageNet Not Found

```bash
✗ ImageNet not found at ./data/imagenet
```

**Solutions**:
- ImageNet requires manual download. Get instructions:
  ```bash
  ./scripts/download_data.sh imagenet
  ```
- Ensure proper directory structure (see ImageNet section above)
- Verify you have both `train/` and `val/` directories

### Issue: Import Errors

```python
ImportError: cannot import name 'build_dataset'
```

**Solutions**:
- Ensure you're running from project root:
  ```bash
  cd /path/to/H-JEPA
  python -m src.data.download cifar10
  ```
- Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

### Issue: Slow Data Loading

**Solutions**:
- Increase `num_workers` in dataloader
- Enable `pin_memory=True` for GPU training
- Use SSD instead of HDD
- Cache dataset in memory if possible

## Advanced Usage

### Custom Dataset Path via Environment Variable

```bash
export DATA_PATH=/mnt/datasets
./scripts/download_data.sh cifar10
# Downloads to /mnt/datasets/cifar10
```

### Batch Verification

```bash
# Verify multiple datasets
./scripts/download_data.sh --verify cifar10 cifar100 stl10
```

### Integration with Training Script

```bash
# Download data
./scripts/download_data.sh cifar10

# Update config
vim configs/default.yaml
# Set: data.dataset = "cifar10"
#      data.data_path = "./data/cifar10"

# Start training
python scripts/train.py --config configs/default.yaml
```

## Performance Tips

1. **Use SSD**: Significantly faster data loading
2. **Increase Workers**: Set `num_workers=8` or higher (match CPU cores)
3. **Pin Memory**: Enable `pin_memory=True` for GPU training
4. **Batch Size**: Larger batches reduce data loading overhead
5. **Preprocessing**: JEPA uses minimal augmentations, so data loading is fast

## Questions?

- Check existing issues: https://github.com/yourusername/H-JEPA/issues
- Create new issue with "data" label
- Include error messages and system info

## Summary

- **Quick Start**: `./scripts/download_data.sh cifar10`
- **All Auto Downloads**: `./scripts/download_data.sh --all-auto`
- **Verify**: `./scripts/download_data.sh --verify cifar10`
- **ImageNet**: Requires manual download (see instructions above)

Happy training! 🚀

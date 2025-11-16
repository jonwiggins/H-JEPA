# DeiT III Augmentation Quick Reference

## TL;DR

```python
from src.data import DeiTIIIAugmentation

# Create augmentation
aug = DeiTIIIAugmentation(image_size=224, num_classes=1000)

# Get transforms
image_transform = aug.get_image_transform()  # For DataLoader
batch_transform = aug.get_batch_transform()  # For training loop

# Use in training
for images, labels in loader:
    images, labels = batch_transform(images, labels)  # Apply Mixup/CutMix
    outputs = model(images)
    loss = criterion(outputs, labels)  # Use soft labels
```

## Component Overview

| Component | Purpose | DeiT III Default | When to Use |
|-----------|---------|------------------|-------------|
| **RandAugment** | Strong automated augmentation | num_ops=2, mag=9 | Always for supervised training |
| **Mixup** | Image/label interpolation | alpha=0.8 | Medium-large datasets |
| **CutMix** | Spatial patch mixing | alpha=1.0 | Medium-large datasets |
| **RandomErasing** | Occlusion robustness | prob=0.25 | Always for robustness |
| **ColorJitter** | Color variations | strength=0.4 | Always for natural images |

## Default Parameters (DeiT III)

```python
DeiTIIIAugmentation(
    # Image preprocessing
    image_size=224,
    color_jitter=0.4,

    # RandAugment
    auto_augment=True,
    rand_aug_num_ops=2,
    rand_aug_magnitude=9,

    # Random Erasing
    random_erasing_prob=0.25,
    random_erasing_scale=(0.02, 0.33),
    random_erasing_ratio=(0.3, 3.3),

    # Mixup/CutMix
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    mixup_cutmix_prob=1.0,
    mixup_switch_prob=0.5,

    # Other
    num_classes=1000,
)
```

## Parameter Tuning Cheatsheet

### By Dataset Size

| Dataset Size | rand_aug_mag | mixup_alpha | cutmix_alpha | random_erasing_prob |
|--------------|--------------|-------------|--------------|---------------------|
| **Tiny** (< 10k) | 12-15 | 0.8-1.0 | 1.0-1.2 | 0.5 |
| **Small** (10k-50k) | 9-12 | 0.6-0.8 | 0.8-1.0 | 0.25-0.5 |
| **Medium** (50k-500k) | 7-9 | 0.6-0.8 | 0.8-1.0 | 0.25 |
| **Large** (> 500k) | 5-7 | 0.4-0.6 | 0.5-0.8 | 0.1-0.25 |

### By Task Type

| Task | Configuration |
|------|---------------|
| **Supervised Training** | Use all defaults |
| **Fine-tuning** | Reduce magnitudes by 30% |
| **Self-supervised (H-JEPA)** | Use JEPATransform instead |
| **Small images (< 64px)** | Reduce rand_aug_magnitude to 5-7 |
| **Medical imaging** | Disable color jitter, reduce magnitude |

## Common Configurations

### Maximum Augmentation (Tiny Datasets)

```python
aug = DeiTIIIAugmentation(
    image_size=224,
    rand_aug_num_ops=3,
    rand_aug_magnitude=15,
    random_erasing_prob=0.5,
    mixup_alpha=1.0,
    cutmix_alpha=1.2,
    num_classes=num_classes,
)
```

### Moderate Augmentation (Balanced)

```python
aug = DeiTIIIAugmentation(
    image_size=224,
    rand_aug_num_ops=2,
    rand_aug_magnitude=7,
    random_erasing_prob=0.25,
    mixup_alpha=0.6,
    cutmix_alpha=0.8,
    num_classes=num_classes,
)
```

### Minimal Augmentation (Fine-tuning)

```python
aug = DeiTIIIAugmentation(
    image_size=224,
    rand_aug_num_ops=1,
    rand_aug_magnitude=5,
    random_erasing_prob=0.1,
    mixup_alpha=0.3,
    cutmix_alpha=0.3,
    num_classes=num_classes,
)
```

### No Strong Augmentation (H-JEPA Style)

```python
from src.data import JEPATransform

# For H-JEPA pretraining, use minimal augmentation
aug = JEPATransform(
    image_size=224,
    color_jitter=0.4,  # Mild color jitter only
)
```

## Individual Components

### Using RandAugment Only

```python
from torchvision import transforms
from src.data import RandAugment

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Using Mixup Only

```python
from src.data import Mixup

mixup = Mixup(alpha=0.8, num_classes=1000)

# In training loop
for images, labels in loader:
    images, labels = mixup(images, labels)
    # Train with soft labels
```

### Using CutMix Only

```python
from src.data import CutMix

cutmix = CutMix(alpha=1.0, num_classes=1000)

# In training loop
for images, labels in loader:
    images, labels = cutmix(images, labels)
    # Train with soft labels
```

### Combining Mixup and CutMix

```python
from src.data import MixupCutmix

mixup_cutmix = MixupCutmix(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    num_classes=1000,
    switch_prob=0.5,  # 50% Mixup, 50% CutMix
)
```

## Integration Patterns

### Pattern 1: Standard Training

```python
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.data import DeiTIIIAugmentation

# Setup
aug = DeiTIIIAugmentation(image_size=224, num_classes=1000)
dataset = ImageFolder("path/to/train", transform=aug.get_image_transform())
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Training loop
batch_transform = aug.get_batch_transform()
for images, labels in loader:
    images, labels = batch_transform(images, labels)
    outputs = model(images)
    loss = criterion(outputs, labels)
```

### Pattern 2: With Existing Dataset Class

```python
from src.data import CIFAR10Dataset, DeiTIIIAugmentation

aug = DeiTIIIAugmentation(image_size=224, num_classes=10)

dataset = CIFAR10Dataset(
    data_path="data",
    split="train",
    transform=aug.get_image_transform(),
)
```

### Pattern 3: Configuration-Based

```python
import yaml
from src.data import build_deit3_transform

# Load config
with open('configs/deit3_augmentation.yaml') as f:
    config = yaml.safe_load(f)

# Build transform
aug_config = config['data']['augmentation']['deit3']
train_transform = build_deit3_transform(is_training=True, config=aug_config)
val_transform = build_deit3_transform(is_training=False)
```

## Loss Function Considerations

### For Soft Labels (Mixup/CutMix)

**Do NOT use**:
```python
criterion = nn.CrossEntropyLoss()  # Expects hard labels
loss = criterion(outputs, labels)  # Will fail with soft labels
```

**Use instead**:
```python
# Method 1: Manual soft cross-entropy
def soft_cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), dim=1))

loss = soft_cross_entropy(outputs, labels)

# Method 2: KL Divergence
criterion = nn.KLDivLoss(reduction='batchmean')
loss = criterion(F.log_softmax(outputs, dim=1), labels)

# Method 3: Custom loss
class SoftTargetCrossEntropy(nn.Module):
    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

criterion = SoftTargetCrossEntropy()
loss = criterion(outputs, labels)
```

## Common Issues & Solutions

### Issue 1: "Expected hard labels but got soft labels"

**Solution**: Use soft cross-entropy loss (see above)

### Issue 2: Training is too slow

**Solution**:
- Reduce `rand_aug_num_ops` from 2 to 1
- Reduce `num_workers` if CPU-bound
- Disable random erasing: `random_erasing_prob=0`

### Issue 3: Model not converging with strong augmentation

**Solution**:
- Increase warmup epochs
- Reduce augmentation strength (lower magnitude, alpha)
- Use learning rate warmup
- Increase total training epochs

### Issue 4: Overfitting despite augmentation

**Solution**:
- Increase augmentation strength
- Increase `rand_aug_magnitude` to 12-15
- Increase `random_erasing_prob` to 0.5
- Add weight decay

### Issue 5: Out of memory errors

**Solution**:
- Reduce batch size
- Use gradient accumulation
- Disable batch augmentation: `mixup_cutmix_prob=0`

## Performance Benchmarks

### Expected Improvements (ImageNet-1K, ViT-Base)

| Configuration | Top-1 Accuracy | Training Time |
|---------------|----------------|---------------|
| No augmentation | ~76% | 1x |
| Basic aug (JEPA) | ~77% | 1.05x |
| DeiT III (full) | ~81% | 1.15x |

### Dataset-Specific Gains

| Dataset | Baseline | + DeiT III | Improvement |
|---------|----------|------------|-------------|
| ImageNet-1K | 76% | 81% | +5% |
| CIFAR-10 | 94% | 97% | +3% |
| CIFAR-100 | 75% | 80% | +5% |
| Tiny ImageNet | 60% | 67% | +7% |

## Files Reference

```
H-JEPA/
├── src/data/
│   ├── transforms.py              # Main implementation
│   └── __init__.py                # Exports
├── examples/
│   └── deit3_augmentation_example.py  # Usage examples
├── configs/
│   └── deit3_augmentation.yaml    # Config template
└── docs/
    ├── DEIT3_AUGMENTATION_IMPLEMENTATION.md  # Full documentation
    └── DEIT3_QUICK_REFERENCE.md   # This file
```

## Import Statements

```python
# All-in-one
from src.data import (
    DeiTIIIAugmentation,      # Main pipeline
    DeiTIIIEvalTransform,     # Eval transform
    build_deit3_transform,    # Builder function
)

# Individual components
from src.data import (
    RandAugment,              # Strong augmentation
    Mixup,                    # Image mixing
    CutMix,                   # Spatial mixing
    MixupCutmix,              # Combined mixing
    RandomErasing,            # Occlusion
)

# For comparison
from src.data import (
    JEPATransform,            # Minimal augmentation
    JEPAEvalTransform,        # JEPA eval
)
```

## Further Reading

- **Full Documentation**: `docs/DEIT3_AUGMENTATION_IMPLEMENTATION.md`
- **Examples**: `examples/deit3_augmentation_example.py`
- **Configuration**: `configs/deit3_augmentation.yaml`
- **DeiT III Paper**: https://arxiv.org/abs/2204.07118
- **RandAugment Paper**: https://arxiv.org/abs/1909.13719

---

**Last Updated**: November 16, 2025

# H-JEPA Training Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Available Configurations](#available-configurations)
3. [Apple Silicon (MPS) Optimization](#apple-silicon-mps-optimization)
4. [Training Commands](#training-commands)
5. [Monitoring Training](#monitoring-training)
6. [Evaluation](#evaluation)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python scripts/download_datasets.py --dataset cifar10
```

### 3. Run Training
```bash
# Quick test (few steps)
./scripts/train_mps_safe.sh configs/quick_test.yaml

# Full training (20 epochs)
./scripts/train_mps_safe.sh configs/m1_max_optimal_20epoch.yaml
```

---

## Available Configurations

### Testing Configurations
| Config | Purpose | Duration |
|--------|---------|----------|
| `configs/quick_test.yaml` | Quick functionality test | ~5 minutes |
| `configs/eval_test.yaml` | Evaluation pipeline test | ~10 minutes |
| `configs/debug_minimal_memsafe.yaml` | Memory-safe debugging | ~30 minutes |

### Training Configurations
| Config | Purpose | Duration | Quality Target |
|--------|---------|----------|----------------|
| `configs/m1_max_5epoch_demo.yaml` | Demo training | ~12 hours | 50-60% accuracy |
| `configs/m1_max_optimal_20epoch.yaml` | Full training | ~50 hours | 70-80% accuracy |
| `configs/mps_optimized_v2.yaml` | MPS optimized | Variable | Production quality |

---

## Apple Silicon (MPS) Optimization

### Environment Variables
```bash
# Recommended for M1/M2 Macs
export PYTORCH_ENABLE_MPS_FALLBACK=0  # No CPU fallback
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5  # Conservative memory

# For debugging memory issues
export PYTORCH_MPS_ALLOCATOR_POLICY=default
```

### MPS-Safe Training Script
The `train_mps_safe.sh` script automatically handles:
- Memory management settings
- Garbage collection
- Error recovery
- Progress logging

Usage:
```bash
./scripts/train_mps_safe.sh <config_file>
```

### Performance Tips for Apple Silicon
1. **Batch Size**: Use 32 for M1 Max/Ultra, 16 for M1/M2
2. **Gradient Accumulation**: Enable for effective batch size > 32
3. **Mixed Precision**: Currently disabled due to MPS bugs
4. **Flash Attention**: Not supported on MPS (automatically disabled)

---

## Training Commands

### Basic Training
```bash
python scripts/train.py --config configs/m1_max_optimal_20epoch.yaml
```

### MPS-Safe Training (Recommended)
```bash
./scripts/train_mps_safe.sh configs/m1_max_optimal_20epoch.yaml 2>&1 | tee logs/training.log
```

### Background Training
```bash
./scripts/train_mps_safe.sh configs/m1_max_optimal_20epoch.yaml 2>&1 | tee logs/training.log &
```

### Resume Training
```bash
python scripts/train.py \
  --config configs/m1_max_optimal_20epoch.yaml \
  --resume results/<experiment>/checkpoints/checkpoint_last.pth
```

---

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir results/
```
Access at: http://localhost:6006

### Real-time Logs
```bash
tail -f logs/training.log
```

### Key Metrics to Monitor
- **Loss Convergence**: Should decrease steadily
- **Learning Rate**: Check warmup and scheduling
- **Memory Usage**: Monitor for OOM issues
- **Gradient Norms**: Should be stable (not exploding/vanishing)

### Expected Training Progress
| Epoch | Expected Loss | Linear Probe Accuracy |
|-------|---------------|----------------------|
| 1 | 8.0-10.0 | 15-25% |
| 5 | 4.0-6.0 | 40-50% |
| 10 | 2.0-4.0 | 55-65% |
| 20 | 1.0-2.0 | 70-80% |

---

## Evaluation

### Evaluate a Checkpoint
```bash
python scripts/evaluate_model.py \
  --checkpoint results/<experiment>/checkpoints/checkpoint_best.pth \
  --config configs/<config>.yaml \
  --linear_probe_epochs 100 \
  --knn_k 20
```

### Evaluation Metrics
- **Linear Probe**: Tests representation quality (target: >70%)
- **KNN Accuracy**: Tests feature clustering (target: >60%)

### Model Quality Benchmarks
| Quality Level | Linear Probe | KNN (k=20) | Training Required |
|--------------|--------------|------------|-------------------|
| Poor | <40% | <30% | Untrained |
| Moderate | 50-65% | 40-55% | 5-10 epochs |
| Good | 70-80% | 60-70% | 20-50 epochs |
| Excellent | >80% | >70% | 100+ epochs |

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM)
```bash
# Reduce batch size in config
batch_size: 16  # Instead of 32

# Or use gradient accumulation
accumulation_steps: 2
```

#### 2. MPS Backend Errors
```bash
# Disable MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=0

# Or use CPU for debugging
device: "cpu"
```

#### 3. Slow Training
- Check batch size (larger is faster but uses more memory)
- Disable logging: `log_frequency: 1000`
- Use optimized config: `mps_optimized_v2.yaml`

#### 4. NaN Losses
- Reduce learning rate
- Enable gradient clipping: `gradient_clip: 1.0`
- Check data augmentations

---

## Best Practices

### 1. Configuration Management
- Start with `quick_test.yaml` to verify setup
- Use `debug_minimal_memsafe.yaml` for development
- Graduate to `m1_max_optimal_20epoch.yaml` for real training

### 2. Checkpointing Strategy
- Save every 500-1000 steps
- Keep best 3 checkpoints by validation accuracy
- Always save last checkpoint for resuming

### 3. Hyperparameter Tuning
```yaml
# Key hyperparameters to tune
learning_rate: 1.5e-4  # Try: 1e-4 to 3e-4
batch_size: 32         # Based on GPU memory
warmup_epochs: 40      # 10-20% of total epochs
weight_decay: 0.04     # Try: 0.01 to 0.1
```

### 4. Data Augmentation
```yaml
transforms:
  train:
    RandomResizedCrop:
      scale: [0.4, 1.0]  # Aggressive cropping
    RandomHorizontalFlip:
      p: 0.5
    ColorJitter:        # Optional
      brightness: 0.4
      contrast: 0.4
```

### 5. Multi-Stage Training
1. **Stage 1**: Train with high learning rate (1.5e-4) for 10 epochs
2. **Stage 2**: Fine-tune with lower LR (5e-5) for 10 epochs
3. **Stage 3**: Final tuning with very low LR (1e-5) for 5 epochs

---

## Advanced Features

### Hierarchical Masking
```yaml
masking:
  hierarchical_masking: true
  mask_at_levels: [1, 2, 3]
  hierarchy_weights: [1.0, 0.8, 0.6]
```

### Feature Pyramid Network (FPN)
```yaml
model:
  use_fpn: true
  fpn_feature_dim: 128
```

### LayerScale (Training Stability)
```yaml
model:
  use_layerscale: true  # Recommended for deep models
```

### SigReg Regularization
```yaml
loss:
  use_sigreg: true
  sigreg_weight: 0.1
```

---

## Performance Optimization

### Memory Optimization
```bash
# Clear cache periodically
clear_cuda_cache_freq: 100

# Use gradient checkpointing
use_gradient_checkpointing: true
```

### Speed Optimization
```yaml
# Larger batch size with accumulation
batch_size: 16
accumulation_steps: 2  # Effective batch size: 32

# Reduce logging overhead
log_frequency: 500
log_images: false
log_gradients: false
```

---

## Example Training Workflow

```bash
# 1. Setup environment
export PYTORCH_ENABLE_MPS_FALLBACK=0

# 2. Quick test to verify everything works
./scripts/train_mps_safe.sh configs/quick_test.yaml

# 3. Run full training
./scripts/train_mps_safe.sh configs/m1_max_optimal_20epoch.yaml \
  2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log &

# 4. Monitor progress
tensorboard --logdir results/

# 5. Evaluate best checkpoint
python scripts/evaluate_model.py \
  --checkpoint results/m1_max_optimal_20epoch/checkpoints/checkpoint_best.pth \
  --config configs/m1_max_optimal_20epoch.yaml

# 6. Generate visualizations
python scripts/visualize_representations.py \
  --checkpoint results/m1_max_optimal_20epoch/checkpoints/checkpoint_best.pth
```

---

## Resources

- [Original JEPA Paper](https://arxiv.org/abs/2301.08243)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Project Repository](https://github.com/jonwiggins/H-JEPA)

---

## Support

For issues or questions:
1. Check the [MODEL_QUALITY_ASSESSMENT.md](MODEL_QUALITY_ASSESSMENT.md) for quality benchmarks
2. Review logs in `logs/` directory
3. Open an issue on GitHub with:
   - Config file used
   - Error message
   - System specs (Mac model, macOS version, PyTorch version)

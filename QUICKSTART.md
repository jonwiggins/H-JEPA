# H-JEPA Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Navigate to project directory
cd /home/user/H-JEPA

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install package
pip install -e .

# 4. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

## Project Overview

H-JEPA is a hierarchical self-supervised learning method that learns visual representations by predicting masked image regions at multiple scales.

**Key Components:**
- Context Encoder: Processes visible patches
- Target Encoder: Processes all patches (EMA updated)
- Hierarchical Predictors: Predict target representations at multiple scales
- Multi-Block Masking: Strategic masking for robust learning

## Quick Commands

### View Configuration
```bash
cat configs/default.yaml
```

### Run Training (when implemented)
```bash
# Small test run
python scripts/train.py --config configs/small_experiment.yaml

# Full training
python scripts/train.py --config configs/default.yaml
```

### Run Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=src
```

### Visualizations (when implemented)
```bash
# Visualize masking strategy
python scripts/visualize.py --visualize-masks

# Visualize predictions
python scripts/visualize.py --checkpoint results/checkpoints/model.pth --image test.jpg --visualize-predictions
```

## File Locations

**Configuration:** `/home/user/H-JEPA/configs/`
**Source Code:** `/home/user/H-JEPA/src/`
**Scripts:** `/home/user/H-JEPA/scripts/`
**Tests:** `/home/user/H-JEPA/tests/`
**Results:** `/home/user/H-JEPA/results/`

## Implementation Priority

1. **Masking** → `src/masks/multi_block.py`
2. **Encoder** → `src/models/encoder.py`
3. **Predictor** → `src/models/predictor.py`
4. **Loss** → `src/losses/hjepa_loss.py`
5. **Data** → `src/data/datasets.py`
6. **Trainer** → `src/trainers/trainer.py`

## Key Configuration Parameters

**Model Size:**
```yaml
model:
  encoder_type: "vit_small_patch16_224"  # or vit_base, vit_large
  num_hierarchies: 3  # Number of hierarchical levels
```

**Training:**
```yaml
training:
  batch_size: 128
  epochs: 300
  lr: 1.5e-4
```

**Masking:**
```yaml
masking:
  num_masks: 4  # Number of target blocks
  mask_scale: [0.15, 0.2]  # Mask size range
```

## Monitoring

**TensorBoard:**
```bash
tensorboard --logdir results/logs
```

**Weights & Biases:**
```bash
# Enable in config
wandb login
# Set wandb.enabled: true in config
```

## Common Tasks

**Change model size:**
Edit `configs/default.yaml` → `model.encoder_type`

**Reduce memory usage:**
- Decrease `batch_size`
- Use `vit_small` instead of `vit_base`
- Enable gradient checkpointing (when implemented)

**Speed up training:**
- Increase `num_workers`
- Enable `use_amp: true` (mixed precision)
- Use multiple GPUs with distributed training

## Getting Help

1. **README.md** - Comprehensive documentation
2. **SETUP_NOTES.md** - Detailed implementation guide
3. **CONTRIBUTING.md** - Development guidelines
4. **GitHub Issues** - Report bugs or ask questions

## Next Steps

1. Read the full README.md
2. Check SETUP_NOTES.md for implementation details
3. Start implementing core components
4. Write tests as you go
5. Train and evaluate!

---

**Project Status:** Structure complete, ready for implementation
**Location:** `/home/user/H-JEPA`

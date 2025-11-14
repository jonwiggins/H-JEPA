# H-JEPA Setup Notes

## Project Structure Created

The complete H-JEPA project structure has been successfully created with the following components:

### Directory Structure

```
H-JEPA/
├── configs/                      # Configuration files
│   ├── default.yaml             # Default configuration for full training
│   └── small_experiment.yaml    # Smaller config for testing
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── models/                  # Model architectures
│   │   └── __init__.py
│   ├── data/                    # Data loading
│   │   └── __init__.py
│   ├── masks/                   # Masking strategies
│   │   └── __init__.py
│   ├── losses/                  # Loss functions
│   │   └── __init__.py
│   ├── trainers/                # Training logic
│   │   └── __init__.py
│   └── utils/                   # Utilities
│       └── __init__.py
│
├── scripts/                     # Executable scripts
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Evaluation script
│   └── visualize.py            # Visualization utilities
│
├── notebooks/                   # Jupyter notebooks
│   └── 01_explore_masking.ipynb
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_masking.py
│   └── test_losses.py
│
├── results/                     # Training outputs
│   ├── checkpoints/            # Model checkpoints
│   │   └── .gitkeep
│   └── logs/                   # Training logs
│       └── .gitkeep
│
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Package configuration
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
└── CONTRIBUTING.md             # Contribution guidelines
```

## Next Steps for Implementation

### Phase 1: Core Components (Priority 1)

1. **Masking Strategy** (`src/masks/multi_block.py`)
   - Implement multi-block masking
   - Generate context and target masks
   - Ensure no overlap between masks
   - Respect aspect ratio and scale constraints

2. **Vision Transformer Encoder** (`src/models/encoder.py`)
   - Context encoder (trainable)
   - Target encoder (EMA updated)
   - Load pre-trained ViT weights from timm
   - Handle patch embedding and positional encoding

3. **Predictor** (`src/models/predictor.py`)
   - Hierarchical predictor architecture
   - Multi-scale processing
   - Cross-attention mechanism for context to target

4. **Loss Function** (`src/losses/hjepa_loss.py`)
   - Prediction loss (MSE or Smooth L1)
   - Hierarchical weighting
   - Normalization options

### Phase 2: Training Infrastructure (Priority 2)

5. **Dataset Loader** (`src/data/datasets.py`)
   - ImageNet dataset wrapper
   - Data augmentation pipeline
   - Efficient loading with caching

6. **Trainer** (`src/trainers/trainer.py`)
   - Main training loop
   - EMA updates for target encoder
   - Gradient accumulation
   - Mixed precision training
   - Checkpointing

7. **Utilities** (`src/utils/`)
   - `checkpoint.py`: Save/load checkpoints
   - `logging.py`: W&B and TensorBoard integration
   - `scheduler.py`: Learning rate scheduling

### Phase 3: Evaluation and Analysis (Priority 3)

8. **Linear Probing** (in `scripts/evaluate.py`)
   - Freeze encoder
   - Train linear classifier
   - Evaluate on downstream tasks

9. **Visualization** (in `scripts/visualize.py`)
   - Mask visualization
   - Attention map visualization
   - Feature space analysis

10. **Unit Tests** (in `tests/`)
    - Test each component
    - Integration tests
    - Gradient flow tests

## Installation Instructions

### 1. Create Virtual Environment

```bash
cd /home/user/H-JEPA
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Option A: Install as package (recommended)
pip install -e .

# Option B: Install requirements only
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import timm; print('timm:', timm.__version__)"
python -c "import wandb; print('wandb:', wandb.__version__)"
```

## Configuration Guide

### Using Different Configurations

**For full training on ImageNet:**
```bash
python scripts/train.py --config configs/default.yaml
```

**For quick testing/debugging:**
```bash
python scripts/train.py --config configs/small_experiment.yaml
```

### Key Configuration Parameters

**Model Size:**
- `vit_small_patch16_224`: ~22M parameters
- `vit_base_patch16_224`: ~86M parameters
- `vit_large_patch16_224`: ~304M parameters

**Hierarchy Levels:**
- `num_hierarchies: 2`: Faster, less memory
- `num_hierarchies: 3`: Better performance, more memory

**Batch Size:**
- Adjust based on GPU memory
- Use gradient accumulation for effective larger batches

## Development Workflow

### 1. Implement a Component

```bash
# Create the file
touch src/models/encoder.py

# Implement the component
# Add tests
touch tests/test_encoder.py
```

### 2. Test Your Implementation

```bash
# Run specific test
pytest tests/test_encoder.py -v

# Run all tests
pytest

# Check coverage
pytest --cov=src --cov-report=html
```

### 3. Format Code

```bash
# Format with black
black src/models/encoder.py

# Sort imports
isort src/models/encoder.py
```

### 4. Run Training

```bash
# Quick test with small config
python scripts/train.py --config configs/small_experiment.yaml

# Full training
python scripts/train.py --config configs/default.yaml
```

## Common Issues and Solutions

### CUDA Out of Memory

**Solutions:**
1. Reduce batch size in config
2. Enable gradient checkpointing
3. Use gradient accumulation
4. Reduce model size (use vit_small instead of vit_base)

### Slow Data Loading

**Solutions:**
1. Increase `num_workers` in config
2. Enable `pin_memory`
3. Use faster storage (SSD)
4. Pre-process and cache dataset

### Loss Diverging

**Solutions:**
1. Reduce learning rate
2. Increase warmup epochs
3. Check EMA momentum settings
4. Verify masking strategy parameters

## Performance Optimization Tips

1. **Mixed Precision Training**
   - Enable `use_amp: true` in config
   - Can reduce memory usage by ~40%

2. **Distributed Training**
   - Use multiple GPUs with DDP
   - Set `distributed.enabled: true`

3. **Efficient Data Loading**
   - Set `num_workers` to number of CPU cores
   - Enable `pin_memory` for GPU training

4. **Gradient Accumulation**
   - Simulate larger batch sizes
   - Set `accumulation_steps` in config

## Monitoring Training

### W&B (Weights & Biases)

```bash
# Login to W&B
wandb login

# Enable in config
# wandb.enabled: true
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir results/logs

# View at http://localhost:6006
```

## Dataset Preparation

### ImageNet

```bash
# Download ImageNet (requires account)
# Structure should be:
# /path/to/imagenet/
#   ├── train/
#   │   ├── n01440764/
#   │   ├── n01443537/
#   │   └── ...
#   └── val/
#       ├── n01440764/
#       ├── n01443537/
#       └── ...

# Update config with path
# data.data_path: "/path/to/imagenet"
```

### CIFAR-10 (for testing)

```python
# Will be downloaded automatically
# Just set: data.dataset: "cifar10"
```

## Git Workflow

```bash
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "Implement encoder module"

# Push
git push origin main
```

## Resources

### Papers
- I-JEPA: https://arxiv.org/abs/2301.08243
- Vision Transformer: https://arxiv.org/abs/2010.11929

### Code References
- timm library: https://github.com/huggingface/pytorch-image-models
- PyTorch: https://pytorch.org/docs/

### Documentation
- W&B: https://docs.wandb.ai/
- TensorBoard: https://www.tensorflow.org/tensorboard

## Support

For issues and questions:
1. Check this document
2. Review README.md
3. Check GitHub issues
4. Open a new issue with details

---

**Status:** Project structure complete, ready for implementation.

**Last Updated:** 2024-11-14

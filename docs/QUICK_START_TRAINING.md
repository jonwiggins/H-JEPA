# H-JEPA Training - Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install W&B for experiment tracking
pip install wandb
wandb login
```

## Basic Training

### 1. Prepare Your Config

Edit `configs/default.yaml`:

```yaml
training:
  epochs: 300
  lr: 1.5e-4
  use_amp: true

logging:
  experiment_name: "my_experiment"
  wandb:
    enabled: true
```

### 2. Run Training

```bash
python scripts/train.py --config configs/default.yaml
```

### 3. Monitor Progress

- **TensorBoard**: `tensorboard --logdir results/logs/tensorboard`
- **W&B**: Check your W&B dashboard
- **Console**: Real-time progress bars and metrics

## Resume Training

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --resume results/checkpoints/checkpoint_latest.pth
```

## Python API Usage

```python
from src.trainers import HJEPATrainer, create_optimizer
from src.utils import MetricsLogger, CheckpointManager

# Create model, data loaders, loss, etc.
model = ...
train_loader = ...
val_loader = ...
loss_fn = ...
masking_fn = ...

# Create optimizer
optimizer = create_optimizer(model, config)

# Create trainer
trainer = HJEPATrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    masking_fn=masking_fn,
    config=config,
    device='cuda',
)

# Train
trainer.train()
```

## Key Configuration Options

### Learning Rate Schedule
```yaml
training:
  lr: 1.5e-4          # Base learning rate
  min_lr: 1.0e-6      # Minimum learning rate
  warmup_epochs: 40   # Warmup period
  lr_schedule: "cosine"  # cosine or linear
```

### Mixed Precision
```yaml
training:
  use_amp: true       # Enable automatic mixed precision
  clip_grad: 3.0      # Gradient clipping threshold
```

### Checkpointing
```yaml
checkpoint:
  save_frequency: 10   # Save every N epochs
  keep_best_n: 3       # Keep best 3 checkpoints
  checkpoint_dir: "results/checkpoints"
```

### Logging
```yaml
logging:
  log_frequency: 100   # Log every N steps
  wandb:
    enabled: true
    project: "h-jepa"
  tensorboard:
    enabled: true
```

## Advanced Usage

### Gradient Accumulation
```yaml
training:
  batch_size: 64
  accumulation_steps: 4  # Effective batch size = 256
```

### Custom Schedulers
```python
from src.utils import CosineScheduler, HierarchicalScheduler

# Different LR per hierarchy level
schedulers = [
    CosineScheduler(1e-3, 1e-6, epochs, steps_per_epoch, warmup_epochs),
    CosineScheduler(5e-4, 5e-7, epochs, steps_per_epoch, warmup_epochs),
    CosineScheduler(2e-4, 2e-7, epochs, steps_per_epoch, warmup_epochs),
]
hier_scheduler = HierarchicalScheduler(schedulers)
```

### Manual Checkpointing
```python
from src.utils import CheckpointManager

ckpt_manager = CheckpointManager(
    checkpoint_dir='results/checkpoints',
    keep_best_n=3,
    metric_name='val_loss',
    mode='min',
)

# Save
ckpt_manager.save_checkpoint(
    epoch=epoch,
    model=model,
    optimizer=optimizer,
    metrics={'val_loss': val_loss},
    is_best=True,
)

# Load
ckpt_manager.load_checkpoint(
    'results/checkpoints/checkpoint_best.pth',
    model=model,
    optimizer=optimizer,
)
```

## Monitoring Metrics

### Training Metrics
- `train/loss`: Training loss
- `train/lr`: Current learning rate
- `train/ema_momentum`: EMA coefficient

### Validation Metrics
- `val/loss`: Validation loss

### Collapse Metrics
- `context_std`: Embedding standard deviation
- `context_norm`: Embedding L2 norm
- `context_rank`: Effective rank

### System Metrics
- `system/gpu0_memory_allocated_gb`: GPU memory
- `system/gpu0_utilization`: GPU usage %

## Troubleshooting

### Training Not Starting
- Check data paths in config
- Verify GPU availability: `nvidia-smi`
- Check dependencies: `pip list`

### Out of Memory
```yaml
training:
  batch_size: 32      # Reduce
  use_amp: true       # Enable
  accumulation_steps: 2  # Use accumulation
```

### Loss Not Decreasing
- Check learning rate (try 1e-4)
- Increase warmup epochs
- Verify data augmentation
- Check masking strategy

### Slow Training
```yaml
data:
  num_workers: 8      # Increase workers
  pin_memory: true    # Enable pin memory

training:
  use_amp: true       # Enable mixed precision
```

## Examples

See `examples/training_example.py` for:
- Basic training setup
- Resume from checkpoint
- Custom schedulers
- Monitoring and logging
- Checkpoint management

## Support

For issues:
1. Check `docs/TRAINING.md` for detailed documentation
2. Review example code in `examples/`
3. Check configuration in `configs/default.yaml`

## Next Steps

1. ✅ Training infrastructure ready
2. ⏳ Implement H-JEPA model
3. ⏳ Implement hierarchical loss
4. ⏳ Implement multi-block masking
5. ⏳ Implement data pipeline
6. ✅ Start training!

# H-JEPA Training Infrastructure

Comprehensive training infrastructure for Hierarchical Joint-Embedding Predictive Architecture (H-JEPA).

## Overview

The training infrastructure provides:

- **Complete Training Loop**: Forward pass, loss computation, backpropagation
- **EMA Updates**: Exponential Moving Average updates for target encoder
- **Mixed Precision Training**: Automatic mixed precision with torch.amp
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Checkpoint Management**: Save/load with best model tracking
- **Logging**: W&B and TensorBoard integration
- **Collapse Monitoring**: Track representation quality metrics
- **Distributed Training Support**: Ready for multi-GPU training

## Components

### 1. Trainer (`src/trainers/trainer.py`)

The `HJEPATrainer` class provides the main training loop.

**Features:**
- Full training and validation loops
- Automatic EMA updates for target encoder
- Gradient clipping and accumulation
- Mixed precision training with GradScaler
- Checkpoint saving/loading
- Metrics logging to W&B and TensorBoard
- Representation collapse monitoring
- Progress tracking with ETA estimation

**Key Methods:**
- `train()`: Main training loop
- `_train_epoch()`: Train for one epoch
- `_validate_epoch()`: Validation loop
- `_update_target_encoder()`: EMA updates
- `_compute_collapse_metrics()`: Monitor representation quality

### 2. Schedulers (`src/utils/scheduler.py`)

Learning rate and EMA momentum schedulers.

**Classes:**
- `CosineScheduler`: Cosine annealing with linear warmup
- `LinearScheduler`: Linear schedule with warmup
- `EMAScheduler`: EMA momentum scheduling
- `HierarchicalScheduler`: Per-level scheduling support

**Example:**
```python
from src.utils import create_lr_scheduler, create_ema_scheduler

# Create learning rate scheduler
lr_scheduler = create_lr_scheduler(
    optimizer_type='adamw',
    base_lr=1.5e-4,
    min_lr=1e-6,
    epochs=300,
    steps_per_epoch=1000,
    warmup_epochs=40,
    schedule_type='cosine',
)

# Create EMA scheduler
ema_scheduler = create_ema_scheduler(
    base_momentum=0.996,
    final_momentum=1.0,
    epochs=300,
    steps_per_epoch=1000,
    warmup_epochs=30,
)

# Use in training loop
for step in range(total_steps):
    lr = lr_scheduler(step)
    ema_momentum = ema_scheduler(step)
```

### 3. Checkpoint Manager (`src/utils/checkpoint.py`)

Handles checkpoint saving, loading, and cleanup.

**Features:**
- Save complete training state (model, optimizer, scheduler, scaler)
- Track and keep best N checkpoints
- Automatic cleanup of old checkpoints
- Resume training from checkpoint
- Support for DataParallel/DistributedDataParallel

**Example:**
```python
from src.utils import CheckpointManager

# Create checkpoint manager
ckpt_manager = CheckpointManager(
    checkpoint_dir='results/checkpoints',
    keep_best_n=3,
    save_frequency=10,
    metric_name='val_loss',
    mode='min',
)

# Save checkpoint
ckpt_manager.save_checkpoint(
    epoch=epoch,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler_state,
    scaler=scaler,
    metrics={'val_loss': val_loss},
    is_best=is_best,
)

# Load checkpoint
metadata = ckpt_manager.load_checkpoint(
    checkpoint_path='results/checkpoints/checkpoint_best.pth',
    model=model,
    optimizer=optimizer,
    scaler=scaler,
)
```

### 4. Metrics Logger (`src/utils/logging.py`)

Unified logging interface for W&B and TensorBoard.

**Features:**
- W&B integration with config tracking
- TensorBoard logging
- Metric aggregation and averaging
- Image and histogram logging
- System monitoring (GPU usage, memory)
- Model gradient/weight tracking

**Example:**
```python
from src.utils import MetricsLogger

# Create logger
logger = MetricsLogger(
    experiment_name='hjepa_experiment',
    log_dir='results/logs',
    config=config,
    use_wandb=True,
    use_tensorboard=True,
    wandb_project='h-jepa',
    wandb_tags=['baseline'],
)

# Log metrics
logger.log_metrics(
    {'loss': 0.5, 'accuracy': 0.85},
    step=100,
    prefix='train/',
)

# Log images
logger.log_image('mask_visualization', image_tensor, step=100)

# Log histograms
logger.log_histogram('gradients/layer1', grad_tensor, step=100)

# System metrics
logger.log_system_metrics(step=100)
```

## Usage

### Basic Training

```python
import torch
from src.trainers import HJEPATrainer, create_optimizer

# Load configuration
config = load_config('configs/default.yaml')

# Create model (example - replace with actual H-JEPA model)
model = create_hjepa_model(config)

# Create data loaders
train_loader = create_train_loader(config)
val_loader = create_val_loader(config)

# Create optimizer
optimizer = create_optimizer(model, config)

# Create loss function
loss_fn = create_loss_function(config)

# Create masking function
masking_fn = create_masking_function(config)

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

# Start training
trainer.train()
```

### Resume Training

```python
# Create trainer with resume checkpoint
trainer = HJEPATrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    masking_fn=masking_fn,
    config=config,
    device='cuda',
    resume_checkpoint='results/checkpoints/checkpoint_latest.pth',
)

# Training automatically resumes from checkpoint
trainer.train()
```

### Command Line Training

```bash
# Basic training
python scripts/train.py --config configs/default.yaml

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume results/checkpoints/checkpoint_latest.pth

# Override device
python scripts/train.py --config configs/default.yaml --device cuda:1

# Distributed training
python -m torch.distributed.launch --nproc_per_node=4 scripts/train.py --config configs/default.yaml --distributed
```

## Configuration

Training is configured via YAML files. Key sections:

### Model Configuration
```yaml
model:
  encoder_type: "vit_base_patch16_224"
  embed_dim: 768
  num_hierarchies: 3
  ema:
    momentum: 0.996
    momentum_end: 1.0
    momentum_warmup_epochs: 30
```

### Training Configuration
```yaml
training:
  epochs: 300
  warmup_epochs: 40
  lr: 1.5e-4
  min_lr: 1.0e-6
  weight_decay: 0.05
  optimizer: "adamw"
  betas: [0.9, 0.95]
  lr_schedule: "cosine"
  clip_grad: 3.0
  use_amp: true
  accumulation_steps: 1
```

### Logging Configuration
```yaml
logging:
  experiment_name: "hjepa_default"
  log_dir: "results/logs"
  log_frequency: 100
  wandb:
    enabled: true
    project: "h-jepa"
    entity: null
    tags: ["baseline"]
  tensorboard:
    enabled: true
```

### Checkpoint Configuration
```yaml
checkpoint:
  save_frequency: 10
  keep_best_n: 3
  checkpoint_dir: "results/checkpoints"
```

## Collapse Monitoring

The trainer automatically monitors representation collapse with these metrics:

- **Standard Deviation**: Should not approach 0
- **Mean L2 Norm**: Measure of embedding magnitude
- **Effective Rank**: Singular value entropy (should be high)

These metrics are logged to W&B/TensorBoard automatically.

## Mixed Precision Training

Enabled by default with `torch.cuda.amp`:

```yaml
training:
  use_amp: true
```

Benefits:
- Faster training (2-3x speedup on modern GPUs)
- Reduced memory usage
- Maintained numerical stability with gradient scaling

## Gradient Accumulation

For training with larger effective batch sizes:

```yaml
training:
  accumulation_steps: 4  # Effective batch size = batch_size * 4
```

## Best Practices

1. **Start with default config**: Use `configs/default.yaml` as a starting point
2. **Enable logging**: Use both W&B and TensorBoard for comprehensive tracking
3. **Monitor collapse metrics**: Check std and rank metrics regularly
4. **Use mixed precision**: Significant speedup on modern GPUs
5. **Save checkpoints frequently**: Use `save_frequency: 10` or lower
6. **Keep multiple best checkpoints**: Set `keep_best_n: 3` or higher
7. **Use gradient clipping**: Prevents training instability (`clip_grad: 3.0`)
8. **Warmup learning rate**: Critical for stability (`warmup_epochs: 40`)

## Troubleshooting

### Training Instability
- Increase warmup epochs
- Reduce learning rate
- Enable gradient clipping
- Check for NaN in loss/gradients

### Representation Collapse
- Monitor std and rank metrics
- Adjust EMA momentum
- Check loss function implementation
- Verify masking strategy

### Out of Memory
- Reduce batch size
- Enable gradient accumulation
- Use mixed precision training
- Reduce model size

### Slow Training
- Enable mixed precision (`use_amp: true`)
- Increase batch size if possible
- Use more workers for data loading
- Profile with PyTorch profiler

## Examples

See `examples/training_example.py` for complete examples:

1. Basic training setup
2. Resume from checkpoint
3. Hierarchical learning rates
4. Custom monitoring
5. Checkpoint management

## Advanced Features

### Distributed Training (Coming Soon)
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/default.yaml \
    --distributed
```

### Custom Loss Functions
```python
class CustomHJEPALoss(nn.Module):
    def forward(self, predictions, targets):
        # Implement custom loss
        loss = compute_loss(predictions, targets)
        loss_dict = {'loss': loss.item()}
        return loss, loss_dict
```

### Custom Callbacks
```python
# Extend HJEPATrainer and override methods
class CustomTrainer(HJEPATrainer):
    def _train_step(self, batch, epoch, step):
        # Custom training step logic
        loss, loss_dict = super()._train_step(batch, epoch, step)
        # Add custom behavior
        return loss, loss_dict
```

## Performance Tips

1. **Data Loading**: Use `num_workers` > 0 and `pin_memory=True`
2. **Batch Size**: Largest that fits in memory
3. **Mixed Precision**: Always enable on modern GPUs
4. **Gradient Checkpointing**: For very large models
5. **Compile Model**: Use `torch.compile()` on PyTorch 2.0+

## License

This training infrastructure is part of the H-JEPA project.

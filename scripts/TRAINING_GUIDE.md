# H-JEPA Training Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch >= 2.0.0
- timm >= 0.9.0
- einops >= 0.7.0
- wandb >= 0.15.0
- PyYAML >= 6.0

### 2. Prepare Your Dataset

The training script supports multiple datasets:

**Automatically downloadable:**
- CIFAR-10
- CIFAR-100
- STL-10

**Manual setup required:**
- ImageNet
- ImageNet-100
- Custom datasets

For ImageNet, organize as:
```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   │   ├── image1.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

### 3. Configure Training

Edit `configs/default.yaml` or create your own config:

```yaml
model:
  encoder_type: "vit_base_patch16_224"
  num_hierarchies: 3

data:
  dataset: "imagenet"
  data_path: "/path/to/dataset"
  batch_size: 128

training:
  epochs: 300
  lr: 1.5e-4
```

### 4. Run Training

**Basic training:**
```bash
python scripts/train.py --config configs/default.yaml
```

**With custom dataset path:**
```bash
python scripts/train.py --config configs/default.yaml --data_path /path/to/data
```

**Quick test (fewer epochs):**
```bash
python scripts/train.py --config configs/default.yaml --epochs 10 --batch_size 32
```

## Usage Examples

### Basic Training

Start training with default configuration:

```bash
python scripts/train.py --config configs/default.yaml
```

Expected output:
```
================================================================================
                         H-JEPA Training Configuration
================================================================================

Experiment:
  Name: hjepa_default
  Seed: 42
  Output: results/checkpoints

Model:
  Encoder: vit_base_patch16_224
  Embedding dim: 768
  Hierarchies: 3
  Predictor depth: 6

...

================================================================================
                              Starting Training
================================================================================
```

### Resume from Checkpoint

Continue training from a saved checkpoint:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --resume results/checkpoints/checkpoint_epoch_100.pth
```

### Override Config Parameters

Override specific parameters without modifying the config file:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --batch_size 64 \
    --lr 1e-4 \
    --epochs 100 \
    --warmup_epochs 10
```

### Multi-GPU Training

**Using torch.distributed.launch (recommended):**

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    scripts/train.py \
    --config configs/default.yaml \
    --distributed
```

**Using torchrun (PyTorch >= 1.10):**

```bash
torchrun --nproc_per_node=4 \
    scripts/train.py \
    --config configs/default.yaml \
    --distributed
```

### Custom Output Directory

Save checkpoints and logs to a custom location:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --output_dir experiments/my_experiment
```

This will create:
- `experiments/my_experiment/checkpoints/`
- `experiments/my_experiment/logs/`

### Disable Weights & Biases

Train without W&B logging:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --no_wandb
```

### Debug Mode

Enable verbose logging for debugging:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --debug
```

### Specific GPU

Train on a specific GPU:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --device cuda:1
```

## Command-Line Arguments

### Required Arguments

- `--config PATH`: Path to configuration YAML file

### Optional Arguments

**General:**
- `--resume PATH`: Resume from checkpoint
- `--device DEVICE`: Device to use (cuda, cuda:0, cpu)
- `--output_dir PATH`: Custom output directory
- `--debug`: Enable debug logging
- `--no_wandb`: Disable W&B logging

**Training Overrides:**
- `--batch_size INT`: Batch size per GPU
- `--epochs INT`: Number of training epochs
- `--lr FLOAT`: Learning rate
- `--weight_decay FLOAT`: Weight decay
- `--warmup_epochs INT`: Warmup epochs

**Data Overrides:**
- `--data_path PATH`: Path to dataset
- `--num_workers INT`: Data loading workers

**Distributed Training:**
- `--distributed`: Enable DDP training
- `--world_size INT`: Number of processes
- `--local_rank INT`: Local rank (auto-set)

## Training Pipeline

The training script executes the following pipeline:

1. **Initialization**
   - Load and validate configuration
   - Setup device and distributed training
   - Set random seeds for reproducibility
   - Create output directories

2. **Data Loading**
   - Build training dataset with augmentations
   - Build validation dataset (optional)
   - Create dataloaders with proper batching

3. **Model Creation**
   - Initialize H-JEPA model (encoder + predictor)
   - Wrap with DistributedDataParallel if needed
   - Move to device

4. **Training Setup**
   - Create hierarchical mask generator
   - Initialize loss function with hierarchy weights
   - Create optimizer (AdamW by default)
   - Setup learning rate schedulers

5. **Training Loop**
   - Forward pass with masking
   - Hierarchical prediction
   - Loss computation
   - Backpropagation with gradient clipping
   - EMA update for target encoder
   - Periodic validation
   - Checkpoint saving
   - Metrics logging (W&B, TensorBoard)

6. **Completion**
   - Save final checkpoint
   - Log final metrics
   - Clean up distributed processes

## Configuration File Structure

Example `configs/default.yaml`:

```yaml
# Model architecture
model:
  encoder_type: "vit_base_patch16_224"
  embed_dim: 768
  num_hierarchies: 3
  predictor:
    depth: 6
    num_heads: 12
    mlp_ratio: 4.0
  ema:
    momentum: 0.996
    momentum_end: 1.0
    momentum_warmup_epochs: 30

# Dataset configuration
data:
  dataset: "imagenet"
  data_path: "/path/to/dataset"
  image_size: 224
  batch_size: 128
  num_workers: 8
  pin_memory: true
  augmentation:
    color_jitter: 0.4
    horizontal_flip: true
    random_crop: true

# Multi-block masking
masking:
  num_masks: 4
  mask_scale: [0.15, 0.2]
  aspect_ratio: [0.75, 1.5]
  num_context_masks: 1
  context_scale: [0.85, 1.0]

# Training hyperparameters
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

# Loss configuration
loss:
  type: "smoothl1"
  hierarchy_weights: [1.0, 0.5, 0.25]
  normalize_embeddings: true

# Checkpointing
checkpoint:
  save_frequency: 10
  keep_best_n: 3
  checkpoint_dir: "results/checkpoints"

# Logging
logging:
  experiment_name: "hjepa_default"
  log_dir: "results/logs"
  log_frequency: 100
  wandb:
    enabled: true
    project: "h-jepa"
    entity: null
    tags: ["baseline", "vit-base"]
  tensorboard:
    enabled: true

# Distributed training
distributed:
  enabled: false
  backend: "nccl"
  world_size: 1

# Evaluation
evaluation:
  eval_frequency: 10

# Reproducibility
seed: 42
device: "cuda"
```

## Output Structure

After training, you'll find:

```
results/
├── checkpoints/
│   ├── checkpoint_epoch_010.pth
│   ├── checkpoint_epoch_020.pth
│   ├── checkpoint_latest.pth
│   └── checkpoint_best.pth
└── logs/
    ├── tensorboard/
    │   └── events.out.tfevents...
    └── train.log
```

Each checkpoint contains:
- Model state dict (context encoder, target encoder, predictor)
- Optimizer state
- Learning rate scheduler state
- Training metadata (epoch, step, loss history)
- Configuration

## Monitoring Training

### TensorBoard

View training metrics in real-time:

```bash
tensorboard --logdir results/logs/tensorboard
```

Metrics logged:
- Training loss (total and per-hierarchy)
- Learning rate
- EMA momentum
- Gradient norms
- Representation collapse metrics
- Validation metrics

### Weights & Biases

If W&B is enabled, view your experiment at:
```
https://wandb.ai/<entity>/<project>/runs/<run_id>
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python scripts/train.py --config configs/default.yaml --batch_size 32
```

Enable gradient accumulation in config:
```yaml
training:
  accumulation_steps: 4  # Effective batch size = 32 * 4 = 128
```

### Slow Data Loading

Increase workers:
```bash
python scripts/train.py --config configs/default.yaml --num_workers 16
```

### CUDA Out of Memory in Multi-GPU

Ensure batch size is per GPU:
```yaml
data:
  batch_size: 64  # This is per GPU, so 64 * 4 = 256 total
```

### Training Collapse

Monitor collapse metrics and adjust:
- Increase EMA momentum warmup
- Reduce learning rate
- Add VICReg loss component
- Adjust hierarchy weights

## Best Practices

1. **Start Small**: Test with smaller batch size and fewer epochs first
2. **Monitor Early**: Check first few epochs for instabilities
3. **Checkpoint Often**: Save checkpoints frequently for long runs
4. **Use Validation**: Enable validation to detect overfitting
5. **Log Everything**: Keep W&B or TensorBoard enabled
6. **Document Changes**: Note config changes in experiment names
7. **Reproducibility**: Fix seeds and document environment

## Common Workflows

### Hyperparameter Search

```bash
for lr in 1e-4 5e-4 1e-3; do
    for bs in 64 128 256; do
        python scripts/train.py \
            --config configs/default.yaml \
            --lr $lr \
            --batch_size $bs \
            --output_dir experiments/search_lr${lr}_bs${bs}
    done
done
```

### Quick Validation

Test configuration with minimal training:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --epochs 2 \
    --batch_size 16 \
    --no_wandb \
    --debug
```

### Production Training

Full training with all features:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    scripts/train.py \
    --config configs/production.yaml \
    --distributed \
    --output_dir experiments/production_run_v1
```

## Next Steps

After training:

1. **Evaluate**: Use evaluation scripts to assess learned representations
2. **Linear Probe**: Train linear classifiers on frozen features
3. **Fine-tune**: Fine-tune on downstream tasks
4. **Visualize**: Analyze attention maps and learned representations
5. **Compare**: Benchmark against other self-supervised methods

## Getting Help

If you encounter issues:

1. Check the debug logs: `--debug`
2. Review configuration validation errors
3. Ensure all dependencies are installed
4. Verify dataset paths and formats
5. Check GPU memory and availability
6. Review the error traceback for specific issues

For more information, see the main README and documentation.

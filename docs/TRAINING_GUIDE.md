# H-JEPA Training Guide

Complete guide to training H-JEPA models for self-supervised visual representation learning.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Training Tips](#training-tips)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Monitoring Training](#monitoring-training)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Quick Start

### Basic Training

```bash
# Train on CIFAR-10 (quick validation)
python3.11 scripts/train.py --config configs/validation_test.yaml

# Train foundation model on multiple datasets
python3.11 scripts/train.py --config configs/foundation_model_cifar_stl.yaml

# Resume from checkpoint
python3.11 scripts/train.py --config configs/my_config.yaml --resume results/my_experiment/checkpoints/checkpoint_epoch_10.pt
```

### Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir results/my_experiment/logs/tensorboard

# Watch training logs
tail -f results/my_experiment/logs/train.log

# Monitor checkpoints and auto-explore
chmod +x scripts/watch_and_explore.sh
./scripts/watch_and_explore.sh
```

---

## Configuration

### Configuration File Structure

```yaml
experiment:
  name: my_experiment           # Experiment name
  seed: 42                       # Random seed for reproducibility
  output_dir: results/my_exp    # Output directory
  save_frequency: 5              # Save checkpoint every N epochs
  eval_frequency: 5              # Run evaluation every N epochs

model:
  encoder_type: vit_base_patch16_224  # Vision Transformer architecture
  embed_dim: 768                      # Embedding dimension
  num_hierarchies: 3                  # Number of hierarchical levels
  use_rope: true                      # Enable RoPE positional encoding
  rope_theta: 10000.0                 # RoPE base frequency
  use_flash_attention: true           # Enable Flash Attention (faster!)
  use_gradient_checkpointing: false   # Save memory (slower)
  ema:
    momentum: 0.996                   # EMA momentum for target encoder
    momentum_end: 1.0                 # Final EMA momentum
    momentum_warmup_epochs: 2         # Warmup period for EMA
  predictor:
    depth: 6                          # Predictor transformer depth
    num_heads: 6                      # Number of attention heads
    mlp_ratio: 4.0                    # MLP expansion ratio
    qkv_bias: true                    # Use bias in QKV projections
    dropout: 0.0                      # Dropout rate

data:
  use_multi_dataset: false      # Single dataset or multi-dataset
  dataset: cifar10              # Dataset name (if single)
  # OR for multi-dataset:
  # datasets:
  #   - name: cifar10
  #     weight: 1.0
  #   - name: stl10
  #     weight: 0.5
  data_path: ./data             # Path to dataset root
  image_size: 224               # Input image size
  batch_size: 32                # Batch size
  num_workers: 4                # DataLoader workers
  transforms:
    crop_scale: [0.8, 1.0]      # Random crop scale range
    horizontal_flip: true       # Random horizontal flip
    color_jitter: 0.1           # Color jitter strength

training:
  epochs: 100                   # Total training epochs
  warmup_epochs: 10             # Learning rate warmup
  optimizer: adamw              # Optimizer (adamw recommended)
  lr: 0.001                     # Peak learning rate
  min_lr: 1.0e-6                # Minimum learning rate
  weight_decay: 0.04            # Weight decay (L2 regularization)
  betas: [0.9, 0.95]            # Adam beta parameters
  lr_schedule: cosine           # LR schedule (cosine recommended)
  min_lr_ratio: 0.01            # Min LR as ratio of peak
  warmup_lr_ratio: 0.001        # Warmup start LR as ratio of peak
  use_amp: true                 # Mixed precision training

masking:
  num_target_masks: 4           # Number of target regions
  mask_scale: [0.15, 0.2]       # Target mask scale range
  aspect_ratio: [0.75, 1.5]     # Target mask aspect ratio range
  num_context_masks: 1          # Number of context masks

loss:
  type: smoothl1                # Loss type (smoothl1, mse, vicreg)
  hierarchy_weights: [1.0, 0.7, 0.5]  # Weights for each hierarchy
  use_vicreg: false             # Add VICReg regularization
  # VICReg parameters (if use_vicreg: true):
  # vicreg_sim_weight: 25.0
  # vicreg_std_weight: 25.0
  # vicreg_cov_weight: 1.0

logging:
  experiment_name: my_experiment
  log_dir: results/my_exp/logs
  use_wandb: false              # Weights & Biases logging
  use_tensorboard: true         # TensorBoard logging
  log_frequency: 10             # Log every N batches
  log_images: true              # Log image examples
  log_attention: true           # Log attention maps
  log_gradients: true           # Log gradient histograms

checkpoint:
  checkpoint_dir: results/my_exp/checkpoints
  save_frequency: 5             # Save every N epochs
  save_best: true               # Save best model
  metric: val_loss              # Metric for best model
  mode: min                     # min or max
  keep_last_k: 5                # Keep last K checkpoints

device: mps                     # Device: mps, cuda, or cpu
```

---

## Training Tips

### Dataset Selection

**Small datasets (quick experiments):**
- CIFAR-10: 50K images, 10 classes - fast iteration
- CIFAR-100: 50K images, 100 classes - more challenging
- STL-10: 100K images, 10 classes - higher resolution (96x96)

**Medium datasets:**
- ImageNet-100: 100 classes from ImageNet - good balance
- Places365-Standard (subset) - scene recognition

**Large datasets (production):**
- ImageNet-1K: 1.2M images - standard benchmark
- ImageNet-21K: 14M images - large-scale pretraining

**Multi-dataset training:**
```yaml
data:
  use_multi_dataset: true
  datasets:
    - name: cifar10
      weight: 1.0      # Full weight
    - name: stl10
      weight: 0.5      # Half weight (sampled less often)
    - name: cifar100
      weight: 0.3
```

### Batch Size Guidelines

**Memory constraints:**
- 8GB VRAM: batch_size = 16-32
- 16GB VRAM: batch_size = 64-128
- 24GB VRAM: batch_size = 128-256
- 40GB+ VRAM: batch_size = 256-512

**For small GPUs:**
- Use `use_gradient_checkpointing: true` to reduce memory
- Reduce `batch_size` and increase `epochs` proportionally
- Use `use_amp: true` for mixed precision (2x memory reduction)

### Learning Rate Scaling

**Rule of thumb:** Scale learning rate linearly with batch size

```
effective_lr = base_lr * (batch_size / 256)
```

**Example:**
- Base config: batch_size=256, lr=0.001
- Your config: batch_size=64, lr=0.00025 (0.001 * 64/256)

### Training Duration

**Minimum training time by dataset:**
- CIFAR-10: 50-100 epochs (~1-2 hours on M1 Max)
- CIFAR-100: 100-200 epochs
- STL-10: 100-200 epochs
- ImageNet-100: 200-400 epochs
- ImageNet-1K: 400-800 epochs

**Diminishing returns:**
Most improvement happens in the first 50% of training. Beyond that, gains are incremental.

---

## Hyperparameter Tuning

### Priority Order

**High impact (tune first):**
1. **Learning rate** - Most important! Try: 1e-4, 3e-4, 1e-3, 3e-3
2. **Batch size** - Affects training dynamics
3. **Weight decay** - Prevents overfitting, try: 0.01, 0.04, 0.1
4. **EMA momentum** - Target encoder update rate, try: 0.99, 0.996, 0.999

**Medium impact:**
5. **Masking strategy** - Number and size of masks
6. **Hierarchy weights** - Balance between levels
7. **Warmup epochs** - Stabilizes early training

**Low impact (use defaults):**
8. Predictor architecture
9. Color jitter strength
10. Crop scale range

### Hyperparameter Ranges

```yaml
# Safe ranges for tuning
training:
  lr: [1e-4, 1e-3, 3e-3]           # Learning rate
  weight_decay: [0.01, 0.04, 0.1]  # Regularization
  batch_size: [32, 64, 128, 256]   # Hardware dependent

model:
  ema:
    momentum: [0.99, 0.996, 0.999]  # EMA momentum

masking:
  num_target_masks: [2, 4, 6]       # Number of targets
  mask_scale: [[0.1, 0.15], [0.15, 0.2], [0.2, 0.3]]
```

### Grid Search Example

```bash
# Simple grid search over LR and weight decay
for lr in 1e-4 3e-4 1e-3; do
  for wd in 0.01 0.04 0.1; do
    python3.11 scripts/train.py \
      --config configs/base.yaml \
      --lr $lr \
      --weight-decay $wd \
      --output-dir results/sweep_lr${lr}_wd${wd}
  done
done
```

---

## Monitoring Training

### TensorBoard Metrics

**Loss curves:**
- `train/total_loss` - Overall training loss (should decrease)
- `train/hierarchy_0_loss`, `train/hierarchy_1_loss`, etc. - Per-hierarchy losses
- `val/loss` - Validation loss (should track training)

**Learning rate:**
- `train/lr` - Current learning rate (should follow schedule)
- `train/ema_momentum` - Current EMA momentum (ramps up during warmup)

**Gradient statistics:**
- `gradients/*` - Gradient norms by layer
- Watch for exploding (>100) or vanishing (<1e-5) gradients

**Model statistics:**
- `embeddings/*` - Embedding norms and distributions
- `attention/*` - Attention patterns (if logged)

### Good Training Signs

**Healthy training:**
- Loss decreases steadily
- Validation loss tracks training loss (no large gap)
- Gradients are stable (norm around 1-10)
- All hierarchy losses decrease

**Warning signs:**
- Validation loss diverges from training (overfitting)
- Loss plateaus early (learning rate too low)
- Loss oscillates wildly (learning rate too high or batch size too small)
- Gradients explode (>1000) or vanish (<1e-6)

### Example Training Curve

```
Epoch   Train Loss   Val Loss    LR         Time
-----   ----------   --------    --------   -----
1       0.0156       0.0158      0.0001     45s
5       0.0089       0.0092      0.0005     44s
10      0.0052       0.0055      0.0010     43s  <- Peak LR
20      0.0031       0.0034      0.0008     43s
50      0.0012       0.0015      0.0003     43s
100     0.0008       0.0011      0.0001     43s  <- Should converge
```

---

## Troubleshooting

### Problem: Loss is NaN

**Causes:**
- Learning rate too high
- Numerical instability
- Bad initialization

**Solutions:**
```yaml
training:
  lr: 1e-4              # Reduce learning rate
  use_amp: false        # Disable mixed precision temporarily
  clip_grad_norm: 1.0   # Add gradient clipping
```

### Problem: Loss not decreasing

**Causes:**
- Learning rate too low
- Model too small
- Not enough training

**Solutions:**
```yaml
training:
  lr: 3e-3              # Increase learning rate
  epochs: 200           # Train longer

model:
  encoder_type: vit_large_patch16_224  # Use larger model
```

### Problem: Out of memory

**Solutions:**
```yaml
data:
  batch_size: 16        # Reduce batch size

model:
  use_gradient_checkpointing: true  # Enable checkpointing

training:
  use_amp: true         # Enable mixed precision
```

### Problem: Training too slow

**Solutions:**
```yaml
model:
  use_flash_attention: true     # Enable Flash Attention (2-3x faster)

data:
  num_workers: 8                # Increase data loading threads

training:
  use_amp: true                 # Mixed precision (faster)
```

### Problem: Overfitting

**Symptoms:**
- Large gap between train and validation loss
- Validation loss increases while training decreases

**Solutions:**
```yaml
training:
  weight_decay: 0.1             # Increase regularization

data:
  transforms:
    color_jitter: 0.2           # Stronger augmentation
    crop_scale: [0.7, 1.0]      # More aggressive crops

masking:
  num_target_masks: 6           # More challenging task
```

---

## Best Practices

### 1. Start Small

Always validate your setup with a quick run:

```yaml
# Quick validation config
training:
  epochs: 10

data:
  dataset: cifar10
  batch_size: 32
```

Run for 10 epochs to ensure:
- Code runs without errors
- Loss decreases
- Checkpoints save correctly
- Logging works

### 2. Use Checkpointing

Always save checkpoints frequently:

```yaml
checkpoint:
  save_frequency: 5      # Don't make this too large!
  save_best: true        # Keep best model
  keep_last_k: 3         # Keep recent checkpoints
```

### 3. Monitor Everything

Enable comprehensive logging:

```yaml
logging:
  use_tensorboard: true
  log_frequency: 10
  log_images: true
  log_attention: true
  log_gradients: true
```

### 4. Reproducibility

Always set seeds:

```yaml
experiment:
  seed: 42
```

And document:
- Exact config file used
- Checkpoint used (if resuming)
- Hardware (GPU model, memory)
- Software versions (PyTorch, CUDA)

### 5. Evaluation

Don't just train - evaluate!

```bash
# After training, evaluate representations
python3.11 scripts/eval_linear_probe.py \
  --checkpoint results/my_exp/checkpoints/best_model.pt \
  --dataset cifar10

python3.11 scripts/eval_knn.py \
  --checkpoint results/my_exp/checkpoints/best_model.pt \
  --dataset cifar10
```

### 6. Version Control

Use git to track:
- Config files
- Training scripts
- Results summaries

```bash
git add configs/my_experiment.yaml
git commit -m "Add config for my_experiment"

# After training
git add results/my_experiment/summary.json
git commit -m "Add results for my_experiment"
```

---

## Common Training Recipes

### Recipe 1: Quick Validation (1-2 hours)

```yaml
experiment:
  name: quick_validation

model:
  encoder_type: vit_base_patch16_224
  use_flash_attention: true

data:
  dataset: cifar10
  batch_size: 64

training:
  epochs: 50
  lr: 0.001
```

### Recipe 2: Production Model (overnight)

```yaml
experiment:
  name: production_v1

model:
  encoder_type: vit_large_patch16_224
  use_flash_attention: true
  num_hierarchies: 3

data:
  use_multi_dataset: true
  datasets:
    - name: cifar10
      weight: 1.0
    - name: stl10
      weight: 0.5
  batch_size: 128

training:
  epochs: 200
  lr: 0.003
  weight_decay: 0.04
```

### Recipe 3: Large-Scale (multi-day)

```yaml
experiment:
  name: imagenet_pretrain

model:
  encoder_type: vit_huge_patch14_224
  use_flash_attention: true
  use_gradient_checkpointing: true

data:
  dataset: imagenet
  batch_size: 256
  num_workers: 16

training:
  epochs: 800
  lr: 0.004
  warmup_epochs: 40
```

---

## FAQ

**Q: How long should I train?**
A: Until validation loss plateaus. Typically 50-200 epochs for small datasets, 400-800 for ImageNet.

**Q: Should I use Flash Attention?**
A: Yes! It's 2-3x faster with identical results. Only disable if you need to extract attention weights.

**Q: What's a good validation accuracy?**
A: This depends on your evaluation protocol. For CIFAR-10 linear probing, expect 70-85% after good pretraining.

**Q: Can I train on CPU?**
A: Yes, but it will be very slow. GPU/MPS is highly recommended.

**Q: How do I know if my hyperparameters are good?**
A: Compare to validation run. If loss is similar after 10 epochs, your settings are probably fine.

**Q: Should I use multi-dataset training?**
A: Yes for production models (better generalization), no for quick experiments (slower).

---

## Next Steps

After successful training:

1. **Evaluate** - Run linear probing and k-NN evaluation
2. **Visualize** - Explore learned features with visualization scripts
3. **Transfer** - Test on downstream tasks
4. **Share** - Document results and share checkpoints

**Further Reading:**
- `docs/ARCHITECTURE.md` - Understanding H-JEPA architecture
- `docs/EVALUATION.md` - Comprehensive evaluation guide
- `notebooks/explore_hjepa.ipynb` - Interactive exploration

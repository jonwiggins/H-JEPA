# H-JEPA Training - Quick Start

## Installation

```bash
# Clone repository (if not already done)
cd H-JEPA

# Install dependencies
pip install -r requirements.txt
```

## 5-Minute Quick Start

### 1. Test with CIFAR-10 (No dataset download needed)

```bash
# CIFAR-10 will auto-download
python scripts/train.py \
    --config configs/default.yaml \
    --data_path ./data \
    --epochs 5 \
    --batch_size 32 \
    --no_wandb
```

This will:
- Download CIFAR-10 automatically
- Train for 5 epochs (quick test)
- Use small batch size
- Skip W&B logging

Expected time: ~5-10 minutes on GPU

### 2. Full Training on ImageNet

```bash
# Make sure ImageNet is at /path/to/imagenet
python scripts/train.py \
    --config configs/default.yaml \
    --data_path /path/to/imagenet
```

This will:
- Train for 300 epochs (config default)
- Use batch size 128 (config default)
- Save checkpoints every 10 epochs
- Log to W&B and TensorBoard

Expected time: Several days on 8 GPUs

## Common Commands

### View Help

```bash
python scripts/train.py --help
```

### View Example Usage

```bash
./scripts/example_usage.sh
```

### Test Configuration

```bash
# Validate config file
python -c "import yaml; print(yaml.safe_load(open('configs/default.yaml')))"
```

## Most Used Commands

### Quick Test Run

```bash
python scripts/train.py --config configs/default.yaml --epochs 2 --batch_size 16 --no_wandb
```

### Resume Training

```bash
python scripts/train.py --config configs/default.yaml --resume results/checkpoints/checkpoint_latest.pth
```

### Multi-GPU (4 GPUs)

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --distributed
```

### Custom Learning Rate

```bash
python scripts/train.py --config configs/default.yaml --lr 1e-4 --epochs 100
```

### Different Dataset Path

```bash
python scripts/train.py --config configs/default.yaml --data_path /my/data/path
```

## Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir results/logs/tensorboard

# Open browser to http://localhost:6006
```

### Watch GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Monitor Training Log

```bash
tail -f results/logs/train.log
```

## File Structure

```
H-JEPA/
├── scripts/
│   ├── train.py              # Main training script
│   ├── TRAINING_GUIDE.md     # Detailed documentation
│   ├── QUICK_START.md        # This file
│   └── example_usage.sh      # Example commands
├── configs/
│   └── default.yaml          # Configuration file
├── src/                      # Source code
│   ├── models/              # H-JEPA model
│   ├── losses/              # Loss functions
│   ├── masks/               # Masking strategies
│   ├── data/                # Datasets
│   ├── trainers/            # Training loop
│   └── utils/               # Utilities
└── results/                 # Created during training
    ├── checkpoints/         # Model checkpoints
    └── logs/                # Training logs
```

## Troubleshooting

### "Module not found" error

```bash
pip install -r requirements.txt
```

### "CUDA out of memory"

```bash
# Reduce batch size
python scripts/train.py --config configs/default.yaml --batch_size 32
```

### "Dataset not found"

```bash
# Specify correct path
python scripts/train.py --config configs/default.yaml --data_path /correct/path
```

### Training is slow

```bash
# Increase workers
python scripts/train.py --config configs/default.yaml --num_workers 16
```

## Configuration

Edit `configs/default.yaml` to change:

```yaml
# Key settings to adjust:

data:
  dataset: "imagenet"        # Dataset name
  data_path: "/path/to/data" # Dataset location
  batch_size: 128            # Batch size per GPU

training:
  epochs: 300                # Total epochs
  lr: 1.5e-4                 # Learning rate

model:
  encoder_type: "vit_base_patch16_224"  # Model architecture
  num_hierarchies: 3         # Hierarchy levels
```

## Next Steps

1. **Read Full Guide**: See `scripts/TRAINING_GUIDE.md` for detailed documentation
2. **Customize Config**: Edit `configs/default.yaml` for your needs
3. **Run Training**: Start with a quick test, then full training
4. **Monitor Progress**: Use TensorBoard or W&B
5. **Evaluate**: Use trained model for downstream tasks

## Support

- **Full Documentation**: `scripts/TRAINING_GUIDE.md`
- **Usage Examples**: `scripts/example_usage.sh`
- **Implementation Summary**: `TRAINING_SCRIPT_SUMMARY.md`

## Quick Reference

| Task | Command |
|------|---------|
| Basic training | `python scripts/train.py --config configs/default.yaml` |
| Quick test | `python scripts/train.py --config configs/default.yaml --epochs 2 --no_wandb` |
| Resume | `python scripts/train.py --config configs/default.yaml --resume results/checkpoints/latest.pth` |
| Multi-GPU | `torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --distributed` |
| Custom LR | `python scripts/train.py --config configs/default.yaml --lr 1e-4` |
| Debug mode | `python scripts/train.py --config configs/default.yaml --debug` |
| View help | `python scripts/train.py --help` |
| TensorBoard | `tensorboard --logdir results/logs/tensorboard` |

---

Ready to train? Start with:

```bash
python scripts/train.py --config configs/default.yaml --epochs 5 --batch_size 32 --no_wandb
```

Good luck with your H-JEPA training!

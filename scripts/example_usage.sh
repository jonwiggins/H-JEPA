#!/bin/bash
# Example usage patterns for H-JEPA training script
# Make executable with: chmod +x scripts/example_usage.sh

set -e  # Exit on error

echo "H-JEPA Training Examples"
echo "========================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Example 1: Basic training
echo -e "${GREEN}Example 1: Basic Training${NC}"
echo "python scripts/train.py --config configs/default.yaml"
echo ""

# Example 2: Quick test run
echo -e "${GREEN}Example 2: Quick Test (10 epochs, small batch)${NC}"
echo "python scripts/train.py \\"
echo "    --config configs/default.yaml \\"
echo "    --epochs 10 \\"
echo "    --batch_size 32 \\"
echo "    --no_wandb"
echo ""

# Example 3: Resume training
echo -e "${GREEN}Example 3: Resume from Checkpoint${NC}"
echo "python scripts/train.py \\"
echo "    --config configs/default.yaml \\"
echo "    --resume results/checkpoints/checkpoint_epoch_100.pth"
echo ""

# Example 4: Override multiple parameters
echo -e "${GREEN}Example 4: Custom Configuration${NC}"
echo "python scripts/train.py \\"
echo "    --config configs/default.yaml \\"
echo "    --data_path /data/imagenet \\"
echo "    --batch_size 64 \\"
echo "    --lr 1e-4 \\"
echo "    --epochs 200 \\"
echo "    --warmup_epochs 20 \\"
echo "    --output_dir experiments/custom_run"
echo ""

# Example 5: Multi-GPU training (torch.distributed.launch)
echo -e "${GREEN}Example 5: Multi-GPU Training (4 GPUs)${NC}"
echo "python -m torch.distributed.launch \\"
echo "    --nproc_per_node=4 \\"
echo "    --use_env \\"
echo "    scripts/train.py \\"
echo "    --config configs/default.yaml \\"
echo "    --distributed"
echo ""

# Example 6: Multi-GPU with torchrun
echo -e "${GREEN}Example 6: Multi-GPU with torchrun (PyTorch >= 1.10)${NC}"
echo "torchrun --nproc_per_node=4 \\"
echo "    scripts/train.py \\"
echo "    --config configs/default.yaml \\"
echo "    --distributed"
echo ""

# Example 7: Specific GPU
echo -e "${GREEN}Example 7: Train on Specific GPU${NC}"
echo "python scripts/train.py \\"
echo "    --config configs/default.yaml \\"
echo "    --device cuda:1"
echo ""

# Example 8: Debug mode
echo -e "${GREEN}Example 8: Debug Mode${NC}"
echo "python scripts/train.py \\"
echo "    --config configs/default.yaml \\"
echo "    --debug \\"
echo "    --epochs 2 \\"
echo "    --batch_size 16"
echo ""

# Example 9: CIFAR-10 training (small dataset)
echo -e "${GREEN}Example 9: Train on CIFAR-10${NC}"
echo "# First, update config or create new config for CIFAR-10"
echo "python scripts/train.py \\"
echo "    --config configs/cifar10.yaml \\"
echo "    --data_path ./data \\"
echo "    --batch_size 256"
echo ""

# Example 10: Hyperparameter sweep
echo -e "${GREEN}Example 10: Hyperparameter Sweep${NC}"
echo "for lr in 1e-4 5e-4 1e-3; do"
echo "    for wd in 0.01 0.05 0.1; do"
echo "        python scripts/train.py \\"
echo "            --config configs/default.yaml \\"
echo "            --lr \$lr \\"
echo "            --weight_decay \$wd \\"
echo "            --epochs 50 \\"
echo "            --output_dir experiments/sweep_lr\${lr}_wd\${wd}"
echo "    done"
echo "done"
echo ""

# Monitoring examples
echo -e "${BLUE}Monitoring Training:${NC}"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir results/logs/tensorboard --port 6006"
echo ""
echo "Watch GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Monitor training log:"
echo "  tail -f results/logs/train.log"
echo ""

# Useful tips
echo -e "${BLUE}Useful Tips:${NC}"
echo ""
echo "1. Check available configurations:"
echo "   ls configs/"
echo ""
echo "2. Validate config without training:"
echo "   python -c 'import yaml; yaml.safe_load(open(\"configs/default.yaml\"))'"
echo ""
echo "3. Check dataset structure:"
echo "   tree -L 2 /path/to/dataset"
echo ""
echo "4. Monitor disk space during training:"
echo "   df -h results/"
echo ""
echo "5. Count parameters in model:"
echo "   # This is logged automatically when training starts"
echo ""

echo "For more details, see scripts/TRAINING_GUIDE.md"

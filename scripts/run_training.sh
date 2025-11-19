#!/bin/bash
# Quick Start Script for H-JEPA CPU Training
# This script automates the setup and execution of the training plan
# See TRAINING_PLAN.md for detailed documentation

set -e  # Exit on error

echo "=========================================="
echo "H-JEPA CPU Training Quick Start"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Step 1: Check system requirements
echo "Step 1: Checking system requirements..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Python version: $PYTHON_VERSION"

# Check available RAM
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
FREE_RAM=$(free -g | awk '/^Mem:/{print $7}')
print_status "RAM: ${FREE_RAM}GB free / ${TOTAL_RAM}GB total"

if [ "$FREE_RAM" -lt 8 ]; then
    print_warning "Less than 8GB RAM available. Consider closing other applications."
fi

# Check disk space
DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
print_status "Disk space available: $DISK_SPACE"

# Check GPU (expected to be none)
if command -v nvidia-smi &> /dev/null; then
    print_warning "GPU detected. Consider using GPU config for faster training."
else
    print_status "CPU-only mode (as expected)"
fi

echo ""

# Step 2: Install dependencies
echo "Step 2: Installing dependencies..."

if [ -f "requirements.txt" ]; then
    print_status "Installing Python packages..."
    pip install -q -r requirements.txt
    print_status "Dependencies installed"
else
    print_error "requirements.txt not found!"
    exit 1
fi

echo ""

# Step 3: Download CIFAR-10
echo "Step 3: Preparing CIFAR-10 dataset..."

if [ -d "data/cifar10/cifar-10-batches-py" ]; then
    print_status "CIFAR-10 already downloaded"
else
    print_status "Downloading CIFAR-10 (~170MB)..."
    python3 -c "from torchvision import datasets; datasets.CIFAR10('./data/cifar10', download=True, train=True); datasets.CIFAR10('./data/cifar10', download=True, train=False)"
    print_status "CIFAR-10 downloaded successfully"
fi

echo ""

# Step 4: Verify configuration
echo "Step 4: Verifying configuration..."

if [ -f "configs/cpu_cifar10.yaml" ]; then
    print_status "Configuration file found: configs/cpu_cifar10.yaml"
else
    print_error "Configuration file not found!"
    print_error "Please ensure configs/cpu_cifar10.yaml exists"
    exit 1
fi

echo ""

# Step 5: Create output directories
echo "Step 5: Creating output directories..."

mkdir -p results/checkpoints/cpu_cifar10
mkdir -p results/logs/cpu_cifar10
mkdir -p backups

print_status "Output directories created"

echo ""

# Step 6: Run test
echo "Step 6: Running quick test (1 batch)..."

print_status "Testing model initialization and forward pass..."
python3 -c "
import torch
import sys
sys.path.insert(0, '.')
from src.models.hjepa import create_hjepa
from src.data import build_dataset, build_dataloader

# Test model creation
print('Creating model...')
model = create_hjepa(
    encoder_type='vit_tiny_patch16_224',
    embed_dim=192,
    num_hierarchies=2,
    predictor_depth=2,
    predictor_num_heads=3,
    predictor_mlp_ratio=4.0,
    ema_momentum=0.996
)
print(f'Model created: {sum(p.numel() for p in model.parameters()):,} parameters')

# Test data loading
print('Loading test batch...')
dataset = build_dataset('cifar10', './data/cifar10', 'train', download=False)
loader = build_dataloader(dataset, batch_size=2, num_workers=0, shuffle=False)
batch = next(iter(loader))
print(f'Test batch loaded: {batch[0].shape}')

# Test forward pass
print('Running forward pass...')
with torch.no_grad():
    context_mask = torch.ones(2, 196).bool()
    context_mask[:, :150] = False  # Mask 150 patches
    target_mask = torch.ones(2, 196).bool()
    target_mask[:, 150:170] = False  # Predict 20 patches

    outputs = model(batch[0], context_mask, target_mask)
    print(f'Forward pass successful')

print('✓ All tests passed!')
"

if [ $? -eq 0 ]; then
    print_status "Test completed successfully"
else
    print_error "Test failed! Please check your installation"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start training (recommended in screen/tmux):"
echo "   $ screen -S hjepa_training"
echo "   $ python scripts/train.py --config configs/cpu_cifar10.yaml --device cpu"
echo "   $ # Press Ctrl+A, then D to detach"
echo ""
echo "2. Monitor progress (in another terminal):"
echo "   $ tail -f results/logs/cpu_cifar10/training.log"
echo ""
echo "3. View TensorBoard:"
echo "   $ tensorboard --logdir results/logs/cpu_cifar10 --port 6006"
echo ""
echo "4. Reattach to training session:"
echo "   $ screen -r hjepa_training"
echo ""
echo "Expected training time: 18-24 hours"
echo "See TRAINING_PLAN.md for detailed information"
echo ""

#!/bin/bash
# Safe training script for Apple Silicon (MPS) with memory management

# Note: PYTORCH_MPS_HIGH_WATERMARK_RATIO causes issues with PyTorch MPS backend
# Leaving it unset and relying on smaller batch sizes for memory management
# export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5

# Disable MPS fallback to CPU for unsupported ops (can cause memory issues)
export PYTORCH_ENABLE_MPS_FALLBACK=0

# Enable memory debugging
export PYTORCH_MPS_ALLOCATOR_POLICY=default

echo "Starting training with MPS memory safety settings..."
echo "Memory management: Using small batch size (4) and periodic GC"
echo "PYTORCH_ENABLE_MPS_FALLBACK=0 (no CPU fallback)"
echo ""

# Parse command line arguments
CONFIG_FILE=${1:-configs/debug_minimal.yaml}

echo "Using config: $CONFIG_FILE"
echo ""

# Run training with Python 3.11
python3.11 scripts/train.py --config "$CONFIG_FILE"

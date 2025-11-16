#!/bin/bash
# Download ImageNet-100 for H-JEPA training
#
# ImageNet-100 is a subset of ImageNet-1K with 100 classes
# Size: ~15GB, 126,689 images
# Much better than CIFAR-10 for hierarchical learning

set -e

echo "======================================================================"
echo "ImageNet-100 Download for H-JEPA"
echo "======================================================================"
echo ""
echo "Dataset: ImageNet-100"
echo "Size: ~15GB"
echo "Images: 126,689 training images"
echo "Resolution: Variable (will be resized to 224x224)"
echo "Classes: 100"
echo ""

# Check if data directory exists
DATA_DIR="./data"
mkdir -p "$DATA_DIR"

echo "📁 Data directory: $DATA_DIR"
echo ""

# Use the built-in download script
echo "🌐 Downloading ImageNet-100..."
echo ""
echo "Note: This will download ImageNet-100 automatically."
echo "If you prefer to download manually, press Ctrl+C now."
echo ""
read -p "Press Enter to continue with automatic download..."

python3.11 scripts/download_data.py \
    --dataset imagenet100 \
    --data-path "$DATA_DIR"

echo ""
echo "======================================================================"
echo "✅ ImageNet-100 Download Complete!"
echo "======================================================================"
echo ""
echo "Dataset location: $DATA_DIR/imagenet100/"
echo ""
echo "Next steps:"
echo "1. Run validation to test the dataset:"
echo "   python3.11 scripts/train.py --config configs/m1_max_imagenet100_100epoch.yaml --epochs 1"
echo ""
echo "2. Start full training:"
echo "   python3.11 scripts/train.py --config configs/m1_max_imagenet100_100epoch.yaml"
echo ""
echo "Expected training time: ~10-15 hours for 100 epochs on M1 Max"
echo "Expected results: 60-70% linear probe accuracy"
echo ""

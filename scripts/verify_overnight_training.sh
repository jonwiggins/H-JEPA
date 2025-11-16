#!/bin/bash
# Overnight Training Verification Script
# Run this before starting your overnight training to ensure everything is ready

set -e  # Exit on error

echo "================================================"
echo "H-JEPA Overnight Training Pre-Flight Check"
echo "================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PASSED=0
WARNINGS=0
FAILED=0

# Function to print status
print_status() {
    local status=$1
    local message=$2

    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $message"
        ((PASSED++))
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
        ((WARNINGS++))
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}✗${NC} $message"
        ((FAILED++))
    else
        echo "  $message"
    fi
}

echo "1. CHECKING CONFIGURATION FILES"
echo "--------------------------------"

# Check conservative config
if [ -f "configs/overnight_training_conservative.yaml" ]; then
    print_status "PASS" "Conservative config exists"

    # Check key settings
    if grep -q "use_flash_attention: true" configs/overnight_training_conservative.yaml; then
        print_status "PASS" "  Flash Attention enabled"
    else
        print_status "WARN" "  Flash Attention not enabled"
    fi

    if grep -q "use_layerscale: true" configs/overnight_training_conservative.yaml; then
        print_status "PASS" "  LayerScale enabled"
    else
        print_status "WARN" "  LayerScale not enabled"
    fi
else
    print_status "FAIL" "Conservative config missing"
fi

# Check aggressive config
if [ -f "configs/overnight_training_aggressive.yaml" ]; then
    print_status "PASS" "Aggressive config exists"

    if grep -q "imagenet100" configs/overnight_training_aggressive.yaml; then
        print_status "PASS" "  ImageNet-100 configured"
    else
        print_status "WARN" "  ImageNet-100 not configured"
    fi
else
    print_status "FAIL" "Aggressive config missing"
fi

echo ""
echo "2. CHECKING DOCUMENTATION"
echo "-------------------------"

if [ -f "OVERNIGHT_TRAINING_GUIDE.md" ]; then
    print_status "PASS" "Main guide exists ($(wc -c < OVERNIGHT_TRAINING_GUIDE.md) bytes)"
else
    print_status "WARN" "Main guide missing"
fi

if [ -f "OVERNIGHT_TRAINING_QUICKREF.md" ]; then
    print_status "PASS" "Quick reference exists"
else
    print_status "WARN" "Quick reference missing"
fi

if [ -f "OVERNIGHT_TRAINING_RECOMMENDATION.md" ]; then
    print_status "PASS" "Recommendation guide exists"
else
    print_status "WARN" "Recommendation guide missing"
fi

echo ""
echo "3. CHECKING SYSTEM RESOURCES"
echo "-----------------------------"

# Check disk space
FREE_SPACE=$(df -h . | tail -1 | awk '{print $4}')
FREE_SPACE_GB=$(df -k . | tail -1 | awk '{print int($4/1024/1024)}')
if [ "$FREE_SPACE_GB" -gt 10 ]; then
    print_status "PASS" "Disk space: $FREE_SPACE free (>10GB required)"
else
    print_status "FAIL" "Disk space: $FREE_SPACE free (need >10GB)"
fi

# Check if running on macOS (for memory check)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Get total RAM on macOS
    TOTAL_RAM_GB=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}')
    print_status "PASS" "Total RAM: ${TOTAL_RAM_GB}GB"

    if [ "$TOTAL_RAM_GB" -ge 32 ]; then
        print_status "PASS" "  RAM sufficient for aggressive config (32GB+)"
    elif [ "$TOTAL_RAM_GB" -ge 16 ]; then
        print_status "WARN" "  RAM marginal for aggressive config (16-32GB)"
        print_status "WARN" "  Recommend conservative config or reduce batch size"
    else
        print_status "FAIL" "  RAM insufficient (<16GB)"
    fi
else
    print_status "WARN" "Not macOS - cannot verify RAM automatically"
fi

echo ""
echo "4. CHECKING PYTHON ENVIRONMENT"
echo "-------------------------------"

# Check if Python is available
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_status "PASS" "Python found: $PYTHON_VERSION"

    # Check PyTorch
    if python -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        print_status "PASS" "PyTorch installed: $TORCH_VERSION"

        # Check MPS availability
        if python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
            print_status "PASS" "  MPS backend available"
        else
            print_status "WARN" "  MPS backend not available"
            print_status "WARN" "  Will fall back to CPU (much slower)"
        fi
    else
        print_status "FAIL" "PyTorch not installed"
    fi

    # Check other dependencies
    DEPS=("yaml" "timm" "einops" "numpy")
    for dep in "${DEPS[@]}"; do
        if python -c "import $dep" 2>/dev/null; then
            print_status "PASS" "  $dep installed"
        else
            print_status "FAIL" "  $dep not installed"
        fi
    done
else
    print_status "FAIL" "Python not found in PATH"
fi

echo ""
echo "5. CHECKING DATASETS"
echo "--------------------"

# Check CIFAR-10 (for conservative)
if [ -d "data/cifar-10-batches-py" ] || [ -d "data/cifar10" ]; then
    print_status "PASS" "CIFAR-10 dataset found"
else
    print_status "WARN" "CIFAR-10 not found (will auto-download)"
fi

# Check STL-10 (for conservative)
if [ -d "data/stl10_binary" ] || [ -d "data/stl10" ]; then
    print_status "PASS" "STL-10 dataset found"
else
    print_status "WARN" "STL-10 not found (will auto-download)"
fi

# Check ImageNet-100 (for aggressive)
if [ -d "data/imagenet/train" ]; then
    NUM_CLASSES=$(ls -d data/imagenet/train/* 2>/dev/null | wc -l)
    NUM_IMAGES=$(find data/imagenet/train -name "*.JPEG" 2>/dev/null | wc -l)

    if [ "$NUM_CLASSES" -ge 90 ]; then
        print_status "PASS" "ImageNet directory found with $NUM_CLASSES classes"

        if [ "$NUM_IMAGES" -ge 100000 ]; then
            print_status "PASS" "  $NUM_IMAGES images found (sufficient for ImageNet-100)"
        else
            print_status "WARN" "  Only $NUM_IMAGES images found (expected ~127K)"
        fi
    else
        print_status "WARN" "ImageNet directory found but only $NUM_CLASSES classes"
    fi
else
    print_status "WARN" "ImageNet not found (required for aggressive config)"
    print_status "WARN" "  Use conservative config or download ImageNet-100"
fi

echo ""
echo "6. CHECKING OUTPUT DIRECTORIES"
echo "-------------------------------"

# Create output directories if they don't exist
mkdir -p results/overnight_conservative/{checkpoints,logs,visualizations}
mkdir -p results/overnight_aggressive/{checkpoints,logs,visualizations}

if [ -d "results/overnight_conservative" ]; then
    print_status "PASS" "Conservative output directory ready"
else
    print_status "FAIL" "Could not create conservative output directory"
fi

if [ -d "results/overnight_aggressive" ]; then
    print_status "PASS" "Aggressive output directory ready"
else
    print_status "FAIL" "Could not create aggressive output directory"
fi

echo ""
echo "7. CHECKING TRAINING SCRIPT"
echo "---------------------------"

if [ -f "scripts/train.py" ]; then
    print_status "PASS" "Training script exists"

    # Quick syntax check
    if python -m py_compile scripts/train.py 2>/dev/null; then
        print_status "PASS" "  Python syntax valid"
    else
        print_status "FAIL" "  Python syntax error in train.py"
    fi
else
    print_status "FAIL" "Training script not found"
fi

echo ""
echo "================================================"
echo "VERIFICATION SUMMARY"
echo "================================================"
echo ""
echo -e "${GREEN}Passed:${NC}   $PASSED checks"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS checks"
echo -e "${RED}Failed:${NC}   $FAILED checks"
echo ""

# Final recommendation
if [ "$FAILED" -eq 0 ]; then
    if [ "$WARNINGS" -eq 0 ]; then
        echo -e "${GREEN}✓ READY TO START TRAINING${NC}"
        echo ""
        echo "Recommended next step:"
        echo "  screen -S hjepa"
        echo "  python scripts/train.py --config configs/overnight_training_conservative.yaml --device mps"
        echo ""
    else
        echo -e "${YELLOW}⚠ MOSTLY READY - CHECK WARNINGS${NC}"
        echo ""
        echo "Review warnings above. Most are non-critical."
        echo "Conservative config should work if warnings are only about:"
        echo "  - ImageNet-100 (not needed for conservative)"
        echo "  - Dataset auto-download (will happen on first run)"
        echo ""
        echo "Recommended next step:"
        echo "  Review warnings, then start conservative training:"
        echo "  screen -S hjepa"
        echo "  python scripts/train.py --config configs/overnight_training_conservative.yaml --device mps"
        echo ""
    fi
else
    echo -e "${RED}✗ NOT READY - FIX FAILURES FIRST${NC}"
    echo ""
    echo "Critical issues found. Fix the failures above before starting."
    echo ""
    echo "Common fixes:"
    echo "  - Install dependencies: pip install -r requirements.txt"
    echo "  - Free up disk space"
    echo "  - Check Python/PyTorch installation"
    echo ""
    exit 1
fi

echo "For detailed information, see:"
echo "  - OVERNIGHT_TRAINING_GUIDE.md (complete guide)"
echo "  - OVERNIGHT_TRAINING_QUICKREF.md (quick reference)"
echo "  - OVERNIGHT_TRAINING_RECOMMENDATION.md (decision guide)"
echo ""
echo "Good luck! 🚀"

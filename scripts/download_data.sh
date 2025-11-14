#!/bin/bash
################################################################################
# Dataset Download Script for H-JEPA
#
# This script automates dataset downloading and preparation for H-JEPA training.
# It handles automatic downloads where possible and provides instructions for
# manual downloads when required.
#
# Usage:
#   ./scripts/download_data.sh [options] [dataset1] [dataset2] ...
#
# Examples:
#   ./scripts/download_data.sh                    # Show dataset summary
#   ./scripts/download_data.sh cifar10 cifar100   # Download CIFAR datasets
#   ./scripts/download_data.sh --all-auto         # Download all auto-downloadable datasets
#   ./scripts/download_data.sh --verify           # Verify existing datasets
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DATA_PATH="${DATA_PATH:-./data}"
VERIFY_AFTER_DOWNLOAD=true
FORCE_DOWNLOAD=false
VERIFY_ONLY=false

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

show_usage() {
    cat << EOF
Usage: $0 [options] [datasets...]

Download and prepare datasets for H-JEPA training.

DATASETS:
  cifar10       CIFAR-10 (170 MB, auto-download)
  cifar100      CIFAR-100 (170 MB, auto-download)
  stl10         STL-10 (2.5 GB, auto-download)
  imagenet      ImageNet ILSVRC2012 (150 GB, manual download required)
  imagenet100   ImageNet-100 subset (15 GB, manual download required)

  If no datasets specified, shows summary of all datasets.

OPTIONS:
  -h, --help              Show this help message
  -d, --data-path PATH    Set data directory (default: ./data)
  -a, --all-auto          Download all auto-downloadable datasets
  -v, --verify            Verify existing datasets only (no download)
  -f, --force             Force re-download even if exists
  -n, --no-verify         Skip verification after download

ENVIRONMENT VARIABLES:
  DATA_PATH               Default data directory (default: ./data)

EXAMPLES:
  # Show summary of available datasets
  $0

  # Download CIFAR-10 and CIFAR-100
  $0 cifar10 cifar100

  # Download all auto-downloadable datasets
  $0 --all-auto

  # Download to custom location
  $0 --data-path /mnt/datasets cifar10

  # Verify existing datasets
  $0 --verify cifar10 cifar100

  # Force re-download
  $0 --force cifar10

EOF
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not found"
        echo "Please install Python 3 and required packages:"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
}

check_dependencies() {
    print_info "Checking dependencies..."

    # Check Python
    check_python

    # Check if we can import required modules
    python3 -c "import torch; import torchvision; import tqdm" 2>/dev/null || {
        print_error "Required Python packages not found"
        echo ""
        echo "Please install required packages:"
        echo "  cd $PROJECT_ROOT"
        echo "  pip install -r requirements.txt"
        echo ""
        exit 1
    }

    print_success "All dependencies found"
}

check_disk_space() {
    local path="$1"
    local required_gb="$2"

    # Get available space in GB
    if command -v df &> /dev/null; then
        available_gb=$(df -BG "$path" | awk 'NR==2 {print $4}' | sed 's/G//')

        echo ""
        print_info "Disk space check:"
        echo "  Location: $path"
        echo "  Required: ${required_gb} GB (minimum)"
        echo "  Available: ${available_gb} GB"

        if (( available_gb < required_gb )); then
            print_warning "Low disk space! You may need to free up space."
            echo ""
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Cancelled by user"
                exit 1
            fi
        else
            print_success "Sufficient disk space available"
        fi
    fi
}

setup_data_directory() {
    print_info "Setting up data directory: $DATA_PATH"

    # Create directory if it doesn't exist
    mkdir -p "$DATA_PATH"

    # Convert to absolute path
    DATA_PATH=$(cd "$DATA_PATH" && pwd)

    print_success "Data directory ready: $DATA_PATH"
}

################################################################################
# Main Functions
################################################################################

download_datasets() {
    local datasets=("$@")

    if [ ${#datasets[@]} -eq 0 ]; then
        # No datasets specified, show summary
        print_header "H-JEPA Dataset Summary"
        python3 -m src.data.download --data-path "$DATA_PATH"
        return 0
    fi

    # Build Python command arguments
    local python_args=("--data-path" "$DATA_PATH")

    if [ "$VERIFY_ONLY" = true ]; then
        python_args+=("--verify-only")
    fi

    if [ "$FORCE_DOWNLOAD" = true ]; then
        python_args+=("--force")
    fi

    if [ "$VERIFY_AFTER_DOWNLOAD" = false ]; then
        python_args+=("--no-verify")
    fi

    # Add datasets
    python_args+=("${datasets[@]}")

    # Run download script
    print_header "Downloading Datasets"
    echo "Data path: $DATA_PATH"
    echo "Datasets: ${datasets[*]}"
    echo ""

    cd "$PROJECT_ROOT"
    python3 -m src.data.download "${python_args[@]}"
}

download_all_auto() {
    print_header "Downloading All Auto-Downloadable Datasets"
    print_info "This will download: CIFAR-10, CIFAR-100, STL-10"
    print_warning "Total size: ~3 GB"
    echo ""

    # Check disk space (3 GB minimum + 5 GB buffer)
    check_disk_space "$DATA_PATH" 8

    # Download each dataset
    download_datasets cifar10 cifar100 stl10
}

verify_datasets() {
    local datasets=("$@")

    if [ ${#datasets[@]} -eq 0 ]; then
        print_error "No datasets specified for verification"
        echo "Usage: $0 --verify cifar10 cifar100 ..."
        exit 1
    fi

    print_header "Verifying Datasets"

    cd "$PROJECT_ROOT"
    python3 -m src.data.download --data-path "$DATA_PATH" --verify-only "${datasets[@]}"
}

################################################################################
# Argument Parsing
################################################################################

DATASETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -d|--data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        -a|--all-auto)
            ALL_AUTO=true
            shift
            ;;
        -v|--verify)
            VERIFY_ONLY=true
            shift
            ;;
        -f|--force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        -n|--no-verify)
            VERIFY_AFTER_DOWNLOAD=false
            shift
            ;;
        -*)
            print_error "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
        *)
            DATASETS+=("$1")
            shift
            ;;
    esac
done

################################################################################
# Main Execution
################################################################################

print_header "H-JEPA Dataset Download Script"

# Check dependencies
check_dependencies

# Setup data directory
setup_data_directory

# Execute based on options
if [ "$ALL_AUTO" = true ]; then
    download_all_auto
elif [ "$VERIFY_ONLY" = true ]; then
    verify_datasets "${DATASETS[@]}"
else
    download_datasets "${DATASETS[@]}"
fi

# Print summary
echo ""
print_header "Summary"
echo "Data directory: $DATA_PATH"
echo ""

if [ ${#DATASETS[@]} -gt 0 ] || [ "$ALL_AUTO" = true ]; then
    print_success "Dataset operations completed!"
    echo ""
    echo "Next steps:"
    echo "  1. Update config file with data path: configs/default.yaml"
    echo "  2. Start training: python scripts/train.py --config configs/default.yaml"
    echo ""
fi

print_info "To see all available datasets: $0"
print_info "To verify datasets: $0 --verify cifar10 cifar100"
echo ""

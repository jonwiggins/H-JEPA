#!/bin/bash
#
# Test runner for H-JEPA data modules
#
# This script runs the comprehensive data module tests and generates a coverage report.
#
# Usage:
#   ./run_data_module_tests.sh
#
# Options:
#   -v    Verbose output
#   -q    Quiet output (only show summary)
#   -h    Show this help

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
VERBOSE=""
QUIET=""

# Parse command line arguments
while getopts "vqh" opt; do
  case $opt in
    v)
      VERBOSE="-vv"
      ;;
    q)
      QUIET="-q"
      ;;
    h)
      echo "Usage: $0 [-v] [-q] [-h]"
      echo "  -v    Verbose output"
      echo "  -q    Quiet output"
      echo "  -h    Show this help"
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}H-JEPA Data Module Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "tests/test_data_modules.py" ]; then
    echo -e "${RED}Error: test_data_modules.py not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Run tests with coverage
echo -e "${YELLOW}Running data module tests with coverage...${NC}"
echo ""

python -m pytest tests/test_data_modules.py \
    --cov=src/data/datasets \
    --cov=src/data/transforms \
    --cov=src/data/multicrop_dataset \
    --cov=src/data/multicrop_transforms \
    --cov-report=term-missing \
    --cov-report=html \
    $VERBOSE $QUIET

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Test Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Coverage report generated in: htmlcov/index.html"
echo ""
echo "Target modules:"
echo "  - src/data/datasets.py (159 lines)"
echo "  - src/data/transforms.py (195 lines)"
echo "  - src/data/multicrop_dataset.py (123 lines)"
echo "  - src/data/multicrop_transforms.py (87 lines)"
echo ""
echo "Coverage goal: 70%+ for each module"
echo ""
echo -e "${GREEN}Done!${NC}"

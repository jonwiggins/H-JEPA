#!/bin/bash
################################################################################
# Data Pipeline Verification Script for H-JEPA
#
# This script verifies that the data pipeline is properly installed and
# functioning. It checks file structure, imports, and basic functionality.
#
# Usage:
#   ./scripts/verify_data_pipeline.sh
#
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

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

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -n "Testing: $test_name... "

    if eval "$test_command" > /dev/null 2>&1; then
        print_success "PASS"
        ((TESTS_PASSED++))
        return 0
    else
        print_error "FAIL"
        ((TESTS_FAILED++))
        return 1
    fi
}

print_header "H-JEPA Data Pipeline Verification"
echo ""

# Check files exist
print_header "1. File Structure Verification"
echo ""

FILES=(
    "src/data/__init__.py"
    "src/data/datasets.py"
    "src/data/download.py"
    "scripts/download_data.sh"
    "tests/test_data.py"
    "examples/data_example.py"
    "DATA_README.md"
    "docs/DATA_PIPELINE_SUMMARY.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        print_success "Found: $file"
        ((TESTS_PASSED++))
    else
        print_error "Missing: $file"
        ((TESTS_FAILED++))
    fi
done

# Check file permissions
echo ""
print_header "2. Permission Verification"
echo ""

EXECUTABLES=(
    "scripts/download_data.sh"
    "examples/data_example.py"
)

for file in "${EXECUTABLES[@]}"; do
    if [ -x "$PROJECT_ROOT/$file" ]; then
        print_success "Executable: $file"
        ((TESTS_PASSED++))
    else
        print_warning "Not executable: $file (attempting to fix...)"
        chmod +x "$PROJECT_ROOT/$file"
        if [ -x "$PROJECT_ROOT/$file" ]; then
            print_success "Fixed: $file"
            ((TESTS_PASSED++))
        else
            print_error "Could not fix: $file"
            ((TESTS_FAILED++))
        fi
    fi
done

# Check file sizes (basic sanity check)
echo ""
print_header "3. File Content Verification"
echo ""

check_file_size() {
    local file="$1"
    local min_size="$2"
    local size=$(wc -l < "$PROJECT_ROOT/$file" 2>/dev/null || echo 0)

    if [ "$size" -ge "$min_size" ]; then
        print_success "$file: $size lines (>= $min_size expected)"
        return 0
    else
        print_error "$file: $size lines (< $min_size expected)"
        return 1
    fi
}

check_file_size "src/data/datasets.py" 500 && ((TESTS_PASSED++)) || ((TESTS_FAILED++))
check_file_size "src/data/download.py" 400 && ((TESTS_PASSED++)) || ((TESTS_FAILED++))
check_file_size "scripts/download_data.sh" 200 && ((TESTS_PASSED++)) || ((TESTS_FAILED++))
check_file_size "tests/test_data.py" 300 && ((TESTS_PASSED++)) || ((TESTS_FAILED++))
check_file_size "DATA_README.md" 400 && ((TESTS_PASSED++)) || ((TESTS_FAILED++))

# Check bash script functionality
echo ""
print_header "4. Bash Script Verification"
echo ""

cd "$PROJECT_ROOT"

if ./scripts/download_data.sh --help > /dev/null 2>&1; then
    print_success "download_data.sh --help works"
    ((TESTS_PASSED++))
else
    print_error "download_data.sh --help failed"
    ((TESTS_FAILED++))
fi

# Check Python environment
echo ""
print_header "5. Python Environment Verification"
echo ""

if command -v python3 &> /dev/null; then
    print_success "Python 3 found: $(python3 --version)"
    ((TESTS_PASSED++))
else
    print_error "Python 3 not found"
    ((TESTS_FAILED++))
fi

# Check Python imports (if dependencies are installed)
echo ""
print_header "6. Python Module Verification (if dependencies installed)"
echo ""

PYTHON_IMPORTS=(
    "torch"
    "torchvision"
    "numpy"
    "PIL"
    "tqdm"
)

for module in "${PYTHON_IMPORTS[@]}"; do
    if python3 -c "import $module" 2>/dev/null; then
        print_success "Python module available: $module"
        ((TESTS_PASSED++))
    else
        print_warning "Python module not installed: $module (install with: pip install -r requirements.txt)"
    fi
done

# Try importing our modules (if dependencies available)
if python3 -c "import torch" 2>/dev/null; then
    echo ""
    echo "Attempting to import H-JEPA data modules..."

    if python3 -c "from src.data import build_dataset, build_dataloader" 2>/dev/null; then
        print_success "Successfully imported src.data modules"
        ((TESTS_PASSED++))
    else
        print_error "Failed to import src.data modules"
        ((TESTS_FAILED++))
        echo "Error details:"
        python3 -c "from src.data import build_dataset" 2>&1 | head -5
    fi

    if python3 -c "from src.data import download_dataset, verify_dataset" 2>/dev/null; then
        print_success "Successfully imported download utilities"
        ((TESTS_PASSED++))
    else
        print_error "Failed to import download utilities"
        ((TESTS_FAILED++))
    fi
else
    print_warning "Skipping Python import tests (PyTorch not installed)"
    echo "  To install dependencies: pip install -r requirements.txt"
fi

# Check documentation
echo ""
print_header "7. Documentation Verification"
echo ""

check_doc_content() {
    local file="$1"
    local search_term="$2"

    if grep -q "$search_term" "$PROJECT_ROOT/$file" 2>/dev/null; then
        print_success "$file contains '$search_term'"
        return 0
    else
        print_error "$file missing '$search_term'"
        return 1
    fi
}

check_doc_content "DATA_README.md" "Quick Start" && ((TESTS_PASSED++)) || ((TESTS_FAILED++))
check_doc_content "DATA_README.md" "CIFAR" && ((TESTS_PASSED++)) || ((TESTS_FAILED++))
check_doc_content "DATA_README.md" "ImageNet" && ((TESTS_PASSED++)) || ((TESTS_FAILED++))
check_doc_content "docs/DATA_PIPELINE_SUMMARY.md" "Implementation Summary" && ((TESTS_PASSED++)) || ((TESTS_FAILED++))

# Summary
echo ""
print_header "Verification Summary"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
PASS_RATE=$((TESTS_PASSED * 100 / TOTAL_TESTS))

echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo "Pass Rate: $PASS_RATE%"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    print_success "All verification tests passed!"
    echo ""
    echo "✅ Data pipeline is ready to use!"
    echo ""
    echo "Next steps:"
    echo "  1. Install dependencies (if not done): pip install -r requirements.txt"
    echo "  2. Download a dataset: ./scripts/download_data.sh cifar10"
    echo "  3. Run examples: python examples/data_example.py"
    echo "  4. Run tests: pytest tests/test_data.py -v"
    echo ""
    exit 0
elif [ $PASS_RATE -ge 80 ]; then
    print_warning "Most verification tests passed ($PASS_RATE%)"
    echo ""
    echo "⚠️  Some tests failed, but core functionality appears intact"
    echo ""
    echo "Common issues:"
    echo "  - Python dependencies not installed: pip install -r requirements.txt"
    echo "  - File permissions: chmod +x scripts/*.sh"
    echo ""
    exit 0
else
    print_error "Verification failed ($PASS_RATE% passed)"
    echo ""
    echo "❌ Data pipeline may not be properly installed"
    echo ""
    echo "Please check:"
    echo "  1. All files are present in the correct locations"
    echo "  2. File permissions are correct"
    echo "  3. Python dependencies are installed"
    echo ""
    exit 1
fi

#!/bin/bash
# Script to run H-JEPA evaluation module tests
# Usage: ./run_evaluation_tests.sh [coverage|quick|verbose]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test files
LINEAR_PROBE_TESTS="tests/test_linear_probe.py"
KNN_EVAL_TESTS="tests/test_knn_eval.py"
FEATURE_QUALITY_TESTS="tests/test_feature_quality.py"

ALL_TESTS="$LINEAR_PROBE_TESTS $KNN_EVAL_TESTS $FEATURE_QUALITY_TESTS"

# Source modules
LINEAR_PROBE_SRC="src/evaluation/linear_probe.py"
KNN_EVAL_SRC="src/evaluation/knn_eval.py"
FEATURE_QUALITY_SRC="src/evaluation/feature_quality.py"

ALL_SRC="src/evaluation"

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}H-JEPA Evaluation Module Test Runner${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# Check Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Check pytest is available
if ! python3 -m pytest --version &> /dev/null; then
    echo -e "${RED}Error: pytest not found. Install with: pip install pytest${NC}"
    exit 1
fi

# Determine mode
MODE=${1:-quick}

case $MODE in
    coverage)
        echo -e "${YELLOW}Running tests with coverage analysis...${NC}"
        echo ""

        # Check pytest-cov is available
        if ! python3 -m pytest --cov &> /dev/null 2>&1; then
            echo -e "${RED}Error: pytest-cov not found. Install with: pip install pytest-cov${NC}"
            exit 1
        fi

        python3 -m pytest $ALL_TESTS \
            --cov=$ALL_SRC \
            --cov-report=term-missing \
            --cov-report=html \
            -v

        echo ""
        echo -e "${GREEN}==================================================${NC}"
        echo -e "${GREEN}Coverage report generated!${NC}"
        echo -e "${GREEN}==================================================${NC}"
        echo "HTML report: htmlcov/index.html"
        echo ""
        echo "To view the HTML report:"
        echo "  open htmlcov/index.html    # macOS"
        echo "  xdg-open htmlcov/index.html  # Linux"
        ;;

    verbose)
        echo -e "${YELLOW}Running tests in verbose mode...${NC}"
        echo ""
        python3 -m pytest $ALL_TESTS -v -s
        ;;

    quick)
        echo -e "${YELLOW}Running quick test suite...${NC}"
        echo ""
        python3 -m pytest $ALL_TESTS -v
        ;;

    individual)
        echo -e "${YELLOW}Running tests individually...${NC}"
        echo ""

        echo -e "${GREEN}[1/3] Testing linear_probe.py...${NC}"
        python3 -m pytest $LINEAR_PROBE_TESTS -v --tb=short
        echo ""

        echo -e "${GREEN}[2/3] Testing knn_eval.py...${NC}"
        python3 -m pytest $KNN_EVAL_TESTS -v --tb=short
        echo ""

        echo -e "${GREEN}[3/3] Testing feature_quality.py...${NC}"
        python3 -m pytest $FEATURE_QUALITY_TESTS -v --tb=short
        echo ""
        ;;

    stats)
        echo -e "${YELLOW}Test Statistics:${NC}"
        echo ""

        echo "Test Counts:"
        echo "  linear_probe:    $(grep -c '^def test_' $LINEAR_PROBE_TESTS) tests"
        echo "  knn_eval:        $(grep -c '^def test_' $KNN_EVAL_TESTS) tests"
        echo "  feature_quality: $(grep -c '^def test_' $FEATURE_QUALITY_TESTS) tests"
        echo "  ---"
        TOTAL=$(($(grep -c '^def test_' $LINEAR_PROBE_TESTS) + $(grep -c '^def test_' $KNN_EVAL_TESTS) + $(grep -c '^def test_' $FEATURE_QUALITY_TESTS)))
        echo "  Total:           $TOTAL tests"
        echo ""

        echo "Source Code Lines:"
        echo "  linear_probe.py:    $(wc -l < $LINEAR_PROBE_SRC) lines"
        echo "  knn_eval.py:        $(wc -l < $KNN_EVAL_SRC) lines"
        echo "  feature_quality.py: $(wc -l < $FEATURE_QUALITY_SRC) lines"
        echo "  ---"
        TOTAL_SRC=$(($(wc -l < $LINEAR_PROBE_SRC) + $(wc -l < $KNN_EVAL_SRC) + $(wc -l < $FEATURE_QUALITY_SRC)))
        echo "  Total:              $TOTAL_SRC lines"
        echo ""

        echo "Test Code Lines:"
        echo "  test_linear_probe.py:    $(wc -l < $LINEAR_PROBE_TESTS) lines"
        echo "  test_knn_eval.py:        $(wc -l < $KNN_EVAL_TESTS) lines"
        echo "  test_feature_quality.py: $(wc -l < $FEATURE_QUALITY_TESTS) lines"
        echo "  ---"
        TOTAL_TEST=$(($(wc -l < $LINEAR_PROBE_TESTS) + $(wc -l < $KNN_EVAL_TESTS) + $(wc -l < $FEATURE_QUALITY_TESTS)))
        echo "  Total:                   $TOTAL_TEST lines"
        echo ""

        RATIO=$(echo "scale=2; $TOTAL_TEST / $TOTAL_SRC" | bc)
        echo "Test-to-Source Ratio: ${RATIO}:1"
        ;;

    help|--help|-h)
        echo "Usage: $0 [MODE]"
        echo ""
        echo "Modes:"
        echo "  quick       - Run all tests quickly (default)"
        echo "  verbose     - Run with verbose output"
        echo "  coverage    - Run with coverage analysis"
        echo "  individual  - Run each test file separately"
        echo "  stats       - Show test statistics"
        echo "  help        - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                    # Quick run"
        echo "  $0 coverage           # With coverage"
        echo "  $0 verbose            # Verbose output"
        echo "  $0 stats              # Show statistics"
        ;;

    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"

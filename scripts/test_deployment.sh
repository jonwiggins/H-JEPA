#!/bin/bash
# Test H-JEPA Model Server Deployment
# Usage: ./scripts/test_deployment.sh [OPTIONS]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
HOST="localhost"
PORT=8000
TEST_IMAGE=""
VERBOSE=false

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -H, --host HOST          Server host [default: localhost]"
    echo "  -p, --port PORT          Server port [default: 8000]"
    echo "  -i, --image PATH         Path to test image (optional)"
    echo "  -v, --verbose            Verbose output"
    echo "  -h, --help               Show this help message"
    echo ""
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -i|--image)
            TEST_IMAGE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

BASE_URL="http://$HOST:$PORT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}H-JEPA Model Server Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Server: ${YELLOW}$BASE_URL${NC}"
echo ""

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to run test
run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -n "Testing $test_name... "

    if $VERBOSE; then
        echo ""
        echo "Command: $test_command"
    fi

    if eval "$test_command" > /tmp/test_output.txt 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))

        if $VERBOSE; then
            cat /tmp/test_output.txt
        fi
    else
        echo -e "${RED}FAILED${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))

        echo -e "${RED}Error output:${NC}"
        cat /tmp/test_output.txt
    fi
}

# Wait for server to be ready
echo -e "${YELLOW}Waiting for server to be ready...${NC}"
RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $RETRIES ]; do
    if curl -s -f "$BASE_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Server is ready!${NC}"
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $RETRIES ]; then
        echo -e "${RED}Server failed to start within timeout${NC}"
        exit 1
    fi

    sleep 1
done

echo ""

# Test 1: Root endpoint
run_test "Root endpoint" \
    "curl -s -f $BASE_URL/ | jq -e '.message'"

# Test 2: Health check
run_test "Health check" \
    "curl -s -f $BASE_URL/health | jq -e '.status == \"healthy\"'"

# Test 3: Model info
run_test "Model info" \
    "curl -s -f $BASE_URL/info | jq -e '.model_type == \"H-JEPA\"'"

# Test 4: Metrics endpoint
run_test "Metrics endpoint" \
    "curl -s -f $BASE_URL/metrics | grep -q 'hjepa_requests_total'"

# Test 5: Feature extraction (if test image provided)
if [ -n "$TEST_IMAGE" ] && [ -f "$TEST_IMAGE" ]; then
    echo ""
    echo -e "${YELLOW}Testing feature extraction with provided image...${NC}"

    # Single image extraction
    run_test "Feature extraction (single)" \
        "curl -s -f -X POST -F 'file=@$TEST_IMAGE' -F 'hierarchy_level=0' $BASE_URL/extract | jq -e '.shape'"

    # Batch extraction
    run_test "Feature extraction (batch)" \
        "curl -s -f -X POST -F 'files=@$TEST_IMAGE' -F 'files=@$TEST_IMAGE' -F 'hierarchy_level=0' $BASE_URL/extract_batch | jq -e '.batch_size == 2'"

    # Different hierarchy levels
    for level in 0 1 2; do
        run_test "Feature extraction (level $level)" \
            "curl -s -f -X POST -F 'file=@$TEST_IMAGE' -F 'hierarchy_level=$level' $BASE_URL/extract | jq -e '.hierarchy_level == $level'"
    done
else
    echo ""
    echo -e "${YELLOW}Skipping feature extraction tests (no test image provided)${NC}"
    echo -e "Provide a test image with: ${YELLOW}-i /path/to/image.jpg${NC}"
fi

# Test 6: API documentation
run_test "API documentation" \
    "curl -s -f $BASE_URL/docs | grep -q 'FastAPI'"

# Test 7: Concurrent requests
echo ""
echo -e "${YELLOW}Testing concurrent requests...${NC}"
if [ -n "$TEST_IMAGE" ] && [ -f "$TEST_IMAGE" ]; then
    run_test "Concurrent requests (5)" \
        "for i in {1..5}; do curl -s -f -X POST -F 'file=@$TEST_IMAGE' $BASE_URL/extract & done; wait"
else
    echo "Skipping concurrent test (no test image)"
fi

# Print summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Test Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi

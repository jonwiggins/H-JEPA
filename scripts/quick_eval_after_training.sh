#!/bin/bash
# Quick Evaluation Script for H-JEPA Models
# Run this after training completes to get comprehensive metrics

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}H-JEPA Post-Training Evaluation${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Default values
CHECKPOINT="${1:-results/checkpoints/checkpoint_best.pth}"
DATASET="${2:-cifar10}"
OUTPUT_DIR="${3:-results/evaluation}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${YELLOW}Warning: Checkpoint not found at $CHECKPOINT${NC}"
    echo "Looking for latest checkpoint..."
    CHECKPOINT=$(ls -t results/checkpoints/checkpoint_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT" ]; then
        echo -e "${YELLOW}Error: No checkpoints found${NC}"
        exit 1
    fi
    echo -e "${GREEN}Found: $CHECKPOINT${NC}"
fi

echo -e "${GREEN}Checkpoint: $CHECKPOINT${NC}"
echo -e "${GREEN}Dataset: $DATASET${NC}"
echo -e "${GREEN}Output: $OUTPUT_DIR${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# 1. Linear Probe Evaluation
echo -e "${BLUE}[1/5] Running Linear Probe Evaluation...${NC}"
python3.11 scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --eval-type linear_probe \
    --hierarchy-levels 0 1 \
    --output-dir "$OUTPUT_DIR/linear_probe" \
    2>&1 | tee "$OUTPUT_DIR/linear_probe.log"

echo -e "${GREEN}✓ Linear probe complete${NC}"
echo ""

# 2. k-NN Evaluation
echo -e "${BLUE}[2/5] Running k-NN Evaluation...${NC}"
python3.11 scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --eval-type knn \
    --hierarchy-levels 0 \
    --output-dir "$OUTPUT_DIR/knn" \
    2>&1 | tee "$OUTPUT_DIR/knn.log"

echo -e "${GREEN}✓ k-NN evaluation complete${NC}"
echo ""

# 3. Feature Quality Analysis
echo -e "${BLUE}[3/5] Analyzing Feature Quality...${NC}"
python3.11 scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --eval-type feature_quality \
    --hierarchy-levels 0 1 \
    --output-dir "$OUTPUT_DIR/feature_quality" \
    2>&1 | tee "$OUTPUT_DIR/feature_quality.log"

echo -e "${GREEN}✓ Feature quality analysis complete${NC}"
echo ""

# 4. Generate Visualizations
echo -e "${BLUE}[4/5] Generating Visualizations...${NC}"
python3.11 scripts/visualize.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR/visualizations" \
    --num-samples 16 \
    2>&1 | tee "$OUTPUT_DIR/visualizations.log"

echo -e "${GREEN}✓ Visualizations generated${NC}"
echo ""

# 5. Generate Summary Report
echo -e "${BLUE}[5/5] Generating Summary Report...${NC}"

REPORT_FILE="$OUTPUT_DIR/EVALUATION_SUMMARY.md"

cat > "$REPORT_FILE" << 'REPORT_HEADER'
# H-JEPA Evaluation Summary

**Generated:** $(date)
**Checkpoint:** CHECKPOINT_PLACEHOLDER
**Dataset:** DATASET_PLACEHOLDER

---

## Quick Results

REPORT_HEADER

# Replace placeholders
sed -i.bak "s|CHECKPOINT_PLACEHOLDER|$CHECKPOINT|g" "$REPORT_FILE"
sed -i.bak "s|DATASET_PLACEHOLDER|$DATASET|g" "$REPORT_FILE"
rm "$REPORT_FILE.bak"

# Extract metrics from logs
echo "### Linear Probe" >> "$REPORT_FILE"
echo '```' >> "$REPORT_FILE"
grep -A 5 "Final Results" "$OUTPUT_DIR/linear_probe.log" 2>/dev/null >> "$REPORT_FILE" || echo "Results pending..." >> "$REPORT_FILE"
echo '```' >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "### k-NN Evaluation" >> "$REPORT_FILE"
echo '```' >> "$REPORT_FILE"
grep -A 5 "k-NN" "$OUTPUT_DIR/knn.log" 2>/dev/null >> "$REPORT_FILE" || echo "Results pending..." >> "$REPORT_FILE"
echo '```' >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "### Feature Quality" >> "$REPORT_FILE"
echo '```' >> "$REPORT_FILE"
grep -A 10 "Feature Quality" "$OUTPUT_DIR/feature_quality.log" 2>/dev/null >> "$REPORT_FILE" || echo "Results pending..." >> "$REPORT_FILE"
echo '```' >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "---" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "## Output Locations" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- Linear probe results: \`$OUTPUT_DIR/linear_probe/\`" >> "$REPORT_FILE"
echo "- k-NN results: \`$OUTPUT_DIR/knn/\`" >> "$REPORT_FILE"
echo "- Feature quality: \`$OUTPUT_DIR/feature_quality/\`" >> "$REPORT_FILE"
echo "- Visualizations: \`$OUTPUT_DIR/visualizations/\`" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo -e "${GREEN}✓ Summary report generated: $REPORT_FILE${NC}"
echo ""

# Print summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  Evaluation Complete!${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
echo -e "${GREEN}Summary report: $REPORT_FILE${NC}"
echo ""
echo "View results:"
echo "  cat $REPORT_FILE"
echo ""
echo "Open visualizations:"
echo "  open $OUTPUT_DIR/visualizations/"
echo ""

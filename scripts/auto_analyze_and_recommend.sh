#!/bin/bash
# Automatic analysis and recommendation after training completion
#
# Usage: ./scripts/auto_analyze_and_recommend.sh [log_file]

set -e

LOG_FILE="${1:-training_run.log}"
OUTPUT_DIR="results/validation_analysis"

echo "======================================================================"
echo "H-JEPA Validation Run: Auto-Analysis"
echo "======================================================================"
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found: $LOG_FILE"
    echo "Usage: $0 [log_file]"
    exit 1
fi

echo "📋 Log file: $LOG_FILE"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Wait for training to complete (check if process is still running)
echo "⏳ Checking training status..."
if pgrep -f "train.py.*m1_max_quick_val" > /dev/null; then
    echo "⚠️  Training is still running. Waiting for completion..."
    echo ""

    # Monitor training progress
    while pgrep -f "train.py.*m1_max_quick_val" > /dev/null; do
        # Extract last progress line
        LAST_LINE=$(tail -n 5 "$LOG_FILE" | grep -E "Epoch [0-9]+/[0-9]+:" | tail -n 1 || echo "")

        if [ -n "$LAST_LINE" ]; then
            # Extract progress percentage
            PROGRESS=$(echo "$LAST_LINE" | grep -oE "[0-9]+%" | head -n 1)
            LOSS=$(echo "$LAST_LINE" | grep -oE "loss=[0-9.]+" | head -n 1)
            SPEED=$(echo "$LAST_LINE" | grep -oE "[0-9.]+it/s" | head -n 1)

            echo -ne "\r  Progress: $PROGRESS | $LOSS | $SPEED    "
        fi

        sleep 10
    done

    echo ""
    echo ""
    echo "✅ Training completed!"
else
    echo "✅ Training already completed"
fi

echo ""
echo "======================================================================"
echo "Running Analysis"
echo "======================================================================"
echo ""

# Install matplotlib if needed
if ! python3.11 -c "import matplotlib" 2>/dev/null; then
    echo "📦 Installing matplotlib for plots..."
    pip3.11 install matplotlib
    echo ""
fi

# Run analysis
python3.11 scripts/analyze_validation_run.py \
    --log "$LOG_FILE" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "======================================================================"
echo "Analysis Complete!"
echo "======================================================================"
echo ""

# Display the report
if [ -f "$OUTPUT_DIR/validation_report.md" ]; then
    echo "📊 Validation Report:"
    echo ""
    cat "$OUTPUT_DIR/validation_report.md"
    echo ""
fi

echo "======================================================================"
echo "Next Steps"
echo "======================================================================"
echo ""
echo "1. Review the full report: $OUTPUT_DIR/validation_report.md"
echo "2. Check training curves: $OUTPUT_DIR/validation_training_curves.png"
echo "3. Review detailed analysis: $OUTPUT_DIR/analysis.json"
echo ""

# Extract recommendation
if [ -f "$OUTPUT_DIR/analysis.json" ]; then
    RECOMMENDED_CONFIG=$(python3.11 -c "
import json
with open('$OUTPUT_DIR/analysis.json') as f:
    data = json.load(f)
    rec = data.get('recommendations', {}).get('recommended_config')
    if rec:
        print(rec)
    else:
        print('Not available')
" 2>/dev/null || echo "Not available")

    if [ "$RECOMMENDED_CONFIG" != "Not available" ]; then
        echo "🚀 Recommended next training run:"
        echo ""
        echo "   python3.11 scripts/train.py --config $RECOMMENDED_CONFIG"
        echo ""
    fi
fi

echo "======================================================================"
echo ""

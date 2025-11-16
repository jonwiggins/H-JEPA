#!/bin/bash
# Watch for new checkpoints and automatically explore them

CHECKPOINT_DIR="results/validation_test/checkpoints"
OUTPUT_DIR="results/exploration"
LAST_PROCESSED=""

echo "==================================================================="
echo "H-JEPA Checkpoint Monitor & Explorer"
echo "==================================================================="
echo "Watching: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

while true; do
    # Find the most recent checkpoint
    LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)

    if [ -n "$LATEST_CHECKPOINT" ] && [ "$LATEST_CHECKPOINT" != "$LAST_PROCESSED" ]; then
        EPOCH=$(basename "$LATEST_CHECKPOINT" | sed 's/checkpoint_epoch_\([0-9]*\)\.pt/\1/')

        echo ""
        echo "==================================================================="
        echo "NEW CHECKPOINT DETECTED: Epoch $EPOCH"
        echo "==================================================================="
        echo "Checkpoint: $LATEST_CHECKPOINT"
        echo "Processing at: $(date)"
        echo ""

        # Create epoch-specific output directory
        EPOCH_OUTPUT="$OUTPUT_DIR/epoch_$EPOCH"
        mkdir -p "$EPOCH_OUTPUT"

        # Run exploration
        echo "Running exploration script..."
        python3.11 scripts/explore_model.py \
            --checkpoint "$LATEST_CHECKPOINT" \
            --device mps \
            --output-dir "$EPOCH_OUTPUT" \
            --sample-idx 0

        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Exploration complete for epoch $EPOCH!"
            echo "  Results: $EPOCH_OUTPUT/"
            echo ""

            # List generated files
            echo "Generated visualizations:"
            ls -lh "$EPOCH_OUTPUT"/*.png 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
        else
            echo "✗ Exploration failed for epoch $EPOCH"
        fi

        LAST_PROCESSED="$LATEST_CHECKPOINT"

        echo ""
        echo "Waiting for next checkpoint..."
    fi

    # Check every 30 seconds
    sleep 30
done

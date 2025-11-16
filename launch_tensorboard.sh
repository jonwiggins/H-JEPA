#!/bin/bash
# Launch TensorBoard for H-JEPA training monitoring

echo "Launching TensorBoard..."
echo "TensorBoard will be available at: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop TensorBoard"

tensorboard --logdir results/foundation_model/logs/tensorboard --port 6006

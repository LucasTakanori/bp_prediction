#!/bin/bash
# CNN-Based BP Predictor Training Script

echo "🚀 Starting CNN-Based BP Predictor Training"
echo "============================================"

# Set paths
CONFIG_PATH="configs/cnn_bp_predictor.yaml"
SCRIPT_PATH="scripts/train_cnn_bp_predictor_fixed.py"

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

# Run training
python "$SCRIPT_PATH" --config "$CONFIG_PATH"

echo "✅ Training completed!" 
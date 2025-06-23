#!/bin/bash

# CNN-Based BP Predictor Evaluation Script
# Usage: ./scripts/run_cnn_bp_evaluation.sh [model_timestamp]

set -e

echo "🧠 Starting CNN-Based BP Predictor Evaluation..."

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="$PROJECT_ROOT/configs/cnn_bp_predictor.yaml"

# Get model timestamp from argument or find the latest
if [ $# -eq 1 ]; then
    TIMESTAMP="$1"
    echo "📅 Using provided timestamp: $TIMESTAMP"
else
    # Find the latest CNN experiment directory
    LATEST_DIR=$(find "$PROJECT_ROOT/experiments" -name "cnn_bp_predictor_*" -type d | sort -r | head -n 1)
    if [ -z "$LATEST_DIR" ]; then
        echo "❌ No CNN BP predictor experiments found in $PROJECT_ROOT/experiments/"
        echo "💡 Run training first with: ./scripts/run_cnn_bp_training.sh"
        exit 1
    fi
    TIMESTAMP=$(basename "$LATEST_DIR" | sed 's/cnn_bp_predictor_//')
    echo "📅 Auto-detected latest timestamp: $TIMESTAMP"
fi

# Set paths
MODEL_PATH="$PROJECT_ROOT/experiments/cnn_bp_predictor_$TIMESTAMP/checkpoints/best_model.pt"
OUTPUT_DIR="$PROJECT_ROOT/evaluation_results_cnn_bp_$TIMESTAMP"

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model checkpoint not found: $MODEL_PATH"
    echo "💡 Available CNN experiments:"
    find "$PROJECT_ROOT/experiments" -name "cnn_bp_predictor_*" -type d | sort -r
    exit 1
fi

# Verify config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "📁 Model: $MODEL_PATH"
echo "⚙️  Config: $CONFIG_PATH"
echo "📂 Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
cd "$PROJECT_ROOT"
python scripts/evaluate_cnn_bp_predictor.py \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples 500

echo "✅ Evaluation completed! Results saved to: $OUTPUT_DIR" 
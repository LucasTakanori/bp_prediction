#!/bin/bash

# Run Improved BP Predictor Evaluation
# This script evaluates the improved BP predictor model trained with continuous configuration

echo "🩺 Starting Improved BP Predictor Evaluation..."
echo "=" * 60

# Set paths
MODEL_PATH="/home/lucas_takanori/phd/bp_prediction/experiments/improved_bp_predictor_continuous/checkpoints/best_model.pt"
CONFIG_PATH="/home/lucas_takanori/phd/bp_prediction/configs/sophisticated_bp_predictor_improved.yaml"
OUTPUT_DIR="./evaluation_results_improved_bp_$(date +%Y%m%d_%H%M%S)"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    echo "Please check the path or train the model first."
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

# Run evaluation
python3 scripts/evaluate_improved_bp_predictor.py \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples 500

echo ""
echo "✅ Evaluation completed!"
echo "📁 Results saved to: $OUTPUT_DIR" 
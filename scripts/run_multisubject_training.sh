#!/bin/bash
#
# Multi-Subject Training Pipeline
# Orchestrates VAE training followed by BiLSTM training for optimal BP prediction
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="/home/lucas_takanori/phd/data"
EXPERIMENTS_ROOT="$PROJECT_ROOT/experiments"

# Default parameters
MAX_SUBJECTS=""
MAX_SAMPLES_PER_SUBJECT=""
USE_WANDB=""
DEVICE="auto"
VAE_EPOCHS="80"
BILSTM_EPOCHS="100"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-subjects)
            MAX_SUBJECTS="--max-subjects $2"
            shift 2
            ;;
        --max-samples-per-subject)
            MAX_SAMPLES_PER_SUBJECT="--max-samples-per-subject $2"
            shift 2
            ;;
        --wandb)
            USE_WANDB="--wandb"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --vae-epochs)
            VAE_EPOCHS="$2"
            shift 2
            ;;
        --bilstm-epochs)
            BILSTM_EPOCHS="$2"
            shift 2
            ;;
        --help)
            echo "Multi-Subject BP Prediction Training Pipeline"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --max-subjects N              Limit to N subjects (default: all available)"
            echo "  --max-samples-per-subject N   Limit to N samples per subject (default: no limit)"
            echo "  --wandb                       Enable Weights & Biases logging"
            echo "  --device DEVICE               Device to use (auto|cuda|cpu, default: auto)"
            echo "  --vae-epochs N                VAE training epochs (default: 80)"
            echo "  --bilstm-epochs N             BiLSTM training epochs (default: 100)"
            echo "  --help                        Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --max-subjects 10 --wandb --device cuda"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if running in the correct environment
print_status "Checking environment..."

if ! command -v python &> /dev/null; then
    print_error "Python not found. Please activate your environment."
    exit 1
fi

if ! python -c "import torch" &> /dev/null; then
    print_error "PyTorch not found. Please install required dependencies."
    exit 1
fi

# Check data availability
print_status "Checking data availability..."
DATA_FILES=$(ls -1 "$DATA_ROOT"/subject*_baseline_masked.h5 2>/dev/null | wc -l)

if [ "$DATA_FILES" -eq 0 ]; then
    print_error "No data files found in $DATA_ROOT"
    exit 1
fi

print_success "Found $DATA_FILES subject data files"

# Create experiments directory
mkdir -p "$EXPERIMENTS_ROOT"

print_status "🚀 Starting Multi-Subject Training Pipeline"
print_status "=============================================="
print_status "📁 Project root: $PROJECT_ROOT"
print_status "📊 Data root: $DATA_ROOT"
print_status "🧪 Experiments root: $EXPERIMENTS_ROOT"
print_status "📈 Available subjects: $DATA_FILES"
print_status "🔧 Device: $DEVICE"
print_status "=============================================="

# Step 1: Train VAE on all subjects
print_status "STEP 1: Training VAE on multi-subject data"
print_status "==========================================="

VAE_CONFIG="$PROJECT_ROOT/configs/vae_multisubject.yaml"
VAE_SCRIPT="$SCRIPT_DIR/train_vae_multisubject.py"

if [ ! -f "$VAE_CONFIG" ]; then
    print_error "VAE config not found: $VAE_CONFIG"
    exit 1
fi

if [ ! -f "$VAE_SCRIPT" ]; then
    print_error "VAE training script not found: $VAE_SCRIPT"
    exit 1
fi

# Update VAE epochs in config if specified
if [ "$VAE_EPOCHS" != "80" ]; then
    print_status "Setting VAE epochs to $VAE_EPOCHS"
    # Create temporary config with updated epochs
    VAE_CONFIG_TEMP="$PROJECT_ROOT/configs/vae_multisubject_temp.yaml"
    sed "s/num_epochs: 80/num_epochs: $VAE_EPOCHS/" "$VAE_CONFIG" > "$VAE_CONFIG_TEMP"
    VAE_CONFIG="$VAE_CONFIG_TEMP"
fi

print_status "Starting VAE training..."
VAE_CMD="python '$VAE_SCRIPT' --config '$VAE_CONFIG' --device $DEVICE $MAX_SUBJECTS $MAX_SAMPLES_PER_SUBJECT $USE_WANDB"

echo "Running: $VAE_CMD"
if eval $VAE_CMD; then
    print_success "VAE training completed successfully"
else
    print_error "VAE training failed"
    exit 1
fi

# Find the latest VAE checkpoint
VAE_EXPERIMENT_DIR=$(ls -td "$EXPERIMENTS_ROOT"/multisubject_vae_* | head -n1)
if [ -z "$VAE_EXPERIMENT_DIR" ]; then
    print_error "Could not find VAE experiment directory"
    exit 1
fi

VAE_CHECKPOINT="$VAE_EXPERIMENT_DIR/checkpoints/vae_best.pt"
if [ ! -f "$VAE_CHECKPOINT" ]; then
    print_error "VAE checkpoint not found: $VAE_CHECKPOINT"
    exit 1
fi

print_success "VAE checkpoint: $VAE_CHECKPOINT"

# Step 2: Train BiLSTM using VAE features
print_status "STEP 2: Training BiLSTM with VAE features"
print_status "========================================="

BILSTM_CONFIG="$PROJECT_ROOT/configs/bilstm_multisubject.yaml"
BILSTM_SCRIPT="$SCRIPT_DIR/train_bilstm_multisubject.py"

if [ ! -f "$BILSTM_CONFIG" ]; then
    print_error "BiLSTM config not found: $BILSTM_CONFIG"
    exit 1
fi

if [ ! -f "$BILSTM_SCRIPT" ]; then
    print_error "BiLSTM training script not found: $BILSTM_SCRIPT"
    exit 1
fi

# Update BiLSTM epochs in config if specified
if [ "$BILSTM_EPOCHS" != "100" ]; then
    print_status "Setting BiLSTM epochs to $BILSTM_EPOCHS"
    # Create temporary config with updated epochs
    BILSTM_CONFIG_TEMP="$PROJECT_ROOT/configs/bilstm_multisubject_temp.yaml"
    sed "s/num_epochs: 100/num_epochs: $BILSTM_EPOCHS/" "$BILSTM_CONFIG" > "$BILSTM_CONFIG_TEMP"
    BILSTM_CONFIG="$BILSTM_CONFIG_TEMP"
fi

print_status "Starting BiLSTM training with VAE checkpoint..."
BILSTM_CMD="python '$BILSTM_SCRIPT' --config '$BILSTM_CONFIG' --vae-checkpoint '$VAE_CHECKPOINT' --device $DEVICE $MAX_SUBJECTS $MAX_SAMPLES_PER_SUBJECT $USE_WANDB"

echo "Running: $BILSTM_CMD"
if eval $BILSTM_CMD; then
    print_success "BiLSTM training completed successfully"
else
    print_error "BiLSTM training failed"
    exit 1
fi

# Find the latest BiLSTM checkpoint
BILSTM_EXPERIMENT_DIR=$(ls -td "$EXPERIMENTS_ROOT"/multisubject_bp_* | head -n1)
if [ -z "$BILSTM_EXPERIMENT_DIR" ]; then
    print_error "Could not find BiLSTM experiment directory"
    exit 1
fi

BILSTM_CHECKPOINT="$BILSTM_EXPERIMENT_DIR/checkpoints/best_model.pt"
if [ ! -f "$BILSTM_CHECKPOINT" ]; then
    print_error "BiLSTM checkpoint not found: $BILSTM_CHECKPOINT"
    exit 1
fi

print_success "BiLSTM checkpoint: $BILSTM_CHECKPOINT"

# Step 3: Generate evaluation report
print_status "STEP 3: Generating evaluation report"
print_status "===================================="

# Get final metrics from the training
if [ -f "$BILSTM_EXPERIMENT_DIR/config.yaml" ]; then
    print_status "Training configuration saved to: $BILSTM_EXPERIMENT_DIR/config.yaml"
fi

# Create summary report
SUMMARY_FILE="$BILSTM_EXPERIMENT_DIR/training_summary.txt"

cat > "$SUMMARY_FILE" << EOF
Multi-Subject BP Prediction Training Summary
============================================

Training Date: $(date)
Project Root: $PROJECT_ROOT
Data Root: $DATA_ROOT

VAE Training:
- Experiment: $(basename "$VAE_EXPERIMENT_DIR")
- Checkpoint: $VAE_CHECKPOINT
- Epochs: $VAE_EPOCHS

BiLSTM Training:
- Experiment: $(basename "$BILSTM_EXPERIMENT_DIR")
- Checkpoint: $BILSTM_CHECKPOINT
- Epochs: $BILSTM_EPOCHS

Configuration:
- Subjects: $DATA_FILES available
- Device: $DEVICE
- WandB: $([ -n "$USE_WANDB" ] && echo "Enabled" || echo "Disabled")

Performance Targets:
- Systolic R²: > 0.85
- Diastolic R²: > 0.80
- Systolic MAE: < 8.0 mmHg
- Diastolic MAE: < 6.0 mmHg

Next Steps:
1. Evaluate the model performance
2. Generate Bland-Altman plots
3. Compare with existing models
4. Consider hyperparameter tuning if needed

EOF

print_success "Training summary saved to: $SUMMARY_FILE"

# Clean up temporary configs
if [ -f "$VAE_CONFIG_TEMP" ]; then
    rm "$VAE_CONFIG_TEMP"
fi
if [ -f "$BILSTM_CONFIG_TEMP" ]; then
    rm "$BILSTM_CONFIG_TEMP"
fi

print_status "🎉 MULTI-SUBJECT TRAINING PIPELINE COMPLETED!"
print_status "=============================================="
print_success "VAE Model: $VAE_CHECKPOINT"
print_success "BiLSTM Model: $BILSTM_CHECKPOINT"
print_success "Experiment Directory: $BILSTM_EXPERIMENT_DIR"
print_success "Summary Report: $SUMMARY_FILE"
print_status "=============================================="

print_status "📊 To evaluate the trained model, run:"
print_status "python scripts/evaluate_multisubject_bp.py --model-checkpoint '$BILSTM_CHECKPOINT' --config '$BILSTM_CONFIG'"

print_status "🎯 To compare with existing models, check the evaluation results in:"
print_status "$BILSTM_EXPERIMENT_DIR/results/" 
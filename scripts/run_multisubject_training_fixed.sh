#!/bin/bash
#
# Fixed Multi-Subject Training Pipeline
# Uses proven working components from train_enhanced.py and vae_baseline.yaml
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
USE_WANDB=""
DEVICE="auto"
VAE_EPOCHS="20"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-subjects)
            MAX_SUBJECTS="--max-subjects $2"
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
        --help)
            echo "Fixed Multi-Subject BP Prediction Training Pipeline"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --max-subjects N              Limit to N subjects (default: all available)"
            echo "  --wandb                       Enable Weights & Biases logging"
            echo "  --device DEVICE               Device to use (auto|cuda|cpu, default: auto)"
            echo "  --vae-epochs N                VAE training epochs (default: 20)"
            echo "  --help                        Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --max-subjects 5 --wandb --device cuda"
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

print_status "🚀 Starting Fixed Multi-Subject Training Pipeline"
print_status "================================================="
print_status "📁 Project root: $PROJECT_ROOT"
print_status "📊 Data root: $DATA_ROOT"
print_status "🧪 Experiments root: $EXPERIMENTS_ROOT"
print_status "📈 Available subjects: $DATA_FILES"
print_status "🔧 Device: $DEVICE"
print_status "================================================="

# Step 1: Train VAE on all subjects using the fixed script
print_status "STEP 1: Training VAE on multi-subject data (FIXED)"
print_status "=================================================="

VAE_CONFIG="$PROJECT_ROOT/configs/vae_multisubject_fixed.yaml"
VAE_SCRIPT="$SCRIPT_DIR/train_vae_multisubject_fixed.py"

if [ ! -f "$VAE_CONFIG" ]; then
    print_error "VAE config not found: $VAE_CONFIG"
    exit 1
fi

if [ ! -f "$VAE_SCRIPT" ]; then
    print_error "VAE training script not found: $VAE_SCRIPT"
    exit 1
fi

print_status "Starting VAE training with proven components..."
VAE_CMD="python '$VAE_SCRIPT' --config '$VAE_CONFIG' --device $DEVICE $MAX_SUBJECTS $USE_WANDB"

echo "Running: $VAE_CMD"
if eval $VAE_CMD; then
    print_success "VAE training completed successfully"
else
    print_error "VAE training failed"
    exit 1
fi

# Find the latest VAE checkpoint
VAE_EXPERIMENT_DIR=$(ls -td "$EXPERIMENTS_ROOT"/multisubject_vae_*subjects 2>/dev/null | head -n1)
if [ -z "$VAE_EXPERIMENT_DIR" ]; then
    print_error "Could not find VAE experiment directory"
    exit 1
fi

VAE_CHECKPOINT="$VAE_EXPERIMENT_DIR/checkpoints/best_model.pt"
if [ ! -f "$VAE_CHECKPOINT" ]; then
    # Try alternative checkpoint names
    VAE_CHECKPOINT="$VAE_EXPERIMENT_DIR/checkpoints/vae_best.pt"
    if [ ! -f "$VAE_CHECKPOINT" ]; then
        VAE_CHECKPOINT=$(ls "$VAE_EXPERIMENT_DIR"/checkpoints/*.pt 2>/dev/null | head -n1)
        if [ -z "$VAE_CHECKPOINT" ]; then
            print_error "No VAE checkpoint found in $VAE_EXPERIMENT_DIR/checkpoints/"
            exit 1
        fi
    fi
fi

print_success "VAE checkpoint found: $VAE_CHECKPOINT"

print_status "================================================="
print_success "Multi-subject VAE training pipeline completed!"
print_status "📁 VAE model saved to: $VAE_CHECKPOINT"
print_status "🔄 Next: Train BiLSTM using this VAE as feature extractor"
print_status "================================================="

echo ""
echo "To use this VAE for BiLSTM training, update your BiLSTM config with:"
echo "vae_checkpoint_path: \"$VAE_CHECKPOINT\"" 
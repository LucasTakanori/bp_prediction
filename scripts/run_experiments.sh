#!/bin/bash

# Blood Pressure Prediction Experiments Runner
# Usage: ./scripts/run_experiments.sh [vae|bilstm|both] [config_path] [data_dir]

set -e  # Exit on any error

# Default values
EXPERIMENT_TYPE=${1:-"both"}
CONFIG_PATH=${2:-""}
DATA_DIR=${3:-"/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data"}
OUTPUT_DIR="experiments"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}       Blood Pressure Prediction Training       ${NC}"
echo -e "${BLUE}================================================${NC}"

# Check if we're in the right directory
if [ ! -f "scripts/train_model.py" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Function to run training
run_training() {
    local model_type=$1
    local config_file=$2
    
    echo -e "${YELLOW}Starting $model_type training...${NC}"
    echo -e "${BLUE}Config: $config_file${NC}"
    echo -e "${BLUE}Data Dir: $DATA_DIR${NC}"
    echo -e "${BLUE}Output Dir: $OUTPUT_DIR${NC}"
    
    python scripts/train_model.py \
        --config "$config_file" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --device "auto"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $model_type training completed successfully${NC}"
    else
        echo -e "${RED}✗ $model_type training failed${NC}"
        return 1
    fi
}

# Main execution
case $EXPERIMENT_TYPE in
    "vae")
        echo -e "${YELLOW}Running VAE experiment only${NC}"
        if [ -z "$CONFIG_PATH" ]; then
            CONFIG_PATH="configs/vae_example.yaml"
        fi
        run_training "VAE" "$CONFIG_PATH"
        ;;
    "bilstm")
        echo -e "${YELLOW}Running BiLSTM experiment only${NC}"
        if [ -z "$CONFIG_PATH" ]; then
            CONFIG_PATH="configs/bilstm_example.yaml"
        fi
        run_training "BiLSTM" "$CONFIG_PATH"
        ;;
    "both")
        echo -e "${YELLOW}Running sequential VAE -> BiLSTM experiments${NC}"
        
        # Step 1: Train VAE
        echo -e "${BLUE}Step 1: Training VAE${NC}"
        VAE_CONFIG="configs/vae_example.yaml"
        run_training "VAE" "$VAE_CONFIG"
        
        # Step 2: Train BiLSTM
        echo -e "${BLUE}Step 2: Training BiLSTM${NC}"
        BILSTM_CONFIG="configs/bilstm_example.yaml"
        run_training "BiLSTM" "$BILSTM_CONFIG"
        ;;
    *)
        echo -e "${RED}Error: Invalid experiment type '$EXPERIMENT_TYPE'${NC}"
        echo -e "${YELLOW}Usage: $0 [vae|bilstm|both] [config_path] [data_dir]${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}           All experiments completed!           ${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "${BLUE}Results saved in: $OUTPUT_DIR${NC}" 
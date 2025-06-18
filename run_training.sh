#!/bin/bash

# Sophisticated BP Predictor - Training Launcher
# Simple script to launch training with different configurations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Sophisticated BP Predictor Training Launcher ===${NC}"
echo ""

# Check if Python environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Warning: No virtual environment detected.${NC}"
    echo -e "${YELLOW}Consider activating your environment: source ../myenv/bin/activate${NC}"
    echo ""
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [CONFIG] [OPTIONS]"
    echo ""
    echo "Available configurations:"
    echo "  quick       - Quick test (5 epochs, small model)"
    echo "  production  - Production training (50 epochs, full model)"
    echo "  full        - Full sophisticated configuration"
    echo ""
    echo "Options:"
    echo "  --dry-run   - Show configuration without training"
    echo "  --verbose   - Verbose output"
    echo "  --epochs N  - Override number of epochs"
    echo "  --batch-size N - Override batch size"
    echo ""
    echo "Examples:"
    echo "  $0 quick                    # Quick test run"
    echo "  $0 production --epochs 100  # Production with 100 epochs"
    echo "  $0 full --dry-run           # Show full config without training"
    echo ""
}

# Parse arguments
CONFIG=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        quick|production|full)
            CONFIG="$1"
            shift
            ;;
        --epochs)
            EXTRA_ARGS="$EXTRA_ARGS --override training_config.num_epochs=$2"
            shift 2
            ;;
        --batch-size)
            EXTRA_ARGS="$EXTRA_ARGS --override training_config.batch_size=$2"
            shift 2
            ;;
        --dry-run|--verbose)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Set default config if none provided
if [[ -z "$CONFIG" ]]; then
    echo -e "${YELLOW}No configuration specified. Using 'quick' by default.${NC}"
    CONFIG="quick"
fi

# Map config names to files
case $CONFIG in
    quick)
        CONFIG_FILE="configs/quick_test.yaml"
        ;;
    production)
        CONFIG_FILE="configs/production.yaml"
        ;;
    full)
        CONFIG_FILE="configs/sophisticated_bp_predictor.yaml"
        ;;
    *)
        echo -e "${RED}Unknown configuration: $CONFIG${NC}"
        show_usage
        exit 1
        ;;
esac

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}Using configuration: $CONFIG_FILE${NC}"
echo -e "${BLUE}Additional arguments: $EXTRA_ARGS${NC}"
echo ""

# Construct the command
CMD="python scripts/train_with_config.py --config $CONFIG_FILE $EXTRA_ARGS"

echo -e "${BLUE}Executing command:${NC}"
echo "$CMD"
echo ""

# Ask for confirmation unless it's a dry run
if [[ "$EXTRA_ARGS" != *"--dry-run"* ]]; then
    read -p "Continue with training? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Training cancelled.${NC}"
        exit 0
    fi
    echo ""
fi

# Execute the command
echo -e "${GREEN}Starting training...${NC}"
echo ""

# Run the training command
eval $CMD

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}=== Training completed successfully! ===${NC}"
else
    echo ""
    echo -e "${RED}=== Training failed! ===${NC}"
    exit 1
fi 
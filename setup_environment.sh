#!/bin/bash
# Environment setup script for BP Prediction Project
# Source this file to set up environment variables

# Activate virtual environment if it exists
VENV_PATH="../myenv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "🐍 Virtual environment activated: myenv"
elif [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
    echo "🐍 Virtual environment activated: venv"
elif [ -f "../venv/bin/activate" ]; then
    source "../venv/bin/activate"
    echo "🐍 Virtual environment activated: ../venv"
else
    echo "⚠️  No virtual environment found. Please activate manually if needed."
fi

# Data paths
export BP_DATA_ROOT="/home/lucas_takanori/phd/data"
export BP_EXPERIMENTS_ROOT="./experiments"
export BP_CACHE_DIR="./cache"
export BP_LOGS_DIR="./logs"

# WandB configuration  
export WANDB_PROJECT="bp-prediction"
export WANDB_MODE="offline"
export WANDB_CONSOLE="off"
export WANDB_SILENT="true"

# Python paths
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create python alias if it doesn't exist
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        alias python=python3
        echo "📍 Created python -> python3 alias"
    fi
fi

# Create directories if they don't exist
mkdir -p "$BP_EXPERIMENTS_ROOT"
mkdir -p "$BP_CACHE_DIR" 
mkdir -p "$BP_LOGS_DIR"

echo "🌟 Environment setup complete!"
echo "📂 Data root: $BP_DATA_ROOT"
echo "🗂️  Experiments: $BP_EXPERIMENTS_ROOT"
echo "💾 Cache: $BP_CACHE_DIR"
echo "📋 Logs: $BP_LOGS_DIR"
echo "📊 WandB project: $WANDB_PROJECT (mode: $WANDB_MODE)"

# Validate data directory exists
if [ ! -d "$BP_DATA_ROOT" ]; then
    echo "⚠️  Warning: Data directory does not exist: $BP_DATA_ROOT"
    echo "   Please ensure the data directory is accessible"
else
    echo "✅ Data directory validated: $BP_DATA_ROOT"
    # List available data files
    echo "📁 Available data files:"
    ls -la "$BP_DATA_ROOT"/*.h5 2>/dev/null | head -5
fi 
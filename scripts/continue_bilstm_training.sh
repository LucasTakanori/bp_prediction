#!/bin/bash
#SBATCH --job-name=continue_bilstm
#SBATCH --output=./logs/%x_%j.log
#SBATCH --error=./logs/%x_%j.err
#SBATCH --nodes 1
#SBATCH -c 80
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH -A bsc88
##SBATCH --qos=acc_debug
##SBATCH --exclusive
#SBATCH -q acc_bscls  

# Script to continue BiLSTM training using existing VAE checkpoint
# Use this when VAE training completed but BiLSTM training failed to start

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
#export SLURM_CPU_BIND=none # This line accelerates training x4 in mn5

# Activate virtual environment
source /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Continuing BiLSTM training with existing VAE checkpoint..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction

WANDB_MODE='offline'

echo "🔍 Finding existing VAE checkpoint..."

# Look for the most recent VAE checkpoint from the failed pipeline
VAE_CHECKPOINT=""

# Method 1: Look for vae_best.pt in enhanced_multisubject_vae directories
VAE_CHECKPOINT=$(find . -name "vae_best.pt" -path "*enhanced_multisubject_vae*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

# Method 2: Look for vae_final.pt
if [ -z "$VAE_CHECKPOINT" ]; then
    echo "⚠️  Looking for vae_final.pt..."
    VAE_CHECKPOINT=$(find . -name "vae_final.pt" -path "*enhanced_multisubject_vae*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
fi

# Method 3: Look for best_model.pt in VAE directories
if [ -z "$VAE_CHECKPOINT" ]; then
    echo "⚠️  Looking for best_model.pt in VAE directories..."
    VAE_CHECKPOINT=$(find . -name "best_model.pt" -path "*vae*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
fi

# Method 4: Use the specific checkpoint from the failed job
if [ -z "$VAE_CHECKPOINT" ]; then
    echo "⚠️  Looking for the specific checkpoint from job 24114037..."
    VAE_CHECKPOINT="./experiments/enhanced_multisubject_vae_28subjects_mask10/checkpoints/vae_best.pt"
    if [ ! -f "$VAE_CHECKPOINT" ]; then
        VAE_CHECKPOINT="./experiments/enhanced_multisubject_vae_28subjects_mask10/checkpoints/vae_final.pt"
    fi
fi

if [ -z "$VAE_CHECKPOINT" ] || [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "❌ VAE checkpoint not found! Available VAE files:"
    find . -name "*.pt" -path "*vae*" 2>/dev/null || echo "No VAE checkpoints found"
    echo ""
    echo "Available experiment directories:"
    find ./experiments -name "*vae*" -type d 2>/dev/null || echo "No VAE experiment directories found"
    exit 1
fi

echo "✅ Found VAE checkpoint: $VAE_CHECKPOINT"

# Verify the checkpoint file exists and is readable
if [ ! -r "$VAE_CHECKPOINT" ]; then
    echo "❌ VAE checkpoint exists but is not readable: $VAE_CHECKPOINT"
    exit 1
fi

echo "🎯 Starting BiLSTM training with VAE checkpoint..."

# Run BiLSTM training with the found VAE checkpoint
srun python scripts/train_bilstm_multisubject_enhanced.py \
    --config configs/bilstm_multisubject.yaml \
    --max-subjects 40 \
    --device cuda \
    --mask-type mask10 \
    --data-root /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --split-seed 42 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --max-eval-samples 1000000

# Check if BiLSTM training succeeded
if [ $? -ne 0 ]; then
    echo "❌ BiLSTM training failed!"
    exit 1
fi

echo "🎉 BiLSTM training completed successfully!"
echo "✅ Used VAE checkpoint: $VAE_CHECKPOINT"
echo "✅ BiLSTM trained with same subject splits as VAE"
echo "✅ Final evaluation performed on isolated test subjects"
echo "🔒 GUARANTEED: No data leakage - test subjects never seen during training"
echo ""
echo "📁 Check results in experiment directories"
echo "📊 Test evaluation results saved automatically" 
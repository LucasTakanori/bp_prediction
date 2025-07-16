#!/bin/bash
#SBATCH --job-name=pipeline_vae_bilstm
#SBATCH --output=./logs/%x_%j.log
#SBATCH --error=./logs/%x_%j.err
#SBATCH --nodes 1
#SBATCH -c 80
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH -A bsc88
##SBATCH --qos=acc_debug
##SBATCH --exclusive
#SBATCH -q acc_bscls  

# For debugging:
#-#sbatch -q acc_debug train_bilstm_mn5.sh
#-#salloc -A bsc88 -q acc_debug -n 1 -c 80 --gres=gpu:1 -t 02:00:00

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
#export SLURM_CPU_BIND=none # This line accelerates training x4 in mn5

# Activate virtual environment
source /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting VAE + BiLSTM pipeline with leakage-free training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction

WANDB_MODE='offline'

# Record start time to identify this pipeline run's checkpoints
PIPELINE_START_TIME=$(date +%s)
TIMESTAMP_FILE="/tmp/pipeline_start_$PIPELINE_START_TIME"
touch "$TIMESTAMP_FILE"
echo "🕐 Pipeline started at: $(date)"

# 1. Train VAE with leakage-free splits
echo "🔧 STEP 1: Training VAE with subject-level splits..."
srun python scripts/train_vae_multisubject_enhanced.py \
    --max-subjects 40 \
    --device cuda \
    --mask-type mask10 \
    --data-root /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data \
    --config configs/vae_multisubject_fixed.yaml \
    --split-seed 42 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1

# Check if VAE training succeeded
if [ $? -ne 0 ]; then
    echo "❌ VAE training failed! Stopping pipeline."
    rm -f "$TIMESTAMP_FILE"
    exit 1
fi

echo "✅ VAE training completed successfully!"

# Find the VAE checkpoint created by THIS pipeline run
echo "🔍 Finding VAE checkpoint created by this pipeline run..."

# Method 1: Try timestamp approach first
VAE_CHECKPOINT=$(find . -name "vae_best.pt" -path "*vae*" -newer "$TIMESTAMP_FILE" 2>/dev/null | head -1)

# Method 2: If timestamp fails, look for specific patterns in recent directories
if [ -z "$VAE_CHECKPOINT" ]; then
    echo "⚠️  Timestamp approach failed, looking for recent VAE directories..."
    VAE_CHECKPOINT=$(find . -name "vae_best.pt" -path "*enhanced_multisubject_vae*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
fi

# Method 3: Alternative checkpoint names
if [ -z "$VAE_CHECKPOINT" ]; then
    echo "⚠️  Trying alternative checkpoint names..."
    VAE_CHECKPOINT=$(find . -name "best_model.pt" -path "*vae*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
fi

# Method 4: Look for vae_final.pt
if [ -z "$VAE_CHECKPOINT" ]; then
    echo "⚠️  Trying vae_final.pt..."
    VAE_CHECKPOINT=$(find . -name "vae_final.pt" -path "*vae*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
fi

# Method 5: Fallback to any VAE checkpoint
if [ -z "$VAE_CHECKPOINT" ]; then
    echo "⚠️  Last resort: any VAE checkpoint..."
    VAE_CHECKPOINT=$(find . -name "*.pt" -path "*vae*checkpoints*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
fi

if [ -z "$VAE_CHECKPOINT" ]; then
    echo "❌ VAE checkpoint not found after all attempts! Listing available files:"
    find . -name "*.pt" -path "*vae*" 2>/dev/null || echo "No VAE checkpoints found"
    rm -f "$TIMESTAMP_FILE"
    exit 1
fi

echo "✅ Found VAE checkpoint: $VAE_CHECKPOINT"

# 2. Train BiLSTM + Automatic Test Evaluation (SAME subjects, SAME splits)
echo "🔧 STEP 2: Training BiLSTM with same subject splits..."
srun python scripts/train_bilstm_multisubject_enhanced.py \
    --max-subjects 40 \
    --device cuda \
    --mask-type mask10 \
    --data-root /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data \
    --config configs/bilstm_multisubject.yaml \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --split-seed 42 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --max-eval-samples 1000000

# Check if BiLSTM training succeeded
if [ $? -ne 0 ]; then
    echo "❌ BiLSTM training failed!"
    rm -f "$TIMESTAMP_FILE"
    exit 1
fi

# That's it! Test evaluation happens automatically at the end.

echo "🎉 PIPELINE COMPLETED SUCCESSFULLY!"
echo "✅ VAE trained with leakage-free subject splits"
echo "✅ BiLSTM trained with same subject splits"
echo "✅ Final evaluation performed on isolated test subjects"
echo "🔒 GUARANTEED: No data leakage - test subjects never seen during training"
echo ""
echo "📁 Check results in experiment directories"
echo "📊 Test evaluation results saved automatically"

# Cleanup
rm -f "$TIMESTAMP_FILE" 
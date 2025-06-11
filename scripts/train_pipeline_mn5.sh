#!/bin/bash
#SBATCH --job-name=BP_Pipeline_Training
#SBATCH --output=./logs/%x_%j.log
#SBATCH --error=./logs/%x_%j.err
#SBATCH --nodes 1
#SBATCH -c 80
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH -A bsc88
##SBATCH --qos=acc_debug
##SBATCH --exclusive
#SBATCH -q acc_bscls  

# For debugging:
#-#sbatch -q acc_debug train_pipeline_mn5.sh
#-#salloc -A bsc88 -q acc_debug -n 1 -c 80 --gres=gpu:2 -t 08:00:00

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
#export SLURM_CPU_BIND=none # This line accelerates training x4 in mn5

# Activate virtual environment
source /gpfs/projects/bsc88/speech/research/scripts/Lucas/vae_lstm_test/test/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting complete BP prediction pipeline training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction

# Step 1: Train VAE
echo "========================================="
echo "STEP 1: Training VAE"
echo "========================================="

srun python scripts/train_model.py \
    --config configs/vae_mn5.yaml \
    --device auto \
    --debug

if [ $? -ne 0 ]; then
    echo "ERROR: VAE training failed!"
    exit 1
fi

echo "VAE training completed successfully!"

# Wait a bit to ensure checkpoint is written
sleep 10

# Check if VAE checkpoint was created
VAE_CHECKPOINT="/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/vae_outputs/checkpoints/vae_best.pt"
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "ERROR: VAE checkpoint not found at $VAE_CHECKPOINT"
    echo "Looking for alternative checkpoint files..."
    find /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/vae_outputs/checkpoints/ -name "*.pt" -ls
    exit 1
fi

echo "VAE checkpoint confirmed: $VAE_CHECKPOINT"

# Step 2: Train BiLSTM
echo "========================================="
echo "STEP 2: Training BiLSTM"
echo "========================================="

srun python scripts/train_model.py \
    --config configs/bilstm_mn5.yaml \
    --device auto \
    --debug

if [ $? -ne 0 ]; then
    echo "ERROR: BiLSTM training failed!"
    exit 1
fi

echo "========================================="
echo "PIPELINE TRAINING COMPLETED SUCCESSFULLY!"
echo "========================================="
echo "VAE results: vae_outputs/"
echo "BiLSTM results: lstm_outputs/"
echo "Job completed at: $(date)" 
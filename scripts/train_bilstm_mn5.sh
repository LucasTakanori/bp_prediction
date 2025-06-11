#!/bin/bash
#SBATCH --job-name=BiLSTM_Training_New
#SBATCH --output=./logs/%x_%j.log
#SBATCH --error=./logs/%x_%j.err
#SBATCH --nodes 1
#SBATCH -c 80
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH -A bsc88
##SBATCH --qos=acc_debug
##SBATCH --exclusive
#SBATCH -q acc_bscls  

# For debugging:
#-#sbatch -q acc_debug train_bilstm_mn5.sh
#-#salloc -A bsc88 -q acc_debug -n 1 -c 80 --gres=gpu:2 -t 06:00:00

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
#export SLURM_CPU_BIND=none # This line accelerates training x4 in mn5

# Activate virtual environment
source /gpfs/projects/bsc88/speech/research/scripts/Lucas/vae_lstm_test/test/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting BiLSTM training with new modular architecture..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction

# Check if VAE checkpoint exists
VAE_CHECKPOINT="/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/vae_outputs/checkpoints/vae_best.pt"
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "Warning: VAE checkpoint not found at $VAE_CHECKPOINT"
    echo "Please train VAE first or update the checkpoint path in configs/bilstm_mn5.yaml"
    exit 1
fi

echo "Using VAE checkpoint: $VAE_CHECKPOINT"

# Run BiLSTM training with new modular system
srun python scripts/train_model.py \
    --config configs/bilstm_mn5.yaml \
    --device auto \
    --debug

echo "BiLSTM training completed!"
echo "Check results in lstm_outputs/ directory" 
#!/bin/bash
#SBATCH --job-name=VAE_Training_New
#SBATCH --output=./logs/%x_%j.log
#SBATCH --error=./logs/%x_%j.err
#SBATCH --nodes 1
#SBATCH -c 80
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -A bsc88
#SBATCH --qos=acc_debug
##SBATCH --exclusive
##SBATCH -q acc_bscls  

# For debugging:
#-#sbatch -q acc_debug train_vae_mn5.sh
#-#salloc -A bsc88 -q acc_debug -n 1 -c 80 --gres=gpu:1 -t 02:00:00

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
#export SLURM_CPU_BIND=none # This line accelerates training x4 in mn5

# Activate virtual environment
source /gpfs/projects/bsc88/speech/research/scripts/Lucas/vae_lstm_test/test/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting VAE training with new modular architecture..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction

WANDB_MODE='offline'

# Run VAE training with new modular system
srun python scripts/train_model.py \
    --config configs/vae_mn5.yaml \
    --device auto \
    --debug

echo "VAE training completed!"
echo "Check results in vae_outputs/ directory" 
#!/bin/bash
#SBATCH --job-name=transformer_Training_New
#SBATCH --output=./logs/%x_%j.log
#SBATCH --error=./logs/%x_%j.err
#SBATCH --nodes 1
#SBATCH -c 80
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH -A bsc88
##SBATCH --qos=acc_debug
##SBATCH --exclusive
#SBATCH -q acc_bscls  

# For debugging:
#-#sbatch -q acc_debug train_transformer_mn5.sh
#-#salloc -A bsc88 -q acc_debug -n 1 -c 80 --gres=gpu:4 -t 02:00:00

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
#export SLURM_CPU_BIND=none # This line accelerates training x4 in mn5

# Activate virtual environment
source /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Transformer training with new modular architecture..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction

WANDB_MODE='offline'

# Run Transformer training with new modular system
srun python scripts/train_transformer_multisubject_enhanced.py \
    --config configs/transformer_multisubject.yaml \
    --max-subjects 32 \
    --device cuda \
    --mask-type mask10 \
    --data-root /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data \
    --vae-checkpoint /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/experiments/enhanced_multisubject_vae_32subjects_mask10/checkpoints/vae_final.pt \
    --max-eval-samples 1000000

echo "Transformer training completed!"
echo "Check results in transformer_outputs/ directory" 
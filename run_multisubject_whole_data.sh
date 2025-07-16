#!/bin/bash
#SBATCH --job-name=vae_whole_data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/VAE_Whole_Data_%j.log
#SBATCH --error=logs/VAE_Whole_Data_%j.err

# Activate virtual environment
source /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting VAE training with WHOLE DATA (no masking)..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction

WANDB_MODE='offline'

# Run the new whole data training script
srun python scripts/train_vae_multisubject_whole_data.py \
    --config configs/vae_multisubject_whole_data.yaml \
    --max-subjects 2 \
    --data-root /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data \
    --device cuda \
    --wandb \
    --debug

echo "VAE WHOLE DATA training completed!"
echo "Expected: 2,101 samples instead of 735 samples (186% increase!)"
echo "Check results in experiments/ directory" 
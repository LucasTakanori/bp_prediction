# Activate virtual environment
source /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting VAE training with new modular architecture..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction

WANDB_MODE='offline'

# Run the new whole data training script
srun python scripts/train_vae_multisubject_enhanced.py --config configs/vae_multisubject_fixed.yaml --max-subjects 2 --device cuda --mask-type mask10 --data-root /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data
echo "VAE training completed!"
echo "Check results in vae_outputs/ directory" 

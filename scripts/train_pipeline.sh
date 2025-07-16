#!/bin/bash
#SBATCH --job-name=bilstm_Training_New
#SBATCH --output=./logs/%x_%j.log
#SBATCH --error=./logs/%x_%j.err
#SBATCH --nodes 1
#SBATCH -c 80
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -A bsc88
#SBATCH --qos=acc_debug
#SBATCH --exclusive
##SBATCH -q acc_bscls  

# For debugging:
#-#sbatch -q acc_debug train_bilstm_mn5.sh
#-#salloc -A bsc88 -q acc_debug -n 1 -c 80 --gres=gpu:1 -t 02:00:00

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
#export SLURM_CPU_BIND=none # This line accelerates training x4 in mn5

# Activate virtual environment
source /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting bilstm training with new modular architecture..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512    # Avoids some OOMs due to memory fragmentation
export NCCL_BUFFSIZE=1048576                            # Avoids some OOM due to NCCL communication overhead
WANDB_MODE='offline'
export NCCL_P2P_DISABLE=1 # Disable NCCL P2P communication to reduce overhead
# Run bilstm training with new modular system
srun python scripts/train_pipeline_vae_bilstm.py \
    --config configs/pipeline_vae_bilstm.yaml \
    --max-subjects 32 \
    --mask-type mask10 \
    --data-root /gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data \
    --device cuda



echo "bilstm training completed!"
echo "Check results in bilstm_outputs/ directory" 
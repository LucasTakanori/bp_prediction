#!/bin/bash
#SBATCH --job-name=lstm_training
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
#-#sbatch -q acc_debug launch.sh
#-#salloc -A bsc81 -q acc_debug -n 1 -c 80 --gres=gpu:4 -t 02:00:00

# adapting script for instruction tuning
# "caller_train_salamandra40b.sh"
# /gpfs/projects/bsc88/text/models/instruction-tuning/it-chat-v1/caller_train_salamandra40b.sh

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
#export SLURM_CPU_BIND=none # This line accelerates training x4 in mn5

source /gpfs/projects/bsc88/speech/research/scripts/Lucas/vae_lstm_test/test/bin/activate

#chmod +x launch.sh
echo "run optimized bilstm"



python optimized_training_script.py \
   --data_path "/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data/subject001_baseline_masked.h5"   \
   --vae_checkpoint "/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/vae_outputs/checkpoints/vae_best.pt"   \
   --output_dir "/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/train/optimized_ouptut" \
   --latent_dim 512 \
   --lstm_hidden_dim 256 \
   --lstm_layers 3 \
   --num_epochs 1 \
   --loss_type composite \
   --attention_dim 128 \
   --batch_size 64 \
   --num_workers 16 \
   --vae_batch_size 128 \
   --wandb_mode disabled


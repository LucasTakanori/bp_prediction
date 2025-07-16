# =============================================================================
# Individual Training Examples - VAE + Enhanced BiLSTM
# =============================================================================

# =============================================================================
# EXAMPLE 1: Basic Training Pipeline
# =============================================================================

# Step 1: Train VAE with recommended parameters
python tuned_vae.py \
    --output_dir "./vae_output" \
    --data_path "~/phd/data/subject001_baseline_masked.h5" \
    --latent_dim 512 \
    --num_epochs 100 \
    --beta_min 0.01 \
    --beta_max 0.8 \
    --beta_warmup_epochs 30 \
    --wandb_project "vae-training" \
    --wandb_name "vae_experiment_001" \
    --wandb_mode offline

# Step 2: Train BiLSTM using the VAE checkpoint
python enhanced_vae_bilstm.py \
    --output_dir "./bilstm_output" \
    --data_path "~/phd/data/subject001_baseline_masked.h5" \
    --vae_checkpoint "./vae_output/checkpoints/vae_best.pt" \
    --latent_dim 256 \
    --lstm_hidden_dim 256 \
    --lstm_layers 3 \
    --num_epochs 25 \
    --batch_size 8 \
    --use_attention \
    --attention_dim 128 \
    --loss_type composite \
    --visualize_attention \
    --wandb_project "bilstm-training" \
    --wandb_name "bilstm_composite_001" \
    --wandb_mode offline

# =============================================================================
# EXAMPLE 2: High-Performance Configuration
# =============================================================================

# Step 1: Train VAE with larger latent space and longer training
python tuned_vae.py \
    --output_dir "./experiments/high_perf/vae" \
    --data_path "~/phd/data/subject001_baseline_masked.h5" \
    --latent_dim 512 \
    --num_epochs 50 \
    --beta_min 0.005 \
    --beta_max 1.2 \
    --beta_warmup_epochs 20 \
    --wandb_project "high-performance-bp" \
    --wandb_name "vae_high_perf" \
    --wandb_mode online

# Step 2: Train enhanced BiLSTM with larger architecture
python enhanced_vae_bilstm.py \
    --output_dir "./experiments/high_perf/bilstm" \
    --data_path "~/phd/data/subject001_baseline_masked.h5" \
    --vae_checkpoint "./experiments/high_perf/vae/checkpoints/vae_best.pt" \
    --latent_dim 512 \
    --lstm_hidden_dim 512 \
    --lstm_layers 4 \
    --num_epochs 40 \
    --batch_size 6 \
    --use_attention \
    --attention_dim 256 \
    --loss_type composite \
    --visualize_attention \
    --wandb_project "high-performance-bp" \
    --wandb_name "bilstm_high_perf" \
    --wandb_mode online

# =============================================================================
# EXAMPLE 3: Systolic Blood Pressure Focus
# =============================================================================

# Step 1: Train VAE (same as basic)
python tuned_vae.py \
    --output_dir "./experiments/systolic_focus/vae" \
    --data_path "~/phd/data/subject001_baseline_masked.h5" \
    --latent_dim 256 \
    --num_epochs 30 \
    --beta_min 0.01 \
    --beta_max 1.0 \
    --beta_warmup_epochs 15 \
    --wandb_project "systolic-focus-bp" \
    --wandb_name "vae_systolic_focus"

# Step 2: Train BiLSTM with systolic distance loss
python enhanced_vae_bilstm.py \
    --output_dir "./experiments/systolic_focus/bilstm" \
    --data_path "~/phd/data/subject001_baseline_masked.h5" \
    --vae_checkpoint "./experiments/systolic_focus/vae/checkpoints/vae_best.pt" \
    --latent_dim 256 \
    --lstm_hidden_dim 256 \
    --lstm_layers 3 \
    --num_epochs 30 \
    --batch_size 8 \
    --use_attention \
    --attention_dim 128 \
    --loss_type systolic_distance \
    --visualize_attention \
    --wandb_project "systolic-focus-bp" \
    --wandb_name "bilstm_systolic_focus"

# =============================================================================
# EXAMPLE 4: Ablation Study - Different Loss Functions
# =============================================================================

# Train VAE once
python tuned_vae.py \
    --output_dir "./experiments/ablation/vae" \
    --data_path "~/phd/data/subject001_baseline_masked.h5" \
    --latent_dim 256 \
    --num_epochs 30 \
    --wandb_project "ablation-study"

# Train multiple BiLSTM models with different loss functions
VAE_CHECKPOINT="./experiments/ablation/vae/checkpoints/vae_best.pt"

# 1. MSE loss (baseline)
python enhanced_vae_bilstm.py \
    --output_dir "./experiments/ablation/bilstm_mse" \
    --vae_checkpoint "$VAE_CHECKPOINT" \
    --loss_type mse \
    --num_epochs 20 \
    --wandb_name "ablation_mse"

# 2. Systolic distance loss
python enhanced_vae_bilstm.py \
    --output_dir "./experiments/ablation/bilstm_systolic" \
    --vae_checkpoint "$VAE_CHECKPOINT" \
    --loss_type systolic_distance \
    --use_attention \
    --num_epochs 20 \
    --wandb_name "ablation_systolic"

# 3. Diastolic distance loss
python enhanced_vae_bilstm.py \
    --output_dir "./experiments/ablation/bilstm_diastolic" \
    --vae_checkpoint "$VAE_CHECKPOINT" \
    --loss_type diastolic_distance \
    --use_attention \
    --num_epochs 20 \
    --wandb_name "ablation_diastolic"

# 4. Composite loss
python enhanced_vae_bilstm.py \
    --output_dir "./experiments/ablation/bilstm_composite" \
    --vae_checkpoint "$VAE_CHECKPOINT" \
    --loss_type composite \
    --use_attention \
    --visualize_attention \
    --num_epochs 25 \
    --wandb_name "ablation_composite"

# =============================================================================
# EXAMPLE 5: Quick Testing Configuration
# =============================================================================

# For rapid prototyping and testing
python tuned_vae.py \
    --output_dir "./test/vae" \
    --data_path "~/phd/data/subject001_baseline_masked.h5" \
    --latent_dim 128 \
    --num_epochs 10 \
    --wandb_mode disabled

python enhanced_vae_bilstm.py \
    --output_dir "./test/bilstm" \
    --vae_checkpoint "./test/vae/checkpoints/vae_best.pt" \
    --latent_dim 128 \
    --lstm_hidden_dim 128 \
    --lstm_layers 2 \
    --num_epochs 5 \
    --batch_size 4 \
    --loss_type mse \
    --wandb_mode disabled

# =============================================================================
# EXAMPLE 6: Multi-Subject Training
# =============================================================================

# Train on multiple subjects sequentially
SUBJECTS=("subject001" "subject002" "subject003")
BASE_DIR="./experiments/multi_subject"

for subject in "${SUBJECTS[@]}"; do
    echo "Training on $subject..."
    
    # Train VAE for this subject
    python tuned_vae.py \
        --output_dir "$BASE_DIR/${subject}/vae" \
        --data_path "~/phd/data/${subject}_baseline_masked.h5" \
        --latent_dim 256 \
        --num_epochs 25 \
        --wandb_project "multi-subject-bp" \
        --wandb_name "vae_${subject}"
    
    # Train BiLSTM for this subject
    python enhanced_vae_bilstm.py \
        --output_dir "$BASE_DIR/${subject}/bilstm" \
        --data_path "~/phd/data/${subject}_baseline_masked.h5" \
        --vae_checkpoint "$BASE_DIR/${subject}/vae/checkpoints/vae_best.pt" \
        --latent_dim 256 \
        --num_epochs 20 \
        --use_attention \
        --loss_type composite \
        --wandb_project "multi-subject-bp" \
        --wandb_name "bilstm_${subject}"
done

# =============================================================================
# Parameter Explanations
# =============================================================================

# VAE Parameters:
# --latent_dim: Size of the latent space (128, 256, 512)
# --num_epochs: Training epochs (20-50 typical)
# --beta_min/max: KL divergence weight range (0.01-1.0)
# --beta_warmup_epochs: Gradual beta increase period

# BiLSTM Parameters:
# --lstm_hidden_dim: LSTM hidden state size (128, 256, 512)
# --lstm_layers: Number of LSTM layers (2-4)
# --attention_dim: Attention mechanism size (64, 128, 256)
# --loss_type: Loss function (mse, systolic_distance, diastolic_distance, composite)
# --use_attention: Enable attention mechanism (recommended)
# --visualize_attention: Generate attention visualizations

# Performance Tips:
# - Start with composite loss for best overall performance
# - Use attention for better temporal modeling
# - Larger latent_dim (256-512) for complex data
# - Batch size 6-8 for good GPU utilization
# - 25-30 epochs usually sufficient for convergence

# Basic pipeline
python training_workflow.py --data_path ~/phd/data/subject001_baseline_masked.h5

# Ablation study
python training_workflow.py --data_path ~/phd/data/subject001_baseline_masked.h5 --ablation

# High-performance configuration
python training_workflow.py \
    --data_path ~/phd/data/subject001_baseline_masked.h5 \
    --latent_dim 512 \
    --vae_epochs 40 \
    --bilstm_epochs 30


 Key Parameters
VAE Parameters:

--latent_dim: 128, 256, 512 (higher = more capacity)
--num_epochs: 20-50 (30 is good default)
--beta_min/max: 0.01-1.0 (KL weight range)

BiLSTM Parameters:

--loss_type: composite (best overall), systolic_distance, diastolic_distance, mse
--use_attention: Enables attention mechanism (recommended)
--lstm_hidden_dim: 256, 512 (higher = more capacity)
--visualize_attention: Generate attention pattern plots

📊 Expected Outputs
Each run creates:

Checkpoints: Best model weights
Results: Training curves, predictions, attention visualizations
Metrics: Detailed evaluation statistics
Logs: W&B experiment tracking
Summary: Complete pipeline report

The pipeline is designed to be robust, well-documented, and production-ready! 🎉
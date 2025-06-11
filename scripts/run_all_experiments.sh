#!/bin/bash
# Batch experiment runner for BP prediction
# Runs VAE and BiLSTM experiments across multiple subjects

set -e  # Exit on error

SUBJECTS=("S01" "S02" "S03")  # Add your subjects here
BASE_OUTPUT_DIR="experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/batch_experiments_${TIMESTAMP}"

# Create log directory
mkdir -p ${LOG_DIR}

echo "Starting batch experiments at $(date)"
echo "Results will be saved to: ${BASE_OUTPUT_DIR}"
echo "Logs will be saved to: ${LOG_DIR}"

# Function to run VAE experiment
run_vae_experiment() {
    local subject=$1
    local config_file="configs/vae_${subject,,}.yaml"
    local output_dir="${BASE_OUTPUT_DIR}/vae_${subject,,}_${TIMESTAMP}"
    
    echo "Training VAE for subject ${subject}..."
    
    # Create subject-specific config if it doesn't exist
    if [[ ! -f $config_file ]]; then
        sed "s/S01/${subject}/g" configs/vae_example.yaml > $config_file
        sed -i "s/bp_prediction_vae/bp_prediction_vae_${subject,,}/g" $config_file
    fi
    
    python scripts/train_model.py \
        --config $config_file \
        --output-dir $output_dir \
        > ${LOG_DIR}/vae_${subject,,}_train.log 2>&1
    
    echo "VAE training completed for ${subject}"
    
    # Evaluate VAE
    echo "Evaluating VAE for subject ${subject}..."
    python scripts/evaluate_model.py \
        --checkpoint ${output_dir}/vae_*/checkpoints/best_model.pth \
        --config $config_file \
        --create-plots \
        --save-predictions \
        --output-dir ${output_dir}/evaluation \
        > ${LOG_DIR}/vae_${subject,,}_eval.log 2>&1
    
    echo "VAE evaluation completed for ${subject}"
}

# Function to run BiLSTM experiment
run_bilstm_experiment() {
    local subject=$1
    local config_file="configs/bilstm_${subject,,}.yaml"
    local output_dir="${BASE_OUTPUT_DIR}/bilstm_${subject,,}_${TIMESTAMP}"
    
    echo "Training BiLSTM for subject ${subject}..."
    
    # Create subject-specific config if it doesn't exist
    if [[ ! -f $config_file ]]; then
        sed "s/S01/${subject}/g" configs/bilstm_example.yaml > $config_file
        sed -i "s/bp_prediction_bilstm/bp_prediction_bilstm_${subject,,}/g" $config_file
    fi
    
    python scripts/train_model.py \
        --config $config_file \
        --output-dir $output_dir \
        > ${LOG_DIR}/bilstm_${subject,,}_train.log 2>&1
    
    echo "BiLSTM training completed for ${subject}"
    
    # Evaluate BiLSTM
    echo "Evaluating BiLSTM for subject ${subject}..."
    python scripts/evaluate_model.py \
        --checkpoint ${output_dir}/bilstm_*/checkpoints/best_model.pth \
        --config $config_file \
        --create-plots \
        --save-predictions \
        --output-dir ${output_dir}/evaluation \
        > ${LOG_DIR}/bilstm_${subject,,}_eval.log 2>&1
    
    echo "BiLSTM evaluation completed for ${subject}"
}

# Main execution
for subject in "${SUBJECTS[@]}"; do
    echo "=== Processing Subject: ${subject} ==="
    
    # Run VAE experiment
    run_vae_experiment $subject
    
    # Run BiLSTM experiment
    run_bilstm_experiment $subject
    
    echo "Completed all experiments for ${subject}"
    echo ""
done

# Generate summary report
echo "=== Experiment Summary ===" > ${LOG_DIR}/summary.txt
echo "Batch started: $(date)" >> ${LOG_DIR}/summary.txt
echo "Subjects processed: ${SUBJECTS[@]}" >> ${LOG_DIR}/summary.txt
echo "Total experiments: $((${#SUBJECTS[@]} * 2))" >> ${LOG_DIR}/summary.txt
echo "Results directory: ${BASE_OUTPUT_DIR}" >> ${LOG_DIR}/summary.txt

echo "All batch experiments completed successfully!"
echo "Check logs in: ${LOG_DIR}"
echo "Check results in: ${BASE_OUTPUT_DIR}" 
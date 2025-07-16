# Improved BP Predictor Evaluation

This document describes how to evaluate the Improved BP Predictor model that was trained with the continuous configuration.

## Model Information

- **Model Path**: `/home/lucas_takanori/phd/bp_prediction/experiments/improved_bp_predictor_continuous/checkpoints/best_model.pt`
- **Configuration**: `configs/sophisticated_bp_predictor_improved.yaml`
- **Architecture**: ImprovedBPPredictor with multi-frame temporal modeling
- **Features**: 
  - VAE-based feature extraction (128-dim latent space)
  - BiLSTM with attention mechanism
  - Multi-head self-attention
  - Physiological feature extraction
  - Separate heads for SBP/DBP prediction

## Quick Start

### Option 1: Use the Shell Script (Recommended)

```bash
# Run the evaluation with default settings
./scripts/run_improved_bp_evaluation.sh
```

### Option 2: Run Python Script Directly

```bash
# Basic evaluation
python scripts/evaluate_improved_bp_predictor.py

# Custom evaluation with specific parameters
python scripts/evaluate_improved_bp_predictor.py \
    --model_path "/home/lucas_takanori/phd/bp_prediction/experiments/improved_bp_predictor_continuous/checkpoints/best_model.pt" \
    --config_path "configs/sophisticated_bp_predictor_improved.yaml" \
    --output_dir "./my_evaluation_results" \
    --max_samples 1000
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | Auto-detected | Path to the trained model checkpoint |
| `--config_path` | Auto-detected | Path to the YAML configuration file |
| `--output_dir` | `./evaluation_results_improved_bp` | Directory to save results |
| `--max_samples` | 500 | Maximum number of samples to evaluate |

## Output Files

The evaluation will generate the following files in the output directory:

### 📊 Visualization
- `improved_bp_evaluation.png` - Comprehensive 9-panel evaluation plot
  - Correlation plots (predicted vs true)
  - Bland-Altman plots (difference vs mean)
  - Error distribution histograms
  - For waveform, systolic, and diastolic predictions

### 📄 Metrics
- `evaluation_metrics.txt` - Human-readable metrics summary
- `evaluation_metrics.json` - Machine-readable metrics in JSON format

### 🔍 Metrics Included

**Accuracy Metrics:**
- R² Score (coefficient of determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- Pearson correlation coefficient

**Clinical Accuracy:**
- Percentage within 5, 10, 15 mmHg tolerance
- For both systolic and diastolic predictions

**Statistical Analysis:**
- Mean bias (systematic error)
- Standard deviation of differences
- Limits of agreement (95% confidence intervals)

## Expected Performance

Based on the improved architecture and training configuration:

- **Systolic BP**: MAE < 15 mmHg, R² > 0.6
- **Diastolic BP**: MAE < 10 mmHg, R² > 0.5
- **Waveform**: MAE < 8 mmHg, R² > 0.7
- **Clinical Accuracy**: >70% within 10 mmHg tolerance

## Model Architecture Details

### Key Improvements Over Previous Versions:
1. **Reduced Data Overlap**: Sequence step size increased to 15 to prevent overfitting
2. **Better Loss Balancing**: Adjusted weights (waveform: 0.5, SBP: 0.25, DBP: 0.25)
3. **Continuous Predictions**: Smooth BP value extraction to prevent discrete patterns
4. **Enhanced Regularization**: Improved dropout and weight initialization
5. **Physiological Constraints**: Pulse pressure validation and smooth extraction

### Architecture Components:
- **VAE Encoder**: 128-dimensional latent space
- **BiLSTM**: 2-layer bidirectional LSTM (256 total hidden units)
- **Multi-Head Attention**: 4 heads with current frame bias
- **Temporal Modeling**: 7-frame sequences with pattern offsets [-4, -3, -2, -1, 0, 1, 2]
- **Output Heads**: Separate prediction heads for waveform, SBP, and DBP

## Troubleshooting

### Common Issues:

1. **Model file not found**
   ```
   ❌ Model file not found: /path/to/model.pt
   ```
   - Ensure the model was trained successfully
   - Check if the path is correct
   - Verify file permissions

2. **CUDA out of memory**
   - Reduce `--max_samples` to a smaller number (e.g., 200)
   - The evaluation uses batch size 16 by default

3. **Config file mismatch**
   - Ensure you're using the same config file that was used for training
   - Check that VAE checkpoint path in config is correct

4. **Import errors**
   - Ensure all dependencies are installed
   - Check that the project root is in your Python path

### Performance Considerations:

- **Memory Usage**: ~2-4 GB GPU memory for 500 samples
- **Evaluation Time**: ~2-5 minutes for 500 samples on GPU
- **CPU Fallback**: Automatic fallback to CPU if CUDA unavailable

## Interpretation of Results

### 🎯 Good Performance Indicators:
- **High R²** (>0.6): Model captures variance well
- **Low MAE** (<15 mmHg): Clinically acceptable accuracy
- **Centered Bland-Altman**: No systematic bias
- **High Clinical Accuracy**: >70% within tolerance

### ⚠️ Warning Signs:
- **Low R²** (<0.3): Poor model fit
- **High MAE** (>20 mmHg): Clinically unacceptable
- **Biased Bland-Altman**: Systematic under/over-prediction
- **Block Patterns**: Discrete predictions instead of continuous

### 🔧 If Performance is Poor:
1. Check if correct model and config files are used
2. Verify VAE checkpoint is loaded properly
3. Consider retraining with different hyperparameters
4. Analyze attention weights for temporal modeling issues

## Data Pipeline

The evaluation uses the same data pipeline as training:
1. Load PVI data from HDF5 file
2. Extract 7-frame sequences with step size 15
3. Apply VAE-compatible normalization [0, 1]
4. Predict BP waveform, SBP, and DBP
5. Compare with ground truth using smooth BP extraction

## Contact & Support

For issues or questions about the evaluation:
- Check the training logs for model performance during training
- Verify that the evaluation dataset matches the training dataset
- Ensure configuration consistency between training and evaluation

---

**Last Updated**: January 2025  
**Model Version**: ImprovedBPPredictor v2.1.0  
**Configuration**: sophisticated_bp_predictor_improved.yaml 
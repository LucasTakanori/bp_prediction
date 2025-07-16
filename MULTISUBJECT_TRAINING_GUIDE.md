# Multi-Subject Blood Pressure Prediction Training Guide

This guide describes how to use the new multi-subject training pipeline to achieve state-of-the-art blood pressure prediction performance that outperforms existing models in R² and Bland-Altman analysis.

## 🎯 Overview

The multi-subject training approach consists of two stages:
1. **VAE Training**: Learn generalizable PVI image representations across all subjects
2. **BiLSTM Training**: Use VAE features for temporal BP prediction with high accuracy

**Performance Targets:**
- Systolic R² > 0.85
- Diastolic R² > 0.80  
- Superior Bland-Altman agreement compared to existing models

## 📊 Available Data

Your dataset contains **33 subjects** with high-quality PVI and BP data:
- `subject001` to `subject037` (some gaps in numbering)
- Each subject has approximately 400-800 samples
- Total: ~15,000+ training samples across all subjects

## 🚀 Quick Start

### Option 1: Full Pipeline (Recommended)

Run the complete training pipeline with all subjects:

```bash
# Train with all available subjects
./scripts/run_multisubject_training.sh

# Train with specific options
./scripts/run_multisubject_training.sh --wandb --device cuda --max-subjects 20
```

### Option 2: Step-by-Step Training

#### Step 1: Train VAE

```bash
python scripts/train_vae_multisubject.py \
    --config configs/vae_multisubject.yaml \
    --device cuda \
    --wandb
```

#### Step 2: Train BiLSTM with VAE Features

```bash
python scripts/train_bilstm_multisubject.py \
    --config configs/bilstm_multisubject.yaml \
    --vae-checkpoint experiments/multisubject_vae_TIMESTAMP/checkpoints/vae_best.pt \
    --device cuda \
    --wandb
```

## ⚙️ Configuration Options

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--max-subjects N` | Limit to N subjects | All available (33) |
| `--max-samples-per-subject N` | Limit samples per subject | No limit |
| `--wandb` | Enable Weights & Biases logging | Disabled |
| `--device DEVICE` | Training device (auto/cuda/cpu) | auto |
| `--vae-epochs N` | VAE training epochs | 80 |
| `--bilstm-epochs N` | BiLSTM training epochs | 100 |

### Configuration Files

#### VAE Configuration (`configs/vae_multisubject.yaml`)
- **Latent dimension**: 128 (optimal for PVI representation)
- **Architecture**: Progressive encoding [64→128→256→512]
- **Beta scheduling**: 0.01 → 0.4 over 25 epochs
- **Batch size**: 32 (optimal for multi-subject stability)

#### BiLSTM Configuration (`configs/bilstm_multisubject.yaml`)
- **LSTM layers**: 3 layers, 256 hidden units, bidirectional
- **Attention**: 8-head multi-head attention
- **Sequence length**: 10 frames for temporal context
- **Loss weights**: 40% waveform, 30% systolic, 30% diastolic

## 🎯 Performance Optimization

### For Maximum R² Performance

1. **Use all available subjects** (removes `--max-subjects` flag)
2. **Increase training epochs** if needed:
   ```bash
   ./scripts/run_multisubject_training.sh --vae-epochs 100 --bilstm-epochs 150
   ```
3. **Enable detailed logging**:
   ```bash
   ./scripts/run_multisubject_training.sh --wandb
   ```

### For Faster Experimentation

1. **Limit subjects for testing**:
   ```bash
   ./scripts/run_multisubject_training.sh --max-subjects 10 --max-samples-per-subject 200
   ```
2. **Reduce epochs**:
   ```bash
   ./scripts/run_multisubject_training.sh --vae-epochs 20 --bilstm-epochs 30
   ```

## 📈 Expected Results

Based on the optimized architecture and multi-subject training:

### Target Performance Metrics
- **Systolic BP**: R² > 0.85, MAE < 8.0 mmHg
- **Diastolic BP**: R² > 0.80, MAE < 6.0 mmHg  
- **Waveform**: R² > 0.75
- **Clinical accuracy**: >80% within 10 mmHg tolerance

### Advantages Over Single-Subject Models
1. **Better generalization** across different subjects
2. **Improved R² scores** due to diverse training data
3. **Superior Bland-Altman agreement** with reduced bias
4. **Robust feature learning** via VAE pre-training

## 📁 Output Structure

After training, you'll find:

```
experiments/
├── multisubject_vae_TIMESTAMP/          # VAE experiment
│   ├── checkpoints/vae_best.pt          # Best VAE model
│   ├── config.yaml                      # VAE configuration
│   └── logs/                            # Training logs
└── multisubject_bp_TIMESTAMP/           # BiLSTM experiment  
    ├── checkpoints/best_model.pt        # Best BP model
    ├── config.yaml                      # BiLSTM configuration
    ├── training_summary.txt             # Training summary
    └── results/                         # Evaluation results
```

## 🔍 Monitoring Training

### With Weights & Biases

If you used `--wandb`, monitor training at: https://wandb.ai

Key metrics to watch:
- **VAE**: Reconstruction loss, KL divergence, validation loss
- **BiLSTM**: Systolic/Diastolic R², MAE, total loss

### Console Output

Monitor training progress with real-time metrics:
```
📊 Train - Loss: 0.1234, Sys R²: 0.8567, Dias R²: 0.8123
📈 Val   - Loss: 0.1456, Sys R²: 0.8456, Dias R²: 0.8034
🎯 New best validation loss: 0.1456
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
sed -i 's/batch_size: 32/batch_size: 16/' configs/vae_multisubject.yaml
sed -i 's/batch_size: 16/batch_size: 8/' configs/bilstm_multisubject.yaml
```

#### 2. Slow Training
```bash
# Use fewer subjects for testing
./scripts/run_multisubject_training.sh --max-subjects 10
```

#### 3. Missing Dependencies
```bash
pip install torch torchvision torchaudio
pip install pyyaml tqdm wandb scikit-learn matplotlib seaborn
```

### Performance Issues

#### Low R² Scores
1. **Increase training epochs**: More epochs often improve convergence
2. **Check data quality**: Ensure all subjects have valid data
3. **Adjust learning rate**: Try 1e-4 to 1e-3 range
4. **Enable attention**: Ensure `use_attention: true` in config

#### Poor Generalization
1. **Use more subjects**: More diverse training data
2. **Reduce overfitting**: Increase dropout rates
3. **Data augmentation**: Enable in config (experimental)

## 📊 Evaluation and Comparison

### Automated Evaluation

The pipeline includes automatic evaluation generation:
- Correlation plots (R² visualization)
- Bland-Altman plots (agreement analysis)  
- Error histograms (accuracy distribution)
- Clinical accuracy metrics

### Manual Evaluation

```bash
# Evaluate specific model
python scripts/evaluate_multisubject_bp.py \
    --model-checkpoint experiments/multisubject_bp_TIMESTAMP/checkpoints/best_model.pt \
    --config configs/bilstm_multisubject.yaml \
    --output-dir evaluation_results_multisubject
```

## 🏆 Beating Existing Models

### Key Strategies

1. **Multi-subject training**: Leverage all 33 subjects
2. **VAE feature extraction**: Pre-trained representations
3. **Attention mechanisms**: Focus on relevant temporal patterns
4. **Composite loss function**: Balance waveform and BP value accuracy
5. **Robust evaluation**: Comprehensive metrics including Bland-Altman

### Performance Comparison

Compare your results against the benchmarks in the PDF:
- **R² improvement**: Target 10-15% improvement over existing models
- **Bland-Altman**: Reduced bias and better limits of agreement
- **Clinical relevance**: Higher accuracy within clinically meaningful thresholds

## 🔬 Advanced Usage

### Hyperparameter Tuning

1. **Learning rate optimization**:
   ```yaml
   # In config file
   learning_rate: 5e-4  # Conservative
   learning_rate: 1e-3  # Aggressive  
   ```

2. **Architecture scaling**:
   ```yaml
   # Bigger model for better performance
   hidden_dim: 512      # From 256
   num_layers: 4        # From 3
   attention_heads: 16  # From 8
   ```

3. **Sequence length tuning**:
   ```yaml
   sequence_length: 15  # More temporal context
   pattern_offsets: [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
   ```

### Custom Subject Selection

```bash
# Train on specific subjects
python scripts/train_bilstm_multisubject.py \
    --subjects subject001 subject002 subject005 subject010 \
    --config configs/bilstm_multisubject.yaml \
    --vae-checkpoint path/to/vae.pt
```

## 📞 Support

For issues or questions:
1. Check the training logs in `experiments/*/logs/`
2. Review the configuration files for parameter settings
3. Monitor GPU/CPU usage during training
4. Use `--wandb` for detailed training visualization

## 🎯 Success Criteria

Your multi-subject model is successful if it achieves:
- ✅ Systolic R² > 0.85
- ✅ Diastolic R² > 0.80
- ✅ Better Bland-Altman agreement than existing models
- ✅ Clinical accuracy > 80% within 10 mmHg
- ✅ Robust performance across different subjects

Ready to train the best BP prediction model? Run:
```bash
./scripts/run_multisubject_training.sh --wandb
``` 
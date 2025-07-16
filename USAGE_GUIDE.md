# BP Prediction Project - Usage Guide

## 🚀 Quick Start

### 1. Environment Setup

First, set up your environment variables:

```bash
# Source the environment setup script
source setup_environment.sh

# Verify environment is set up correctly
python3 scripts/validate_setup.py
```

### 2. Train a VAE Model

```bash
# Train VAE with baseline configuration
python3 scripts/train_enhanced.py --config configs/vae_baseline.yaml

# Train with custom settings
python3 scripts/train_enhanced.py \
    --config configs/vae_baseline.yaml \
    --epochs 50 \
    --batch-size 32 \
    --wandb-mode online
```

### 3. Train a BiLSTM Model

```bash
# First ensure you have a trained VAE model
# Then train BiLSTM
python3 scripts/train_enhanced.py --config configs/bilstm_baseline.yaml
```

## 📋 Configuration System

### Environment Variables

The system uses environment variables for flexible path management:

- `BP_DATA_ROOT`: Root directory containing your data files (default: `/home/lucas_takanori/phd/data`)
- `BP_EXPERIMENTS_ROOT`: Directory for experiment outputs (default: `./experiments`)
- `BP_CACHE_DIR`: Cache directory (default: `./cache`)
- `WANDB_PROJECT`: WandB project name (default: `bp-prediction`)
- `WANDB_MODE`: WandB mode - `online`, `offline`, or `disabled` (default: `offline`)

### Configuration Files

Configuration files follow a standardized structure:

```yaml
# Example configuration structure
environment:
  data_root: "${BP_DATA_ROOT:/home/lucas_takanori/phd/data}"
  experiments_root: "${BP_EXPERIMENTS_ROOT:./experiments}"

data_config:
  subject: "subject001"
  session: "baseline"
  # ... other data settings

model_config:
  model_type: "vae"  # or "bilstm"
  # ... model-specific settings

training_config:
  num_epochs: 30
  batch_size: 16
  # ... training settings

logging_config:
  use_wandb: true
  wandb_mode: "offline"
```

## 🛠️ Advanced Usage

### Command Line Options

The enhanced training script supports many command-line options:

```bash
python3 scripts/train_enhanced.py \
    --config configs/vae_baseline.yaml \
    --data-root /custom/data/path \
    --output-dir /custom/output/path \
    --experiment-name my_experiment \
    --epochs 100 \
    --batch-size 64 \
    --device cuda \
    --wandb-mode online \
    --debug
```

### Validation and Testing

```bash
# Validate configuration without training
python3 scripts/train_enhanced.py \
    --config configs/vae_baseline.yaml \
    --validate-only

# Dry run (setup everything but don't train)
python3 scripts/train_enhanced.py \
    --config configs/vae_baseline.yaml \
    --dry-run

# Debug mode with verbose logging
python3 scripts/train_enhanced.py \
    --config configs/vae_baseline.yaml \
    --debug
```

### Environment Setup Validation

```bash
# Test your complete setup
python3 scripts/validate_setup.py
```

This will check:
- ✅ Environment variables
- ✅ Required Python packages
- ✅ Path management system
- ✅ Data file availability
- ✅ Configuration file validity

## 📊 Experiment Management

### Directory Structure

Each experiment creates a timestamped directory:

```
experiments/
├── vae_baseline_20240101_120000/
│   ├── checkpoints/
│   │   ├── vae_epoch_5.pt
│   │   ├── vae_epoch_10.pt
│   │   └── vae_best.pt
│   ├── logs/
│   │   └── training.log
│   ├── results/
│   │   ├── training_curves.png
│   │   └── metrics.json
│   ├── reconstructions/
│   │   ├── recon_epoch_1.png
│   │   └── recon_epoch_5.png
│   └── configs/
│       └── config.yaml
```

### WandB Integration

The system integrates with Weights & Biases for experiment tracking:

- **Offline mode**: Logs stored locally, can sync later
- **Online mode**: Real-time syncing to WandB cloud
- **Disabled mode**: No WandB logging

## 🔧 Troubleshooting

### Common Issues

1. **Data not found**:
   ```bash
   # Check your data path
   echo $BP_DATA_ROOT
   ls -la $BP_DATA_ROOT/*.h5
   ```

2. **Configuration errors**:
   ```bash
   # Validate your config
python3 scripts/train_enhanced.py --config configs/your_config.yaml --validate-only
   ```

3. **Import errors**:
   ```bash
   # Check dependencies
python3 scripts/validate_setup.py
   ```

4. **Permission errors**:
   ```bash
   # Make scripts executable
   chmod +x setup_environment.sh scripts/*.py
   ```

### Debug Mode

Use debug mode for detailed logging:

```bash
python3 scripts/train_enhanced.py \
    --config configs/vae_baseline.yaml \
    --debug
```

## 🎯 Best Practices

1. **Always validate first**:
   ```bash
   source setup_environment.sh
python3 scripts/validate_setup.py
   ```

2. **Use configuration files** for reproducible experiments

3. **Monitor with WandB** for experiment tracking

4. **Use meaningful experiment names**:
   ```bash
   --experiment-name vae_latent512_lr1e3
   ```

5. **Keep data organized** with consistent subject/session naming

## 🔄 Migration from Old Scripts

If you have existing training scripts, you can gradually migrate:

1. **Immediate**: Use new config files with environment variables
2. **Short-term**: Switch to `train_enhanced.py` for better logging
3. **Long-term**: Adopt the full training pipeline architecture

The system is backward-compatible with existing `tuned_vae.py` and `bilstm.py` scripts. 
# Blood Pressure Prediction Training on MareNostrum 5

## New Modular Architecture

The project now uses a clean, modular architecture with:
- **`src/`**: Modular source code (data, models, training, utils)
- **`configs/`**: YAML configuration files
- **`scripts/`**: Training scripts and SBATCH files

## Available SBATCH Scripts

### 1. Individual Training Scripts

#### VAE Training
```bash
sbatch scripts/train_vae_mn5.sh
```
- **Time**: 2 hours
- **Resources**: 1 GPU, 80 CPUs
- **Output**: `vae_outputs/`
- **Config**: `configs/vae_mn5.yaml`

#### BiLSTM Training
```bash
sbatch scripts/train_bilstm_mn5.sh
```
- **Time**: 6 hours
- **Resources**: 2 GPUs, 80 CPUs
- **Output**: `lstm_outputs/`
- **Config**: `configs/bilstm_mn5.yaml`
- **Requires**: VAE checkpoint from previous training

#### Complete Pipeline
```bash
sbatch scripts/train_pipeline_mn5.sh
```
- **Time**: 8 hours
- **Resources**: 2 GPUs, 80 CPUs
- **Runs**: VAE → BiLSTM sequentially
- **Output**: Both `vae_outputs/` and `lstm_outputs/`

### 2. Queue Options

For **debugging** (short jobs, quick allocation):
```bash
# Edit the script to uncomment:
# #SBATCH --qos=acc_debug
# and comment out:
# #SBATCH -q acc_bscls
```

For **production** (long jobs, normal queue):
```bash
# Keep default settings with:
# #SBATCH -q acc_bscls
```

## Configuration Files

### VAE Configuration (`configs/vae_mn5.yaml`)
```yaml
model_config:
  model_type: "vae"
  latent_dim: 512
  input_channels: 1
  input_height: 32
  input_width: 32

training_config:
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 16
```

### BiLSTM Configuration (`configs/bilstm_mn5.yaml`)
```yaml
model_config:
  model_type: "bilstm"
  latent_dim: 512  # Must match VAE
  hidden_dim: 256
  num_layers: 3
  use_attention: true

training_config:
  num_epochs: 20
  learning_rate: 0.0001
  batch_size: 32
```

## Key Improvements Over Old System

1. **No Hardcoded Paths**: All paths in configuration files
2. **Proper Error Handling**: Validates data and configurations
3. **Efficient Data Loading**: Caching and optimized loaders
4. **Modular Architecture**: Easy to modify and extend
5. **Clinical Metrics**: Proper BP evaluation metrics
6. **Resource Optimization**: Mixed precision, gradient clipping

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

View logs:
```bash
tail -f logs/VAE_Training_New_JOBID.log
tail -f logs/BiLSTM_Training_New_JOBID.log
```

Cancel job:
```bash
scancel JOBID
```

## Output Structure

```
vae_outputs/
├── checkpoints/
│   ├── vae_best.pt
│   └── vae_last.pt
├── results/
└── logs/

lstm_outputs/
├── checkpoints/
│   ├── bilstm_best.pt
│   └── bilstm_last.pt
├── results/
└── logs/
```

## Troubleshooting

1. **Data not found**: Check paths in config files
2. **VAE checkpoint missing**: Train VAE first or update path in BiLSTM config
3. **Out of memory**: Reduce batch_size in config files
4. **Job fails immediately**: Check logs in `logs/` directory

## Example Usage Workflow

1. **Start with VAE training**:
   ```bash
   sbatch scripts/train_vae_mn5.sh
   ```

2. **Monitor progress**:
   ```bash
   squeue -u $USER
   tail -f logs/VAE_Training_New_*.log
   ```

3. **After VAE completes, train BiLSTM**:
   ```bash
   sbatch scripts/train_bilstm_mn5.sh
   ```

4. **Or run complete pipeline**:
   ```bash
   sbatch scripts/train_pipeline_mn5.sh
   ``` 
# Blood Pressure Prediction - New Modular Architecture

This document describes the new modular architecture implemented for the blood pressure prediction project. The architecture addresses the critical issues identified in the original codebase and provides a clean, scalable, and maintainable solution.

## Architecture Overview

The new architecture follows a clean separation of concerns with the following structure:

```
src/
├── data/           # Data handling and preprocessing
├── models/         # Model definitions and architectures
├── training/       # Training infrastructure
└── utils/          # Utilities and configuration

scripts/           # Training and evaluation scripts
configs/           # Configuration files
docs/             # Documentation
```

## Key Improvements

### 1. **Modular Design**
- Clear separation between data, models, training, and utilities
- Each module has well-defined responsibilities
- Easy to test, extend, and maintain individual components

### 2. **Configuration Management**
- YAML-based configuration system
- Environment variable support for paths
- Separate configs for data, model, and training parameters

### 3. **Efficient Data Loading**
- Proper caching and memory management
- Batched processing instead of frame-by-frame
- Data validation and integrity checks
- Flexible sequence building strategies

### 4. **Improved Model Architecture**
- Unified VAE implementation with proper loss scaling
- BiLSTM with actual attention mechanisms
- Combined VAE-BiLSTM architecture option
- Proper batch processing throughout

### 5. **Comprehensive Training Infrastructure**
- Mixed precision training support
- Advanced callbacks (early stopping, checkpointing, scheduling)
- Comprehensive metrics and validation
- Integration with TensorBoard and Weights & Biases

## Components

### Data Module (`src/data/`)

#### Datasets (`datasets.py`)
- `PVIDataset`: Handles PVI data loading with validation and caching
- `BPDataset`: Creates sequences from PVI data for training
- `DataValidator`: Ensures data integrity

#### Data Loaders (`loaders.py`)
- `DataManager`: Central data management with train/val/test splits
- `CollateFunction`: Custom batching for variable-length sequences
- `SequentialSampler`: Maintains temporal ordering

#### Preprocessing (`preprocessing.py`)
- `ImageProcessor`: Frame preprocessing with CLAHE and normalization
- `BPProcessor`: Signal filtering and feature extraction
- `DataAugmentor`: Training-time data augmentation
- `SequenceBuilder`: Flexible sequence creation strategies

### Models Module (`src/models/`)

#### VAE (`vae.py`)
- Modern VAE implementation with configurable architecture
- Proper loss formulation with β-VAE support
- Efficient batch processing

#### BiLSTM (`bilstm.py`)
- Bidirectional LSTM with actual attention mechanisms
- CNN feature extractor for frame processing
- Residual connections and dropout for regularization

#### Losses (`losses.py`)
- Clinical BP loss with systolic/diastolic weighting
- Waveform shape and temporal consistency losses
- Combined loss for VAE-BiLSTM training

### Training Module (`src/training/`)

#### Trainers (`trainers.py`)
- `BaseTrainer`: Common training functionality
- `VAETrainer`: Specialized for VAE training
- `BiLSTMTrainer`: Specialized for BP prediction
- Mixed precision, gradient clipping, checkpointing

#### Metrics (`metrics.py`)
- Comprehensive clinical metrics (AAMI, BHS standards)
- Waveform similarity measures
- Error statistics and correlation analysis

#### Callbacks (`callbacks.py`)
- Early stopping with customizable criteria
- Model checkpointing with best model saving
- Learning rate scheduling
- Metrics logging and visualization

### Utilities (`src/utils/`)

#### Configuration (`config.py`)
- `DataConfig`, `ModelConfig`, `TrainingConfig` classes
- YAML loading with environment variable substitution
- Configuration validation

#### Logging (`logging.py`)
- Structured logging setup
- File and console output
- Configurable log levels

## Usage

### 1. Training a Model

```bash
# Train a VAE model
python scripts/train_model.py --config configs/vae_example.yaml

# Train a BiLSTM model with custom data directory
python scripts/train_model.py \
    --config configs/bilstm_example.yaml \
    --data-dir /path/to/your/data \
    --output-dir experiments/my_experiment

# Resume training from checkpoint
python scripts/train_model.py \
    --config configs/bilstm_example.yaml \
    --resume experiments/bilstm_20231201_143022/checkpoints/best_model.pth
```

### 2. Evaluating a Model

```bash
# Evaluate on test set with plots
python scripts/evaluate_model.py \
    --checkpoint experiments/bilstm_20231201_143022/checkpoints/best_model.pth \
    --config configs/bilstm_example.yaml \
    --create-plots \
    --save-predictions

# Evaluate on validation set
python scripts/evaluate_model.py \
    --checkpoint path/to/checkpoint.pth \
    --split val \
    --output-dir evaluation_results
```

### 3. Configuration Examples

#### VAE Configuration (`configs/vae_example.yaml`)
```yaml
data_config:
  data_root: "/path/to/data"
  subject: "S01"
  session: "Session1"
  sequence_length: 3
  pattern_offsets: [-7, 0, 3]

model_config:
  model_type: "vae"
  input_channels: 1
  latent_dim: 128
  hidden_dims: [32, 64, 128, 256]

training_config:
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 32
  vae_beta: 1.0
```

#### BiLSTM Configuration (`configs/bilstm_example.yaml`)
```yaml
data_config:
  data_root: "/path/to/data"
  subject: "S01"
  session: "Session1"

model_config:
  model_type: "bilstm"
  hidden_dim: 256
  num_layers: 2
  use_attention: true
  output_dim: 100

training_config:
  num_epochs: 200
  learning_rate: 0.0005
  batch_size: 16
  systolic_weight: 2.0
  diastolic_weight: 2.0
```

## Advanced Usage

### Custom Model Development

```python
from src.models import BaseModel
from src.training import BaseTrainer

class CustomModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        # Your custom architecture
    
    def forward(self, x):
        # Your forward pass
        return output

class CustomTrainer(BaseTrainer):
    def train_epoch(self):
        # Custom training logic
        pass
```

### Custom Data Processing

```python
from src.data.preprocessing import DataPreprocessor

class CustomPreprocessor(DataPreprocessor):
    def preprocess_sample(self, frames, bp_signal, augment=False):
        # Custom preprocessing logic
        return processed_frames, processed_bp, metadata
```

### Custom Metrics

```python
from src.training.metrics import MetricsCalculator

class CustomMetrics(MetricsCalculator):
    def compute_custom_metric(self, predictions, targets):
        # Your custom metric computation
        return metric_value
```

## Performance Considerations

### Memory Optimization
- Use data caching for repeated experiments
- Enable mixed precision training for larger batch sizes
- Configure appropriate number of data loading workers

### Training Efficiency
- Use appropriate learning rate scheduling
- Enable gradient clipping for stable training
- Monitor training with early stopping

### Evaluation
- Use batch processing for faster evaluation
- Save predictions for detailed post-analysis
- Generate comprehensive plots for clinical validation

## Migration from Old Code

To migrate from the old codebase:

1. **Update data paths**: Remove hardcoded paths, use configuration files
2. **Replace data loading**: Use new `DataManager` instead of custom loaders
3. **Update model definitions**: Use new unified model architectures
4. **Replace training loops**: Use new trainer classes with proper validation
5. **Update evaluation**: Use comprehensive metrics and plotting utilities

## Best Practices

1. **Always use configuration files** instead of hardcoding parameters
2. **Validate data integrity** before training
3. **Monitor training metrics** and use early stopping
4. **Save comprehensive results** including configurations
5. **Use version control** for experiments and configurations
6. **Document model changes** and experimental setups

## Troubleshooting

### Common Issues
- **Data loading errors**: Check paths in configuration files
- **Memory issues**: Reduce batch size or enable gradient checkpointing
- **Slow training**: Increase number of workers or use mixed precision
- **Poor convergence**: Adjust learning rate or model architecture

### Debug Mode
```bash
python scripts/train_model.py --config config.yaml --debug --dry-run
```

This enables detailed logging and runs validation without actual training.

## Contributing

When adding new features:
1. Follow the modular architecture patterns
2. Add comprehensive tests
3. Update documentation
4. Use type hints and docstrings
5. Ensure backward compatibility where possible

## References

- Original codebase analysis and improvements
- Clinical BP prediction standards (AAMI, BHS)
- Modern deep learning best practices
- PyTorch training recommendations 
# Blood Pressure Prediction Project - Code Organization Report

## Executive Summary

This report analyzes the current codebase for the blood pressure prediction project and proposes comprehensive improvements focusing on path management, logging standardization, configuration validation, and training pipeline optimization based on the working `tuned_vae.py` implementation.

## Current State Analysis

### 🔍 **Strengths**
- ✅ Working VAE and BiLSTM implementations (`tuned_vae.py`, `bilstm.py`)
- ✅ Modular architecture with separated concerns (`src/` structure)
- ✅ WandB integration for experiment tracking
- ✅ Configuration management system

### ⚠️ **Issues Identified**

#### 1. Path Management
- **Inconsistent paths**: Mix of absolute/relative paths across configs
- **Hardcoded paths**: Environment-specific paths in config files
- **No path validation**: Missing checks for data directory existence

#### 2. Configuration Management
- **Config inconsistency**: Different structures between example and MN5 configs
- **Missing validation**: No validation of config parameters
- **Path coupling**: Configs tightly coupled to specific environments

#### 3. Logging & Monitoring
- **Inconsistent logging**: Mixed logging approaches across modules
- **WandB integration**: Not standardized across all training scripts
- **Error handling**: Insufficient error logging and recovery

#### 4. Training Pipeline
- **Code duplication**: Similar training loops across different scripts
- **Missing abstractions**: No common base for different model trainers
- **Checkpoint management**: Inconsistent checkpoint saving strategies

## Proposed Improvements

### 🎯 **Phase 1: Path Management & Environment Setup**

#### 1.1 Environment Configuration
```python
# .env file support
BP_DATA_ROOT=/home/lucas_takanori/phd/data
BP_EXPERIMENTS_ROOT=./experiments
BP_CACHE_DIR=./cache
WANDB_PROJECT=bp-prediction
WANDB_ENTITY=your-entity
```

#### 1.2 Centralized Path Management
```python
# src/utils/paths.py
class PathManager:
    def __init__(self, config: DataConfig):
        self.config = config
        self.validate_paths()
    
    def validate_paths(self):
        """Validate all required paths exist"""
        if not Path(self.config.root_path).exists():
            raise FileNotFoundError(f"Data root not found: {self.config.root_path}")
    
    @property
    def data_file_path(self) -> Path:
        return Path(self.config.root_path) / self.config.data_file
```

### 🎯 **Phase 2: Configuration Standardization**

#### 2.1 Unified Config Structure
```yaml
# Standard config template
environment:
  data_root: "${BP_DATA_ROOT:/home/lucas_takanori/phd/data}"
  experiments_root: "${BP_EXPERIMENTS_ROOT:./experiments}"
  cache_dir: "${BP_CACHE_DIR:./cache}"

data_config:
  subject: "subject001"
  session: "baseline"
  # ... rest of data config

model_config:
  model_type: "vae"  # vae, bilstm, vae_bilstm
  # ... model specific config

training_config:
  # ... training config
  
logging_config:
  level: "INFO"
  use_wandb: true
  wandb_project: "${WANDB_PROJECT:bp-prediction}"
  wandb_mode: "offline"
```

#### 2.2 Config Validation
```python
# src/utils/config_validator.py
class ConfigValidator:
    def validate(self, config) -> List[str]:
        errors = []
        errors.extend(self._validate_data_config(config.data))
        errors.extend(self._validate_model_config(config.model))
        errors.extend(self._validate_training_config(config.training))
        return errors
```

### 🎯 **Phase 3: Standardized Logging with WandB**

#### 3.1 Logger Factory
```python
# src/utils/logger.py
class LoggerFactory:
    @staticmethod
    def create_logger(name: str, config: LoggingConfig) -> logging.Logger:
        logger = logging.getLogger(name)
        # Setup file and console handlers
        # Configure WandB if enabled
        return logger

    @staticmethod
    def setup_wandb(config: TrainingConfig, experiment_dir: Path):
        if config.use_wandb:
            import wandb
            wandb.init(
                project=config.project_name,
                mode=config.wandb_mode,
                dir=str(experiment_dir),
                config=config.__dict__
            )
```

### 🎯 **Phase 4: Enhanced Training Pipeline (Based on tuned_vae.py)**

#### 4.1 Base Trainer Class
```python
# src/training/base_trainer.py
class BaseTrainer:
    def __init__(self, model, config, data_loaders, device):
        self.model = model
        self.config = config
        self.train_loader, self.val_loader = data_loaders
        self.device = device
        
        self.logger = LoggerFactory.create_logger(self.__class__.__name__, config.logging)
        self.setup_training_components()
        self.setup_callbacks()
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, loss function"""
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch - to be implemented by subclasses"""
        raise NotImplementedError
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch - to be implemented by subclasses"""
        raise NotImplementedError
    
    def train(self):
        """Main training loop with proper error handling"""
        try:
            for epoch in range(self.config.num_epochs):
                train_metrics = self.train_epoch()
                val_metrics = self.validate_epoch()
                
                # Combine metrics
                metrics = {**train_metrics, **val_metrics}
                
                # Log to wandb and files
                self.log_metrics(epoch, metrics)
                
                # Run callbacks
                if self.callbacks.on_epoch_end(epoch, self.model, metrics):
                    break
                    
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
```

#### 4.2 VAE Trainer (Based on tuned_vae.py)
```python
# src/training/vae_trainer.py
class VAETrainer(BaseTrainer):
    def setup_training_components(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6
        )
    
    def vae_loss(self, recon_x, x, mu, logvar, beta):
        """VAE loss from tuned_vae.py"""
        # Implementation from working code
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """Training epoch with multiple frames like tuned_vae.py"""
        self.model.train()
        # Implementation based on tuned_vae.py
        pass
```

## Implementation Priority

### 🚀 **High Priority (Week 1)**
1. ✅ Path management with environment variables
2. ✅ Config validation and standardization
3. ✅ Unified logging with WandB integration

### 🔄 **Medium Priority (Week 2)**
1. ✅ Base trainer refactoring
2. ✅ VAE trainer based on tuned_vae.py
3. ✅ Enhanced callback system

### 📊 **Low Priority (Week 3)**
1. ✅ Advanced metrics and visualization
2. ✅ Performance optimization
3. ✅ Documentation improvements

## File Structure Changes

```
project/
├── configs/
│   ├── base/
│   │   ├── data.yaml
│   │   ├── model.yaml
│   │   └── training.yaml
│   ├── experiments/
│   │   ├── vae_baseline.yaml
│   │   ├── bilstm_baseline.yaml
│   │   └── vae_bilstm.yaml
│   └── environments/
│       ├── local.yaml
│       ├── hpc.yaml
│       └── cloud.yaml
├── src/
│   ├── utils/
│   │   ├── paths.py          # NEW: Path management
│   │   ├── logger.py         # NEW: Logging factory
│   │   └── config_validator.py  # NEW: Config validation
│   ├── training/
│   │   ├── base_trainer.py   # ENHANCED: Base trainer
│   │   ├── vae_trainer.py    # NEW: VAE trainer based on tuned_vae.py
│   │   └── bilstm_trainer.py # NEW: BiLSTM trainer
│   └── ...
└── scripts/
    ├── train_model.py        # ENHANCED: Main training script
    └── validate_setup.py    # NEW: Setup validation
```

## Migration Strategy

### Step 1: Implement Core Infrastructure
- [ ] Path management system
- [ ] Config validation
- [ ] Logger factory

### Step 2: Refactor Training Pipeline
- [ ] Extract base trainer from tuned_vae.py
- [ ] Implement VAE trainer
- [ ] Update training scripts

### Step 3: Update Configurations
- [ ] Standardize all config files
- [ ] Add environment variable support
- [ ] Validate all configurations

### Step 4: Testing & Validation
- [ ] Test with existing data
- [ ] Validate wandb integration
- [ ] Performance benchmarking

## Success Metrics

1. **Code Quality**: Reduced duplication, improved maintainability
2. **Configuration**: Consistent, validated, environment-agnostic configs
3. **Logging**: Comprehensive logging with WandB integration
4. **Training**: Robust, resumable training pipeline
5. **Portability**: Easy deployment across different environments

## Conclusion

These improvements will create a more robust, maintainable, and scalable codebase while preserving the working functionality from `tuned_vae.py` and `bilstm.py`. The focus on standardization and proper abstractions will make the project easier to extend and deploy across different environments. 
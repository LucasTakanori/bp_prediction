"""
Configuration validation for BP prediction project.
Validates configuration parameters and ensures consistency across different config types.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
# Optional import for torch - will be needed for training but not validation
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for configuration validation errors"""
    pass


class ConfigValidator:
    """Comprehensive configuration validator"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_data_config(self, data_config) -> List[str]:
        """Validate data configuration"""
        errors = []
        
        # Validate paths
        if not hasattr(data_config, 'root_path') or not data_config.root_path:
            errors.append("Data root path is required")
        elif not Path(data_config.root_path).exists():
            errors.append(f"Data root path does not exist: {data_config.root_path}")
        
        # Validate subject format
        if hasattr(data_config, 'subject'):
            if not data_config.subject.startswith('subject'):
                errors.append("Subject must start with 'subject'")
        
        # Validate session
        if not hasattr(data_config, 'session') or not data_config.session:
            errors.append("Session is required")
        
        # Validate sequence parameters
        if hasattr(data_config, 'sequence_length'):
            if data_config.sequence_length <= 0:
                errors.append("Sequence length must be positive")
        
        # Validate pattern offsets
        if hasattr(data_config, 'pattern_offsets') and data_config.pattern_offsets:
            if not isinstance(data_config.pattern_offsets, list):
                errors.append("Pattern offsets must be a list")
            elif len(data_config.pattern_offsets) == 0:
                errors.append("Pattern offsets cannot be empty")
        
        # Validate BP normalization
        if hasattr(data_config, 'bp_normalization'):
            if not isinstance(data_config.bp_normalization, (list, tuple)) or len(data_config.bp_normalization) != 2:
                errors.append("BP normalization must be a tuple/list of [min, max]")
            elif data_config.bp_normalization[0] >= data_config.bp_normalization[1]:
                errors.append("BP normalization min must be less than max")
        
        # Validate frame normalization
        if hasattr(data_config, 'frame_normalization'):
            valid_normalizations = ['minmax', 'standardize', 'none']
            if data_config.frame_normalization not in valid_normalizations:
                errors.append(f"Frame normalization must be one of {valid_normalizations}")
        
        return errors
    
    def validate_model_config(self, model_config) -> List[str]:
        """Validate model configuration"""
        errors = []
        
        # Model type validation
        valid_types = ['vae', 'bilstm', 'vae_bilstm']
        if model_config.model_type not in valid_types:
            errors.append(f"Invalid model_type '{model_config.model_type}'. Must be one of {valid_types}")
        
        # VAE specific validations
        if model_config.model_type in ['vae', 'vae_bilstm']:
            if model_config.latent_dim <= 0:
                errors.append("latent_dim must be positive")
            
            if model_config.input_channels <= 0:
                errors.append("input_channels must be positive")
                
            if model_config.input_height <= 0 or model_config.input_width <= 0:
                errors.append("input_height and input_width must be positive")
        
        # BiLSTM specific validations
        if model_config.model_type in ['bilstm', 'vae_bilstm']:
            if model_config.hidden_dim <= 0:
                errors.append("hidden_dim must be positive")
                
            if model_config.num_layers <= 0:
                errors.append("num_layers must be positive")
                
            if model_config.output_dim <= 0:
                errors.append("output_dim must be positive")
            
            # VAE checkpoint validation (optional - warn if missing but don't fail)
            if hasattr(model_config, 'vae_checkpoint_path') and model_config.vae_checkpoint_path:
                checkpoint_path = Path(model_config.vae_checkpoint_path)
                if not checkpoint_path.exists():
                    logger.warning(f"VAE checkpoint not found (will need to train VAE first): {checkpoint_path}")
                    # Don't add to errors - this is just a warning
        
        # Dropout validation
        if hasattr(model_config, 'dropout_rate'):
            if not (0 <= model_config.dropout_rate <= 1):
                errors.append("dropout_rate must be between 0 and 1")
        
        return errors
    
    def validate_training_config(self, training_config) -> List[str]:
        """Validate training configuration"""
        errors = []
        
        # Basic training parameters
        if hasattr(training_config, 'num_epochs') and training_config.num_epochs <= 0:
            errors.append("Number of epochs must be positive")
        
        if hasattr(training_config, 'learning_rate') and training_config.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if hasattr(training_config, 'batch_size') and training_config.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Optimizer validation
        if hasattr(training_config, 'optimizer'):
            valid_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
            if training_config.optimizer.lower() not in valid_optimizers:
                errors.append(f"Optimizer must be one of {valid_optimizers}")
        
        # Weight decay
        if hasattr(training_config, 'weight_decay') and training_config.weight_decay < 0:
            errors.append("Weight decay must be non-negative")
        
        # VAE-specific parameters
        if hasattr(training_config, 'vae_beta'):
            if training_config.vae_beta < 0:
                errors.append("VAE beta must be non-negative")
        
        # Scheduler validation
        if hasattr(training_config, 'use_scheduler') and training_config.use_scheduler:
            if hasattr(training_config, 'scheduler_type'):
                valid_schedulers = ['cosine', 'reduce_on_plateau', 'step', 'exponential']
                if training_config.scheduler_type not in valid_schedulers:
                    errors.append(f"Scheduler type must be one of {valid_schedulers}")
            
            if hasattr(training_config, 'scheduler_patience') and training_config.scheduler_patience <= 0:
                errors.append("Scheduler patience must be positive")
        
        # Early stopping validation
        if hasattr(training_config, 'early_stopping_patience') and training_config.early_stopping_patience <= 0:
            errors.append("Early stopping patience must be positive")
        
        if hasattr(training_config, 'early_stopping_min_delta') and training_config.early_stopping_min_delta < 0:
            errors.append("Early stopping min delta must be non-negative")
        
        # Gradient clipping
        if hasattr(training_config, 'grad_clip_norm') and training_config.grad_clip_norm <= 0:
            errors.append("Gradient clipping norm must be positive")
        
        # Output directory
        if hasattr(training_config, 'output_dir') and training_config.output_dir:
            output_path = Path(training_config.output_dir)
            if output_path.exists() and not output_path.is_dir():
                errors.append(f"Output path exists but is not a directory: {training_config.output_dir}")
        
        # WandB validation
        if hasattr(training_config, 'use_wandb') and training_config.use_wandb:
            if hasattr(training_config, 'wandb_mode'):
                valid_modes = ['online', 'offline', 'disabled']
                if training_config.wandb_mode not in valid_modes:
                    errors.append(f"WandB mode must be one of {valid_modes}")
            
            if not hasattr(training_config, 'project_name') or not training_config.project_name:
                errors.append("WandB project name is required when wandb is enabled")
        
        return errors
    
    def validate_config_consistency(self, data_config, model_config, training_config) -> List[str]:
        """Validate consistency between different config sections"""
        errors = []
        
        # Check if VAE latent dim matches BiLSTM input dim for combined models
        if hasattr(model_config, 'model_type') and model_config.model_type == 'vae_bilstm':
            vae_latent = getattr(model_config, 'latent_dim', None)
            bilstm_input = getattr(model_config, 'input_dim', None)
            
            if vae_latent and bilstm_input and vae_latent != bilstm_input:
                errors.append(f"VAE latent dim ({vae_latent}) must match BiLSTM input dim ({bilstm_input})")
        
        # Check sequence length consistency
        data_seq_len = getattr(data_config, 'sequence_length', None)
        model_seq_len = getattr(model_config, 'sequence_length', None)
        
        if data_seq_len and model_seq_len and data_seq_len != model_seq_len:
            errors.append(f"Data sequence length ({data_seq_len}) must match model sequence length ({model_seq_len})")
        
        # Check pattern offsets consistency
        data_offsets = getattr(data_config, 'pattern_offsets', None)
        model_offsets = getattr(model_config, 'pattern_offsets', None)
        
        if data_offsets and model_offsets and data_offsets != model_offsets:
            errors.append("Pattern offsets in data config must match model config")
        
        # Check if data file exists for specified subject and session
        if hasattr(data_config, 'root_path') and hasattr(data_config, 'subject') and hasattr(data_config, 'session'):
            from .paths import PathManager, PathConfig
            try:
                path_manager = PathManager(PathConfig(data_root=data_config.root_path))
                if not path_manager.validate_data_file(data_config.subject, data_config.session):
                    errors.append(f"Data file not found for {data_config.subject}_{data_config.session}")
            except Exception as e:
                errors.append(f"Error validating data file: {str(e)}")
        
        return errors
    
    def validate_complete_config(self, data_config, model_config, training_config) -> Dict[str, List[str]]:
        """Perform complete validation of all configuration sections"""
        self.errors = []
        self.warnings = []
        
        # Validate individual sections
        data_errors = self.validate_data_config(data_config)
        model_errors = self.validate_model_config(model_config)
        training_errors = self.validate_training_config(training_config)
        consistency_errors = self.validate_config_consistency(data_config, model_config, training_config)
        
        # Collect all errors
        all_errors = {
            'data_config': data_errors,
            'model_config': model_errors,
            'training_config': training_errors,
            'consistency': consistency_errors
        }
        
        # Log results
        total_errors = sum(len(errors) for errors in all_errors.values())
        
        if total_errors > 0:
            logger.error(f"Configuration validation failed with {total_errors} errors:")
            for section, errors in all_errors.items():
                for error in errors:
                    logger.error(f"  [{section}] {error}")
        else:
            logger.info("Configuration validation passed")
        
        return all_errors
    
    def is_valid(self, data_config, model_config, training_config) -> bool:
        """Check if configuration is valid"""
        errors = self.validate_complete_config(data_config, model_config, training_config)
        return sum(len(error_list) for error_list in errors.values()) == 0


def validate_config(data_config, model_config, training_config, raise_on_error: bool = True) -> bool:
    """Convenience function for config validation"""
    validator = ConfigValidator()
    is_valid = validator.is_valid(data_config, model_config, training_config)
    
    if not is_valid and raise_on_error:
        errors = validator.validate_complete_config(data_config, model_config, training_config)
        error_messages = []
        for section, section_errors in errors.items():
            for error in section_errors:
                error_messages.append(f"[{section}] {error}")
        
        raise ValidationError(f"Configuration validation failed:\n" + "\n".join(error_messages))
    
    return is_valid 
#!/usr/bin/env python3
"""
Main training script for blood pressure prediction models.
Demonstrates the new modular architecture with proper configuration management.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import yaml
from datetime import datetime

# Add project root to path (not src directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import DataConfig, ModelConfig, TrainingConfig, create_config_from_yaml
from src.data.loaders import create_data_loaders
from src.models import create_model
from src.training import create_trainer, create_standard_callbacks
from src.utils import setup_logging

# Configure logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train BP prediction model")
    
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--data-dir", type=str, 
                       help="Override data directory from config")
    parser.add_argument("--output-dir", type=str,
                       help="Override output directory from config") 
    parser.add_argument("--resume", type=str,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run without actual training (for testing)")
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup and return the appropriate device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device(device_arg)
        logger.info(f"Using specified device: {device}")
    
    return device


def validate_config(data_config: DataConfig, model_config: ModelConfig, 
                   training_config: TrainingConfig) -> bool:
    """Validate configuration consistency."""
    issues = []
    
    # Check data paths
    if not Path(data_config.root_path).exists():
        issues.append(f"Data root does not exist: {data_config.root_path}")
    
    # Check model compatibility
    if model_config.model_type == "vae_bilstm":
        if not hasattr(model_config, 'vae_config') or not hasattr(model_config, 'bilstm_config'):
            issues.append("VAE-BiLSTM model requires both vae_config and bilstm_config")
    
    # Check training config
    if training_config.num_epochs <= 0:
        issues.append("Number of epochs must be positive")
    
    if training_config.learning_rate <= 0:
        issues.append("Learning rate must be positive")
    
    if issues:
        for issue in issues:
            logger.error(f"Configuration issue: {issue}")
        return False
    
    return True


def create_experiment_dir(output_dir: Path, model_type: str) -> Path:
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model_type}_{timestamp}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    
    return exp_dir


def save_configs(exp_dir: Path, data_config: DataConfig, 
                model_config: ModelConfig, training_config: TrainingConfig):
    """Save configurations to experiment directory."""
    configs = {
        'data_config': data_config.__dict__,
        'model_config': model_config.__dict__,
        'training_config': training_config.__dict__
    }
    
    config_file = exp_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(configs, f, default_flow_style=False)
    
    logger.info(f"Configurations saved to {config_file}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    logger.info("Starting BP prediction model training")
    logger.info(f"Arguments: {args}")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        configs = create_config_from_yaml(args.config)
        #logger.info(f"Configs: {configs}")
        data_config, model_config, training_config = configs
        
        # Set WANDB environment variables for HPC compatibility
        if hasattr(training_config, 'wandb_mode'):
            os.environ['WANDB_MODE'] = training_config.wandb_mode
        if hasattr(training_config, 'use_wandb') and not training_config.use_wandb:
            os.environ['WANDB_MODE'] = 'disabled'
        
        # Additional wandb settings for HPC environments
        #os.environ['WANDB_SILENT'] = 'true'
        #os.environ['WANDB_CONSOLE'] = 'off'
        
        # Override config with command line arguments
        if args.data_dir:
            data_config.root_path = str(Path(args.data_dir))
        if args.output_dir:
            training_config.output_dir = Path(args.output_dir)
        
        # Validate configuration
        if not validate_config(data_config, model_config, training_config):
            logger.error("Configuration validation failed")
            return 1
        
        # Setup device
        device = setup_device(args.device)
        
        # Create experiment directory
        exp_dir = create_experiment_dir(
            training_config.output_dir, 
            model_config.model_type
        )
        logger.info(f"Experiment directory: {exp_dir}")
        
        # Update training config with experiment paths
        training_config.checkpoint_dir = exp_dir / "checkpoints"
        
        # Save configurations
        save_configs(exp_dir, data_config, model_config, training_config)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        data_loaders, data_manager = create_data_loaders(
            data_config=data_config,
            cache_dir=exp_dir / "cache",
            splits=['train', 'val', 'test'],
            verbose=True
        )
        
        if args.dry_run:
            logger.info("Dry run completed successfully")
            return 0
        
        # Validate data integrity
        logger.info("Validating data integrity...")
        integrity_results = data_manager.validate_data_integrity()
        if not all(integrity_results.values()):
            logger.error(f"Data integrity check failed: {integrity_results}")
            return 1
        
        # Create model
        logger.info(f"Creating {model_config.model_type} model...")
        model = create_model(model_config)
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer(
            model_type=model_config.model_type,
            model=model,
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            config=training_config,
            model_config=model_config,
            device=device
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(Path(args.resume))
        
        # Start training
        logger.info("Starting training...")
        training_results = trainer.fit()
        
        # Save final results
        results_file = exp_dir / "results" / "training_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(training_results, f, default_flow_style=False)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best metric: {training_results['best_metric']:.6f} at epoch {training_results['best_epoch']}")
        logger.info(f"Results saved to: {exp_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
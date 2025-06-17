#!/usr/bin/env python3
"""
Enhanced training script for BP prediction project.
Uses improved path management, config validation, and standardized training pipeline.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import torch
from types import SimpleNamespace

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from src.utils.paths import setup_paths, PathConfig
from src.utils.config import load_config, create_config_from_yaml
from src.utils.config_validator import validate_config, ValidationError
from src.utils.logging import setup_logging, get_logger
from utils.data_utils import DataPathManager, PviDataset, load_subjects
from src.training.vae_trainer import VAETrainer

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments with improved options"""
    parser = argparse.ArgumentParser(
        description="Enhanced BP Prediction Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration YAML file"
    )
    
    # Override arguments
    parser.add_argument(
        "--data-root",
        type=str,
        help="Override data root directory (overrides config and env vars)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory for experiments"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Override experiment name"
    )
    
    # Training control
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size"
    )
    
    # Logging and debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and setup without training"
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        help="Override WandB logging mode"
    )
    
    # Environment setup
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and paths, then exit"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup environment variables and paths"""
    # Set environment variables if provided
    if args.data_root:
        os.environ['BP_DATA_ROOT'] = args.data_root
    if args.output_dir:
        os.environ['BP_EXPERIMENTS_ROOT'] = args.output_dir
    if args.wandb_mode:
        os.environ['WANDB_MODE'] = args.wandb_mode
    
    # Setup path management
    path_config = PathConfig(
        data_root=args.data_root,
        experiments_root=args.output_dir
    )
    path_manager = setup_paths(path_config)
    
    logger.info("Environment setup completed")
    logger.info(f"Data root: {path_manager.data_root}")
    logger.info(f"Experiments root: {path_manager.experiments_root}")
    
    return path_manager


def load_and_validate_config(config_path: str, args):
    """Load and validate configuration"""
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        # Load base configuration (returns data_config, model_config, training_config)
        data_config, model_config, training_config = create_config_from_yaml(config_path)
        
        # Apply command line overrides
        if args.data_root:
            data_config.root_path = args.data_root
        if args.output_dir:
            training_config.output_dir = Path(args.output_dir)
        if args.experiment_name:
            training_config.experiment_name = args.experiment_name
        if args.epochs:
            training_config.num_epochs = args.epochs
        if args.batch_size:
            training_config.batch_size = args.batch_size
        if args.wandb_mode:
            training_config.wandb_mode = args.wandb_mode
        
        # Validate configuration
        logger.info("Validating configuration...")
        is_valid = validate_config(
            data_config, 
            model_config, 
            training_config, 
            raise_on_error=True
        )
        
        if is_valid:
            logger.info("✅ Configuration validation passed")
        
        # Return a tuple for downstream use
        return data_config, model_config, training_config
        
    except ValidationError as e:
        logger.error(f"❌ Configuration validation failed:")
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"❌ Error loading configuration: {e}")
        raise


def setup_device(device_arg: str):
    """Setup and validate device"""
    import torch
    
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.info("💻 Using CPU device")
    else:
        device = torch.device(device_arg)
        logger.info(f"📱 Using specified device: {device}")
    
    return device


def setup_data_loaders(data_config, training_config, path_manager):
    """Setup data loaders with validation"""
    logger.info("Setting up data loaders...")
    
    # Validate data file exists
    if not path_manager.validate_data_file(data_config.subject, data_config.session):
        raise FileNotFoundError(
            f"Data file not found: {path_manager.get_data_file_path(data_config.subject, data_config.session)}"
        )
    
    # Create data manager
    data_manager = DataPathManager(
        subject=data_config.subject,
        session=data_config.session,
        root=data_config.root_path
    )
    # Create dataset
    dataset = PviDataset(str(data_manager._h5_path))
    # Split dataset (80/20 train/val)
    n = len(dataset)
    n_train = int(n * 0.8)
    indices = np.random.permutation(n)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=training_config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=training_config.batch_size, shuffle=False)
    logger.info(f"✅ Data loaders created:")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Validation batches: {len(val_loader)}")
    logger.info(f"   Batch size: {training_config.batch_size}")
    return train_loader, val_loader, data_manager


def create_model(config, device):
    """Create and initialize model"""
    logger.info(f"Creating {config.model_type} model...")
    if config.model_type == "vae":
        from train.tuned_vae import VAE  # Use working implementation
        model = VAE(latent_dim=config.latent_dim)
    elif config.model_type == "bilstm":
        from train.bilstm import create_bilstm  # Use working implementation
        model = create_bilstm(config)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
    # Move model to device
    model = model.to(device)
    
    # Log model summary
    logger.info(f"✅ Model created and moved to {device}")
    logger.info(f"   Model type: {config.model_type}")
    if config.model_type == "vae":
        logger.info(f"   Latent dimension: {config.latent_dim}")
    
    return model


def create_trainer(model, train_loader, val_loader, model_config, training_config, device, experiment_dir):
    """Create appropriate trainer based on model type"""
    logger.info("Creating trainer...")
    
    if model_config.model_type == "vae":
        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=SimpleNamespace(
                model=model_config,
                training=training_config
            ),
            device=device,
            experiment_dir=experiment_dir
        )
    else:
        raise ValueError(f"Unsupported model type for trainer: {model_config.model_type}")
    
    logger.info(f"✅ Trainer created: {trainer.__class__.__name__}")
    return trainer


def create_experiment_directory(path_manager, training_config, config_path=None):
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = getattr(training_config, 'experiment_name', f"experiment_{timestamp}")
    experiment_dir = path_manager.experiments_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config file if provided
    if config_path:
        import shutil
        shutil.copy2(config_path, experiment_dir / "config.yaml")
    
    logger.info(f"✅ Experiment directory created: {experiment_dir}")
    return experiment_dir


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=level)
    
    try:
        # Setup environment and paths
        path_manager = setup_environment(args)
        
        # Load and validate configuration
        data_config, model_config, training_config = load_and_validate_config(args.config, args)
        
        if args.validate_only:
            logger.info("✅ Configuration validation passed, exiting as requested")
            return
        
        # Setup device
        device = setup_device(args.device)
        
        # Setup data loaders
        train_loader, val_loader, data_manager = setup_data_loaders(data_config, training_config, path_manager)
        
        # Create model
        model = create_model(model_config, device)
        
        # Create experiment directory
        experiment_dir = create_experiment_directory(path_manager, training_config, args.config)
        
        # Create trainer
        trainer = create_trainer(model, train_loader, val_loader, model_config, training_config, device, experiment_dir)
        
        if args.dry_run:
            logger.info("✅ Dry run completed successfully")
            return
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("✅ Training completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 
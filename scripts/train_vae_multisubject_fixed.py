#!/usr/bin/env python3
"""
Fixed Multi-Subject VAE Training Script
Based on the working train_enhanced.py approach, adapted for multiple subjects
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import numpy as np
import torch
from types import SimpleNamespace

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import working modules from the proven system
from src.utils.paths import setup_paths, PathConfig
from src.utils.config import load_config, create_config_from_yaml
from src.utils.logging import setup_logging, get_logger
from utils.data_utils import DataPathManager, PviDataset
from src.training.vae_trainer import VAETrainer
from torch.utils.data import ConcatDataset, DataLoader, random_split

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Subject VAE Training (Fixed)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to VAE configuration YAML file"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to use (default: all available)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        help="Override data root directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup environment variables and paths"""
    if args.data_root:
        os.environ['BP_DATA_ROOT'] = args.data_root
    
    path_config = PathConfig(
        data_root=args.data_root,
        experiments_root="./experiments"
    )
    path_manager = setup_paths(path_config)
    
    logger.info("Environment setup completed")
    logger.info(f"Data root: {path_manager.data_root}")
    logger.info(f"Experiments root: {path_manager.experiments_root}")
    
    return path_manager


def get_available_subjects(data_root: str) -> List[str]:
    """Get list of available subjects from data directory"""
    data_path = Path(data_root)
    subjects = []
    
    for file_path in data_path.glob("subject*_baseline_masked.h5"):
        subject = file_path.stem.split('_')[0]
        subjects.append(subject)
    
    return sorted(subjects)


def setup_device(device_arg: str):
    """Setup and validate device"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("💻 Using CPU device")
    else:
        device = torch.device(device_arg)
        logger.info(f"📱 Using specified device: {device}")
    
    return device


def create_multisubject_dataset(data_root: str, subjects: List[str], session: str = "baseline"):
    """Create a combined dataset from multiple subjects"""
    logger.info(f"Creating multi-subject dataset from {len(subjects)} subjects...")
    
    datasets = []
    successful_subjects = []
    
    for subject in subjects:
        try:
            # Create data manager for this subject
            data_manager = DataPathManager(
                subject=subject,
                session=session,
                root=data_root
            )
            
            if not data_manager._h5_path.exists():
                logger.warning(f"⚠️  Data file not found for {subject}, skipping")
                continue
            
            # Create dataset using the working PviDataset
            dataset = PviDataset(str(data_manager._h5_path))
            
            datasets.append(dataset)
            successful_subjects.append(subject)
            logger.info(f"✅ Loaded {subject}: {len(dataset)} samples")
            
        except Exception as e:
            logger.warning(f"❌ Failed to load {subject}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets were loaded!")
    
    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)
    
    total_samples = len(combined_dataset)
    logger.info(f"🎯 Multi-subject dataset ready:")
    logger.info(f"   📊 Successful subjects: {len(successful_subjects)}")
    logger.info(f"   📈 Total samples: {total_samples}")
    logger.info(f"   📝 Average per subject: {total_samples / len(successful_subjects):.1f}")
    
    return combined_dataset, successful_subjects


def create_model(model_config, device):
    """Create VAE model using the working implementation"""
    logger.info(f"Creating VAE model...")
    
    # Use the working VAE implementation
    from train.tuned_vae import VAE
    model = VAE(latent_dim=model_config.latent_dim)
    
    # Move model to device
    model = model.to(device)
    
    # Log model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"✅ VAE model created and moved to {device}")
    logger.info(f"   Latent dimension: {model_config.latent_dim}")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")
    
    return model


def create_experiment_directory(path_manager, experiment_name=None):
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"multisubject_vae_{timestamp}"
    
    experiment_dir = path_manager.experiments_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    
    logger.info(f"✅ Experiment directory created: {experiment_dir}")
    return experiment_dir


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=level)
    
    try:
        logger.info("🚀 Starting Multi-Subject VAE Training (Fixed)")
        logger.info("=" * 60)
        
        # Setup environment and paths
        path_manager = setup_environment(args)
        
        # Load configuration using the working system
        logger.info(f"Loading configuration from: {args.config}")
        data_config, model_config, training_config = create_config_from_yaml(args.config)
        
        # Override data root if provided
        if args.data_root:
            data_config.root_path = args.data_root
        
        # Setup device
        device = setup_device(args.device)
        
        # Get available subjects
        subjects = get_available_subjects(data_config.root_path)
        if args.max_subjects:
            subjects = subjects[:args.max_subjects]
        
        logger.info(f"📊 Training subjects: {subjects}")
        logger.info(f"📈 Total subjects: {len(subjects)}")
        
        # Create multi-subject dataset
        combined_dataset, successful_subjects = create_multisubject_dataset(
            data_config.root_path, 
            subjects, 
            getattr(data_config, 'session', 'baseline')
        )
        
        # Split dataset (80/20 train/val)
        train_size = int(0.8 * len(combined_dataset))
        val_size = len(combined_dataset) - train_size
        train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"✅ Data loaders created:")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Validation batches: {len(val_loader)}")
        logger.info(f"   Batch size: {training_config.batch_size}")
        
        # Create model using working implementation
        model = create_model(model_config, device)
        
        # Create experiment directory
        experiment_dir = create_experiment_directory(
            path_manager, 
            f"multisubject_vae_{len(successful_subjects)}subjects"
        )
        
        # Enable wandb if requested
        if args.wandb:
            training_config.wandb_mode = "online"
            training_config.use_wandb = True
        
        # Create trainer using the working VAETrainer
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
        
        logger.info(f"✅ Trainer created: {trainer.__class__.__name__}")
        
        # Start training
        logger.info("🎯 Starting training...")
        logger.info("=" * 60)
        
        trainer.train()
        
        logger.info("=" * 60)
        logger.info("✅ Multi-subject VAE training completed successfully!")
        logger.info(f"📁 Results saved to: {experiment_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 
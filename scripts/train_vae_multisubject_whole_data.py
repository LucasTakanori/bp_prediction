#!/usr/bin/env python3
"""
Multi-Subject VAE Training Script with Whole Data (No Masking)
Based on train_enhanced.py approach but for multiple subjects
Uses PviDataset directly without any masking - image in, image out
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
import gc
from types import SimpleNamespace

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import working modules
from src.utils.paths import setup_paths, PathConfig
from src.utils.config import load_config, create_config_from_yaml
from src.utils.logging import setup_logging, get_logger
from utils.data_utils import DataPathManager, PviDataset  # Use basic PviDataset
from src.training.vae_trainer import VAETrainer
from torch.utils.data import ConcatDataset, DataLoader, random_split

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Subject VAE Training with Whole Data",
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


def create_multisubject_whole_data_dataset(data_root: str, subjects: List[str], session: str = "baseline"):
    """Create a combined dataset from multiple subjects using whole data (no masking)"""
    logger.info(f"Creating multi-subject dataset with WHOLE DATA from {len(subjects)} subjects...")
    logger.info("🔥 NO MASKING - Using all available data like train_enhanced.py")
    
    datasets = []
    successful_subjects = []
    failed_subjects = []
    
    for subject in subjects:
        try:
            # Create data path manager
            data_manager = DataPathManager(
                subject=subject,
                session=session,
                root=data_root
            )
            
            # Check if data file exists
            if not data_manager._h5_path.exists():
                logger.warning(f"⚠️  Data file not found for {subject}, skipping")
                failed_subjects.append(subject)
                continue
            
            # Load dataset using basic PviDataset (no masking)
            dataset = PviDataset(str(data_manager._h5_path))
            
            datasets.append(dataset)
            successful_subjects.append(subject)
            logger.info(f"✅ Loaded {subject}: {len(dataset)} samples (WHOLE DATA)")
            
        except Exception as e:
            logger.warning(f"❌ Failed to load {subject}: {e}")
            failed_subjects.append(subject)
            continue
    
    if not datasets:
        raise ValueError("No valid datasets were loaded!")
    
    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)
    
    # Memory optimization
    del datasets
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    total_samples = len(combined_dataset)
    logger.info(f"🎯 Multi-subject WHOLE DATA dataset ready:")
    logger.info(f"   📊 Successful subjects: {len(successful_subjects)}")
    logger.info(f"   📊 Failed subjects: {len(failed_subjects)}")
    logger.info(f"   📈 Total samples: {total_samples}")
    logger.info(f"   📝 Average per successful subject: {total_samples / len(successful_subjects):.1f}")
    logger.info(f"   🔥 Data type: WHOLE DATA (no masking)")
    
    return combined_dataset, successful_subjects, failed_subjects


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
        experiment_name = f"multisubject_vae_whole_data_{timestamp}"
    
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
        logger.info("🚀 Starting Multi-Subject VAE Training with WHOLE DATA")
        logger.info("🔥 NO MASKING - Image in, Image out like train_enhanced.py")
        logger.info("=" * 70)
        
        # Setup environment and paths
        path_manager = setup_environment(args)
        
        # Get available subjects
        subjects = get_available_subjects(args.data_root or os.getenv('BP_DATA_ROOT', '/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data'))
        if args.max_subjects:
            subjects = subjects[:args.max_subjects]
        
        logger.info(f"📊 Target subjects: {subjects}")
        logger.info(f"📈 Total subjects: {len(subjects)}")
        
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        data_config, model_config, training_config = create_config_from_yaml(args.config)
        
        # Override data root if provided
        if args.data_root:
            data_config.root_path = args.data_root
        
        # Setup device
        device = setup_device(args.device)
        
        # Create multi-subject WHOLE DATA dataset
        combined_dataset, successful_subjects, failed_subjects = create_multisubject_whole_data_dataset(
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
            num_workers=0,  # Disable multiprocessing to reduce memory usage
            pin_memory=False  # Disable pinned memory to reduce GPU memory usage
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=False, 
            num_workers=0,  # Disable multiprocessing 
            pin_memory=False  # Disable pinned memory
        )
        
        logger.info(f"✅ Data loaders created:")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Validation batches: {len(val_loader)}")
        logger.info(f"   Batch size: {training_config.batch_size}")
        
        # Create model
        model = create_model(model_config, device)
        
        # Create experiment directory
        experiment_name = f"multisubject_vae_whole_data_{len(successful_subjects)}subjects"
        experiment_dir = create_experiment_directory(path_manager, experiment_name)
        
        # Save experiment metadata
        experiment_metadata = {
            'successful_subjects': successful_subjects,
            'failed_subjects': failed_subjects,
            'data_type': 'whole_data_no_masking',
            'total_samples': len(combined_dataset),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset)
        }
        
        import json
        with open(experiment_dir / "experiment_metadata.json", 'w') as f:
            json.dump(experiment_metadata, f, indent=2, default=str)
        
        # Enable wandb if requested
        if args.wandb:
            training_config.wandb_mode = "offline"
            training_config.use_wandb = True
        
        # Create trainer
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
        logger.info("🎯 Starting training with WHOLE DATA...")
        logger.info("=" * 70)
        
        trainer.train()
        
        logger.info("=" * 70)
        logger.info("✅ Multi-subject VAE training with WHOLE DATA completed successfully!")
        logger.info(f"📁 Results saved to: {experiment_dir}")
        logger.info(f"📊 Successfully trained on {len(successful_subjects)} subjects")
        logger.info(f"🔥 Used WHOLE DATA (no masking)")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 
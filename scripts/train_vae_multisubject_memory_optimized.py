"""
Memory-Optimized Multi-Subject VAE Training
Handles large datasets with memory constraints
"""

import argparse
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import yaml
import gc
from types import SimpleNamespace

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.paths import setup_paths
from src.utils.logging import setup_logging
from src.training.vae_trainer import VAETrainer
from utils.data_utils_memory_optimized import (
    create_memory_efficient_multisubject_dataset,
    estimate_memory_usage
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Memory-Optimized Multi-Subject VAE Training')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--max-subjects', type=int, default=10,
                       help='Maximum number of subjects to include')
    parser.add_argument('--max-samples-per-subject', type=int, default=300,
                       help='Maximum samples per subject (for memory control)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--mask-type', type=str, default='mask10',
                       help='Mask type to use')
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only estimate memory usage, don\'t train')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (lower for memory optimization)')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_arg: str = 'auto') -> torch.device:
    """Get appropriate device"""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def get_subject_list(max_subjects: int) -> List[str]:
    """Get list of subjects to process"""
    # You can customize this based on your available subjects
    subjects = [f'subject{i:03d}' for i in range(1, max_subjects + 1)]
    
    # Skip subject011, 012, 013 as they don't exist
    subjects = [s for s in subjects if s not in ['subject011', 'subject012', 'subject013']]
    
    return subjects


def setup_wandb(config: Dict, args: argparse.Namespace) -> None:
    """Setup Weights & Biases logging"""
    if not args.wandb:
        return
    
    wandb.init(
        project="bp-prediction-multisubject-memory-optimized",
        config={
            **config,
            'script': 'train_vae_multisubject_memory_optimized.py',
            'max_subjects': args.max_subjects,
            'max_samples_per_subject': args.max_samples_per_subject,
            'mask_type': args.mask_type,
            'device': str(args.device),
            'batch_size': args.batch_size,
        },
        tags=['multisubject', 'vae', 'memory-optimized']
    )


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("train_multisubject_memory_optimized")
    logger.info("🚀 Starting Memory-Optimized Multi-Subject VAE Training")
    logger.info("=" * 70)
    
    # Setup paths
    paths = setup_paths()
    logger.info("Environment setup completed")
    logger.info(f"Data root: {paths.data_root}")
    logger.info(f"Experiments root: {paths.experiments_root}")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loading configuration from: {args.config}")
    
    # Setup device
    device = get_device(args.device)
    logger.info(f"📱 Using device: {device}")
    
    # Get subjects
    subjects = get_subject_list(args.max_subjects)
    logger.info(f"📊 Target subjects: {subjects}")
    logger.info(f"📈 Total subjects: {len(subjects)}")
    
    # Memory estimation
    logger.info("🧮 Estimating memory requirements...")
    memory_info = estimate_memory_usage(
        str(paths.data_root), 
        subjects, 
        args.mask_type
    )
    
    if args.estimate_only:
        logger.info("📊 Memory estimation complete. Exiting.")
        return
    
    # Memory safety check
    if memory_info['estimated_memory_gb'] > 16:
        logger.warning(f"⚠️  High memory usage estimated: {memory_info['estimated_memory_gb']:.1f} GB")
        logger.warning(f"⚠️  Consider reducing --max-subjects or --max-samples-per-subject")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            logger.info("Training cancelled by user")
            return
    
    # Create memory-efficient dataset
    logger.info("Creating memory-efficient multi-subject dataset...")
    logger.info(f"Max samples per subject: {args.max_samples_per_subject}")
    logger.info(f"Using mask type: {args.mask_type}")
    
    try:
        combined_dataset, successful_subjects, failed_subjects = \
            create_memory_efficient_multisubject_dataset(
                str(paths.data_root),
                subjects,
                mask_type=args.mask_type,
                max_samples_per_subject=args.max_samples_per_subject,
                session="baseline"
            )
        
        logger.info(f"✅ Dataset created successfully")
        logger.info(f"   📊 Successful subjects: {len(successful_subjects)}")
        logger.info(f"   📊 Failed subjects: {len(failed_subjects)}")
        logger.info(f"   📈 Total samples: {len(combined_dataset)}")
        
        if failed_subjects:
            logger.warning(f"⚠️  Failed subjects: {failed_subjects}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create dataset: {e}")
        return
    
    # Setup experiment tracking
    setup_wandb(config, args)
    
    # Update config for memory optimization
    if 'training_config' not in config:
        config['training_config'] = {}
    if 'data_config' not in config:
        config['data_config'] = {}
    
    config['training_config']['batch_size'] = args.batch_size
    config['data_config']['num_subjects'] = len(successful_subjects)
    config['data_config']['total_samples'] = len(combined_dataset)
    config['data_config']['max_samples_per_subject'] = args.max_samples_per_subject
    
    # Split dataset into train/val
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])
    
    # Create train and validation dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False
    )
    
    logger.info(f"✅ Train/Val split created:")
    logger.info(f"   📦 Train samples: {len(train_dataset)}")
    logger.info(f"   📦 Val samples: {len(val_dataset)}")
    logger.info(f"   📈 Train batches: {len(train_dataloader)}")
    logger.info(f"   📈 Val batches: {len(val_dataloader)}")
    
    # Create experiment directory
    paths = setup_paths()
    experiment_name = f"multisubject_vae_memory_opt_{len(successful_subjects)}subj_{len(combined_dataset)}samples"
    experiment_dir = paths.create_experiment_dir(experiment_name)
    
    # Initialize model (we need to create it based on the config)
    from train.vae_clean import VAE  # Import your VAE model
    
    model = VAE(
        latent_dim=config['model_config']['latent_dim']
    ).to(device)
    
    # Convert config dict to object format expected by trainer
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    
    config_obj = dict_to_namespace(config)
    
    # Initialize trainer
    try:
        trainer = VAETrainer(
            model=model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            config=config_obj,
            device=device,
            experiment_dir=experiment_dir
        )
        logger.info("✅ VAE Trainer initialized")
        
        # Start training
        logger.info("🚀 Starting training...")
        trainer.train()
        
        logger.info("🎉 Training completed successfully!")
        
        # Save experiment metadata
        experiment_info = {
            'successful_subjects': successful_subjects,
            'failed_subjects': failed_subjects,
            'total_samples': len(combined_dataset),
            'memory_info': memory_info,
            'config': config,
            'args': vars(args)
        }
        
        # Save to experiment directory
        with open(experiment_dir / 'experiment_metadata.yaml', 'w') as f:
            yaml.dump(experiment_info, f, default_flow_style=False)
        
        logger.info(f"💾 Experiment metadata saved to: {experiment_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main() 
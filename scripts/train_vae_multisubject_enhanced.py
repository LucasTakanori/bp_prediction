#!/usr/bin/env python3
"""
Enhanced Multi-Subject VAE Training Script
Handles different mask formats automatically
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import numpy as np
import torch
import gc  # Add garbage collection
from types import SimpleNamespace

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import working modules
from src.utils.paths import setup_paths, PathConfig
from src.utils.config import load_config, create_config_from_yaml
from src.utils.logging import setup_logging, get_logger
from utils.data_utils_enhanced import EnhancedPviDataset, load_dataset_with_best_mask, analyze_all_mask_formats
from src.training.vae_trainer import VAETrainer
from torch.utils.data import ConcatDataset, DataLoader, random_split

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Subject VAE Training",
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
        "--mask-type",
        type=str,
        default="auto",
        choices=["auto", "metadata", "mask01", "mask05", "mask10", "mask15"],
        help="Mask type to use for all subjects"
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
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze mask formats, don't train"
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


def analyze_subjects_masks(data_root: str, subjects: List[str]) -> Dict:
    """Analyze mask formats for given subjects"""
    logger.info("🔍 Analyzing mask formats for all subjects...")
    
    analysis = analyze_all_mask_formats(data_root)
    
    # Filter to requested subjects
    filtered_analysis = {s: analysis.get(s, {}) for s in subjects}
    
    # Categorize subjects
    metadata_mask_subjects = []
    masks_group_subjects = []
    no_mask_subjects = []
    error_subjects = []
    
    for subject, info in filtered_analysis.items():
        if 'error' in info:
            error_subjects.append(subject)
            logger.warning(f"❌ {subject}: {info['error']}")
            continue
            
        if info.get('has_metadata_mask', False):
            metadata_mask_subjects.append(subject)
        elif info.get('has_masks_group', False):
            masks_group_subjects.append(subject)
        else:
            no_mask_subjects.append(subject)
        
        logger.info(f"📊 {subject}:")
        logger.info(f"   Data periods: {info.get('data_periods', 'unknown')}")
        logger.info(f"   Metadata mask: {'✅' if info.get('has_metadata_mask', False) else '❌'}")
        if info.get('has_metadata_mask', False):
            logger.info(f"   Metadata mask count: {info.get('metadata_mask_count', 'unknown')}")
        logger.info(f"   Masks group: {'✅' if info.get('has_masks_group', False) else '❌'}")
        if info.get('has_masks_group', False):
            logger.info(f"   Available masks: {info.get('masks_group_keys', [])}")
    
    summary = {
        'metadata_mask_subjects': metadata_mask_subjects,
        'masks_group_subjects': masks_group_subjects,
        'no_mask_subjects': no_mask_subjects,
        'error_subjects': error_subjects,
        'analysis': filtered_analysis
    }
    
    logger.info(f"\n📈 MASK ANALYSIS SUMMARY:")
    logger.info(f"   Subjects with metadata masks: {len(metadata_mask_subjects)}")
    logger.info(f"   Subjects with masks group: {len(masks_group_subjects)}")
    logger.info(f"   Subjects with no masks: {len(no_mask_subjects)}")
    logger.info(f"   Subjects with errors: {len(error_subjects)}")
    
    return summary


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


def create_enhanced_multisubject_dataset(data_root: str, subjects: List[str], mask_type: str = "auto", session: str = "baseline"):
    """Create a combined dataset from multiple subjects using enhanced loader"""
    logger.info(f"Creating enhanced multi-subject dataset from {len(subjects)} subjects...")
    logger.info(f"Using mask type: {mask_type}")
    
    datasets = []
    successful_subjects = []
    failed_subjects = []
    
    for subject in subjects:
        try:
            # Create file path
            file_path = Path(data_root) / f"{subject}_{session}_masked.h5"
            
            if not file_path.exists():
                logger.warning(f"⚠️  Data file not found for {subject}, skipping")
                failed_subjects.append(subject)
                continue
            
            # Load dataset using enhanced loader
            dataset = load_dataset_with_best_mask(str(file_path), mask_type)
            
            datasets.append(dataset)
            successful_subjects.append(subject)
            logger.info(f"✅ Loaded {subject}: {len(dataset)} samples")
            
            # Log mask info
            mask_info = dataset.get_mask_info()
            logger.info(f"   Available masks: {mask_info.get('available_masks', [])}")
            logger.info(f"   Used mask: {mask_info.get('recommended_mask', 'unknown')}")
            
        except Exception as e:
            logger.warning(f"❌ Failed to load {subject}: {e}")
            failed_subjects.append(subject)
            continue
    
    if not datasets:
        raise ValueError("No valid datasets were loaded!")
    
    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)
    
    # Memory optimization: clear references and force garbage collection
    del datasets  # Remove reference to individual datasets
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache() if torch.cuda.is_available() else None  # Clear GPU cache
    
    total_samples = len(combined_dataset)
    logger.info(f"🎯 Enhanced multi-subject dataset ready:")
    logger.info(f"   📊 Successful subjects: {len(successful_subjects)}")
    logger.info(f"   📊 Failed subjects: {len(failed_subjects)}")
    logger.info(f"   📈 Total samples: {total_samples}")
    logger.info(f"   📝 Average per successful subject: {total_samples / len(successful_subjects):.1f}")
    
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
        experiment_name = f"enhanced_multisubject_vae_{timestamp}"
    
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
        logger.info("🚀 Starting Enhanced Multi-Subject VAE Training")
        logger.info("=" * 70)
        
        # Setup environment and paths
        path_manager = setup_environment(args)
        
        # Get available subjects
        subjects = get_available_subjects(args.data_root or os.getenv('BP_DATA_ROOT', '/home/lucas_takanori/phd/data'))
        if args.max_subjects:
            subjects = subjects[:args.max_subjects]
        
        logger.info(f"📊 Target subjects: {subjects}")
        logger.info(f"📈 Total subjects: {len(subjects)}")
        
        # Analyze mask formats
        mask_analysis = analyze_subjects_masks(
            args.data_root or os.getenv('BP_DATA_ROOT', '/home/lucas_takanori/phd/data'), 
            subjects
        )
        
        if args.analyze_only:
            logger.info("✅ Analysis complete. Exiting (--analyze-only mode)")
            return
        
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        data_config, model_config, training_config = create_config_from_yaml(args.config)
        
        # Override data root if provided
        if args.data_root:
            data_config.root_path = args.data_root
        
        # Setup device
        device = setup_device(args.device)
        
        # Create enhanced multi-subject dataset
        combined_dataset, successful_subjects, failed_subjects = create_enhanced_multisubject_dataset(
            data_config.root_path, 
            subjects,
            args.mask_type,
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
        experiment_name = f"enhanced_multisubject_vae_{len(successful_subjects)}subjects_{args.mask_type}"
        experiment_dir = create_experiment_directory(path_manager, experiment_name)
        
        # Save experiment metadata
        experiment_metadata = {
            'successful_subjects': successful_subjects,
            'failed_subjects': failed_subjects,
            'mask_type': args.mask_type,
            'mask_analysis': mask_analysis,
            'total_samples': len(combined_dataset),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset)
        }
        
        import json
        with open(experiment_dir / "experiment_metadata.json", 'w') as f:
            json.dump(experiment_metadata, f, indent=2, default=str)
        
        # Enable wandb if requested
        if args.wandb:
            training_config.wandb_mode = "online"
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
        logger.info("🎯 Starting training...")
        logger.info("=" * 70)
        
        trainer.train()
        
        logger.info("=" * 70)
        logger.info("✅ Enhanced multi-subject VAE training completed successfully!")
        logger.info(f"📁 Results saved to: {experiment_dir}")
        logger.info(f"📊 Successfully trained on {len(successful_subjects)} subjects")
        logger.info(f"📊 Mask type used: {args.mask_type}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 
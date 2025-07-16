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
from typing import Optional, List, Dict, Tuple
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
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for reproducible subject-level splits (CRITICAL for no data leakage)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio (default: 0.8 for 80%)"
    )
    parser.add_argument(
        "--val-ratio", 
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2 for 20%)"
    )
    parser.add_argument(
        "--test-ratio", 
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2 for 20%)"
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


def create_subject_level_splits(subjects: List[str], train_ratio: float = 0.6, 
                               val_ratio: float = 0.2, test_ratio: float = 0.2, 
                               seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Create subject-level train/val/test splits to prevent data leakage
    
    Args:
        subjects: List of all available subjects
        train_ratio: Ratio for training split (default: 0.6 for 60%)
        val_ratio: Ratio for validation split (default: 0.2 for 20%)
        test_ratio: Ratio for test split (default: 0.2 for 20%)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    logger.info(f"🔒 Creating SUBJECT-LEVEL splits to prevent data leakage:")
    logger.info(f"   Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    logger.info(f"   Total subjects: {len(subjects)}")
    logger.info(f"   Random seed: {seed}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Shuffle subjects deterministically
    shuffled_subjects = subjects.copy()
    np.random.shuffle(shuffled_subjects)
    
    # Calculate split sizes
    n_total = len(shuffled_subjects)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Test gets the remaining (to ensure all subjects are used)
    
    # Create splits
    train_subjects = shuffled_subjects[:n_train]
    val_subjects = shuffled_subjects[n_train:n_train + n_val]
    test_subjects = shuffled_subjects[n_train + n_val:]
    
    logger.info(f"📊 Subject distribution:")
    logger.info(f"   Train: {len(train_subjects)} subjects: {train_subjects}")
    logger.info(f"   Val:   {len(val_subjects)} subjects: {val_subjects}")
    logger.info(f"   Test:  {len(test_subjects)} subjects: {test_subjects}")
    
    # Validate no overlap (CRITICAL)
    all_sets = [set(train_subjects), set(val_subjects), set(test_subjects)]
    for i, set1 in enumerate(all_sets):
        for j, set2 in enumerate(all_sets[i+1:], i+1):
            overlap = set1 & set2
            set_names = ['train', 'val', 'test']
            assert len(overlap) == 0, f"❌ LEAKAGE DETECTED: {set_names[i]}/{set_names[j]} overlap: {overlap}"
    
    logger.info("✅ Subject-level splits created successfully - NO DATA LEAKAGE")
    logger.info("🔒 TEST SET ISOLATED: Test subjects will not be seen during training")
    
    return train_subjects, val_subjects, test_subjects


def create_datasets_from_subject_splits(train_subjects: List[str], val_subjects: List[str], 
                                       data_root: str, mask_type: str, session: str = "baseline"):
    """Create separate datasets from train and val subject lists"""
    logger.info(f"🏗️ Creating datasets from subject splits:")
    logger.info(f"   Train subjects: {len(train_subjects)}")
    logger.info(f"   Val subjects: {len(val_subjects)}")
    
    # Create training dataset
    train_datasets = []
    successful_train_subjects = []
    
    logger.info("📈 Loading TRAINING subjects...")
    for subject in train_subjects:
        try:
            file_path = Path(data_root) / f"{subject}_{session}_masked.h5"
            if not file_path.exists():
                logger.warning(f"⚠️  Train data file not found for {subject}, skipping")
                continue
                
            dataset = load_dataset_with_best_mask(str(file_path), mask_type)
            train_datasets.append(dataset)
            successful_train_subjects.append(subject)
            logger.info(f"✅ Train: {subject} - {len(dataset)} samples")
            
        except Exception as e:
            logger.warning(f"❌ Failed to load train {subject}: {e}")
            continue
    
    # Create validation dataset
    val_datasets = []
    successful_val_subjects = []
    
    logger.info("📉 Loading VALIDATION subjects...")
    for subject in val_subjects:
        try:
            file_path = Path(data_root) / f"{subject}_{session}_masked.h5"
            if not file_path.exists():
                logger.warning(f"⚠️  Val data file not found for {subject}, skipping")
                continue
                
            dataset = load_dataset_with_best_mask(str(file_path), mask_type)
            val_datasets.append(dataset)
            successful_val_subjects.append(subject)
            logger.info(f"✅ Val: {subject} - {len(dataset)} samples")
            
        except Exception as e:
            logger.warning(f"❌ Failed to load val {subject}: {e}")
            continue
    
    # Combine datasets within each split
    train_combined = ConcatDataset(train_datasets) if train_datasets else None
    val_combined = ConcatDataset(val_datasets) if val_datasets else None
    
    if train_combined is None or val_combined is None:
        raise ValueError("Failed to create train or validation datasets!")
    
    # Memory cleanup
    del train_datasets, val_datasets
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    logger.info(f"✅ Datasets created from subject splits:")
    logger.info(f"   Train: {len(train_combined)} samples from {len(successful_train_subjects)} subjects")
    logger.info(f"   Val: {len(val_combined)} samples from {len(successful_val_subjects)} subjects")
    logger.info(f"🔒 GUARANTEE: Zero subject overlap between train/val")
    
    return train_combined, val_combined, successful_train_subjects, successful_val_subjects


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


# Note: This function was replaced by create_datasets_from_subject_splits for leakage-free training


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
        subjects = get_available_subjects(args.data_root or os.getenv('BP_DATA_ROOT', '/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data'))
        if args.max_subjects:
            subjects = subjects[:args.max_subjects]
        
        logger.info(f"📊 Target subjects: {subjects}")
        logger.info(f"📈 Total subjects: {len(subjects)}")
        
        # Analyze mask formats
        mask_analysis = analyze_subjects_masks(
            args.data_root or os.getenv('BP_DATA_ROOT', '/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data'), 
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
        
        # 🔒 CRITICAL: Create subject-level splits to prevent data leakage
        logger.info("=" * 80)
        logger.info("🔒 PREVENTING DATA LEAKAGE WITH SUBJECT-LEVEL SPLITS")
        logger.info("=" * 80)
        
        train_subjects, val_subjects, test_subjects = create_subject_level_splits(
            subjects, 
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio, 
            seed=args.split_seed
        )
        
        # Create datasets from subject splits (NO LEAKAGE)
        train_dataset, val_dataset, successful_train_subjects, successful_val_subjects = create_datasets_from_subject_splits(
            train_subjects, 
            val_subjects,
            data_config.root_path, 
            args.mask_type,
            getattr(data_config, 'session', 'baseline')
        )
        
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
        total_successful_subjects = len(successful_train_subjects) + len(successful_val_subjects)
        experiment_name = f"enhanced_multisubject_vae_{total_successful_subjects}subjects_{args.mask_type}"
        experiment_dir = create_experiment_directory(path_manager, experiment_name)
        
        # Save experiment metadata with NO LEAKAGE guarantee
        experiment_metadata = {
            'experiment_type': 'leakage_free_subject_level_splits',
            'split_configuration': {
                'method': 'subject_level_splits',
                'seed': args.split_seed,
                'train_ratio': args.train_ratio,
                'val_ratio': args.val_ratio,
                'test_ratio': args.test_ratio,
                'train_subjects': successful_train_subjects,
                'val_subjects': successful_val_subjects,
                'test_subjects': test_subjects,  # CRITICAL: Save test subjects for final evaluation
                'train_subject_count': len(successful_train_subjects),
                'val_subject_count': len(successful_val_subjects),
                'test_subject_count': len(test_subjects)
            },
            'leakage_prevention': {
                'subject_level_splits': True,
                'no_subject_overlap': len(set(successful_train_subjects) & set(successful_val_subjects)) == 0,
                'reproducible_seed': args.split_seed,
                'validation_passed': True
            },
            'data_info': {
                'mask_type': args.mask_type,
                'mask_analysis': mask_analysis,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'total_samples': len(train_dataset) + len(val_dataset)
            },
            'command_line_args': vars(args)
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
        
        logger.info("=" * 80)
        logger.info("✅ LEAKAGE-FREE multi-subject VAE training completed successfully!")
        logger.info("🔒 GUARANTEED: No data leakage - subject-level splits enforced")
        logger.info("=" * 80)
        logger.info(f"📁 Results saved to: {experiment_dir}")
        logger.info(f"📊 Training subjects: {len(successful_train_subjects)} subjects")
        logger.info(f"📊 Validation subjects: {len(successful_val_subjects)} subjects")
        logger.info(f"🧪 TEST subjects: {len(test_subjects)} subjects (ISOLATED for evaluation)")
        logger.info(f"📊 Mask type used: {args.mask_type}")
        logger.info(f"🔢 Split seed used: {args.split_seed} (use same seed for BiLSTM!)")
        logger.info(f"📋 Train subjects: {successful_train_subjects}")
        logger.info(f"📋 Val subjects: {successful_val_subjects}")
        logger.info(f"🧪 TEST subjects: {test_subjects}")
        logger.info("⚠️  CRITICAL: Test subjects saved in metadata for final evaluation!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Continuous VAE + BiLSTM Training Pipeline
Trains VAE first, then uses it as frozen encoder for BiLSTM training
Ensures no data leakage with consistent 90/10 split across both phases
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import gc
import json
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import psutil
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import working modules
from src.utils.paths import setup_paths, PathConfig
from src.utils.config import load_config, create_config_from_yaml
from src.utils.logging import setup_logging, get_logger
from utils.data_utils_enhanced import EnhancedPviDataset, load_dataset_with_best_mask, analyze_all_mask_formats

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments for the complete pipeline"""
    parser = argparse.ArgumentParser(
        description="Continuous VAE + BiLSTM Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to merged pipeline configuration YAML file"
    )
    
    # Pipeline mode control
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        default="full_pipeline",
        choices=["vae_only", "bilstm_only", "full_pipeline"],
        help="Which part of the pipeline to run"
    )
    
    # Data configuration
    parser.add_argument(
        "--data-root",
        type=str,
        help="Override data root directory"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to use"
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="mask10",
        choices=["auto", "metadata", "mask01", "mask05", "mask10", "mask15"],
        help="Mask type to use for all subjects"
    )
    parser.add_argument(
        "--data-split-seed",
        type=int,
        default=42,
        help="Seed for reproducible train/val split"
    )
    
    # VAE-specific arguments
    parser.add_argument(
        "--vae-output-dir",
        type=str,
        help="Directory to save trained VAE model"
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        help="Path to pre-trained VAE checkpoint (for bilstm_only mode)"
    )
    
    # Hardware configuration
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    
    # Logging and debugging
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
    
    # Evaluation configuration
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=500,
        help="Maximum number of samples for evaluation"
    )
    
    return parser.parse_args()


def load_pipeline_config(config_path: str) -> Dict:
    """Load and validate merged pipeline configuration"""
    logger.info(f"Loading pipeline configuration from: {config_path}")
    
    with open(config_path, 'r') as file:
        config_text = file.read()
    
    # Replace timestamp placeholder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_text = config_text.replace('{timestamp}', timestamp)
    
    config = yaml.safe_load(config_text)
    
    # Validate required sections
    required_sections = ['pipeline_config', 'vae_phase', 'bilstm_phase']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    logger.info("✅ Pipeline configuration loaded and validated")
    return config


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


def create_data_split(subjects: List[str], split_ratio: float = 0.9, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Create deterministic train/val split and save split info to prevent leakage
    
    Args:
        subjects: List of all available subjects
        split_ratio: Ratio for training split (e.g., 0.9 for 90/10 split)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_subjects, val_subjects)
    """
    logger.info(f"🎯 Creating data split with {split_ratio:.1%} training ratio")
    logger.info(f"📊 Total subjects: {len(subjects)}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Shuffle subjects deterministically
    shuffled_subjects = subjects.copy()
    np.random.shuffle(shuffled_subjects)
    
    # Calculate split sizes
    train_size = int(len(shuffled_subjects) * split_ratio)
    val_size = len(shuffled_subjects) - train_size
    
    # Create splits
    train_subjects = shuffled_subjects[:train_size]
    val_subjects = shuffled_subjects[train_size:]
    
    logger.info(f"📈 Training subjects: {len(train_subjects)}")
    logger.info(f"📉 Validation subjects: {len(val_subjects)}")
    logger.info(f"🔢 Split seed: {seed}")
    
    # Log subject distribution
    logger.info("📋 Subject distribution:")
    logger.info(f"   Train: {train_subjects}")
    logger.info(f"   Val: {val_subjects}")
    
    return train_subjects, val_subjects


def save_data_split_info(train_subjects: List[str], val_subjects: List[str], 
                        experiment_dir: Path, seed: int):
    """Save data split information for reproducibility"""
    split_info = {
        'split_timestamp': datetime.now().isoformat(),
        'split_seed': seed,
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'train_count': len(train_subjects),
        'val_count': len(val_subjects),
        'split_ratio': len(train_subjects) / (len(train_subjects) + len(val_subjects))
    }
    
    split_file = experiment_dir / 'config' / 'data_split_info.json'
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logger.info(f"✅ Data split information saved: {split_file}")


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"🧹 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")


def create_experiment_directory(path_manager, experiment_name: str = None) -> Path:
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"pipeline_vae_bilstm_{timestamp}"
    
    experiment_dir = path_manager.experiments_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = ["checkpoints", "logs", "results", "evaluation", "config", "vae_phase", "bilstm_phase"]
    for subdir in subdirs:
        (experiment_dir / subdir).mkdir(exist_ok=True)
    
    logger.info(f"✅ Experiment directory created: {experiment_dir}")
    return experiment_dir


# VAE Model Definition (matching the trained checkpoint)
class VAE(nn.Module):
    """VAE model for image feature extraction"""
    
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(512 * 2 * 2, latent_dim)
        
        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, 512 * 2 * 2)
        self.bn_dec = nn.BatchNorm1d(512 * 2 * 2)
        
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_dec1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_dec2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_dec3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.dropout = nn.Dropout(0.2)
        
    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        x = self.fc_decoder(z)
        x = F.relu(self.bn_dec(x))
        x = x.view(x.size(0), 512, 2, 2)
        
        x = F.relu(self.bn_dec1(self.deconv1(x)))
        x = F.relu(self.bn_dec2(self.deconv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_dec3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def create_vae_datasets(train_subjects: List[str], val_subjects: List[str], 
                       data_root: str, mask_type: str, session: str = "baseline"):
    """Create VAE training datasets from subject lists"""
    logger.info(f"🏗️  Creating VAE datasets...")
    logger.info(f"📊 Train subjects: {len(train_subjects)}")
    logger.info(f"📊 Val subjects: {len(val_subjects)}")
    
    # Create training dataset
    train_datasets = []
    successful_train_subjects = []
    
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
    
    if not train_datasets:
        raise ValueError("No valid training datasets were loaded for VAE!")
    if not val_datasets:
        raise ValueError("No valid validation datasets were loaded for VAE!")
    
    # Combine datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)
    
    # Memory cleanup
    del train_datasets, val_datasets
    gc.collect()
    
    logger.info(f"🎯 VAE datasets created:")
    logger.info(f"   📈 Training: {len(combined_train_dataset)} samples from {len(successful_train_subjects)} subjects")
    logger.info(f"   📉 Validation: {len(combined_val_dataset)} samples from {len(successful_val_subjects)} subjects")
    
    return combined_train_dataset, combined_val_dataset, successful_train_subjects, successful_val_subjects


def train_vae_phase(train_subjects: List[str], val_subjects: List[str], 
                   config: Dict, args, experiment_dir: Path, device) -> str:
    """Train VAE model and return checkpoint path"""
    logger.info("🏗️  Starting VAE training phase...")
    
    # Extract VAE configuration
    vae_config = config['vae_phase']
    model_config = vae_config['model_config']
    training_config = vae_config['training_config']
    
    # Create VAE datasets
    data_root = args.data_root or config['environment']['data_root']
    session = config['pipeline_config'].get('session', 'baseline')
    
    train_dataset, val_dataset, successful_train_subjects, successful_val_subjects = create_vae_datasets(
        train_subjects, val_subjects, data_root, args.mask_type, session
    )
    
    # Create data loaders
    batch_size = int(training_config['batch_size'])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    logger.info(f"✅ VAE data loaders created:")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Batch size: {batch_size}")
    
    # Create VAE model
    latent_dim = int(model_config['latent_dim'])
    model = VAE(latent_dim=latent_dim).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ VAE model created:")
    logger.info(f"   Latent dim: {latent_dim}")
    logger.info(f"   Total params: {total_params:,}")
    logger.info(f"   Trainable params: {trainable_params:,}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay'])
    )
    
    # Setup scheduler
    if training_config.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=int(training_config['num_epochs'])
        )
    
    # Training loop
    num_epochs = int(training_config['num_epochs'])
    beta_min = float(training_config['beta_min'])
    beta_max = float(training_config['beta_max'])
    beta_warmup_epochs = int(training_config['beta_warmup_epochs'])
    
    best_val_loss = float('inf')
    vae_training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'learning_rates': [],
        'beta_values': []
    }
    
    logger.info(f"🎯 Starting VAE training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nVAE Epoch {epoch}/{num_epochs}")
        
        # Calculate beta for KL annealing
        if epoch <= beta_warmup_epochs:
            beta = beta_min + (beta_max - beta_min) * (epoch / beta_warmup_epochs)
        else:
            beta = beta_max
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        num_train_batches = 0
        
        train_pbar = tqdm(train_loader, desc="VAE Training")
        for batch in train_pbar:
            try:
                # Get images
                images = batch['pviHP']['img']  # [batch_size, 32, 32, num_frames]
                batch_size_actual = images.shape[0]
                
                # Process each frame in the batch
                total_batch_loss = 0.0
                total_batch_mae = 0.0
                frame_count = 0
                
                for frame_idx in range(images.shape[-1]):
                    frame_batch = images[:, :, :, frame_idx]  # [batch_size, 32, 32]
                    frame_batch = frame_batch.unsqueeze(1)  # [batch_size, 1, 32, 32]
                    frame_batch = frame_batch.float().to(device)
                    
                    # Normalize frames
                    frame_batch = torch.nan_to_num(frame_batch, nan=0.0)
                    for i in range(frame_batch.shape[0]):
                        frame_min = frame_batch[i].min()
                        frame_max = frame_batch[i].max()
                        if frame_max > frame_min:
                            frame_batch[i] = (frame_batch[i] - frame_min) / (frame_max - frame_min)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    recon_batch, mu, logvar = model(frame_batch)
                    
                    # Compute losses
                    recon_loss = F.mse_loss(recon_batch, frame_batch, reduction='mean')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / frame_batch.shape[0]
                    
                    loss = recon_loss + beta * kl_loss
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config['grad_clip_norm']))
                    
                    optimizer.step()
                    
                    # Calculate MAE
                    mae = F.l1_loss(recon_batch, frame_batch, reduction='mean')
                    
                    total_batch_loss += loss.item()
                    total_batch_mae += mae.item()
                    frame_count += 1
                
                avg_batch_loss = total_batch_loss / frame_count
                avg_batch_mae = total_batch_mae / frame_count
                
                train_loss += avg_batch_loss
                train_mae += avg_batch_mae
                num_train_batches += 1
                
                train_pbar.set_postfix({
                    'Loss': f"{avg_batch_loss:.4f}",
                    'MAE': f"{avg_batch_mae:.4f}",
                    'Beta': f"{beta:.4f}",
                    'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except Exception as e:
                logger.warning(f"VAE training error: {e}")
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="VAE Validation")
            for batch in val_pbar:
                try:
                    images = batch['pviHP']['img']
                    
                    total_batch_loss = 0.0
                    total_batch_mae = 0.0
                    frame_count = 0
                    
                    for frame_idx in range(images.shape[-1]):
                        frame_batch = images[:, :, :, frame_idx]
                        frame_batch = frame_batch.unsqueeze(1).float().to(device)
                        
                        # Normalize frames
                        frame_batch = torch.nan_to_num(frame_batch, nan=0.0)
                        for i in range(frame_batch.shape[0]):
                            frame_min = frame_batch[i].min()
                            frame_max = frame_batch[i].max()
                            if frame_max > frame_min:
                                frame_batch[i] = (frame_batch[i] - frame_min) / (frame_max - frame_min)
                        
                        recon_batch, mu, logvar = model(frame_batch)
                        
                        recon_loss = F.mse_loss(recon_batch, frame_batch, reduction='mean')
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / frame_batch.shape[0]
                        
                        loss = recon_loss + beta * kl_loss
                        mae = F.l1_loss(recon_batch, frame_batch, reduction='mean')
                        
                        total_batch_loss += loss.item()
                        total_batch_mae += mae.item()
                        frame_count += 1
                    
                    avg_batch_loss = total_batch_loss / frame_count
                    avg_batch_mae = total_batch_mae / frame_count
                    
                    val_loss += avg_batch_loss
                    val_mae += avg_batch_mae
                    num_val_batches += 1
                    
                except Exception as e:
                    logger.warning(f"VAE validation error: {e}")
                    continue
        
        # Calculate epoch metrics
        if num_train_batches > 0:
            train_loss /= num_train_batches
            train_mae /= num_train_batches
        
        if num_val_batches > 0:
            val_loss /= num_val_batches
            val_mae /= num_val_batches
        
        # Update scheduler
        if training_config.get('use_scheduler', True):
            scheduler.step()
        
        # Store history
        vae_training_history['train_loss'].append(train_loss)
        vae_training_history['val_loss'].append(val_loss)
        vae_training_history['train_mae'].append(train_mae)
        vae_training_history['val_mae'].append(val_mae)
        vae_training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        vae_training_history['beta_values'].append(beta)
        
        logger.info(f"VAE Epoch {epoch} | Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            vae_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': vae_config,
                'training_history': vae_training_history,
                'successful_train_subjects': successful_train_subjects,
                'successful_val_subjects': successful_val_subjects
            }
            
            vae_checkpoint_path = experiment_dir / 'vae_phase' / 'best_vae_model.pt'
            torch.save(vae_checkpoint, vae_checkpoint_path)
            logger.info(f"✅ Best VAE model saved! Val Loss: {best_val_loss:.4f}")
    
    # Save final training history
    history_path = experiment_dir / 'vae_phase' / 'vae_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(vae_training_history, f, indent=2, default=str)
    
    logger.info("✅ VAE training phase completed!")
    logger.info(f"📁 VAE checkpoint saved: {vae_checkpoint_path}")
    logger.info(f"📊 Best validation loss: {best_val_loss:.4f}")
    
    return str(vae_checkpoint_path)


# BiLSTM Dataset for temporal sequences
class BiLSTMDataset(Dataset):
    """BiLSTM dataset with 10-frame temporal window"""
    
    def __init__(self, subjects: List[str], data_root: str, mask_type: str = "mask10",
                 pattern_offsets: List[int] = None, max_samples_per_subject: int = 200,
                 sequence_step_size: int = 15, session: str = "baseline"):
        
        self.subjects = subjects
        self.data_root = data_root
        self.mask_type = mask_type
        self.pattern_offsets = pattern_offsets or [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
        self.max_samples_per_subject = max_samples_per_subject
        self.sequence_step_size = sequence_step_size
        self.session = session
        
        self.sequences = []
        self.targets = []
        self.subject_labels = []
        
        self._load_all_subjects()
        
        if len(self.sequences) == 0:
            raise ValueError("No sequences loaded for BiLSTM training!")
    
    def _load_all_subjects(self):
        """Load temporal sequences from all subjects"""
        successful_subjects = []
        failed_subjects = []
        
        for subject in self.subjects:
            try:
                file_path = Path(self.data_root) / f"{subject}_{self.session}_masked.h5"
                if not file_path.exists():
                    logger.warning(f"⚠️  BiLSTM data file not found for {subject}, skipping")
                    failed_subjects.append(subject)
                    continue
                
                dataset = load_dataset_with_best_mask(str(file_path), self.mask_type)
                subject_sequences, subject_targets = self._process_subject_data(dataset, subject)
                
                if len(subject_sequences) > 0:
                    self.sequences.extend(subject_sequences)
                    self.targets.extend(subject_targets)
                    self.subject_labels.extend([subject] * len(subject_sequences))
                    successful_subjects.append(subject)
                    logger.info(f"✅ BiLSTM {subject}: {len(subject_sequences)} sequences")
                else:
                    failed_subjects.append(subject)
                    
            except Exception as e:
                logger.warning(f"❌ BiLSTM failed to load {subject}: {e}")
                failed_subjects.append(subject)
        
        logger.info(f"BiLSTM loaded {len(successful_subjects)} subjects, {len(self.sequences)} sequences")
        gc.collect()
    
    def _process_subject_data(self, dataset, subject):
        """Process a single subject's data with 10-frame temporal window"""
        subject_sequences = []
        subject_targets = []
        
        num_samples = min(len(dataset), self.max_samples_per_subject)
        
        for sample_idx in range(num_samples):
            try:
                sample = dataset[sample_idx]
                pvi_img = sample['pviHP']['img']  # [32, 32, num_frames]
                bp_signal = sample['bp']['signal']  # [num_frames]
                
                num_frames = pvi_img.shape[-1]
                
                # Valid central indices
                min_offset = min(self.pattern_offsets)
                max_offset = max(self.pattern_offsets)
                valid_start = max(0, -min_offset)
                valid_end = min(num_frames, num_frames - max_offset)
                
                central_indices = list(range(valid_start, valid_end, self.sequence_step_size))
                
                for central_idx in central_indices:
                    seq_frames = []
                    valid_sequence = True
                    
                    for offset in self.pattern_offsets:
                        frame_idx = central_idx + offset
                        if 0 <= frame_idx < num_frames:
                            frame = pvi_img[:, :, frame_idx]  # [32, 32]
                            frame = torch.tensor(frame, dtype=torch.float32)
                            frame = torch.nan_to_num(frame, nan=0.0)
                            
                            # Normalize to [0, 1]
                            frame_min = frame.min()
                            frame_max = frame.max()
                            if frame_max > frame_min:
                                frame = (frame - frame_min) / (frame_max - frame_min)
                            
                            frame = frame.unsqueeze(0)  # [1, 32, 32]
                            seq_frames.append(frame)
                        else:
                            valid_sequence = False
                            break
                    
                    if valid_sequence:
                        sequence = torch.stack(seq_frames)  # [10, 1, 32, 32]
                        
                        # Extract BP signal
                        if bp_signal.dim() == 1:
                            target_bp = bp_signal
                        else:
                            target_bp = bp_signal[central_idx] if central_idx < bp_signal.shape[0] else bp_signal[0]
                        
                        # Ensure target is exactly 50 samples
                        if target_bp.dim() > 1:
                            target_bp = target_bp.flatten()
                        
                        current_length = target_bp.shape[0]
                        if current_length != 50:
                            if current_length > 50:
                                indices = torch.linspace(0, current_length - 1, 50).long()
                                target_bp = target_bp[indices]
                            else:
                                target_bp = torch.nn.functional.interpolate(
                                    target_bp.unsqueeze(0).unsqueeze(0),
                                    size=50,
                                    mode='linear',
                                    align_corners=False
                                ).squeeze()
                        
                        target_bp = target_bp.float()
                        
                        subject_sequences.append(sequence)
                        subject_targets.append(target_bp)
                        
            except Exception as e:
                logger.warning(f"Error processing BiLSTM sample {sample_idx} for {subject}: {e}")
                continue
        
        return subject_sequences, subject_targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': self.sequences[idx],
            'targets': self.targets[idx],
            'subject': self.subject_labels[idx]
        }


# BiLSTM Model Components
class MultiHeadAttention(nn.Module):
    """Multi-head attention with current frame bias"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2,
                 current_frame_bias: float = 3.0):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.current_frame_bias = current_frame_bias
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, pattern_offsets=None):
        batch_size, seq_len, hidden_dim = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Add bias for current frame (t+0)
        if pattern_offsets is not None and self.current_frame_bias > 0:
            try:
                current_frame_idx = pattern_offsets.index(0)
                bias_matrix = torch.zeros_like(scores[0, 0])
                bias_matrix[:, current_frame_idx] += self.current_frame_bias * 2.0
                bias_matrix[current_frame_idx, :] += self.current_frame_bias * 3.0
                bias_matrix[current_frame_idx, current_frame_idx] += self.current_frame_bias * 2.0
                bias_matrix = bias_matrix.unsqueeze(0).unsqueeze(0)
                scores = scores + bias_matrix.to(scores.device)
            except ValueError:
                pass
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.output_proj(attended)
        
        return self.layer_norm(x + output), attention_weights.mean(dim=1)


class BiLSTMBPPredictor(nn.Module):
    """BiLSTM BP predictor with frozen VAE encoder"""
    
    def __init__(self, vae_model, latent_dim: int = 128, hidden_dim: int = 256,
                 num_layers: int = 3, num_heads: int = 8, dropout: float = 0.3,
                 use_attention: bool = True, pattern_offsets: List[int] = None,
                 current_frame_bias: float = 3.0):
        super(BiLSTMBPPredictor, self).__init__()
        
        self.vae = vae_model
        self.use_attention = use_attention
        self.pattern_offsets = pattern_offsets or [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        lstm_output_dim = hidden_dim * 2
        
        # Attention mechanism
        if use_attention:
            self.attention = MultiHeadAttention(
                hidden_dim=lstm_output_dim,
                num_heads=num_heads,
                dropout=dropout * 0.5,
                current_frame_bias=current_frame_bias
            )
        
        # Temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(lstm_output_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        # Prediction heads
        self.waveform_head = nn.Linear(hidden_dim // 4, 50)
        
        self.bp_feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3)
        )
        
        self.systolic_head = nn.Sequential(
            nn.Linear(hidden_dim // 8, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )
        
        self.diastolic_head = nn.Sequential(
            nn.Linear(hidden_dim // 8, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better BP prediction"""
        import math
        
        with torch.no_grad():
            # Initialize BP heads for physiological range
            fan_in = self.systolic_head[-1].in_features
            std = math.sqrt(2.0 / fan_in)
            self.systolic_head[-1].weight.data.normal_(0, std)
            self.systolic_head[-1].bias.data.uniform_(110, 130)
            
            self.diastolic_head[-1].weight.data.normal_(0, std)
            self.diastolic_head[-1].bias.data.uniform_(70, 90)
            
            if self.waveform_head.weight.dim() >= 2:
                torch.nn.init.kaiming_normal_(self.waveform_head.weight, mode='fan_out', nonlinearity='relu')
            if self.waveform_head.bias is not None:
                self.waveform_head.bias.data.uniform_(60, 80)
    
    def forward(self, x_seq, return_attention=False):
        batch_size, seq_len = x_seq.shape[:2]
        
        # Encode frames with frozen VAE
        latent_seq = []
        with torch.no_grad():
            for t in range(seq_len):
                mu_t, _ = self.vae.encode(x_seq[:, t])
                latent_seq.append(mu_t)
        
        latent_seq = torch.stack(latent_seq, dim=1)  # [batch_size, seq_len, latent_dim]
        
        # Process through BiLSTM
        projected_seq = self.input_projection(latent_seq)
        lstm_out, _ = self.lstm(projected_seq)
        
        # Apply attention
        attention_weights = None
        if self.attention is not None:
            lstm_out, attention_weights = self.attention(lstm_out, self.pattern_offsets)
        
        # Temporal convolution and aggregation
        conv_input = lstm_out.transpose(1, 2)
        conv_out = self.temporal_conv(conv_input)
        features = torch.mean(conv_out, dim=2)
        
        # Generate predictions
        final_features = self.output_layers(features)
        bp_features = self.bp_feature_extractor(final_features)
        
        outputs = {
            'waveform': self.waveform_head(final_features),
            'systolic': self.systolic_head(bp_features),
            'diastolic': self.diastolic_head(bp_features)
        }
        
        if return_attention and attention_weights is not None:
            outputs['attention_weights'] = attention_weights
        
        return outputs


# BiLSTM Loss Function
class BiLSTMBPLoss(nn.Module):
    """Composite loss function for BiLSTM BP prediction"""
    
    def __init__(self, waveform_weight=0.4, systolic_weight=0.3, diastolic_weight=0.3,
                 huber_delta=1.0, physiological_constraint=True, pulse_pressure_weight=0.1):
        super(BiLSTMBPLoss, self).__init__()
        self.waveform_weight = waveform_weight
        self.systolic_weight = systolic_weight
        self.diastolic_weight = diastolic_weight
        self.huber_delta = huber_delta
        self.physiological_constraint = physiological_constraint
        self.pulse_pressure_weight = pulse_pressure_weight
        
        # Normalize weights
        total = waveform_weight + systolic_weight + diastolic_weight
        self.waveform_weight /= total
        self.systolic_weight /= total
        self.diastolic_weight /= total
    
    def forward(self, predictions, targets):
        pred_waveform = predictions['waveform']
        
        # Extract ground truth values
        target_systolic, target_diastolic = self.extract_bp_values(targets)
        
        # Waveform loss
        waveform_loss = nn.functional.huber_loss(pred_waveform, targets, delta=self.huber_delta)
        
        # SBP/DBP losses
        pred_systolic = predictions['systolic'].squeeze()
        systolic_loss = nn.functional.huber_loss(pred_systolic, target_systolic, delta=self.huber_delta)
        
        pred_diastolic = predictions['diastolic'].squeeze()
        diastolic_loss = nn.functional.huber_loss(pred_diastolic, target_diastolic, delta=self.huber_delta)
        
        # Physiological constraint
        if self.physiological_constraint:
            invalid_bp = (pred_systolic <= pred_diastolic).float()
            physiological_penalty = torch.mean(invalid_bp * torch.abs(pred_systolic - pred_diastolic))
            systolic_loss += physiological_penalty
        
        # Pulse pressure constraint
        pulse_pressure_loss = 0.0
        if self.physiological_constraint:
            pred_pp = pred_systolic - pred_diastolic
            target_pp = target_systolic - target_diastolic
            pulse_pressure_loss = nn.functional.huber_loss(pred_pp, target_pp, delta=self.huber_delta)
        
        # Total loss
        total_loss = (self.waveform_weight * waveform_loss +
                     self.systolic_weight * systolic_loss +
                     self.diastolic_weight * diastolic_loss +
                     self.pulse_pressure_weight * pulse_pressure_loss)
        
        return {
            'total_loss': total_loss,
            'waveform_loss': waveform_loss,
            'systolic_loss': systolic_loss,
            'diastolic_loss': diastolic_loss,
            'pulse_pressure_loss': pulse_pressure_loss
        }
    
    def extract_bp_values(self, waveform):
        """Extract systolic and diastolic values from waveform"""
        batch_size, signal_length = waveform.shape
        device = waveform.device
        
        systolic_values = torch.zeros(batch_size, device=device)
        diastolic_values = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            signal = waveform[i]
            
            # Systolic: maximum
            sys_val = torch.max(signal)
            sys_idx = torch.argmax(signal)
            
            # Diastolic: minimum in post-systolic region
            search_start = max(sys_idx + 1, int(signal_length * 0.6))
            search_end = min(signal_length, int(signal_length * 0.95))
            
            if search_start < search_end:
                diastolic_window = signal[search_start:search_end]
                dias_val = torch.min(diastolic_window)
            else:
                dias_val = torch.min(signal)
            
            # Ensure physiological constraint
            pulse_pressure = sys_val - dias_val
            min_pulse_pressure = 15.0
            
            if pulse_pressure < min_pulse_pressure:
                center_pressure = (sys_val + dias_val) / 2
                sys_val = center_pressure + min_pulse_pressure / 2
                dias_val = center_pressure - min_pulse_pressure / 2
            
            systolic_values[i] = sys_val
            diastolic_values[i] = dias_val
        
        return systolic_values, diastolic_values


def train_bilstm_phase(train_subjects: List[str], val_subjects: List[str],
                      vae_checkpoint_path: str, config: Dict, args, experiment_dir: Path, device) -> str:
    """Train BiLSTM model using frozen VAE and return checkpoint path"""
    logger.info("🧠 Starting BiLSTM training phase...")
    
    # Load pre-trained VAE
    logger.info(f"Loading VAE from: {vae_checkpoint_path}")
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
    
    # Extract VAE configuration
    vae_config = config['vae_phase']['model_config']
    latent_dim = int(vae_config['latent_dim'])
    
    # Create and load VAE
    vae = VAE(latent_dim=latent_dim)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.to(device)
    vae.eval()
    
    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False
    
    logger.info(f"✅ VAE loaded and frozen (latent_dim: {latent_dim})")
    
    # Extract BiLSTM configuration
    bilstm_config = config['bilstm_phase']
    data_config = bilstm_config['data_config']
    model_config = bilstm_config['model_config']
    training_config = bilstm_config['training_config']
    loss_config = config['loss_config']
    
    # Create BiLSTM datasets
    data_root = args.data_root or config['environment']['data_root']
    session = config['pipeline_config'].get('session', 'baseline')
    
    pattern_offsets = data_config['pattern_offsets']
    max_samples_per_subject = int(data_config['max_samples_per_subject'])
    sequence_step_size = int(data_config['sequence_step_size'])
    
    train_dataset = BiLSTMDataset(
        subjects=train_subjects,
        data_root=data_root,
        mask_type=args.mask_type,
        pattern_offsets=pattern_offsets,
        max_samples_per_subject=max_samples_per_subject,
        sequence_step_size=sequence_step_size,
        session=session
    )
    
    val_dataset = BiLSTMDataset(
        subjects=val_subjects,
        data_root=data_root,
        mask_type=args.mask_type,
        pattern_offsets=pattern_offsets,
        max_samples_per_subject=max_samples_per_subject,
        sequence_step_size=sequence_step_size,
        session=session
    )
    
    # Create data loaders
    batch_size = int(training_config['batch_size'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"✅ BiLSTM datasets created:")
    logger.info(f"   Train: {len(train_dataset)} sequences")
    logger.info(f"   Val: {len(val_dataset)} sequences")
    logger.info(f"   Batch size: {batch_size}")
    
    # Create BiLSTM model
    bilstm_model_config = model_config['bilstm_config']
    attention_config = model_config['attention_config']
    
    model = BiLSTMBPPredictor(
        vae_model=vae,
        latent_dim=latent_dim,
        hidden_dim=int(bilstm_model_config['hidden_dim']),
        num_layers=int(bilstm_model_config['num_layers']),
        num_heads=int(attention_config['num_attention_heads']),
        dropout=float(bilstm_model_config['dropout_rate']),
        use_attention=bool(attention_config['use_attention']),
        pattern_offsets=pattern_offsets,
        current_frame_bias=float(attention_config['current_frame_bias'])
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ BiLSTM model created with {total_params:,} trainable parameters")
    
    # Create loss function
    criterion = BiLSTMBPLoss(
        waveform_weight=float(loss_config['waveform_weight']),
        systolic_weight=float(loss_config['systolic_weight']),
        diastolic_weight=float(loss_config['diastolic_weight']),
        huber_delta=float(loss_config['huber_delta']),
        physiological_constraint=bool(loss_config['physiological_constraint']),
        pulse_pressure_weight=float(loss_config['pulse_pressure_weight'])
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay']),
        betas=training_config['optimizer_config']['betas'],
        amsgrad=bool(training_config['optimizer_config']['amsgrad'])
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=int(training_config['scheduler_patience']),
        factor=float(training_config['factor']),
        min_lr=float(training_config['min_lr']),
        verbose=True
    )
    
    # Training loop
    num_epochs = int(training_config['num_epochs'])
    best_val_mae = float('inf')
    bilstm_training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'val_systolic_mae': [],
        'val_diastolic_mae': [],
        'learning_rates': []
    }
    
    logger.info(f"🎯 Starting BiLSTM training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nBiLSTM Epoch {epoch}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        num_train_batches = 0
        
        train_pbar = tqdm(train_loader, desc="BiLSTM Training")
        for batch in train_pbar:
            try:
                sequences = batch['sequences'].to(device)
                targets = batch['targets'].to(device)
                
                optimizer.zero_grad()
                
                outputs = model(sequences)
                loss_dict = criterion(outputs, targets)
                total_loss = loss_dict['total_loss']
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config['grad_clip_norm']))
                
                optimizer.step()
                
                # Calculate MAE
                mae = torch.mean(torch.abs(outputs['waveform'] - targets))
                
                train_loss += total_loss.item()
                train_mae += mae.item()
                num_train_batches += 1
                
                train_pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'MAE': f"{mae.item():.4f}",
                    'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except Exception as e:
                logger.warning(f"BiLSTM training error: {e}")
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_systolic_predictions = []
        val_diastolic_predictions = []
        val_systolic_targets = []
        val_diastolic_targets = []
        num_val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="BiLSTM Validation")
            for batch in val_pbar:
                try:
                    sequences = batch['sequences'].to(device)
                    targets = batch['targets'].to(device)
                    
                    outputs = model(sequences)
                    loss_dict = criterion(outputs, targets)
                    
                    val_loss += loss_dict['total_loss'].item()
                    
                    # Calculate MAE
                    mae = torch.mean(torch.abs(outputs['waveform'] - targets))
                    val_mae += mae.item()
                    
                    # Store SBP/DBP predictions
                    val_systolic_predictions.append(outputs['systolic'].cpu().numpy())
                    val_diastolic_predictions.append(outputs['diastolic'].cpu().numpy())
                    
                    target_sys, target_dias = criterion.extract_bp_values(targets)
                    val_systolic_targets.append(target_sys.cpu().numpy())
                    val_diastolic_targets.append(target_dias.cpu().numpy())
                    
                    num_val_batches += 1
                    
                except Exception as e:
                    logger.warning(f"BiLSTM validation error: {e}")
                    continue
        
        # Calculate epoch metrics
        if num_train_batches > 0:
            train_loss /= num_train_batches
            train_mae /= num_train_batches
        
        if num_val_batches > 0:
            val_loss /= num_val_batches
            val_mae /= num_val_batches
        
        # Calculate SBP/DBP MAE
        val_systolic_mae = 0.0
        val_diastolic_mae = 0.0
        if val_systolic_predictions and val_systolic_targets:
            all_sys_pred = np.concatenate(val_systolic_predictions, axis=0).flatten()
            all_sys_target = np.concatenate(val_systolic_targets, axis=0).flatten()
            val_systolic_mae = np.mean(np.abs(all_sys_pred - all_sys_target))
            
            all_dias_pred = np.concatenate(val_diastolic_predictions, axis=0).flatten()
            all_dias_target = np.concatenate(val_diastolic_targets, axis=0).flatten()
            val_diastolic_mae = np.mean(np.abs(all_dias_pred - all_dias_target))
        
        # Update scheduler
        scheduler.step(val_mae)
        
        # Store history
        bilstm_training_history['train_loss'].append(train_loss)
        bilstm_training_history['val_loss'].append(val_loss)
        bilstm_training_history['train_mae'].append(train_mae)
        bilstm_training_history['val_mae'].append(val_mae)
        bilstm_training_history['val_systolic_mae'].append(val_systolic_mae)
        bilstm_training_history['val_diastolic_mae'].append(val_diastolic_mae)
        bilstm_training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        logger.info(f"BiLSTM Epoch {epoch} | Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
        logger.info(f"  └─ SBP MAE: {val_systolic_mae:.2f} | DBP MAE: {val_diastolic_mae:.2f}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            
            bilstm_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_mae': best_val_mae,
                'config': bilstm_config,
                'training_history': bilstm_training_history,
                'vae_checkpoint_path': vae_checkpoint_path,
                'pattern_offsets': pattern_offsets
            }
            
            bilstm_checkpoint_path = experiment_dir / 'bilstm_phase' / 'best_bilstm_model.pt'
            torch.save(bilstm_checkpoint, bilstm_checkpoint_path)
            logger.info(f"✅ Best BiLSTM model saved! Val MAE: {best_val_mae:.4f}")
    
    # Save training history
    history_path = experiment_dir / 'bilstm_phase' / 'bilstm_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(bilstm_training_history, f, indent=2, default=str)
    
    logger.info("✅ BiLSTM training phase completed!")
    logger.info(f"📁 BiLSTM checkpoint saved: {bilstm_checkpoint_path}")
    logger.info(f"📊 Best validation MAE: {best_val_mae:.4f}")
    
    return str(bilstm_checkpoint_path)


def evaluate_pipeline(val_subjects: List[str], vae_checkpoint_path: str, bilstm_checkpoint_path: str,
                     config: Dict, args, experiment_dir: Path, device) -> Dict:
    """Evaluate the complete pipeline on validation set"""
    logger.info("📊 Starting pipeline evaluation...")
    
    # Load models
    vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
    bilstm_checkpoint = torch.load(bilstm_checkpoint_path, map_location=device, weights_only=False)
    
    # Create VAE
    vae_config = config['vae_phase']['model_config']
    latent_dim = int(vae_config['latent_dim'])
    vae = VAE(latent_dim=latent_dim)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.to(device)
    vae.eval()
    
    # Create BiLSTM
    bilstm_config = config['bilstm_phase']
    model_config = bilstm_config['model_config']
    bilstm_model_config = model_config['bilstm_config']
    attention_config = model_config['attention_config']
    
    pattern_offsets = bilstm_checkpoint['pattern_offsets']
    
    model = BiLSTMBPPredictor(
        vae_model=vae,
        latent_dim=latent_dim,
        hidden_dim=int(bilstm_model_config['hidden_dim']),
        num_layers=int(bilstm_model_config['num_layers']),
        num_heads=int(attention_config['num_attention_heads']),
        dropout=float(bilstm_model_config['dropout_rate']),
        use_attention=bool(attention_config['use_attention']),
        pattern_offsets=pattern_offsets,
        current_frame_bias=float(attention_config['current_frame_bias'])
    ).to(device)
    
    model.load_state_dict(bilstm_checkpoint['model_state_dict'])
    model.eval()
    
    # Create loss function
    loss_config = config['loss_config']
    criterion = BiLSTMBPLoss(
        waveform_weight=float(loss_config['waveform_weight']),
        systolic_weight=float(loss_config['systolic_weight']),
        diastolic_weight=float(loss_config['diastolic_weight']),
        huber_delta=float(loss_config['huber_delta']),
        physiological_constraint=bool(loss_config['physiological_constraint']),
        pulse_pressure_weight=float(loss_config['pulse_pressure_weight'])
    )
    
    # Create evaluation dataset
    data_root = args.data_root or config['environment']['data_root']
    session = config['pipeline_config'].get('session', 'baseline')
    data_config = bilstm_config['data_config']
    
    eval_dataset = BiLSTMDataset(
        subjects=val_subjects,
        data_root=data_root,
        mask_type=args.mask_type,
        pattern_offsets=pattern_offsets,
        max_samples_per_subject=int(data_config['max_samples_per_subject']),
        sequence_step_size=int(data_config['sequence_step_size']),
        session=session
    )
    
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Run evaluation
    max_samples = args.max_eval_samples or config['evaluation_config']['max_eval_samples']
    
    all_predictions = []
    all_targets = []
    sample_count = 0
    
    logger.info(f"🔍 Running pipeline evaluation with max {max_samples} samples...")
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            if sample_count >= max_samples:
                break
            
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(sequences)
            pred_waveforms = outputs['waveform']
            pred_systolic = outputs['systolic'].squeeze()
            pred_diastolic = outputs['diastolic'].squeeze()
            
            target_systolic, target_diastolic = criterion.extract_bp_values(targets)
            
            for i in range(len(targets)):
                if sample_count >= max_samples:
                    break
                
                pred_dict = {
                    'waveform': pred_waveforms[i].cpu().numpy(),
                    'systolic': pred_systolic[i].cpu().numpy(),
                    'diastolic': pred_diastolic[i].cpu().numpy()
                }
                
                target_dict = {
                    'waveform': targets[i].cpu().numpy(),
                    'systolic': target_systolic[i].cpu().numpy(),
                    'diastolic': target_diastolic[i].cpu().numpy()
                }
                
                all_predictions.append(pred_dict)
                all_targets.append(target_dict)
                sample_count += 1
    
    # Calculate metrics
    metrics = calculate_evaluation_metrics(all_predictions, all_targets)
    
    # Create evaluation plots
    create_evaluation_plots(all_predictions, all_targets, experiment_dir / 'evaluation', "Pipeline VAE+BiLSTM")
    
    # Save results
    save_evaluation_results(metrics, experiment_dir / 'evaluation')
    
    logger.info("✅ Pipeline evaluation completed!")
    logger.info(f"📊 Evaluated {len(all_predictions)} samples")
    logger.info(f"📈 Systolic MAE: {metrics['systolic']['mae']:.2f} mmHg (R²={metrics['systolic']['r2']:.3f})")
    logger.info(f"📈 Diastolic MAE: {metrics['diastolic']['mae']:.2f} mmHg (R²={metrics['diastolic']['r2']:.3f})")
    
    return metrics


def calculate_evaluation_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict:
    """Calculate comprehensive evaluation metrics"""
    pred_sys = np.array([float(p['systolic']) for p in predictions])
    pred_dias = np.array([float(p['diastolic']) for p in predictions])
    pred_waveforms = np.array([p['waveform'] for p in predictions])
    
    true_sys = np.array([float(t['systolic']) for t in targets])
    true_dias = np.array([float(t['diastolic']) for t in targets])
    true_waveforms = np.array([t['waveform'] for t in targets])
    
    metrics = {}
    
    # Systolic metrics
    metrics['systolic'] = {
        'r2': r2_score(true_sys, pred_sys),
        'mae': mean_absolute_error(true_sys, pred_sys),
        'rmse': np.sqrt(mean_squared_error(true_sys, pred_sys)),
        'pearson_r': stats.pearsonr(true_sys, pred_sys)[0]
    }
    
    # Diastolic metrics
    metrics['diastolic'] = {
        'r2': r2_score(true_dias, pred_dias),
        'mae': mean_absolute_error(true_dias, pred_dias),
        'rmse': np.sqrt(mean_squared_error(true_dias, pred_dias)),
        'pearson_r': stats.pearsonr(true_dias, pred_dias)[0]
    }
    
    # Waveform metrics
    waveform_r2_scores = []
    for i in range(len(pred_waveforms)):
        waveform_r2_scores.append(r2_score(true_waveforms[i], pred_waveforms[i]))
    
    metrics['waveform'] = {
        'r2': np.mean(waveform_r2_scores),
        'mae': mean_absolute_error(true_waveforms.flatten(), pred_waveforms.flatten()),
        'rmse': np.sqrt(mean_squared_error(true_waveforms.flatten(), pred_waveforms.flatten()))
    }
    
    # Clinical accuracy
    metrics['clinical'] = {
        'systolic_5mmhg': np.mean(np.abs(pred_sys - true_sys) <= 5) * 100,
        'systolic_10mmhg': np.mean(np.abs(pred_sys - true_sys) <= 10) * 100,
        'systolic_15mmhg': np.mean(np.abs(pred_sys - true_sys) <= 15) * 100,
        'diastolic_5mmhg': np.mean(np.abs(pred_dias - true_dias) <= 5) * 100,
        'diastolic_10mmhg': np.mean(np.abs(pred_dias - true_dias) <= 10) * 100,
        'diastolic_15mmhg': np.mean(np.abs(pred_dias - true_dias) <= 15) * 100
    }
    
    return metrics


def create_evaluation_plots(predictions: List[Dict], targets: List[Dict], 
                           output_dir: Path, model_name: str = "Pipeline VAE+BiLSTM"):
    """Create comprehensive evaluation plots"""
    output_dir.mkdir(exist_ok=True)
    
    # Extract data
    pred_sys = np.array([float(p['systolic']) for p in predictions])
    pred_dias = np.array([float(p['diastolic']) for p in predictions])
    pred_waveforms = np.array([p['waveform'] for p in predictions])
    
    true_sys = np.array([float(t['systolic']) for t in targets])
    true_dias = np.array([float(t['diastolic']) for t in targets])
    true_waveforms = np.array([t['waveform'] for t in targets])
    
    # Create plots (similar to BiLSTM evaluation)
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Systolic correlation
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(true_sys, pred_sys, alpha=0.6, s=20)
    min_val, max_val = min(true_sys.min(), pred_sys.min()), max(true_sys.max(), pred_sys.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
    ax1.set_xlabel('True Systolic (mmHg)')
    ax1.set_ylabel('Predicted Systolic (mmHg)')
    ax1.set_title('Systolic BP Correlation')
    r2_sys = r2_score(true_sys, pred_sys)
    ax1.text(0.05, 0.95, f'R²={r2_sys:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Diastolic correlation
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(true_dias, pred_dias, alpha=0.6, s=20)
    min_val, max_val = min(true_dias.min(), pred_dias.min()), max(true_dias.max(), pred_dias.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
    ax2.set_xlabel('True Diastolic (mmHg)')
    ax2.set_ylabel('Predicted Diastolic (mmHg)')
    ax2.set_title('Diastolic BP Correlation')
    r2_dias = r2_score(true_dias, pred_dias)
    ax2.text(0.05, 0.95, f'R²={r2_dias:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Waveform example
    ax3 = plt.subplot(3, 3, 3)
    example_idx = 0
    ax3.plot(true_waveforms[example_idx], 'b-', label='True', linewidth=2)
    ax3.plot(pred_waveforms[example_idx], 'r--', label='Predicted', linewidth=2)
    ax3.set_xlabel('Time Points')
    ax3.set_ylabel('BP (mmHg)')
    ax3.set_title('Waveform Example')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Systolic error histogram
    ax4 = plt.subplot(3, 3, 4)
    sys_errors = np.abs(pred_sys - true_sys)
    ax4.hist(sys_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(5, color='green', linestyle='--', linewidth=2)
    ax4.axvline(10, color='orange', linestyle='--', linewidth=2)
    ax4.axvline(15, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Absolute Error (mmHg)')
    ax4.set_ylabel('Count')
    ax4.set_title('Systolic Error Distribution')
    mae_sys = np.mean(sys_errors)
    acc_5_sys = np.mean(sys_errors <= 5) * 100
    acc_10_sys = np.mean(sys_errors <= 10) * 100
    ax4.text(0.7, 0.9, f'MAE: {mae_sys:.2f}\n≤5mmHg: {acc_5_sys:.1f}%\n≤10mmHg: {acc_10_sys:.1f}%',
             transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Diastolic error histogram
    ax5 = plt.subplot(3, 3, 5)
    dias_errors = np.abs(pred_dias - true_dias)
    ax5.hist(dias_errors, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax5.axvline(5, color='green', linestyle='--', linewidth=2)
    ax5.axvline(10, color='orange', linestyle='--', linewidth=2)
    ax5.axvline(15, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Absolute Error (mmHg)')
    ax5.set_ylabel('Count')
    ax5.set_title('Diastolic Error Distribution')
    mae_dias = np.mean(dias_errors)
    acc_5_dias = np.mean(dias_errors <= 5) * 100
    acc_10_dias = np.mean(dias_errors <= 10) * 100
    ax5.text(0.7, 0.9, f'MAE: {mae_dias:.2f}\n≤5mmHg: {acc_5_dias:.1f}%\n≤10mmHg: {acc_10_dias:.1f}%',
             transform=ax5.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Bland-Altman Systolic
    ax6 = plt.subplot(3, 3, 6)
    sys_diff = pred_sys - true_sys
    sys_mean = (pred_sys + true_sys) / 2
    ax6.scatter(sys_mean, sys_diff, alpha=0.6, s=20)
    mean_diff = np.mean(sys_diff)
    std_diff = np.std(sys_diff)
    ax6.axhline(mean_diff, color='blue', linestyle='-', label=f'Mean: {mean_diff:.2f}')
    ax6.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', label=f'+1.96SD: {mean_diff + 1.96*std_diff:.2f}')
    ax6.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', label=f'-1.96SD: {mean_diff - 1.96*std_diff:.2f}')
    ax6.set_xlabel('Mean Systolic (mmHg)')
    ax6.set_ylabel('Difference (mmHg)')
    ax6.set_title('Bland-Altman Systolic')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Bland-Altman Diastolic
    ax7 = plt.subplot(3, 3, 7)
    dias_diff = pred_dias - true_dias
    dias_mean = (pred_dias + true_dias) / 2
    ax7.scatter(dias_mean, dias_diff, alpha=0.6, s=20)
    mean_diff = np.mean(dias_diff)
    std_diff = np.std(dias_diff)
    ax7.axhline(mean_diff, color='blue', linestyle='-', label=f'Mean: {mean_diff:.2f}')
    ax7.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', label=f'+1.96SD: {mean_diff + 1.96*std_diff:.2f}')
    ax7.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', label=f'-1.96SD: {mean_diff - 1.96*std_diff:.2f}')
    ax7.set_xlabel('Mean Diastolic (mmHg)')
    ax7.set_ylabel('Difference (mmHg)')
    ax7.set_title('Bland-Altman Diastolic')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Combined performance
    ax8 = plt.subplot(3, 3, 8)
    metrics = ['SBP MAE', 'DBP MAE', 'SBP R²', 'DBP R²']
    values = [mae_sys, mae_dias, r2_sys, r2_dias]
    colors = ['red', 'red', 'green', 'green']
    bars = ax8.bar(metrics, values, color=colors, alpha=0.7)
    ax8.set_ylabel('Value')
    ax8.set_title('Performance Summary')
    ax8.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 9: Clinical accuracy
    ax9 = plt.subplot(3, 3, 9)
    thresholds = ['≤5mmHg', '≤10mmHg', '≤15mmHg']
    sys_accs = [acc_5_sys, acc_10_sys, np.mean(sys_errors <= 15) * 100]
    dias_accs = [acc_5_dias, acc_10_dias, np.mean(dias_errors <= 15) * 100]
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    ax9.bar(x - width/2, sys_accs, width, label='Systolic', alpha=0.8)
    ax9.bar(x + width/2, dias_accs, width, label='Diastolic', alpha=0.8)
    
    ax9.set_xlabel('Error Threshold')
    ax9.set_ylabel('Accuracy (%)')
    ax9.set_title('Clinical Accuracy')
    ax9.set_xticks(x)
    ax9.set_xticklabels(thresholds)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} Pipeline Evaluation | {len(predictions)} samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'pipeline_evaluation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Evaluation plots saved: {plot_path}")


def save_evaluation_results(metrics: Dict, output_dir: Path):
    """Save evaluation results to files"""
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics to text file
    metrics_path = output_dir / 'pipeline_evaluation_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("PIPELINE VAE+BiLSTM EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        for category in ['systolic', 'diastolic', 'waveform']:
            if category in metrics:
                f.write(f"{category.upper()} METRICS:\n")
                for metric, value in metrics[category].items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        
        if 'clinical' in metrics:
            f.write("CLINICAL ACCURACY:\n")
            for metric, value in metrics['clinical'].items():
                f.write(f"  {metric}: {value:.2f}%\n")
    
    # Save to JSON
    json_path = output_dir / 'pipeline_evaluation_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"📄 Evaluation results saved: {metrics_path}")


def main():
    """Main pipeline orchestrator"""
    args = parse_arguments()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=level)
    
    try:
        logger.info("🚀 Starting Continuous VAE + BiLSTM Training Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Load configuration and setup environment
        config = load_pipeline_config(args.config)
        path_manager = setup_environment(args)
        device = setup_device(args.device)
        
        # Step 2: Get available subjects and create data split
        pipeline_config = config['pipeline_config']
        
        # Override with command line arguments
        max_subjects = args.max_subjects or pipeline_config.get('max_subjects')
        data_split_seed = args.data_split_seed or pipeline_config.get('data_split_seed', 42)
        data_split_ratio = pipeline_config.get('data_split_ratio', 0.9)
        
        # Get available subjects
        data_root = args.data_root or config['environment']['data_root']
        subjects = get_available_subjects(data_root)
        
        if max_subjects:
            subjects = subjects[:max_subjects]
            logger.info(f"🎯 Limited to {max_subjects} subjects")
        
        # Create deterministic data split
        train_subjects, val_subjects = create_data_split(subjects, data_split_ratio, data_split_seed)
        
        # Step 3: Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"pipeline_vae_bilstm_{len(subjects)}subj_{args.mask_type}_{timestamp}"
        experiment_dir = create_experiment_directory(path_manager, experiment_name)
        
        # Save data split info for reproducibility
        save_data_split_info(train_subjects, val_subjects, experiment_dir, data_split_seed)
        
        # Save pipeline configuration
        config_save_path = experiment_dir / 'config' / 'pipeline_config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Save command line arguments
        args_save_path = experiment_dir / 'config' / 'command_line_args.json'
        with open(args_save_path, 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)
        
        logger.info(f"✅ Configuration saved to: {experiment_dir / 'config'}")
        
        # Initialize pipeline results
        pipeline_results = {
            'experiment_name': experiment_name,
            'experiment_dir': str(experiment_dir),
            'config': config,
            'args': vars(args),
            'subjects': {
                'total': len(subjects),
                'train': train_subjects,
                'val': val_subjects
            },
            'data_split_info': {
                'seed': data_split_seed,
                'ratio': data_split_ratio
            },
            'phases_completed': [],
            'timestamps': {
                'pipeline_start': datetime.now().isoformat()
            }
        }
        
        # Step 4: Execute pipeline phases based on mode
        vae_checkpoint_path = None
        bilstm_checkpoint_path = None
        
        if args.pipeline_mode in ["vae_only", "full_pipeline"]:
            logger.info("\n" + "=" * 80)
            logger.info("🏗️  PHASE 1: VAE TRAINING")
            logger.info("=" * 80)
            
            vae_checkpoint_path = train_vae_phase(
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                config=config,
                args=args,
                experiment_dir=experiment_dir,
                device=device
            )
            
            pipeline_results['phases_completed'].append('vae_training')
            pipeline_results['timestamps']['vae_completed'] = datetime.now().isoformat()
            pipeline_results['vae_checkpoint'] = str(vae_checkpoint_path)
            
            # Clear GPU memory after VAE training
            clear_gpu_memory()
            
        elif args.pipeline_mode == "bilstm_only":
            if not args.vae_checkpoint:
                raise ValueError("--vae-checkpoint required for bilstm_only mode")
            vae_checkpoint_path = args.vae_checkpoint
            logger.info(f"Using pre-trained VAE checkpoint: {vae_checkpoint_path}")
        
        if args.pipeline_mode in ["bilstm_only", "full_pipeline"]:
            logger.info("\n" + "=" * 80)
            logger.info("🧠 PHASE 2: BiLSTM TRAINING")
            logger.info("=" * 80)
            
            bilstm_checkpoint_path = train_bilstm_phase(
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                vae_checkpoint_path=vae_checkpoint_path,
                config=config,
                args=args,
                experiment_dir=experiment_dir,
                device=device
            )
            
            pipeline_results['phases_completed'].append('bilstm_training')
            pipeline_results['timestamps']['bilstm_completed'] = datetime.now().isoformat()
            pipeline_results['bilstm_checkpoint'] = str(bilstm_checkpoint_path)
            
            # Clear GPU memory after BiLSTM training
            clear_gpu_memory()
        
        # Step 5: Final pipeline evaluation
        if args.pipeline_mode == "full_pipeline" and vae_checkpoint_path and bilstm_checkpoint_path:
            logger.info("\n" + "=" * 80)
            logger.info("📊 PHASE 3: PIPELINE EVALUATION")
            logger.info("=" * 80)
            
            evaluation_results = evaluate_pipeline(
                val_subjects=val_subjects,
                vae_checkpoint_path=vae_checkpoint_path,
                bilstm_checkpoint_path=bilstm_checkpoint_path,
                config=config,
                args=args,
                experiment_dir=experiment_dir,
                device=device
            )
            
            pipeline_results['phases_completed'].append('pipeline_evaluation')
            pipeline_results['timestamps']['evaluation_completed'] = datetime.now().isoformat()
            pipeline_results['evaluation_results'] = evaluation_results
        
        # Step 6: Save complete pipeline results
        pipeline_results['timestamps']['pipeline_completed'] = datetime.now().isoformat()
        
        results_file = experiment_dir / 'pipeline_results.json'
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        logger.info("=" * 80)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"📁 Results saved to: {experiment_dir}")
        logger.info(f"📊 Phases completed: {pipeline_results['phases_completed']}")
        logger.info(f"🎯 Total subjects: {len(subjects)}")
        logger.info(f"📈 Training subjects: {len(train_subjects)}")
        logger.info(f"📉 Validation subjects: {len(val_subjects)}")
        logger.info(f"🔢 Data split seed: {data_split_seed}")
        logger.info(f"⏱️  Pipeline duration: {pipeline_results['timestamps']['pipeline_completed']}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 
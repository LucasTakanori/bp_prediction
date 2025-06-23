#!/usr/bin/env python3
"""
Multi-Subject VAE Training Script
Trains VAE model using data from all available subjects for better generalization
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import yaml
from tqdm import tqdm
import wandb
from typing import List, Tuple, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utilities
from utils.data_utils import PviDataset, DataPathManager
from train.tuned_vae import VAE  # Use the working VAE implementation


class MultiSubjectVAEDataset(Dataset):
    """Dataset that combines multiple subjects for VAE training"""
    
    def __init__(self, 
                 data_root: str,
                 subjects: List[str],
                 session: str = "baseline",
                 max_samples_per_subject: Optional[int] = None):
        
        self.data_root = data_root
        self.subjects = subjects
        self.session = session
        self.max_samples_per_subject = max_samples_per_subject
        
        print(f"🔄 Loading multi-subject dataset from {len(subjects)} subjects...")
        
        # Load all subject datasets
        self.datasets = []
        self.subject_info = []
        
        for subject in subjects:
            try:
                path_manager = DataPathManager(
                    subject=subject,
                    session=session,
                    root=data_root
                )
                
                if not path_manager._h5_path.exists():
                    print(f"⚠️  Skipping {subject}: file not found")
                    continue
                
                dataset = PviDataset(
                    str(path_manager._h5_path),
                    device=torch.device('cpu')  # Load on CPU first, move to GPU later
                )
                
                # Limit samples per subject if specified
                if max_samples_per_subject and len(dataset) > max_samples_per_subject:
                    indices = np.random.choice(len(dataset), max_samples_per_subject, replace=False)
                    dataset = torch.utils.data.Subset(dataset, indices)
                
                self.datasets.append(dataset)
                self.subject_info.append({
                    'subject': subject,
                    'samples': len(dataset),
                    'file_path': str(path_manager._h5_path)
                })
                
                print(f"✅ Loaded {subject}: {len(dataset)} samples")
                
            except Exception as e:
                print(f"❌ Failed to load {subject}: {e}")
                continue
        
        if not self.datasets:
            raise ValueError("No valid datasets were loaded!")
        
        # Combine all datasets
        self.combined_dataset = ConcatDataset(self.datasets)
        
        total_samples = len(self.combined_dataset)
        print(f"🎯 Multi-subject dataset ready:")
        print(f"   📊 Total subjects: {len(self.datasets)}")
        print(f"   📈 Total samples: {total_samples}")
        print(f"   📝 Average per subject: {total_samples / len(self.datasets):.1f}")
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        sample = self.combined_dataset[idx]
        
        # Extract PVI image data (32x32x500 -> 32x32)
        pvi_data = sample['pviHP']['img']  # Shape: (32, 32, 500)
        
        # Take middle frame for VAE training
        frame_idx = pvi_data.shape[2] // 2
        pvi_frame = pvi_data[:, :, frame_idx]  # Shape: (32, 32)
        
        # Normalize to [-1, 1] range to match VAE tanh output
        pvi_frame = (pvi_frame - pvi_frame.min()) / (pvi_frame.max() - pvi_frame.min() + 1e-8)  # First to [0, 1]
        pvi_frame = 2.0 * pvi_frame - 1.0  # Then to [-1, 1]
        
        # Add channel dimension: (32, 32) -> (1, 32, 32)
        pvi_frame = pvi_frame.unsqueeze(0)
        
        return {
            'image': pvi_frame.float(),
            'subject_idx': idx % len(self.datasets)  # Which subject this sample came from
        }


class MultiSubjectVAETrainer:
    """Trainer for multi-subject VAE"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: torch.device,
                 experiment_dir: Path):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        
        # Training parameters
        training_config = config['training_config']
        self.num_epochs = training_config['num_epochs']
        self.learning_rate = float(training_config['learning_rate'])  # Ensure it's a float
        self.beta_min = float(training_config.get('beta_min', 0.01))
        self.beta_max = float(training_config.get('beta_max', 0.5))
        self.beta_warmup_epochs = int(training_config.get('beta_warmup_epochs', 20))
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=float(training_config.get('weight_decay', 1e-5))
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6
        )
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Checkpointing
        self.checkpoint_dir = experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Multi-subject VAE trainer initialized")
        print(f"   📊 Training samples: {len(train_loader.dataset)}")
        print(f"   📈 Validation samples: {len(val_loader.dataset)}")
        print(f"   🎯 Target epochs: {self.num_epochs}")
        print(f"   ⚙️  Learning rate: {self.learning_rate}")
        print(f"   🔧 Beta range: {self.beta_min} → {self.beta_max}")
    
    def compute_loss(self, batch, beta):
        """Compute VAE loss (reconstruction + KL divergence)"""
        images = batch['image'].to(self.device)
        
        # Forward pass
        recon_images, mu, logvar = self.model(images)
        
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_images, images, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
        
        # Total loss with beta weighting
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': beta
        }
    
    def get_beta(self, epoch):
        """Beta scheduling for KL divergence weight"""
        if epoch < self.beta_warmup_epochs:
            return self.beta_min + (self.beta_max - self.beta_min) * (epoch / self.beta_warmup_epochs)
        else:
            return self.beta_max
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        beta = self.get_beta(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Compute loss
            loss_dict = self.compute_loss(batch, beta)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update running averages
            total_loss += loss.item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Recon': f"{loss_dict['recon_loss'].item():.4f}",
                'KL': f"{loss_dict['kl_loss'].item():.4f}",
                'Beta': f"{beta:.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_recon_loss = total_recon_loss / len(self.train_loader)
        avg_kl_loss = total_kl_loss / len(self.train_loader)
        
        return {
            'total_loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
            'beta': beta
        }
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        beta = self.get_beta(epoch)
        
        with torch.no_grad():
            for batch in self.val_loader:
                loss_dict = self.compute_loss(batch, beta)
                
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['recon_loss'].item()
                total_kl_loss += loss_dict['kl_loss'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_recon_loss = total_recon_loss / len(self.val_loader)
        avg_kl_loss = total_kl_loss / len(self.val_loader)
        
        return {
            'total_loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
            'beta': beta
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"vae_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "vae_best.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved: {best_path}")
        
        # Save latest model
        latest_path = self.checkpoint_dir / "vae_latest.pt"
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting multi-subject VAE training...")
        
        for epoch in range(self.num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics)
            
            # Validate epoch
            val_metrics = self.validate_epoch(epoch)
            self.val_losses.append(val_metrics)
            
            # Update learning rate
            self.scheduler.step(val_metrics['total_loss'])
            
            # Print metrics
            print(f"📊 Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Recon: {train_metrics['recon_loss']:.4f}, "
                  f"KL: {train_metrics['kl_loss']:.4f}")
            print(f"📈 Val   - Loss: {val_metrics['total_loss']:.4f}, "
                  f"Recon: {val_metrics['recon_loss']:.4f}, "
                  f"KL: {val_metrics['kl_loss']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
                print(f"🎯 New best validation loss: {self.best_val_loss:.4f}")
            
            # Save every 5 epochs and when best
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Log to wandb if enabled
            if self.config['logging_config'].get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'train_recon_loss': train_metrics['recon_loss'],
                    'train_kl_loss': train_metrics['kl_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'val_recon_loss': val_metrics['recon_loss'],
                    'val_kl_loss': val_metrics['kl_loss'],
                    'beta': train_metrics['beta'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        print("\n🎉 Training completed!")
        print(f"📊 Best validation loss: {self.best_val_loss:.4f}")
        print(f"💾 Model saved to: {self.checkpoint_dir}")
        
        return self.best_val_loss


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_available_subjects(data_root: str) -> List[str]:
    """Get list of available subjects from data directory"""
    data_path = Path(data_root)
    subjects = []
    
    for file_path in data_path.glob("subject*_baseline_masked.h5"):
        subject = file_path.stem.split('_')[0]
        subjects.append(subject)
    
    return sorted(subjects)


def create_experiment_dir(config: Dict) -> Path:
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"multisubject_vae_{timestamp}"
    
    output_dir = Path(config['environment']['experiments_root'])
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    
    return experiment_dir


def main():
    parser = argparse.ArgumentParser(description='Multi-Subject VAE Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--subjects', nargs='+', 
                       help='Specific subjects to train on (default: all available)')
    parser.add_argument('--max-subjects', type=int, default=None,
                       help='Maximum number of subjects to use')
    parser.add_argument('--max-samples-per-subject', type=int, default=None,
                       help='Maximum samples per subject')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Multi-Subject VAE Training")
    print(f"📱 Device: {device}")
    print(f"⚙️  Config: {args.config}")
    
    # Get available subjects
    data_root = config['data_config']['root_path']
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = get_available_subjects(data_root)
    
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    
    print(f"📊 Training subjects: {subjects}")
    print(f"📈 Total subjects: {len(subjects)}")
    
    # Initialize wandb if requested
    if args.wandb:
        config['logging_config']['use_wandb'] = True
        wandb.init(
            project=config['logging_config'].get('wandb_project', 'multisubject-vae'),
            name=f"multisubject_vae_{len(subjects)}subjects",
            config=config
        )
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(config)
    print(f"📁 Experiment directory: {experiment_dir}")
    
    # Save config
    with open(experiment_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create dataset
    dataset = MultiSubjectVAEDataset(
        data_root=data_root,
        subjects=subjects,
        session=config['data_config']['session'],
        max_samples_per_subject=args.max_samples_per_subject
    )
    
    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    batch_size = config['training_config']['batch_size']
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model_config = config['model_config']
    model = VAE(latent_dim=model_config['latent_dim'])
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create trainer
    trainer = MultiSubjectVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_dir=experiment_dir
    )
    
    # Train model
    best_loss = trainer.train()
    
    print(f"\n✅ Training completed!")
    print(f"📊 Best validation loss: {best_loss:.4f}")
    print(f"💾 Model saved to: {experiment_dir}")
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main() 
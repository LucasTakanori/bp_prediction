#!/usr/bin/env python3
"""
Multi-Subject BiLSTM + VAE Training Script - CREATING NEW FILE
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
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utilities
from utils.data_utils import PviDataset, DataPathManager
from train.tuned_vae import VAE  # Use the working VAE implementation


class MultiSubjectBPDataset(Dataset):
    """Dataset that combines multiple subjects for VAE+BiLSTM BP prediction"""
    
    def __init__(self, 
                 data_root: str,
                 subjects: List[str],
                 session: str = "baseline",
                 sequence_length: int = 10,
                 pattern_offsets: List[int] = None,
                 max_samples_per_subject: Optional[int] = None,
                 sequence_step_size: int = 5,
                 cache_dir: Optional[str] = None):
        
        self.data_root = data_root
        self.subjects = subjects
        self.session = session
        self.sequence_length = sequence_length
        self.pattern_offsets = pattern_offsets or [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        self.max_samples_per_subject = max_samples_per_subject
        self.sequence_step_size = sequence_step_size
        
        print(f"🔄 Loading multi-subject BP dataset from {len(subjects)} subjects...")
        print(f"   📊 Sequence length: {sequence_length}")
        print(f"   📈 Pattern offsets: {pattern_offsets}")
        print(f"   🔀 Step size: {sequence_step_size}")
        
        # Load all subject datasets and create sequences
        self.sequences = []
        self.targets = []
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
                    cache_dir=cache_dir
                )
                
                # Create sequences for this subject
                subject_sequences, subject_targets = self._create_sequences_from_dataset(
                    dataset, subject, max_samples_per_subject
                )
                
                self.sequences.extend(subject_sequences)
                self.targets.extend(subject_targets)
                
                self.subject_info.append({
                    'subject': subject,
                    'samples': len(subject_sequences),
                    'file_path': str(path_manager._h5_path)
                })
                
                print(f"✅ Loaded {subject}: {len(subject_sequences)} sequences")
                
            except Exception as e:
                print(f"❌ Failed to load {subject}: {e}")
                continue
        
        if not self.sequences:
            raise ValueError("No valid sequences were created!")
        
        total_sequences = len(self.sequences)
        print(f"🎯 Multi-subject BP dataset ready:")
        print(f"   📊 Total subjects: {len(self.subject_info)}")
        print(f"   📈 Total sequences: {total_sequences}")
        print(f"   📝 Average per subject: {total_sequences / len(self.subject_info):.1f}")
        
        # Convert to tensors
        self.sequences = torch.stack(self.sequences)
        self.targets = torch.stack(self.targets)
        
        print(f"   🔢 Sequence shape: {self.sequences.shape}")
        print(f"   🎯 Target shape: {self.targets.shape}")
    
    def _create_sequences_from_dataset(self, dataset, subject, max_samples):
        """Create sequences from a single subject's dataset"""
        sequences = []
        targets = []
        
        available_samples = len(dataset)
        if max_samples:
            available_samples = min(available_samples, max_samples)
        
        # Generate sequences with step size
        for i in range(0, available_samples - self.sequence_length, self.sequence_step_size):
            try:
                # Create sequence of PVI frames
                sequence_frames = []
                sequence_bp = []
                
                for j in range(self.sequence_length):
                    sample_idx = i + j
                    if sample_idx >= len(dataset):
                        break
                    
                    sample = dataset[sample_idx]
                    
                    # Extract PVI frame (middle frame)
                    pvi_data = sample['pviHP']['img']  # Shape: (32, 32, 500)
                    frame_idx = pvi_data.shape[2] // 2
                    pvi_frame = pvi_data[:, :, frame_idx]  # Shape: (32, 32)
                    
                    # Normalize frame
                    pvi_frame = (pvi_frame - pvi_frame.min()) / (pvi_frame.max() - pvi_frame.min() + 1e-8)
                    
                    # Add channel dimension: (32, 32) -> (1, 32, 32)
                    pvi_frame = pvi_frame.unsqueeze(0)
                    sequence_frames.append(pvi_frame)
                    
                    # Extract BP signal (target)
                    bp_signal = sample['bp']['signal']  # Shape: (50,)
                    sequence_bp.append(bp_signal)
                
                if len(sequence_frames) == self.sequence_length:
                    # Stack frames: (sequence_length, 1, 32, 32)
                    sequence_tensor = torch.stack(sequence_frames)
                    
                    # Target is the BP signal from the center frame
                    target_idx = self.sequence_length // 2
                    target_bp = sequence_bp[target_idx]
                    
                    sequences.append(sequence_tensor)
                    targets.append(target_bp)
                
            except Exception as e:
                print(f"⚠️  Error creating sequence {i} for {subject}: {e}")
                continue
        
        return sequences, targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': self.sequences[idx].float(),  # (sequence_length, 1, 32, 32)
            'targets': self.targets[idx].float(),      # (50,) BP signal
            'subject_idx': idx % len(self.subject_info)
        }


class VAEBiLSTMModel(nn.Module):
    """VAE + BiLSTM model for BP prediction"""
    
    def __init__(self, 
                 vae_model: nn.Module,
                 latent_dim: int = 128,
                 lstm_hidden_dim: int = 256,
                 lstm_num_layers: int = 3,
                 lstm_dropout: float = 0.3,
                 output_dim: int = 50,
                 freeze_vae: bool = True,
                 use_attention: bool = True,
                 attention_heads: int = 8):
        super().__init__()
        
        self.vae_model = vae_model
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.output_dim = output_dim
        self.freeze_vae = freeze_vae
        self.use_attention = use_attention
        
        # Freeze VAE if requested
        if freeze_vae:
            for param in self.vae_model.parameters():
                param.requires_grad = False
            self.vae_model.eval()
        
        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_hidden_dim * 2,  # Bidirectional
                num_heads=attention_heads,
                dropout=0.2,
                batch_first=True
            )
        
        # Output layers
        attention_output_dim = lstm_hidden_dim * 2 if use_attention else lstm_hidden_dim * 2
        
        self.output_layers = nn.Sequential(
            nn.Linear(attention_output_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden_dim // 2, output_dim)
        )
        
        # Auxiliary heads for systolic/diastolic prediction
        self.systolic_head = nn.Sequential(
            nn.Linear(attention_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self.diastolic_head = nn.Sequential(
            nn.Linear(attention_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        print(f"🏗️  VAE+BiLSTM Model Architecture:")
        print(f"   🔧 VAE latent dim: {latent_dim}")
        print(f"   🔄 LSTM hidden dim: {lstm_hidden_dim}")
        print(f"   📚 LSTM layers: {lstm_num_layers}")
        print(f"   🎯 Output dim: {output_dim}")
        print(f"   ❄️  VAE frozen: {freeze_vae}")
        print(f"   👁️  Attention: {use_attention}")
    
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, sequence_length, 1, 32, 32)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for VAE processing: (batch_size * seq_len, 1, 32, 32)
        x_reshaped = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features using VAE encoder
        with torch.set_grad_enabled(not self.freeze_vae):
            latent_features = self.vae_model.encode(x_reshaped)  # (batch_size * seq_len, latent_dim)
        
        # Reshape back to sequence: (batch_size, seq_len, latent_dim)
        latent_features = latent_features.view(batch_size, seq_len, self.latent_dim)
        
        # BiLSTM processing
        lstm_out, (h_n, c_n) = self.lstm(latent_features)  # (batch_size, seq_len, lstm_hidden_dim * 2)
        
        # Attention mechanism
        if self.use_attention:
            attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
            # Use the center frame output
            center_idx = seq_len // 2
            final_features = attended_out[:, center_idx, :]  # (batch_size, lstm_hidden_dim * 2)
        else:
            # Use the center frame output
            center_idx = seq_len // 2
            final_features = lstm_out[:, center_idx, :]  # (batch_size, lstm_hidden_dim * 2)
        
        # Generate outputs
        waveform_output = self.output_layers(final_features)  # (batch_size, output_dim)
        systolic_output = self.systolic_head(final_features)  # (batch_size, 1)
        diastolic_output = self.diastolic_head(final_features)  # (batch_size, 1)
        
        return {
            'waveform': waveform_output,
            'systolic': systolic_output,
            'diastolic': diastolic_output,
            'features': final_features
        }


class ImprovedBPLoss(nn.Module):
    """Improved loss function for BP prediction"""
    
    def __init__(self, 
                 waveform_weight: float = 0.4,
                 systolic_weight: float = 0.3,
                 diastolic_weight: float = 0.3,
                 huber_delta: float = 1.0):
        super().__init__()
        self.waveform_weight = waveform_weight
        self.systolic_weight = systolic_weight
        self.diastolic_weight = diastolic_weight
        self.huber_delta = huber_delta
        
        self.huber_loss = nn.SmoothL1Loss(reduction='mean', beta=huber_delta)
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def extract_bp_values(self, waveform):
        """Extract systolic and diastolic values from waveform"""
        batch_size = waveform.shape[0]
        device = waveform.device
        
        systolic_values = torch.zeros(batch_size, device=device)
        diastolic_values = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            signal = waveform[i]
            
            # Systolic: maximum value
            sys_val = torch.max(signal)
            
            # Diastolic: minimum in second half
            second_half_start = len(signal) // 2
            dias_val = torch.min(signal[second_half_start:])
            
            # Ensure physiological constraint
            if dias_val >= sys_val:
                dias_val = sys_val - 10.0
            
            systolic_values[i] = sys_val
            diastolic_values[i] = dias_val
        
        return systolic_values, diastolic_values
    
    def forward(self, predictions, targets):
        """
        Compute composite loss
        """
        pred_waveform = predictions['waveform']
        pred_systolic = predictions['systolic'].squeeze()
        pred_diastolic = predictions['diastolic'].squeeze()
        
        # Waveform loss
        waveform_loss = self.huber_loss(pred_waveform, targets)
        
        # Extract true BP values
        true_systolic, true_diastolic = self.extract_bp_values(targets)
        
        # BP value losses
        systolic_loss = self.mse_loss(pred_systolic, true_systolic)
        diastolic_loss = self.mse_loss(pred_diastolic, true_diastolic)
        
        # Composite loss
        total_loss = (
            self.waveform_weight * waveform_loss +
            self.systolic_weight * systolic_loss +
            self.diastolic_weight * diastolic_loss
        )
        
        return {
            'total_loss': total_loss,
            'waveform_loss': waveform_loss,
            'systolic_loss': systolic_loss,
            'diastolic_loss': diastolic_loss
        }


class MultiSubjectBPTrainer:
    """Trainer for multi-subject VAE+BiLSTM BP prediction"""
    
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
        self.learning_rate = training_config['learning_rate']
        
        # Loss function
        loss_config = config['loss_config']
        self.criterion = ImprovedBPLoss(
            waveform_weight=loss_config.get('waveform_weight', 0.4),
            systolic_weight=loss_config.get('systolic_weight', 0.3),
            diastolic_weight=loss_config.get('diastolic_weight', 0.3),
            huber_delta=loss_config.get('huber_delta', 1.0)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=training_config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=8, min_lr=1e-7
        )
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        # Checkpointing
        self.checkpoint_dir = experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Multi-subject BP trainer initialized")
        print(f"   📊 Training samples: {len(train_loader.dataset)}")
        print(f"   📈 Validation samples: {len(val_loader.dataset)}")
        print(f"   🎯 Target epochs: {self.num_epochs}")
        print(f"   ⚙️  Learning rate: {self.learning_rate}")
    
    def compute_metrics(self, predictions, targets):
        """Compute evaluation metrics"""
        pred_waveform = predictions['waveform'].cpu().numpy()
        pred_systolic = predictions['systolic'].squeeze().cpu().numpy()
        pred_diastolic = predictions['diastolic'].squeeze().cpu().numpy()
        
        target_waveform = targets.cpu().numpy()
        
        # Extract true BP values
        true_systolic = []
        true_diastolic = []
        
        for i in range(len(target_waveform)):
            signal = target_waveform[i]
            sys_val = np.max(signal)
            dias_val = np.min(signal[len(signal)//2:])
            if dias_val >= sys_val:
                dias_val = sys_val - 10.0
            true_systolic.append(sys_val)
            true_diastolic.append(dias_val)
        
        true_systolic = np.array(true_systolic)
        true_diastolic = np.array(true_diastolic)
        
        # Compute metrics
        metrics = {
            'waveform_r2': r2_score(target_waveform.flatten(), pred_waveform.flatten()),
            'waveform_mae': mean_absolute_error(target_waveform.flatten(), pred_waveform.flatten()),
            'systolic_r2': r2_score(true_systolic, pred_systolic),
            'systolic_mae': mean_absolute_error(true_systolic, pred_systolic),
            'diastolic_r2': r2_score(true_diastolic, pred_diastolic),
            'diastolic_mae': mean_absolute_error(true_diastolic, pred_diastolic),
        }
        
        return metrics
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_waveform_loss = 0
        total_systolic_loss = 0
        total_diastolic_loss = 0
        
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch in pbar:
            sequences = batch['sequences'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(sequences)
            
            # Compute loss
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update running averages
            total_loss += loss.item()
            total_waveform_loss += loss_dict['waveform_loss'].item()
            total_systolic_loss += loss_dict['systolic_loss'].item()
            total_diastolic_loss += loss_dict['diastolic_loss'].item()
            
            # Collect for metrics
            all_predictions.append(predictions)
            all_targets.append(targets)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Wave': f"{loss_dict['waveform_loss'].item():.4f}",
                'Sys': f"{loss_dict['systolic_loss'].item():.4f}",
                'Dias': f"{loss_dict['diastolic_loss'].item():.4f}"
            })
        
        # Compute epoch metrics
        all_pred_concat = {
            'waveform': torch.cat([p['waveform'] for p in all_predictions]),
            'systolic': torch.cat([p['systolic'] for p in all_predictions]),
            'diastolic': torch.cat([p['diastolic'] for p in all_predictions])
        }
        all_targets_concat = torch.cat(all_targets)
        
        metrics = self.compute_metrics(all_pred_concat, all_targets_concat)
        
        return {
            'total_loss': total_loss / len(self.train_loader),
            'waveform_loss': total_waveform_loss / len(self.train_loader),
            'systolic_loss': total_systolic_loss / len(self.train_loader),
            'diastolic_loss': total_diastolic_loss / len(self.train_loader),
            'metrics': metrics
        }
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_waveform_loss = 0
        total_systolic_loss = 0
        total_diastolic_loss = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                sequences = batch['sequences'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Compute loss
                loss_dict = self.criterion(predictions, targets)
                
                total_loss += loss_dict['total_loss'].item()
                total_waveform_loss += loss_dict['waveform_loss'].item()
                total_systolic_loss += loss_dict['systolic_loss'].item()
                total_diastolic_loss += loss_dict['diastolic_loss'].item()
                
                # Collect for metrics
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Compute epoch metrics
        all_pred_concat = {
            'waveform': torch.cat([p['waveform'] for p in all_predictions]),
            'systolic': torch.cat([p['systolic'] for p in all_predictions]),
            'diastolic': torch.cat([p['diastolic'] for p in all_predictions])
        }
        all_targets_concat = torch.cat(all_targets)
        
        metrics = self.compute_metrics(all_pred_concat, all_targets_concat)
        
        return {
            'total_loss': total_loss / len(self.val_loader),
            'waveform_loss': total_waveform_loss / len(self.val_loader),
            'systolic_loss': total_systolic_loss / len(self.val_loader),
            'diastolic_loss': total_diastolic_loss / len(self.val_loader),
            'metrics': metrics
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
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"bp_model_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved: {best_path}")
        
        # Save latest model
        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting multi-subject VAE+BiLSTM training...")
        
        for epoch in range(self.num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{self.num_epochs}")
            print("-" * 60)
            
            # Train epoch
            train_results = self.train_epoch(epoch)
            self.train_losses.append(train_results)
            self.train_metrics.append(train_results['metrics'])
            
            # Validate epoch
            val_results = self.validate_epoch(epoch)
            self.val_losses.append(val_results)
            self.val_metrics.append(val_results['metrics'])
            
            # Update learning rate
            self.scheduler.step(val_results['total_loss'])
            
            # Print metrics
            print(f"📊 Train - Loss: {train_results['total_loss']:.4f}, "
                  f"Sys R²: {train_results['metrics']['systolic_r2']:.4f}, "
                  f"Dias R²: {train_results['metrics']['diastolic_r2']:.4f}")
            print(f"📈 Val   - Loss: {val_results['total_loss']:.4f}, "
                  f"Sys R²: {val_results['metrics']['systolic_r2']:.4f}, "
                  f"Dias R²: {val_results['metrics']['diastolic_r2']:.4f}")
            
            # Save checkpoint
            is_best = val_results['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_results['total_loss']
                print(f"🎯 New best validation loss: {self.best_val_loss:.4f}")
                print(f"🏆 Best Systolic R²: {val_results['metrics']['systolic_r2']:.4f}")
                print(f"🏆 Best Diastolic R²: {val_results['metrics']['diastolic_r2']:.4f}")
            
            # Save every 5 epochs and when best
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Log to wandb if enabled
            if self.config['logging_config'].get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_results['total_loss'],
                    'val_loss': val_results['total_loss'],
                    'train_systolic_r2': train_results['metrics']['systolic_r2'],
                    'val_systolic_r2': val_results['metrics']['systolic_r2'],
                    'train_diastolic_r2': train_results['metrics']['diastolic_r2'],
                    'val_diastolic_r2': val_results['metrics']['diastolic_r2'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        print("\n🎉 Training completed!")
        print(f"📊 Best validation loss: {self.best_val_loss:.4f}")
        print(f"💾 Model saved to: {self.checkpoint_dir}")
        
        # Print final metrics
        if self.val_metrics:
            final_metrics = self.val_metrics[-1]
            print(f"\n🏆 FINAL PERFORMANCE:")
            print(f"   📈 Systolic R²: {final_metrics['systolic_r2']:.4f}")
            print(f"   📈 Diastolic R²: {final_metrics['diastolic_r2']:.4f}")
            print(f"   📊 Systolic MAE: {final_metrics['systolic_mae']:.2f} mmHg")
            print(f"   📊 Diastolic MAE: {final_metrics['diastolic_mae']:.2f} mmHg")
        
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


def load_pretrained_vae(vae_checkpoint_path: str, latent_dim: int = 128, device: torch.device = None):
    """Load pre-trained VAE model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔄 Loading pre-trained VAE from: {vae_checkpoint_path}")
    
    # Create VAE model
    vae_model = VAE(latent_dim=latent_dim)
    
    # Load checkpoint
    checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        vae_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        vae_model.load_state_dict(checkpoint)
    
    vae_model = vae_model.to(device)
    vae_model.eval()
    
    print(f"✅ VAE loaded successfully")
    return vae_model


def create_experiment_dir(config: Dict) -> Path:
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"multisubject_bp_{timestamp}"
    
    output_dir = Path(config['environment']['experiments_root'])
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    
    return experiment_dir


def main():
    parser = argparse.ArgumentParser(description='Multi-Subject VAE+BiLSTM BP Prediction Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                       help='Path to pre-trained VAE checkpoint')
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
    
    print(f"🚀 Multi-Subject VAE+BiLSTM BP Prediction Training")
    print(f"📱 Device: {device}")
    print(f"⚙️  Config: {args.config}")
    print(f"🧠 VAE checkpoint: {args.vae_checkpoint}")
    
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
            project=config['logging_config'].get('wandb_project', 'multisubject-bp-prediction'),
            name=f"multisubject_bp_{len(subjects)}subjects",
            config=config
        )
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(config)
    print(f"📁 Experiment directory: {experiment_dir}")
    
    # Save config
    with open(experiment_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Load pre-trained VAE
    model_config = config['model_config']
    vae_model = load_pretrained_vae(
        args.vae_checkpoint, 
        latent_dim=model_config['vae_config']['latent_dim'],
        device=device
    )
    
    # Create dataset
    dataset = MultiSubjectBPDataset(
        data_root=data_root,
        subjects=subjects,
        session=config['data_config']['session'],
        sequence_length=config['data_config']['sequence_length'],
        pattern_offsets=config['data_config']['pattern_offsets'],
        max_samples_per_subject=args.max_samples_per_subject,
        sequence_step_size=config['data_config'].get('sequence_step_size', 5),
        cache_dir=config['environment'].get('cache_dir')
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
    
    # Create VAE+BiLSTM model
    bilstm_config = model_config['bilstm_config']
    model = VAEBiLSTMModel(
        vae_model=vae_model,
        latent_dim=model_config['vae_config']['latent_dim'],
        lstm_hidden_dim=bilstm_config['hidden_dim'],
        lstm_num_layers=bilstm_config['num_layers'],
        lstm_dropout=bilstm_config['dropout_rate'],
        output_dim=bilstm_config['output_dim'],
        freeze_vae=model_config['vae_config']['freeze_vae'],
        use_attention=bilstm_config.get('use_attention', True),
        attention_heads=bilstm_config.get('attention_heads', 8)
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create trainer
    trainer = MultiSubjectBPTrainer(
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
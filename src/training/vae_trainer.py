"""
VAE Trainer - Based on working tuned_vae.py implementation
Implements VAE training with proper logging, WandB integration, and robust error handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import wandb
from torch.cuda.amp import GradScaler, autocast

from .base_trainer import BaseTrainer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VAETrainer(BaseTrainer):
    """VAE trainer based on tuned_vae.py working implementation"""
    
    def __init__(self, model, train_loader, val_loader, config, device, experiment_dir):
        self.logger = get_logger(self.__class__.__name__)
        super().__init__(model, train_loader, val_loader, config, device, experiment_dir)
        
        self.use_mixed_precision = config.training.use_mixed_precision
        self._determine_frame_indices()
        
        # VAE-specific parameters from tuned_vae.py
        self.beta_min = config.training.beta_min
        self.beta_max = config.training.beta_max
        self.beta_warmup_epochs = config.training.beta_warmup_epochs
        
        # Initialize best model tracking
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None
        
        # Track losses for visualization
        self.train_losses = []
        self.val_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.learning_rates = []
        
        # Setup training components
        self.setup_training_components()
        
        # Initialize mixed precision training if enabled
        self.scaler = GradScaler() if self.use_mixed_precision and torch.cuda.is_available() else None
        
        # Setup wandb if enabled
        self.use_wandb = getattr(config.training, 'use_wandb', True)
        if self.use_wandb:
            # Convert config to JSON-serializable format
            wandb_config = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__'):
                    # Handle nested config objects
                    nested_config = {}
                    for nested_key, nested_value in value.__dict__.items():
                        # Convert Path objects to strings
                        if hasattr(nested_value, '__fspath__'):
                            nested_config[nested_key] = str(nested_value)
                        else:
                            nested_config[nested_key] = nested_value
                    wandb_config[key] = nested_config
                elif hasattr(value, '__fspath__'):
                    # Convert Path objects to strings
                    wandb_config[key] = str(value)
                else:
                    wandb_config[key] = value
            
            wandb.init(
                project=getattr(config.training, 'project_name', 'bp_prediction'),
                config=wandb_config,
                mode=getattr(config.training, 'wandb_mode', 'offline')
            )
        
        self.logger.info(f"VAE Trainer initialized with:")
        self.logger.info(f"  Frame indices: {self.frame_indices}")
        self.logger.info(f"  Beta range: {self.beta_min} -> {self.beta_max}")
        self.logger.info(f"  Beta warmup epochs: {self.beta_warmup_epochs}")
        self.logger.info(f"  Mixed precision: {self.use_mixed_precision}")
        self.logger.info(f"  WandB logging: {self.use_wandb}")
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, and loss function based on tuned_vae.py"""
        # Optimizer with weight decay for regularization (from tuned_vae.py)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.training.learning_rate,
            weight_decay=getattr(self.config.training, 'weight_decay', 1e-5)
        )
        
        # CosineAnnealingLR for smooth decay (from tuned_vae.py)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.num_epochs,
            eta_min=1e-6
        )
        
        self.logger.info("Training components setup completed")
    
    def _determine_frame_indices(self):
        """Determine which frame indices to use for training and validation - USE ALL FRAMES."""
        self.logger.info("Determining frame indices from validation data...")
        try:
            sample_batch = next(iter(self.val_loader))
            
            data_tensor = None
            if isinstance(sample_batch, dict):
                if 'img' in sample_batch:
                    data_tensor = sample_batch['img']
                elif 'pviHP' in sample_batch:
                    if isinstance(sample_batch['pviHP'], dict) and 'img' in sample_batch['pviHP']:
                        data_tensor = sample_batch['pviHP']['img']
                    else:
                        data_tensor = sample_batch['pviHP']
                else:
                    self.logger.warning("Could not find 'img' or 'pviHP' in batch dict.")
            elif torch.is_tensor(sample_batch):
                data_tensor = sample_batch

            if data_tensor is not None:
                if data_tensor.ndim == 4:  # [batch, height, width, frames]
                    num_frames = data_tensor.shape[-1]
                    # USE ALL FRAMES for maximum temporal information
                    self.frame_indices = list(range(num_frames))
                    self.logger.info(f"Using ALL {num_frames} frames for training")
                elif data_tensor.ndim == 5:  # [batch, channels, height, width, frames]
                    num_frames = data_tensor.shape[-1]
                    # USE ALL FRAMES for maximum temporal information
                    self.frame_indices = list(range(num_frames))
                    self.logger.info(f"Using ALL {num_frames} frames for training")
                else:
                    # Default to a single frame if structure is not as expected
                    self.frame_indices = [0]
                    self.logger.warning("Unexpected data structure, using single frame")
            else:
                # Default to a single frame if structure is not as expected
                self.frame_indices = [0]
                self.logger.warning("Could not determine data structure, using single frame")
        except Exception as e:
            self.logger.warning(f"Could not automatically determine frame indices: {e}. Defaulting to [0].")
            self.frame_indices = [0]
        self.logger.info(f"Total frame indices to use: {len(self.frame_indices)}")
    
    def extract_frame(self, batch_data, frame_idx=0):
        """Extract a specific frame from the batch data (following tuned_vae.py approach)"""
        if isinstance(batch_data, dict) and 'input' in batch_data:
            # PviBatchServer returns data with 'input' key
            # Input shape: [batch_size, 32, 32, 500]
            pvi_data = batch_data['input']
            # Extract frame at specific time index
            frames = pvi_data[:, :, :, frame_idx].unsqueeze(1)  # Add channel dimension: [batch, 1, 32, 32]
        elif isinstance(batch_data, dict) and 'pviHP' in batch_data:
            # Handle nested structure: batch_data['pviHP']['img']
            if isinstance(batch_data['pviHP'], dict) and 'img' in batch_data['pviHP']:
                pvi_data = batch_data['pviHP']['img']
                frames = pvi_data[:, :, :, frame_idx].unsqueeze(1)  # [batch, 1, 32, 32]
            else:
                # Direct dataset access (for compatibility)
                pvi_data = batch_data['pviHP']
                frames = pvi_data[:, 0, :, :, frame_idx].unsqueeze(1)
        else:
            # If batch is just the tensor
            pvi_data = batch_data
            frames = pvi_data[:, 0, :, :, frame_idx].unsqueeze(1)
        
        # Replace NaN values with zeros (simple approach from tuned_vae.py)
        frames = torch.nan_to_num(frames, nan=0.0)
        
        return frames
    
    def vae_loss(self, recon_x, x, mu, logvar, beta):
        """VAE loss = reconstruction loss + beta * KL divergence (from tuned_vae.py)"""
        # Ensure inputs are valid for tanh range (between -1 and 1)
        x = torch.clamp(x, -1.0, 1.0)
        recon_x = torch.clamp(recon_x, -1.0, 1.0)
        
        # MSE loss - works well with tanh's [-1, 1] range
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') * 100  # Scale factor to make comparable to BCE
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_div
        
        return total_loss, recon_loss, kl_div
    
    def calculate_beta(self, epoch):
        """Calculate beta value for current epoch (from tuned_vae.py)"""
        if epoch <= self.beta_warmup_epochs:
            beta = self.beta_min + (self.beta_max - self.beta_min) * (epoch - 1) / (self.beta_warmup_epochs - 1)
        else:
            beta = self.beta_max
        return beta
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch using multiple frames (from tuned_vae.py)"""
        self.model.train()
        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        # Calculate beta for this epoch
        beta = self.calculate_beta(epoch)
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        
        self.logger.info(f"Epoch {epoch}, beta = {beta:.4f}, LR = {current_lr:.6f}")
        
        num_batches = 0
        total_samples = 0
        
        # Training loop with multiple frames (from tuned_vae.py)
        for batch_idx, batch_data in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            for frame_idx in self.frame_indices:
                try:
                    # Extract and normalize frames from this batch
                    frames = self.extract_frame(batch_data, frame_idx).to(self.device)
                    
                    # Forward pass with mixed precision if enabled
                    self.optimizer.zero_grad()
                    if self.use_mixed_precision and self.scaler is not None:
                        with autocast():
                            recon_batch, mu, logvar = self.model(frames)
                            loss, recon, kl = self.vae_loss(recon_batch, frames, mu, logvar, beta)
                        
                        # Backward pass with gradient scaling
                        self.scaler.scale(loss).backward()
                        
                        # Gradient clipping
                        if hasattr(self.config.training, 'grad_clip_norm'):
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                max_norm=self.config.training.grad_clip_norm
                            )
                        
                        # Optimizer step with gradient scaling
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Standard training without mixed precision
                        recon_batch, mu, logvar = self.model(frames)
                        loss, recon, kl = self.vae_loss(recon_batch, frames, mu, logvar, beta)
                        loss.backward()
                        
                        # Gradient clipping
                        if hasattr(self.config.training, 'grad_clip_norm'):
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                max_norm=self.config.training.grad_clip_norm
                            )
                        
                        self.optimizer.step()
                    
                    # Accumulate losses
                    train_loss += loss.item()
                    recon_loss_total += recon.item()
                    kl_loss_total += kl.item()
                    total_samples += frames.size(0)
                    
                    # Log batch-level metrics to wandb
                    if self.use_wandb and batch_idx % 10 == 0:
                        wandb.log({
                            "batch_loss": loss.item() / frames.size(0),
                            "batch_recon_loss": recon.item() / frames.size(0),
                            "batch_kl_loss": kl.item() / frames.size(0),
                            "batch": epoch * len(self.train_loader) + batch_idx,
                            "beta": beta,
                            "learning_rate": current_lr
                        })
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}, frame {frame_idx}: {e}")
                    continue
            
            num_batches += 1
        
        # Calculate average training losses
        if total_samples > 0:
            avg_loss = train_loss / total_samples
            avg_recon = recon_loss_total / total_samples
            avg_kl = kl_loss_total / total_samples
        else:
            avg_loss = avg_recon = avg_kl = 0.0
        
        # Record training losses
        self.train_losses.append(avg_loss)
        self.recon_losses.append(avg_recon)
        self.kl_losses.append(avg_kl)
        
        return {
            'train_loss': avg_loss,
            'train_recon_loss': avg_recon,
            'train_kl_loss': avg_kl,
            'beta': beta,
            'learning_rate': current_lr
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch using multiple frames (from tuned_vae.py)"""
        self.model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        total_samples = 0
        
        # Calculate beta for this epoch
        beta = self.calculate_beta(epoch)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.val_loader, desc=f"Validation Epoch {epoch}")):
                for frame_idx in self.frame_indices:
                    try:
                        # Extract and normalize frames from this batch
                        frames = self.extract_frame(batch_data, frame_idx).to(self.device)
                        
                        # Forward pass
                        recon_batch, mu, logvar = self.model(frames)
                        
                        # Calculate loss with current beta
                        loss, recon, kl = self.vae_loss(recon_batch, frames, mu, logvar, beta)
                        
                        # Accumulate losses
                        val_loss += loss.item()
                        val_recon_loss += recon.item()
                        val_kl_loss += kl.item()
                        total_samples += frames.size(0)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing validation batch {batch_idx}, frame {frame_idx}: {e}")
                        continue
        
        # Calculate average validation losses
        if total_samples > 0:
            avg_val_loss = val_loss / total_samples
            avg_val_recon = val_recon_loss / total_samples
            avg_val_kl = val_kl_loss / total_samples
        else:
            avg_val_loss = avg_val_recon = avg_val_kl = 0.0
        
        # Record validation losses
        self.val_losses.append(avg_val_loss)
        
        return {
            'val_loss': avg_val_loss,
            'val_recon_loss': avg_val_recon,
            'val_kl_loss': avg_val_kl
        }
    
    def visualize_reconstructions(self, epoch: int, num_images: int = 5, frame_idx: int = None):
        """Visualize reconstructions for a few samples"""
        self.model.eval()
        with torch.no_grad():
            # Use a valid frame index
            if frame_idx is None:
                frame_idx = self.frame_indices[0] if self.frame_indices else 0
            
            # Get a batch of validation data
            batch_data = next(iter(self.val_loader))
            frames = self.extract_frame(batch_data, frame_idx).to(self.device)
            
            # Get reconstructions
            recon_batch, _, _ = self.model(frames)
            
            # Create figure
            fig, axes = plt.subplots(num_images, 2, figsize=(10, 2*num_images))
            
            for i in range(num_images):
                # Original
                axes[i, 0].imshow(frames[i, 0].cpu().numpy(), cmap='gray')
                axes[i, 0].set_title('Original')
                axes[i, 0].axis('off')
                
                # Reconstruction
                axes[i, 1].imshow(recon_batch[i, 0].cpu().numpy(), cmap='gray')
                axes[i, 1].set_title('Reconstruction')
                axes[i, 1].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            save_path = self.experiment_dir / 'reconstructions' / f'recon_epoch_{epoch}.png'
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "reconstructions": wandb.Image(str(save_path)),
                    "epoch": epoch
                })
    
    def plot_training_curves(self):
        """Plot training curves and save to file"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training and validation losses
        axes[0, 0].plot(self.train_losses, label='Train')
        axes[0, 0].plot(self.val_losses, label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot reconstruction and KL losses
        axes[0, 1].plot(self.recon_losses, label='Reconstruction')
        axes[0, 1].plot(self.kl_losses, label='KL Divergence')
        axes[0, 1].set_title('Component Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot learning rate
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Plot beta value
        beta_values = [self.calculate_beta(epoch) for epoch in range(1, len(self.train_losses) + 1)]
        axes[1, 1].plot(beta_values)
        axes[1, 1].set_title('Beta Value')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Beta')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.experiment_dir / 'training_curves.png'
        plt.savefig(save_path)
        plt.close()
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "training_curves": wandb.Image(str(save_path))
            })
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Handle end of epoch"""
        # Update scheduler
        self.scheduler.step()
        
        # Log metrics to wandb
        if self.use_wandb:
            wandb.log(metrics)
        
        # Save checkpoint every N epochs
        if epoch % getattr(self.config.training, 'save_every_n_epochs', 5) == 0:
            checkpoint_path = self.experiment_dir / 'checkpoints' / f'vae_epoch_{epoch}.pt'
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': metrics['val_loss'],
                'beta': self.calculate_beta(epoch)
            }, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Visualize reconstructions
        self.visualize_reconstructions(epoch)
        
        # Early stopping check
        if metrics['val_loss'] < self.best_loss:
            self.best_loss = metrics['val_loss']
            self.best_epoch = epoch
            self.best_model_state = self.model.state_dict().copy()
            
            # Save best model
            best_model_path = self.experiment_dir / 'checkpoints' / 'vae_best.pt'
            torch.save(self.model.state_dict(), best_model_path)
            self.logger.info(f"New best model saved: {best_model_path}")
        
        # Check early stopping
        patience = getattr(self.config.training, 'early_stopping_patience', 15)
        min_delta = getattr(self.config.training, 'early_stopping_min_delta', 1e-4)
        
        if epoch - self.best_epoch > patience:
            self.logger.info(f"Early stopping triggered. Best epoch: {self.best_epoch}")
            return True
        
        return False
    
    def on_training_end(self):
        """Handle end of training"""
        # Plot final training curves
        self.plot_training_curves()
        
        # Save final model
        final_model_path = self.experiment_dir / 'checkpoints' / 'vae_final.pt'
        torch.save(self.model.state_dict(), final_model_path)
        self.logger.info(f"Final model saved: {final_model_path}")
        
        # Save best model if different from final
        if self.best_model_state is not None:
            best_model_path = self.experiment_dir / 'checkpoints' / 'vae_best.pt'
            torch.save(self.best_model_state, best_model_path)
            self.logger.info(f"Best model saved: {best_model_path}")
        
        # Close wandb run
        if self.use_wandb:
            wandb.finish()
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best model was at epoch {self.best_epoch} with loss {self.best_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(epoch)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Log epoch metrics
            self.logger.info(
                f"Epoch {epoch}/{self.config.training.num_epochs} - "
                f"Train Loss: {metrics['train_loss']:.4f} - "
                f"Val Loss: {metrics['val_loss']:.4f}"
            )
            
            # Handle end of epoch
            should_stop = self.on_epoch_end(epoch, metrics)
            if should_stop:
                break
        
        # Handle end of training
        self.on_training_end() 
"""
Training utilities for blood pressure prediction models.
Implements trainers for VAE and BiLSTM with proper validation and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import wandb
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import time
import json
from collections import defaultdict
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC

# Import model types for type hints
try:
    from ..models.vae import VAE, VAELoss, BetaScheduler
    from ..models.bilstm import BiLSTM
    from ..models.losses import BPLoss, BPMetrics
    from ..utils.config import TrainingConfig, ModelConfig
    
    # Try to import optional training utilities
    try:
        from .metrics import MetricsCalculator
    except ImportError:
        MetricsCalculator = None
        
    try:
        from .callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        EarlyStopping = None
        ModelCheckpoint = None
        
except ImportError:
    # Fallback for cases where relative imports fail
    VAE = None
    VAELoss = None
    BetaScheduler = None
    BiLSTM = None
    BPLoss = None
    BPMetrics = None
    TrainingConfig = None
    ModelConfig = None
    MetricsCalculator = None
    EarlyStopping = None
    ModelCheckpoint = None

from .callbacks import CallbackManager, LearningRateScheduler, PeriodicCheckpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Keeps track of training state and metrics for plotting."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    
    # Loss tracking
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    # VAE-specific tracking
    train_recon_losses: List[float] = field(default_factory=list)
    val_recon_losses: List[float] = field(default_factory=list)
    train_kl_losses: List[float] = field(default_factory=list)
    val_kl_losses: List[float] = field(default_factory=list)
    beta_values: List[float] = field(default_factory=list)
    
    # BiLSTM-specific tracking
    train_systolic_losses: List[float] = field(default_factory=list)
    val_systolic_losses: List[float] = field(default_factory=list)
    train_diastolic_losses: List[float] = field(default_factory=list)
    val_diastolic_losses: List[float] = field(default_factory=list)


class BaseTrainer:
    """Base trainer class with common functionality."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        model_config: ModelConfig,
        device: torch.device = None,
        logger_name: str = "trainer"
    ):
        """
        Initialize base trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            model_config: Model configuration
            device: Computing device
            logger_name: Logger name
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_config = model_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(logger_name)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler(device='cuda') if config.use_mixed_precision and torch.cuda.is_available() else None
        
        # Initialize metrics and callbacks
        self.metrics_calculator = MetricsCalculator()
        self.callback_manager = CallbackManager()
        self._setup_callbacks()
        
        # Initialize logging
        self.use_wandb = config.use_wandb
        
        # Training state
        self.state = TrainingState()
        
        self.logger.info(f"Trainer initialized on device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.adam_betas
            )
        elif self.config.optimizer.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.adam_betas
            )
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if not self.config.use_scheduler:
            return None
        
        if self.config.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.scheduler_patience,
                verbose=True
            )
        elif self.config.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=0.1
            )
        else:
            return None
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        # Early stopping
        if self.config.early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                mode='min'
            )
            self.callback_manager.add_callback('early_stopping', early_stopping)
        
        # Model checkpointing (best model only)
        if self.config.checkpoint_dir:
            checkpoint_callback = ModelCheckpoint(
                checkpoint_dir=self.config.checkpoint_dir,
                save_best_only=True,
                mode='min',
                verbose=True
            )
            self.callback_manager.add_callback('model_checkpoint', checkpoint_callback)
            
            # Periodic checkpointing (every N epochs)
            save_frequency = getattr(self.config, 'save_every_n_epochs', 5)
            periodic_callback = PeriodicCheckpoint(
                checkpoint_dir=self.config.checkpoint_dir,
                save_frequency=save_frequency,
                verbose=True
            )
            periodic_callback.set_trainer(self)
            self.callback_manager.add_callback('periodic_checkpoint', periodic_callback)
        
        # Learning rate scheduling
        if self.scheduler:
            lr_scheduler = LearningRateScheduler(self.scheduler)
            self.callback_manager.add_callback('lr_scheduler', lr_scheduler)
    
    def save_checkpoint(self, filepath: Path, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.state.best_metric,
            'best_epoch': self.state.best_epoch,
            'config': self.config,
            'model_config': self.model_config,
            'training_state': self.state
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.parent / 'best_model.pth'
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: Path) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if 'training_state' in checkpoint:
            self.state = checkpoint['training_state']
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to wandb."""
        if self.use_wandb and wandb.run is not None:
            log_dict = {}
            for name, value in metrics.items():
                metric_name = f"{prefix}/{name}" if prefix else name
                log_dict[metric_name] = value
            wandb.log(log_dict, step=step)
    
    def generate_training_plots(self, save_dir: Path):
        """Generate comprehensive training plots at the end of training."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the main training plot
        self._create_training_summary_plot(save_dir)
        
        # Generate model-specific plots
        if hasattr(self, '_generate_model_specific_plots'):
            self._generate_model_specific_plots(save_dir)
    
    def _create_training_summary_plot(self, save_dir: Path):
        """Create a comprehensive training summary plot."""
        # Basic validation
        if len(self.state.train_losses) < 2:
            self.logger.warning("Not enough training data to generate plots")
            return
        
        epochs = list(range(len(self.state.train_losses)))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Training vs Validation Loss
        axes[0, 0].plot(epochs, self.state.train_losses, 'b-', label='Train', linewidth=2)
        # Only plot validation if we have the same number of data points
        if len(self.state.val_losses) == len(self.state.train_losses) and len(self.state.val_losses) > 0:
            axes[0, 0].plot(epochs, self.state.val_losses, 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss vs. Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate Schedule
        if len(self.state.learning_rates) > 0:
            lr_epochs = list(range(len(self.state.learning_rates)))
            axes[0, 1].plot(lr_epochs, self.state.learning_rates, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate vs. Epoch')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Learning Rate vs. Epoch')
        
        # Plot 3 & 4: Will be filled by subclasses
        axes[1, 0].text(0.5, 0.5, 'Model-Specific\nMetrics', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Loss Components')
        
        # Training Summary
        summary_text = f"Training Completed\n\n"
        summary_text += f"Total Epochs: {len(self.state.train_losses)}\n"
        summary_text += f"Best Metric: {self.state.best_metric:.6f}\n"
        summary_text += f"Best Epoch: {self.state.best_epoch}\n"
        summary_text += f"Final Train Loss: {self.state.train_losses[-1]:.6f}\n"
        if len(self.state.val_losses) > 0:
            summary_text += f"Final Val Loss: {self.state.val_losses[-1]:.6f}\n"
        
        axes[1, 1].text(0.1, 0.5, summary_text, ha='left', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=11, fontfamily='monospace')
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = save_dir / "training_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training summary plot saved: {plot_path}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. To be implemented by subclasses."""
        raise NotImplementedError
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch. To be implemented by subclasses."""
        raise NotImplementedError
    
    def fit(self) -> Dict[str, Any]:
        """Main training loop."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=self.config.project_name,
                name=f"{self.model_config.model_type}_{int(time.time())}",
                config={
                    **{k: v for k, v in self.config.__dict__.items() if not k.startswith('_')},
                    **{k: v for k, v in self.model_config.__dict__.items() if not k.startswith('_')}
                },
                tags=[self.model_config.model_type, "training"],
                mode=self.config.wandb_mode,
                settings=wandb.Settings(init_timeout=120)
            )
            # Log model architecture
            wandb.watch(self.model, log="all", log_freq=100)
        
        try:
            for epoch in range(self.state.epoch, self.config.num_epochs):
                self.state.epoch = epoch
                
                # Training
                train_metrics = self.train_epoch()
                self.state.train_losses.append(train_metrics['loss'])
                
                # Validation
                val_metrics = self.validate_epoch()
                self.state.val_losses.append(val_metrics['loss'])
                
                # Track learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.state.learning_rates.append(current_lr)
                
                # Combine metrics
                epoch_metrics = {
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    'learning_rate': current_lr,
                    'epoch': epoch
                }
                
                # Log metrics
                self.log_metrics(epoch_metrics, self.state.global_step)
                
                # Callbacks
                callback_metrics = {'val_loss': val_metrics['loss']}
                should_stop = self.callback_manager.on_epoch_end(
                    epoch, self.model, callback_metrics
                )
                
                # Update best metric
                current_metric = val_metrics['loss']
                if current_metric < self.state.best_metric:
                    self.state.best_metric = current_metric
                    self.state.best_epoch = epoch
                    
                    # Save best checkpoint
                    if self.config.checkpoint_dir:
                        checkpoint_path = Path(self.config.checkpoint_dir) / f"epoch_{epoch:03d}.pth"
                        self.save_checkpoint(checkpoint_path, is_best=True)
                
                # Print progress
                self.logger.info(
                    f"Epoch {epoch:3d}/{self.config.num_epochs:3d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"LR: {current_lr:.6f}"
                )
                
                # Early stopping check
                if should_stop:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                # Memory cleanup
                if epoch % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Generate final training plots
            if self.config.checkpoint_dir:
                results_dir = Path(self.config.checkpoint_dir).parent / "results"
                self.generate_training_plots(results_dir)
            
            if self.use_wandb and wandb.run is not None:
                # Log final metrics
                wandb.log({
                    "training/final_best_metric": self.state.best_metric,
                    "training/final_best_epoch": self.state.best_epoch,
                    "training/total_time": training_time
                })
                wandb.finish()
        
        return {
            'best_metric': self.state.best_metric,
            'best_epoch': self.state.best_epoch,
            'training_time': training_time,
            'final_plots_generated': True
        }


class VAETrainer(BaseTrainer):
    """Trainer specifically for VAE models."""
    
    def __init__(self, model: VAE, train_loader: DataLoader, val_loader: DataLoader,
                 config: TrainingConfig, model_config: ModelConfig, **kwargs):
        super().__init__(model, train_loader, val_loader, config, model_config, **kwargs)
        
        # VAE-specific loss
        self.criterion = VAELoss(
            beta=config.vae_beta,
            reduction='mean'
        )
        
        # Beta scheduling for VAE
        self.beta_scheduler = self._create_beta_scheduler()
        
        # Initialize scaler with correct device
        self.scaler = GradScaler(device='cuda') if config.use_mixed_precision and torch.cuda.is_available() else None
    
    def _create_beta_scheduler(self):
        """Create beta scheduler for VAE training."""
        if hasattr(self.config, 'beta_warmup_epochs') and self.config.beta_warmup_epochs > 0:
            def beta_schedule(epoch):
                if epoch < self.config.beta_warmup_epochs:
                    return self.config.vae_beta * (epoch / self.config.beta_warmup_epochs)
                return self.config.vae_beta
            return beta_schedule
        return lambda epoch: self.config.vae_beta
    
    def _extract_frame(self, batch_data, frame_idx=0):
        """
        Extract and process a specific frame from batch data.
        Adapted from tuned_vae.py for consistent frame handling.
        
        Args:
            batch_data: Batch data (dict or tensor)
            frame_idx: Frame index to extract (for temporal data)
        
        Returns:
            Processed frames tensor
        """
        if isinstance(batch_data, dict):
            # If batch is a dictionary (like in PVI data)
            pvi_data = batch_data['pviHP'] if 'pviHP' in batch_data else batch_data['sequence']
        else:
            # If batch is just the tensor
            pvi_data = batch_data
        
        # Handle different data shapes
        if len(pvi_data.shape) == 5:  # [batch, sequence, channels, height, width]
            # Extract the specific frame from each sample
            frames = pvi_data[:, 0, :, :, frame_idx].unsqueeze(1) if pvi_data.shape[-1] > 1 else pvi_data[:, 0]
        elif len(pvi_data.shape) == 4:  # [batch, channels, height, width]
            frames = pvi_data
        else:
            raise ValueError(f"Unexpected data shape: {pvi_data.shape}")
        
        # Replace NaN values with zeros
        frames = torch.nan_to_num(frames, nan=0.0)
        
        return frames
    
    def train_epoch(self) -> Dict[str, float]:
        """Train VAE for one epoch."""
        self.model.train()
        running_metrics = defaultdict(float)
        num_batches = len(self.train_loader)
        
        # Update beta for current epoch
        current_beta = self.beta_scheduler(self.state.epoch)
        self.criterion.beta = current_beta
        self.state.beta_values.append(current_beta)
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.state.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            if isinstance(batch, dict):
                x = batch['sequence'] if 'sequence' in batch else batch['pviHP']
            else:
                x = batch[0]
            
            x = x.to(self.device, non_blocking=True)
            
            # Reshape if needed for VAE
            original_shape = x.shape
            if len(x.shape) == 5:  # [batch, sequence, channels, height, width]
                batch_size, seq_len = x.shape[:2]
                x = x.view(-1, *x.shape[2:])  # [batch*sequence, channels, height, width]
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast(device_type='cuda'):
                    recon_x, mu, logvar = self.model(x)
                    loss, recon_loss, kl_loss = self.criterion(recon_x, x, mu, logvar)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                recon_x, mu, logvar = self.model(x)
                loss, recon_loss, kl_loss = self.criterion(recon_x, x, mu, logvar)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            running_metrics['loss'] += loss.item()
            running_metrics['recon_loss'] += recon_loss.item()
            running_metrics['kl_loss'] += kl_loss.item()
            
            self.state.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'beta': current_beta
            })
        
        # Average metrics and store for plotting
        epoch_metrics = {k: v / num_batches for k, v in running_metrics.items()}
        self.state.train_recon_losses.append(epoch_metrics['recon_loss'])
        self.state.train_kl_losses.append(epoch_metrics['kl_loss'])
        
        # Generate reconstruction visualizations periodically
        if self.config.checkpoint_dir and (self.state.epoch == 0 or (self.state.epoch + 1) % 5 == 0):
            try:
                results_dir = Path(self.config.checkpoint_dir).parent / "results"
                self.visualize_reconstructions(results_dir, epoch=self.state.epoch)
                self.logger.info(f"Generated reconstruction visualization for epoch {self.state.epoch}")
            except Exception as e:
                self.logger.warning(f"Could not generate reconstruction visualization: {e}")
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate VAE for one epoch."""
        self.model.eval()
        running_metrics = defaultdict(float)
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    # Use the helper method for consistent frame extraction
                    x = self._extract_frame(batch, frame_idx=0).to(self.device)
                    
                    # Forward pass
                    recon_x, mu, logvar = self.model(x)
                    loss, recon_loss, kl_loss = self.criterion(recon_x, x, mu, logvar)
                    
                    # Update metrics
                    running_metrics['loss'] += loss.item()
                    running_metrics['recon_loss'] += recon_loss.item()
                    running_metrics['kl_loss'] += kl_loss.item()
                    
                except Exception as e:
                    self.logger.error(f"Error during validation: {e}")
                    # Return a default loss if validation fails
                    return {
                        'loss': float('inf'),
                        'recon_loss': float('inf'), 
                        'kl_loss': float('inf')
                    }
        
        # Average metrics and store for plotting
        epoch_metrics = {k: v / num_batches for k, v in running_metrics.items()}
        self.state.val_recon_losses.append(epoch_metrics['recon_loss'])
        self.state.val_kl_losses.append(epoch_metrics['kl_loss'])
        
        return epoch_metrics
    
    def _generate_model_specific_plots(self, save_dir: Path):
        """Generate VAE-specific training plots with enhanced styling."""
        epochs = list(range(len(self.state.train_losses)))
        
        # Create comprehensive VAE plots with improved styling
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle('VAE Training Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Total Loss with improved styling
        axes[0, 0].plot(epochs, self.state.train_losses, 'b-', label='Train', linewidth=2, alpha=0.8)
        # Only plot validation if we have matching data
        if len(self.state.val_losses) == len(epochs) and len(self.state.val_losses) > 0:
            axes[0, 0].plot(epochs, self.state.val_losses, 'r-', label='Validation', linewidth=2, alpha=0.8)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Total Loss', fontsize=12)
        axes[0, 0].set_title('Total Loss vs. Epoch', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Reconstruction Loss with improved styling
        if len(self.state.train_recon_losses) > 0:
            axes[0, 1].plot(epochs, self.state.train_recon_losses, 'b-', label='Train Reconstruction', linewidth=2, alpha=0.8)
            if len(self.state.val_recon_losses) == len(epochs) and len(self.state.val_recon_losses) > 0:
                axes[0, 1].plot(epochs, self.state.val_recon_losses, 'r-', label='Val Reconstruction', linewidth=2, alpha=0.8)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Reconstruction Loss', fontsize=12)
        axes[0, 1].set_title('Reconstruction Loss vs. Epoch', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: KL Divergence with improved styling
        if len(self.state.train_kl_losses) > 0:
            axes[0, 2].plot(epochs, self.state.train_kl_losses, 'b-', label='Train KL', linewidth=2, alpha=0.8)
            if len(self.state.val_kl_losses) == len(epochs) and len(self.state.val_kl_losses) > 0:
                axes[0, 2].plot(epochs, self.state.val_kl_losses, 'r-', label='Val KL', linewidth=2, alpha=0.8)
        axes[0, 2].set_xlabel('Epoch', fontsize=12)
        axes[0, 2].set_ylabel('KL Divergence', fontsize=12)
        axes[0, 2].set_title('KL Divergence vs. Epoch', fontsize=14, fontweight='bold')
        axes[0, 2].legend(fontsize=11)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate with improved styling
        if len(self.state.learning_rates) > 0:
            axes[1, 0].plot(epochs, self.state.learning_rates, 'g-', linewidth=2, alpha=0.8)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 0].set_title('Learning Rate vs. Epoch', fontsize=14, fontweight='bold')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Learning Rate Data', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Learning Rate vs. Epoch', fontsize=14, fontweight='bold')
        
        # Plot 5: Beta Schedule with improved styling
        if len(self.state.beta_values) > 0:
            axes[1, 1].plot(epochs, self.state.beta_values, color='purple', linewidth=2, alpha=0.8)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Beta Value', fontsize=12)
            axes[1, 1].set_title('Beta Schedule vs. Epoch', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Beta Data', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Beta Schedule vs. Epoch', fontsize=14, fontweight='bold')
        
        # Plot 6: Generate enhanced reconstruction samples or loss comparison
        if len(epochs) > 0:
            try:
                # Try to generate reconstruction samples
                self._generate_enhanced_reconstruction_samples(axes[1, 2])
            except Exception as e:
                self.logger.warning(f"Could not generate reconstruction samples: {e}")
                # Fallback to train vs val loss scatter only if we have both datasets
                if len(self.state.train_losses) > 0 and len(self.state.val_losses) == len(self.state.train_losses) and len(self.state.val_losses) > 0:
                    scatter = axes[1, 2].scatter(self.state.train_losses, self.state.val_losses, 
                                               c=epochs, cmap='viridis', alpha=0.7)
                    axes[1, 2].plot([min(self.state.train_losses), max(self.state.train_losses)], 
                                   [min(self.state.train_losses), max(self.state.train_losses)], 
                                   'r--', label='y=x', alpha=0.5)
                    axes[1, 2].set_xlabel('Training Loss', fontsize=12)
                    axes[1, 2].set_ylabel('Validation Loss', fontsize=12)
                    axes[1, 2].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
                    plt.colorbar(scatter, ax=axes[1, 2], label='Epoch')
                    axes[1, 2].legend(fontsize=11)
                    axes[1, 2].grid(True, alpha=0.3)
                else:
                    axes[1, 2].text(0.5, 0.5, 'Validation Failed\nNo Comparison Data', 
                                   horizontalalignment='center', verticalalignment='center',
                                   transform=axes[1, 2].transAxes, fontsize=12)
                    axes[1, 2].set_title('Analysis Plot', fontsize=14, fontweight='bold')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Data Available', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].set_title('Analysis Plot', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the enhanced plot
        plot_path = save_dir / "vae_training_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        self.logger.info(f"Saved VAE training analysis to {plot_path}")
        
        # Log training curves to wandb if available
        if self.use_wandb and wandb.run is not None:
            wandb.log({"vae_training_analysis": wandb.Image(plt)})
        
        plt.close(fig)
        
        # Also generate the enhanced reconstruction plot separately
        if len(epochs) > 0:
            try:
                current_epoch = max(epochs) if epochs else 0
                self.visualize_reconstructions(save_dir, epoch=current_epoch)
            except Exception as e:
                self.logger.warning(f"Could not generate separate reconstruction plot: {e}")
    
    def _generate_enhanced_reconstruction_samples(self, ax):
        """Generate enhanced reconstruction samples for the final plot."""
        self.model.eval()
        with torch.no_grad():
            # Get a batch of validation data
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    x = batch['sequence'] if 'sequence' in batch else batch['pviHP']
                else:
                    x = batch[0]
                
                x = x.to(self.device)
                
                # Reshape if needed
                if len(x.shape) == 5:
                    x = x.view(-1, *x.shape[2:])
                
                # Forward pass
                recon_x, mu, logvar = self.model(x)
                
                # Show first few samples with better layout
                num_samples = min(3, x.shape[0])
                samples_combined = []
                
                for i in range(num_samples):
                    original = x[i].cpu().squeeze().numpy()
                    reconstructed = recon_x[i].cpu().squeeze().numpy()
                    
                    # Stack vertically for better comparison
                    combined_sample = np.vstack([original, reconstructed])
                    samples_combined.append(combined_sample)
                
                # Create horizontal layout of all samples
                if samples_combined:
                    final_combined = np.hstack(samples_combined)
                    im = ax.imshow(final_combined, cmap='viridis', aspect='auto')
                    ax.set_title('Original (top) vs Reconstructed (bottom)', fontsize=12, fontweight='bold')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
                    # Remove axis ticks but keep the frame
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.text(0.5, 0.5, 'No Reconstruction Data', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title('Reconstructions', fontsize=12, fontweight='bold')
                
                break  # Only process first batch
    
    def visualize_reconstructions(self, save_dir: Path, epoch: int = 0, num_images: int = 5):
        """
        Enhanced reconstruction visualization with multiple samples, axes, and colorbars.
        Adapted from tuned_vae.py for better visualization quality.
        """
        reconstructions_dir = save_dir / 'reconstructions'
        reconstructions_dir.mkdir(exist_ok=True)
        
        self.model.eval()
        
        # Get a batch from validation data
        with torch.no_grad():
            for batch in self.val_loader:
                # Use the helper method for consistent frame extraction
                x = self._extract_frame(batch, frame_idx=0).to(self.device)
                
                # Limit to num_images
                x = x[:num_images]
                
                # Get reconstructions
                recon_x, _, _ = self.model(x)
                break
        
        # Create a figure
        fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
        
        # Convert tensors to numpy for plotting
        x_np = x.cpu().numpy()
        recon_np = recon_x.cpu().numpy()
        
        # Plot original images with axes values
        for i in range(num_images):
            # Turn on axis to show coordinates
            axes[0, i].axis('on')
            
            # Plot the image with origin at lower left
            img_plot = axes[0, i].imshow(x_np[i, 0], cmap='viridis', origin='lower')
            
            # Set x and y ticks to show coordinates
            img_height, img_width = x_np[i, 0].shape
            x_ticks = np.linspace(0, img_width-1, 5, dtype=int)  # 5 ticks on x-axis
            y_ticks = np.linspace(0, img_height-1, 5, dtype=int)  # 5 ticks on y-axis
            
            axes[0, i].set_xticks(x_ticks)
            axes[0, i].set_yticks(y_ticks)
            
            # Add colorbar for each plot to show intensity values
            cbar = plt.colorbar(img_plot, ax=axes[0, i], fraction=0.046, pad=0.04)
            cbar.set_label('Intensity')
            
            # Add sample number as title
            axes[0, i].set_title(f'Original Sample {i+1}')
            
        # Plot reconstructed images with axes values
        for i in range(num_images):
            # Turn on axis
            axes[1, i].axis('on')
            
            # Plot the image with origin at lower left
            img_plot = axes[1, i].imshow(recon_np[i, 0], cmap='viridis', origin='lower')
            
            # Set x and y ticks to show coordinates
            img_height, img_width = recon_np[i, 0].shape
            x_ticks = np.linspace(0, img_width-1, 5, dtype=int)  # 5 ticks on x-axis
            y_ticks = np.linspace(0, img_height-1, 5, dtype=int)  # 5 ticks on y-axis
            
            axes[1, i].set_xticks(x_ticks)
            axes[1, i].set_yticks(y_ticks)
            
            # Add colorbar for each plot to show intensity values
            cbar = plt.colorbar(img_plot, ax=axes[1, i], fraction=0.046, pad=0.04)
            cbar.set_label('Intensity')
            
            # Add sample number as title
            axes[1, i].set_title(f'Reconstructed Sample {i+1}')
        
        plt.suptitle(f'Epoch {epoch} - Reconstructions', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        save_path = reconstructions_dir / f'recon_epoch_{epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved reconstructions for epoch {epoch} to {save_path}")
        
        # Log the reconstruction plot to wandb if available
        if self.use_wandb and wandb.run is not None:
            wandb.log({f"reconstructions_epoch_{epoch}": wandb.Image(plt)})
        
        plt.close(fig)
        return recon_np  # Return reconstructions for further analysis if needed

    def _generate_reconstruction_samples(self, ax):
        """Generate reconstruction samples for the final plot (legacy method)."""
        self.model.eval()
        with torch.no_grad():
            # Get a batch of validation data
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    x = batch['sequence'] if 'sequence' in batch else batch['pviHP']
                else:
                    x = batch[0]
                
                x = x.to(self.device)
                
                # Reshape if needed
                if len(x.shape) == 5:
                    x = x.view(-1, *x.shape[2:])
                
                # Forward pass
                recon_x, mu, logvar = self.model(x)
                
                # Show first sample
                original = x[0].cpu().squeeze().numpy()
                reconstructed = recon_x[0].cpu().squeeze().numpy()
                
                # Create side-by-side comparison
                combined = np.hstack([original, reconstructed])
                ax.imshow(combined, cmap='gray')
                ax.set_title('Original | Reconstructed')
                ax.axis('off')
                
                break  # Only process first batch


class BiLSTMTrainer(BaseTrainer):
    """Trainer for BiLSTM models."""
    
    def __init__(self, model: Union[BiLSTM, nn.Module], 
                 train_loader: DataLoader, val_loader: DataLoader,
                 config: TrainingConfig, model_config: ModelConfig, **kwargs):
        super().__init__(model, train_loader, val_loader, config, model_config, **kwargs)
        
        # BP prediction loss
        self.criterion = BPLoss(
            systolic_weight=getattr(config, 'systolic_weight', 1.0),
            diastolic_weight=getattr(config, 'diastolic_weight', 1.0),
            shape_weight=getattr(config, 'shape_weight', 0.5),
            temporal_weight=getattr(config, 'temporal_weight', 0.3)
        )
        
        # Initialize scaler with correct device
        self.scaler = GradScaler(device='cuda') if config.use_mixed_precision and torch.cuda.is_available() else None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train BiLSTM for one epoch."""
        self.model.train()
        running_metrics = defaultdict(float)
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.state.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            x_seq = batch['sequence'].to(self.device, non_blocking=True)
            y_target = batch['target'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.scaler:
                with autocast(device_type='cuda'):
                    output = self.model(x_seq)
                    # Handle models that return attention weights
                    if isinstance(output, tuple):
                        output, attention_weights = output
                    elif isinstance(output, dict):
                        attention_weights = output.get('attention_weights')
                        output = output.get('waveform', output)
                    
                    loss_dict = self.criterion(output, y_target)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(x_seq)
                # Handle models that return attention weights
                if isinstance(output, tuple):
                    output, attention_weights = output
                elif isinstance(output, dict):
                    attention_weights = output.get('attention_weights')
                    output = output.get('waveform', output)
                
                loss_dict = self.criterion(output, y_target)
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            for key, value in loss_dict.items():
                running_metrics[key] += value.item()
            
            # Compute additional metrics
            with torch.no_grad():
                batch_metrics = self.metrics_calculator.compute_bp_metrics(
                    output.cpu(), y_target.cpu()
                )
                for key, value in batch_metrics.items():
                    running_metrics[key] += value
            
            self.state.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'mae': batch_metrics.get('mae', 0)
            })
        
        # Average metrics and store for plotting
        epoch_metrics = {k: v / num_batches for k, v in running_metrics.items()}
        if 'systolic_loss' in epoch_metrics:
            self.state.train_systolic_losses.append(epoch_metrics['systolic_loss'])
        if 'diastolic_loss' in epoch_metrics:
            self.state.train_diastolic_losses.append(epoch_metrics['diastolic_loss'])
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate BiLSTM for one epoch."""
        self.model.eval()
        running_metrics = defaultdict(float)
        num_batches = len(self.val_loader)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                x_seq = batch['sequence'].to(self.device, non_blocking=True)
                y_target = batch['target'].to(self.device, non_blocking=True)
                
                # Forward pass
                output = self.model(x_seq)
                # Handle models that return attention weights
                if isinstance(output, tuple):
                    output, attention_weights = output
                elif isinstance(output, dict):
                    attention_weights = output.get('attention_weights')
                    output = output.get('waveform', output)
                
                loss_dict = self.criterion(output, y_target)
                
                # Update metrics
                for key, value in loss_dict.items():
                    running_metrics[key] += value.item()
                
                # Store predictions for final metrics
                all_predictions.append(output.cpu())
                all_targets.append(y_target.cpu())
        
        # Compute comprehensive metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        final_metrics = self.metrics_calculator.compute_comprehensive_metrics(
            all_predictions, all_targets
        )
        
        # Combine with running metrics
        for key, value in final_metrics.items():
            running_metrics[key] = value
        
        # Average loss metrics and store for plotting
        for key in ['total_loss', 'systolic_loss', 'diastolic_loss', 'shape_loss', 'temporal_loss']:
            if key in running_metrics:
                running_metrics[key] /= num_batches
        
        if 'systolic_loss' in running_metrics:
            self.state.val_systolic_losses.append(running_metrics['systolic_loss'])
        if 'diastolic_loss' in running_metrics:
            self.state.val_diastolic_losses.append(running_metrics['diastolic_loss'])
        
        # Rename total_loss to loss for consistency
        if 'total_loss' in running_metrics:
            running_metrics['loss'] = running_metrics['total_loss']
        
        return dict(running_metrics)


class SimpleMetricsCalculator:
    """Simple metrics calculator for when the full one is not available."""
    
    def compute_bp_metrics(self, pred, target):
        """Compute basic BP metrics."""
        mse = torch.mean((pred - target) ** 2).item()
        mae = torch.mean(torch.abs(pred - target)).item()
        return {
            'mse': mse,
            'mae': mae
        }
    
    def compute_comprehensive_metrics(self, pred, target):
        """Compute comprehensive metrics."""
        mse = torch.mean((pred - target) ** 2).item()
        mae = torch.mean(torch.abs(pred - target)).item()
        return {
            'mse': mse,
            'mae': mae,
            'rmse': mse ** 0.5
        }


def create_trainer(
    model_type: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    model_config: ModelConfig,
    **kwargs
) -> BaseTrainer:
    """
    Factory function to create appropriate trainer.
    
    Args:
        model_type: Type of model ('vae', 'bilstm', 'vae_bilstm')
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        model_config: Model configuration
        **kwargs: Additional arguments
        
    Returns:
        Appropriate trainer instance
    """
    if model_type.lower() == 'vae':
        return VAETrainer(model, train_loader, val_loader, config, model_config, **kwargs)
    elif model_type.lower() in ['bilstm', 'vae_bilstm']:
        return BiLSTMTrainer(model, train_loader, val_loader, config, model_config, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 
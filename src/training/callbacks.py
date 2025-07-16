"""
Training callbacks for blood pressure prediction models.
Includes early stopping, model checkpointing, and learning rate scheduling.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod
import numpy as np
import json
import warnings

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base callback class."""
    
    def on_epoch_end(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> bool:
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            model: The model being trained
            metrics: Dictionary of metrics from this epoch
            
        Returns:
            Boolean indicating whether to stop training
        """
        return False
    
    def on_training_start(self, model: nn.Module):
        """Called at the start of training."""
        pass
    
    def on_training_end(self, model: nn.Module):
        """Called at the end of training."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimizing metric, 'max' for maximizing
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.wait_count = 0
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.stopped_epoch = 0
        
        # Function to check if improvement occurred
        if mode == 'min':
            self.is_better = lambda current, best: current < (best - min_delta)
        else:
            self.is_better = lambda current, best: current > (best + min_delta)
    
    def on_training_start(self, model: nn.Module):
        """Reset state at start of training."""
        self.wait_count = 0
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria are met."""
        # Default to 'val_loss' if available, otherwise use 'loss'
        current_metric = metrics.get('val_loss', metrics.get('loss', None))
        
        if current_metric is None:
            logger.warning("No suitable metric found for early stopping")
            return False
        
        if self.is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.wait_count = 0
            if self.verbose:
                logger.info(f"Early stopping: new best metric {current_metric:.6f}")
        else:
            self.wait_count += 1
            if self.verbose:
                logger.info(
                    f"Early stopping: no improvement for {self.wait_count}/{self.patience} epochs"
                )
        
        if self.wait_count >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                logger.info(f"Early stopping triggered at epoch {epoch}")
            return True
        
        return False


class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        filename_template: str = "epoch_{epoch:03d}_metric_{metric:.4f}.pth",
        save_best_only: bool = True,
        mode: str = 'min',
        verbose: bool = True,
        save_weights_only: bool = False
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            filename_template: Template for checkpoint filenames
            save_best_only: Whether to only save when metric improves
            mode: 'min' for minimizing metric, 'max' for maximizing
            verbose: Whether to print messages
            save_weights_only: Whether to save only model weights
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.filename_template = filename_template
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self.save_weights_only = save_weights_only
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # Function to check if improvement occurred
        if mode == 'min':
            self.is_better = lambda current, best: current < best
        else:
            self.is_better = lambda current, best: current > best
    
    def on_training_start(self, model: nn.Module):
        """Reset state at start of training."""
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
    
    def on_epoch_end(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> bool:
        """Save checkpoint if criteria are met."""
        # Default to 'val_loss' if available, otherwise use 'loss'
        current_metric = metrics.get('val_loss', metrics.get('loss', None))
        
        if current_metric is None:
            logger.warning("No suitable metric found for checkpointing")
            return False
        
        should_save = not self.save_best_only or self.is_better(current_metric, self.best_metric)
        
        if should_save:
            if self.is_better(current_metric, self.best_metric):
                self.best_metric = current_metric
            
            # Create filename
            filename = self.filename_template.format(
                epoch=epoch,
                metric=current_metric
            )
            filepath = self.checkpoint_dir / filename
            
            # Save checkpoint
            if self.save_weights_only:
                torch.save(model.state_dict(), filepath)
            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics,
                    'best_metric': self.best_metric
                }
                torch.save(checkpoint, filepath)
            
            if self.verbose:
                logger.info(f"Checkpoint saved: {filepath}")
            
            # Save best model separately
            if self.is_better(current_metric, self.best_metric):
                best_path = self.checkpoint_dir / "best_model.pth"
                if self.save_weights_only:
                    torch.save(model.state_dict(), best_path)
                else:
                    torch.save(checkpoint, best_path)
                
                if self.verbose:
                    logger.info(f"Best model saved: {best_path}")
        
        return False


class PeriodicCheckpoint(Callback):
    """Save model checkpoints at regular intervals."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_frequency: int = 5,
        filename_template: str = "checkpoint_epoch_{epoch:03d}.pth",
        verbose: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True
    ):
        """
        Initialize periodic checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_frequency: Save every N epochs
            filename_template: Template for checkpoint filenames
            verbose: Whether to print messages
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_frequency = save_frequency
        self.filename_template = filename_template
        self.verbose = verbose
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        # Store trainer components for saving
        self.trainer = None
    
    def set_trainer(self, trainer):
        """Set trainer reference to access optimizer and scheduler."""
        self.trainer = trainer
    
    def on_epoch_end(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> bool:
        """Save checkpoint if it's time to save."""
        if (epoch + 1) % self.save_frequency == 0:
            # Create filename
            filename = self.filename_template.format(epoch=epoch)
            filepath = self.checkpoint_dir / filename
            
            # Create checkpoint dictionary
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }
            
            # Add trainer components if available
            if self.trainer:
                if self.save_optimizer and hasattr(self.trainer, 'optimizer'):
                    checkpoint['optimizer_state_dict'] = self.trainer.optimizer.state_dict()
                
                if self.save_scheduler and hasattr(self.trainer, 'scheduler') and self.trainer.scheduler:
                    checkpoint['scheduler_state_dict'] = self.trainer.scheduler.state_dict()
                
                if hasattr(self.trainer, 'scaler') and self.trainer.scaler:
                    checkpoint['scaler_state_dict'] = self.trainer.scaler.state_dict()
                
                if hasattr(self.trainer, 'state'):
                    checkpoint['training_state'] = {
                        'best_metric': self.trainer.state.best_metric,
                        'best_epoch': self.trainer.state.best_epoch,
                        'train_losses': self.trainer.state.train_losses,
                        'val_losses': self.trainer.state.val_losses
                    }
            
            # Save checkpoint
            torch.save(checkpoint, filepath)
            
            if self.verbose:
                logger.info(f"Periodic checkpoint saved: {filepath}")
        
        return False


class LearningRateScheduler(Callback):
    """Learning rate scheduling callback."""
    
    def __init__(self, scheduler, monitor: str = 'val_loss', verbose: bool = True):
        """
        Initialize learning rate scheduler callback.
        
        Args:
            scheduler: PyTorch learning rate scheduler
            monitor: Metric to monitor for ReduceLROnPlateau
            verbose: Whether to print messages
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose
        
        # Check if this is a ReduceLROnPlateau scheduler
        self.is_plateau_scheduler = hasattr(scheduler, 'step') and \
                                   'ReduceLROnPlateau' in str(type(scheduler))
    
    def on_epoch_end(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> bool:
        """Update learning rate based on scheduler type."""
        if self.is_plateau_scheduler:
            # ReduceLROnPlateau needs a metric value
            metric_value = metrics.get(self.monitor)
            if metric_value is not None:
                old_lr = self.scheduler.optimizer.param_groups[0]['lr']
                self.scheduler.step(metric_value)
                new_lr = self.scheduler.optimizer.param_groups[0]['lr']
                
                if old_lr != new_lr and self.verbose:
                    logger.info(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            else:
                logger.warning(f"Metric '{self.monitor}' not found for LR scheduling")
        else:
            # Other schedulers just need to be stepped
            old_lr = self.scheduler.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.scheduler.optimizer.param_groups[0]['lr']
            
            if old_lr != new_lr and self.verbose:
                logger.info(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")
        
        return False


class MetricsLogger(Callback):
    """Log metrics to file during training."""
    
    def __init__(self, log_file: str, verbose: bool = True):
        """
        Initialize metrics logger callback.
        
        Args:
            log_file: Path to log file
            verbose: Whether to print messages
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("epoch,timestamp")
    
    def on_epoch_end(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> bool:
        """Log metrics to file."""
        import time
        timestamp = time.time()
        
        # Prepare log entry
        log_entry = f"{epoch},{timestamp}"
        for key, value in metrics.items():
            log_entry += f",{value}"
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(f"\n{log_entry}")
        
        if self.verbose and epoch % 10 == 0:
            logger.info(f"Metrics logged to {self.log_file}")
        
        return False


class GradientClipping(Callback):
    """Gradient clipping callback."""
    
    def __init__(self, max_norm: float, norm_type: float = 2.0, verbose: bool = False):
        """
        Initialize gradient clipping callback.
        
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm to use
            verbose: Whether to print clipping information
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.verbose = verbose
        self.clip_count = 0
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return the gradient norm."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )
        
        if grad_norm > self.max_norm:
            self.clip_count += 1
            if self.verbose:
                logger.info(f"Gradients clipped: norm {grad_norm:.4f} -> {self.max_norm}")
        
        return grad_norm.item()


class WarmupScheduler(Callback):
    """Learning rate warmup callback."""
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        target_lr: float,
        warmup_strategy: str = 'linear',
        verbose: bool = True
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for warmup
            target_lr: Target learning rate after warmup
            warmup_strategy: 'linear' or 'exponential'
            verbose: Whether to print messages
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.warmup_strategy = warmup_strategy
        self.verbose = verbose
        
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def on_epoch_end(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> bool:
        """Update learning rate during warmup period."""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            if self.warmup_strategy == 'linear':
                lr = self.initial_lr + (self.target_lr - self.initial_lr) * epoch / self.warmup_epochs
            elif self.warmup_strategy == 'exponential':
                lr = self.initial_lr * (self.target_lr / self.initial_lr) ** (epoch / self.warmup_epochs)
            else:
                raise ValueError(f"Unknown warmup strategy: {self.warmup_strategy}")
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            if self.verbose:
                logger.info(f"Warmup LR: {lr:.6f} (epoch {epoch}/{self.warmup_epochs})")
        
        return False


class CallbackManager:
    """Manages multiple callbacks during training."""
    
    def __init__(self):
        """Initialize callback manager."""
        self.callbacks: Dict[str, Callback] = {}
    
    def add_callback(self, name: str, callback: Callback):
        """Add a callback."""
        self.callbacks[name] = callback
    
    def remove_callback(self, name: str):
        """Remove a callback."""
        if name in self.callbacks:
            del self.callbacks[name]
    
    def on_training_start(self, model: nn.Module):
        """Call on_training_start for all callbacks."""
        for callback in self.callbacks.values():
            callback.on_training_start(model)
    
    def on_training_end(self, model: nn.Module):
        """Call on_training_end for all callbacks."""
        for callback in self.callbacks.values():
            callback.on_training_end(model)
    
    def on_epoch_end(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> bool:
        """
        Call on_epoch_end for all callbacks.
        
        Returns:
            True if any callback requests stopping training
        """
        should_stop = False
        
        for name, callback in self.callbacks.items():
            try:
                stop_signal = callback.on_epoch_end(epoch, model, metrics)
                if stop_signal:
                    logger.info(f"Training stop requested by callback: {name}")
                    should_stop = True
            except Exception as e:
                logger.error(f"Error in callback {name}: {e}")
                # Continue with other callbacks
        
        return should_stop


# Utility functions for creating common callback combinations
def create_standard_callbacks(
    checkpoint_dir: str,
    patience: int = 15,
    min_delta: float = 1e-4,
    verbose: bool = True
) -> CallbackManager:
    """
    Create a standard set of callbacks for training.
    
    Args:
        checkpoint_dir: Directory for saving checkpoints
        patience: Early stopping patience
        min_delta: Minimum delta for early stopping
        verbose: Whether callbacks should be verbose
        
    Returns:
        CallbackManager with standard callbacks
    """
    manager = CallbackManager()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode='min',
        verbose=verbose
    )
    manager.add_callback('early_stopping', early_stopping)
    
    # Model checkpointing
    checkpoint = ModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        save_best_only=True,
        mode='min',
        verbose=verbose
    )
    manager.add_callback('model_checkpoint', checkpoint)
    
    return manager


def create_advanced_callbacks(
    checkpoint_dir: str,
    log_file: str,
    scheduler = None,
    patience: int = 15,
    warmup_epochs: int = 0,
    target_lr: float = None,
    verbose: bool = True
) -> CallbackManager:
    """
    Create an advanced set of callbacks for training.
    
    Args:
        checkpoint_dir: Directory for saving checkpoints
        log_file: Path to metrics log file
        scheduler: Learning rate scheduler
        patience: Early stopping patience
        warmup_epochs: Number of warmup epochs
        target_lr: Target learning rate for warmup
        verbose: Whether callbacks should be verbose
        
    Returns:
        CallbackManager with advanced callbacks
    """
    manager = create_standard_callbacks(checkpoint_dir, patience, verbose=verbose)
    
    # Metrics logging
    metrics_logger = MetricsLogger(log_file, verbose=verbose)
    manager.add_callback('metrics_logger', metrics_logger)
    
    # Learning rate scheduling
    if scheduler:
        lr_scheduler = LearningRateScheduler(scheduler, verbose=verbose)
        manager.add_callback('lr_scheduler', lr_scheduler)
    
    # Warmup scheduling
    if warmup_epochs > 0 and target_lr:
        # Note: This would need the optimizer to be passed
        # warmup = WarmupScheduler(optimizer, warmup_epochs, target_lr, verbose=verbose)
        # manager.add_callback('warmup', warmup)
        pass
    
    return manager 
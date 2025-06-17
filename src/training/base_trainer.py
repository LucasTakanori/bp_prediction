"""
Base trainer class that provides common functionality for all trainers.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import logging
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseTrainer:
    """Base trainer class that provides common functionality for all trainers."""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, config, device: torch.device, experiment_dir: Path):
        """Initialize base trainer.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            device: Device to use for training
            experiment_dir: Directory to save experiment artifacts
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        
        # Setup logging
        self.logger = get_logger(self.__class__.__name__)
        
        # Create experiment directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.experiment_dir / 'reconstructions').mkdir(exist_ok=True)
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, and other training components.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement setup_training_components()")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        Must be implemented by subclasses.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        raise NotImplementedError("Subclasses must implement train_epoch()")
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch.
        Must be implemented by subclasses.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        raise NotImplementedError("Subclasses must implement validate_epoch()")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Handle end of epoch.
        Must be implemented by subclasses.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics from training and validation
            
        Returns:
            True if training should stop, False otherwise
        """
        raise NotImplementedError("Subclasses must implement on_epoch_end()")
    
    def on_training_end(self):
        """Handle end of training.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement on_training_end()")
    
    def train(self):
        """Main training loop.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement train()") 
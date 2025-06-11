"""
Training module for blood pressure prediction.
Provides trainers, metrics, and callbacks for model training.
"""

from .trainers import BaseTrainer, VAETrainer, BiLSTMTrainer, create_trainer
from .metrics import MetricsCalculator, compute_batch_metrics, format_metrics_for_logging
from .callbacks import (
    Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler,
    MetricsLogger, GradientClipping, WarmupScheduler, CallbackManager,
    create_standard_callbacks, create_advanced_callbacks
)

__all__ = [
    # Trainers
    'BaseTrainer', 'VAETrainer', 'BiLSTMTrainer', 'create_trainer',
    
    # Metrics
    'MetricsCalculator', 'compute_batch_metrics', 'format_metrics_for_logging',
    
    # Callbacks
    'Callback', 'EarlyStopping', 'ModelCheckpoint', 'LearningRateScheduler',
    'MetricsLogger', 'GradientClipping', 'WarmupScheduler', 'CallbackManager',
    'create_standard_callbacks', 'create_advanced_callbacks'
] 
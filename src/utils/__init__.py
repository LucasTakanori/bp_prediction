"""
Utility functions and configurations for BP prediction project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import (
    DataConfig, VAEConfig, BiLSTMConfig, LossConfig, TrainingConfig, 
    Config, ModelConfig, create_default_config, load_config, create_config_from_yaml
)

def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional file to log to
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

__all__ = [
    "DataConfig", "VAEConfig", "BiLSTMConfig", "LossConfig", "TrainingConfig",
    "Config", "ModelConfig", "create_default_config", "load_config", "create_config_from_yaml",
    "setup_logging"
] 
"""
Logging utilities for blood pressure prediction project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional file to log to
        format_string: Custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string, '%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string, '%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name) 
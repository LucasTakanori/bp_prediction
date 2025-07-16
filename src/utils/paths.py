"""
Centralized path management for BP prediction project.
Handles path validation, environment variables, and cross-platform compatibility.
"""

import os
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PathConfig:
    """Configuration for path management"""
    data_root: str = None
    experiments_root: str = "./experiments"
    cache_dir: str = "./cache"
    logs_dir: str = "./logs"
    
    def __post_init__(self):
        # Use environment variables with fallbacks to the default string values
        self.data_root = self.data_root or os.getenv('BP_DATA_ROOT') or '/home/lucas_takanori/phd/data'
        self.experiments_root = self.experiments_root or os.getenv('BP_EXPERIMENTS_ROOT') or './experiments'
        self.cache_dir = self.cache_dir or os.getenv('BP_CACHE_DIR') or './cache'
        self.logs_dir = self.logs_dir or os.getenv('BP_LOGS_DIR') or './logs'


class PathManager:
    """Centralized path management with validation and environment support"""
    
    def __init__(self, config: Optional[PathConfig] = None):
        self.config = config or PathConfig()
        self._validate_and_create_paths()
        
    def _validate_and_create_paths(self):
        """Validate required paths exist and create output directories"""
        # Validate data root exists
        if not Path(self.config.data_root).exists():
            raise FileNotFoundError(f"Data root not found: {self.config.data_root}")
        
        # Create output directories if they don't exist
        for dir_path in [self.config.experiments_root, self.config.cache_dir, self.config.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Path validation completed:")
        logger.info(f"  Data root: {self.config.data_root}")
        logger.info(f"  Experiments: {self.config.experiments_root}")
        logger.info(f"  Cache: {self.config.cache_dir}")
        logger.info(f"  Logs: {self.config.logs_dir}")
    
    @property
    def data_root(self) -> Path:
        """Get data root path"""
        return Path(self.config.data_root)
    
    @property
    def experiments_root(self) -> Path:
        """Get experiments root path"""
        return Path(self.config.experiments_root)
    
    @property
    def cache_dir(self) -> Path:
        """Get cache directory path"""
        return Path(self.config.cache_dir)
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory path"""
        return Path(self.config.logs_dir)
    
    def get_data_file_path(self, subject: str, session: str) -> Path:
        """Get path to data file for specific subject and session"""
        filename = f"{subject}_{session}_masked.h5"
        return self.data_root / filename
    
    def create_experiment_dir(self, experiment_name: str) -> Path:
        """Create and return experiment directory"""
        exp_dir = self.experiments_root / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)
        (exp_dir / "configs").mkdir(exist_ok=True)
        
        return exp_dir
    
    def get_checkpoint_path(self, experiment_name: str, checkpoint_name: str = "best_model.pth") -> Path:
        """Get path to checkpoint file"""
        return self.experiments_root / experiment_name / "checkpoints" / checkpoint_name
    
    def validate_data_file(self, subject: str, session: str) -> bool:
        """Validate that data file exists for given subject and session"""
        data_file = self.get_data_file_path(subject, session)
        exists = data_file.exists()
        
        if not exists:
            logger.warning(f"Data file not found: {data_file}")
        else:
            logger.info(f"Data file validated: {data_file}")
            
        return exists
    
    def list_available_subjects(self) -> list:
        """List all available subjects in data directory"""
        subjects = []
        for file_path in self.data_root.glob("subject*_*.h5"):
            subject = file_path.stem.split('_')[0]
            if subject not in subjects:
                subjects.append(subject)
        return sorted(subjects)
    
    def list_available_sessions(self, subject: str) -> list:
        """List all available sessions for a subject"""
        sessions = []
        pattern = f"{subject}_*_masked.h5"
        for file_path in self.data_root.glob(pattern):
            parts = file_path.stem.split('_')
            if len(parts) >= 3:
                session = '_'.join(parts[1:-1])  # Handle multi-word sessions
                if session not in sessions:
                    sessions.append(session)
        return sorted(sessions)
    
    def get_relative_path(self, absolute_path: Union[str, Path], base: Optional[Path] = None) -> Path:
        """Convert absolute path to relative path from base directory"""
        if base is None:
            base = Path.cwd()
        
        try:
            return Path(absolute_path).relative_to(base)
        except ValueError:
            # If paths are on different drives or not related, return as-is
            return Path(absolute_path)
    
    def ensure_path_exists(self, path: Union[str, Path], create_parents: bool = True) -> Path:
        """Ensure path exists, create if necessary"""
        path = Path(path)
        
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        return path


# Global path manager instance
_path_manager: Optional[PathManager] = None


def get_path_manager(config: Optional[PathConfig] = None) -> PathManager:
    """Get or create global path manager instance"""
    global _path_manager
    
    if _path_manager is None or config is not None:
        _path_manager = PathManager(config)
    
    return _path_manager


def setup_paths(config: Optional[PathConfig] = None) -> PathManager:
    """Setup and return path manager"""
    return get_path_manager(config) 
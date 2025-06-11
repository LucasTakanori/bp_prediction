"""
Configuration management for BP prediction project.
Handles paths, model parameters, and training configurations.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass, asdict
import torch


@dataclass
class DataConfig:
    """Data-related configuration"""
    root_path: str = os.getenv('BP_DATA_ROOT', '/gpfs/projects/bsc88/speech/research/data')
    subject: str = "subject001"
    session: str = "baseline"
    cache_dir: Optional[str] = None
    preload_to_memory: bool = False
    use_mmap: bool = True
    force_reload: bool = False
    
    # Sequence parameters
    sequence_length: int = 3
    pattern_offsets: List[int] = None
    
    # Normalization parameters
    bp_normalization: Tuple[float, float] = (40.0, 200.0)
    frame_normalization: str = "minmax"  # "minmax", "standardize", "none"
    
    # Data loading options
    return_metadata: bool = True
    
    def __post_init__(self):
        if self.pattern_offsets is None:
            self.pattern_offsets = [-7, 0, 3]
    
    @property
    def data_file(self) -> str:
        return f"{self.subject}_{self.session}_masked.h5"
    
    @property
    def full_path(self) -> Path:
        return Path(self.root_path) / self.data_file


@dataclass
class VAEConfig:
    """VAE model configuration"""
    latent_dim: int = 256
    input_channels: int = 1
    input_height: int = 32
    input_width: int = 32
    dropout_rate: float = 0.2
    
    # Training parameters
    num_epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-4
    beta_min: float = 0.01
    beta_max: float = 1.0
    beta_warmup_epochs: int = 15
    
    # Architecture parameters
    encoder_channels: list = None
    decoder_channels: list = None
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [64, 128, 256, 512]
        if self.decoder_channels is None:
            self.decoder_channels = [512, 256, 128, 64]


@dataclass
class BiLSTMConfig:
    """BiLSTM model configuration"""
    input_dim: int = 256  # Should match VAE latent_dim
    hidden_dim: int = 256
    num_layers: int = 3
    output_dim: int = 50
    dropout_rate: float = 0.3
    bidirectional: bool = True
    
    # Attention mechanism
    use_attention: bool = True
    attention_dim: int = 128
    num_attention_heads: int = 8
    
    # Training parameters
    num_epochs: int = 25
    batch_size: int = 8
    learning_rate: float = 1e-4
    
    # Sequence parameters
    sequence_length: int = 3
    pattern_offsets: list = None
    
    def __post_init__(self):
        if self.pattern_offsets is None:
            self.pattern_offsets = [-7, 0, 3]


@dataclass
class LossConfig:
    """Loss function configuration"""
    loss_type: str = "composite"  # mse, systolic_distance, diastolic_distance, composite
    
    # Composite loss weights
    systolic_weight: float = 1.0
    diastolic_weight: float = 1.0
    shape_weight: float = 0.5
    temporal_weight: float = 0.3
    
    # BP normalization parameters
    bp_min: float = 40.0
    bp_max: float = 200.0


@dataclass
class TrainingConfig:
    """Training configuration with wandb logging."""
    num_epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "adam"  # adam, adamw, sgd
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-5
    
    # VAE specific
    vae_beta: float = 1.0
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, reduce_on_plateau, step
    scheduler_patience: int = 10  # for reduce_on_plateau
    scheduler_step_size: int = 20  # for step
    
    # Mixed precision and optimization
    use_mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    checkpoint_dir: Optional[Path] = None
    save_every_n_epochs: int = 5
    
    # Wandb logging
    use_wandb: bool = True
    project_name: str = "bp_prediction"
    wandb_mode: str = "offline"  # online, offline, disabled
    
    # Output
    output_dir: Path = Path("experiments")
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert string paths to Path objects
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)


@dataclass
class ModelConfig:
    """Model configuration that can represent different model types."""
    model_type: str = "vae"  # vae, bilstm, vae_bilstm
    
    # VAE parameters
    latent_dim: int = 128
    input_channels: int = 1
    input_height: int = 64
    input_width: int = 64
    hidden_dims: list = None
    activation: str = "relu"
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    
    # BiLSTM parameters  
    hidden_dim: int = 256
    num_layers: int = 3
    bidirectional: bool = True
    use_attention: bool = True
    attention_dim: int = 128
    num_attention_heads: int = 8
    output_dim: int = 50
    use_residual: bool = True
    
    # Feature extractor parameters
    feature_extractor_type: str = "cnn"
    cnn_channels: list = None
    cnn_kernel_sizes: list = None
    cnn_dropout: float = 0.2
    
    # VAE checkpoint path for BiLSTM models
    vae_checkpoint_path: Optional[str] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256]
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3]


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig
    vae: VAEConfig
    bilstm: BiLSTMConfig
    loss: LossConfig
    training: TrainingConfig
    
    output_dir: str = "./experiments"
    experiment_name: str = "bp_prediction"
    
    def __post_init__(self):
        # Ensure VAE and BiLSTM latent dimensions match
        if self.vae.latent_dim != self.bilstm.input_dim:
            self.bilstm.input_dim = self.vae.latent_dim
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'Config':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            vae=VAEConfig(**config_dict.get('vae', {})),
            bilstm=BiLSTMConfig(**config_dict.get('bilstm', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            output_dir=config_dict.get('output_dir', './experiments'),
            experiment_name=config_dict.get('experiment_name', 'bp_prediction')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'data': asdict(self.data),
            'vae': asdict(self.vae),
            'bilstm': asdict(self.bilstm),
            'loss': asdict(self.loss),
            'training': asdict(self.training),
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name
        }
    
    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for training"""
        if self.training.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.training.device)


def create_default_config() -> Config:
    """Create a default configuration"""
    return Config(
        data=DataConfig(),
        vae=VAEConfig(),
        bilstm=BiLSTMConfig(),
        loss=LossConfig(),
        training=TrainingConfig()
    )


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file or create default
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
    
    Returns:
        Configuration object
    """
    if config_path is None:
        return create_default_config()
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
        return Config.from_yaml(config_path)
    elif config_path.suffix.lower() == '.json':
        return Config.from_json(config_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def create_config_from_yaml(yaml_path: Union[str, Path]) -> Tuple[DataConfig, ModelConfig, TrainingConfig]:
    """
    Create separate config objects from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Tuple of (DataConfig, ModelConfig, TrainingConfig)
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create individual config objects
    data_config = DataConfig(**config_dict.get('data_config', {}))
    model_config = ModelConfig(**config_dict.get('model_config', {}))
    training_config = TrainingConfig(**config_dict.get('training_config', {}))
    
    return data_config, model_config, training_config 
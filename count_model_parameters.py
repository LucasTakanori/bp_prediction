#!/usr/bin/env python3
"""
Count VAE Model Parameters
Using the same model and configuration as in the training scripts
"""

import os
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import create_config_from_yaml
from train.tuned_vae import VAE


def count_model_parameters(model):
    """Count and display model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print("=" * 60)
    print("🧠 VAE MODEL PARAMETER COUNT")
    print("=" * 60)
    print(f"📊 Total parameters:      {total_params:,}")
    print(f"🎯 Trainable parameters:  {trainable_params:,}")
    print(f"🔒 Non-trainable params:  {non_trainable_params:,}")
    print("=" * 60)
    
    # Memory usage estimation (assuming float32)
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"💾 Estimated memory:      {memory_mb:.2f} MB")
    print("=" * 60)
    
    return total_params, trainable_params


def print_layer_parameters(model):
    """Print parameters for each layer"""
    print("\n📋 LAYER-BY-LAYER PARAMETER COUNT:")
    print("-" * 60)
    
    total = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total += param_count
        trainable = "✅" if param.requires_grad else "❌"
        print(f"{trainable} {name:30s} {param_count:>10,} params {str(param.shape):>15s}")
    
    print("-" * 60)
    print(f"{'TOTAL':>35s} {total:>10,} params")
    print("-" * 60)


def main():
    """Main function to count model parameters"""
    # Load configuration
    config_path = "configs/vae_multisubject_fixed.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    print(f"📁 Loading configuration from: {config_path}")
    data_config, model_config, training_config = create_config_from_yaml(config_path)
    
    print(f"🔧 Model configuration:")
    print(f"   - Latent dimension: {model_config.latent_dim}")
    print(f"   - Input channels: {getattr(model_config, 'input_channels', 1)}")
    print(f"   - Hidden dimensions: {getattr(model_config, 'hidden_dims', 'Not specified')}")
    
    # Create model (same as in training script)
    print(f"\n🏗️  Creating VAE model...")
    model = VAE(latent_dim=model_config.latent_dim)
    
    # Count parameters
    total_params, trainable_params = count_model_parameters(model)
    
    # Print detailed layer information
    print_layer_parameters(model)
    
    # Model architecture summary
    print("\n🏗️  MODEL ARCHITECTURE SUMMARY:")
    print("-" * 60)
    print("Encoder:")
    print("  - Conv2d layers: 4 (64→128→256→512 channels)")
    print("  - BatchNorm2d layers: 4")
    print("  - Fully connected: 2 (mu & logvar)")
    print("\nDecoder:")
    print("  - Fully connected: 1")
    print("  - ConvTranspose2d layers: 4 (512→256→128→64→1)")
    print("  - BatchNorm layers: 4 (3 Conv + 1 FC)")
    print("  - Dropout layers: 2")
    print("-" * 60)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Sophisticated BP Predictor - Config-Based Training Script
Simple interface to run training using YAML configuration files.

Usage:
    python scripts/train_with_config.py --config configs/sophisticated_bp_predictor.yaml
    python scripts/train_with_config.py --config configs/sophisticated_bp_predictor.yaml --override training_config.num_epochs=50
"""

import sys
import os
import argparse
import yaml
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_config(config_path):
    """Load YAML configuration file with environment variable substitution"""
    with open(config_path, 'r') as file:
        config_str = file.read()
    
    # Simple environment variable substitution
    import re
    def replace_env_vars(match):
        var_expr = match.group(1)
        if ':' in var_expr:
            var_name, default_value = var_expr.split(':', 1)
            return os.getenv(var_name, default_value)
        else:
            return os.getenv(var_expr, '')
    
    config_str = re.sub(r'\$\{([^}]+)\}', replace_env_vars, config_str)
    
    # Load YAML
    config = yaml.safe_load(config_str)
    return config

def override_config(config, overrides):
    """Override configuration values using dot notation"""
    for override in overrides:
        if '=' not in override:
            print(f"Warning: Invalid override format: {override}")
            continue
            
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Navigate to the parent dictionary
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value (attempt to convert to appropriate type)
        final_key = keys[-1]
        try:
            # Try to evaluate as Python literal (int, float, bool, list, etc.)
            current[final_key] = eval(value)
        except:
            # If that fails, keep as string
            current[final_key] = value
        
        print(f"Override: {key_path} = {current[final_key]}")

def convert_config_to_args(config):
    """Convert config dictionary to arguments compatible with bilstm.py"""
    args = argparse.Namespace()
    
    # Data configuration
    data_config = config.get('data_config', {})
    if 'data_file' in data_config:
        data_path = os.path.join(data_config.get('root_path', ''), data_config['data_file'])
    else:
        # Fallback to existing path format
        data_path = os.path.join(data_config.get('root_path', '/home/lucas_takanori/phd/data'), 
                                'subject001_baseline_masked.h5')
    args.data_path = data_path
    
    # Model configuration
    model_config = config.get('model_config', {})
    vae_config = model_config.get('vae_config', {})
    bilstm_config = model_config.get('bilstm_config', {})
    attention_config = model_config.get('attention_config', {})
    
    args.latent_dim = vae_config.get('latent_dim', 128)
    args.vae_checkpoint = vae_config.get('vae_checkpoint_path')
    args.lstm_hidden_dim = bilstm_config.get('hidden_dim', 256)
    args.lstm_layers = bilstm_config.get('num_layers', 3)
    args.use_attention = attention_config.get('use_attention', True)
    args.attention_dim = attention_config.get('attention_dim', 128)
    
    # Training configuration
    training_config = config.get('training_config', {})
    args.num_epochs = training_config.get('num_epochs', 25)
    args.batch_size = training_config.get('batch_size', 16)
    args.learning_rate = training_config.get('learning_rate', 1e-3)
    
    # Loss configuration
    loss_config = config.get('loss_config', {})
    args.loss_type = loss_config.get('loss_type', 'composite')
    
    # Output configuration
    output_config = config.get('output_config', {})
    args.output_dir = output_config.get('output_dir', './experiments/sophisticated_bp_predictor')
    
    # Logging configuration
    logging_config = config.get('logging_config', {})
    visualization = logging_config.get('visualization', {})
    args.visualize_attention = visualization.get('visualize_attention', True)
    
    return args

def main():
    parser = argparse.ArgumentParser(description='Train Sophisticated BP Predictor with Config File')
    parser.add_argument('--config', '-c', required=True, help='Path to configuration YAML file')
    parser.add_argument('--override', '-o', action='append', default=[], 
                       help='Override config values (e.g., training_config.num_epochs=50)')
    parser.add_argument('--dry-run', action='store_true', help='Print configuration and exit')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Apply overrides
    if args.override:
        print(f"Applying {len(args.override)} override(s):")
        override_config(config, args.override)
    
    # Convert config to training arguments
    training_args = convert_config_to_args(config)
    
    if args.verbose or args.dry_run:
        print("\n=== Configuration Summary ===")
        print(f"Data Path: {training_args.data_path}")
        print(f"VAE Checkpoint: {training_args.vae_checkpoint}")
        print(f"Output Directory: {training_args.output_dir}")
        print(f"Epochs: {training_args.num_epochs}")
        print(f"Batch Size: {training_args.batch_size}")
        print(f"Learning Rate: {training_args.learning_rate}")
        print(f"LSTM Hidden Dim: {training_args.lstm_hidden_dim}")
        print(f"LSTM Layers: {training_args.lstm_layers}")
        print(f"Latent Dim: {training_args.latent_dim}")
        print(f"Use Attention: {training_args.use_attention}")
        print(f"Attention Dim: {training_args.attention_dim}")
        print(f"Loss Type: {training_args.loss_type}")
        print(f"Visualize Attention: {training_args.visualize_attention}")
        print("=" * 30)
    
    if args.dry_run:
        print("Dry run mode - exiting without training")
        return
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Save the configuration for reproducibility
    config_save_path = os.path.join(training_args.output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"Configuration saved to: {config_save_path}")
    
    # Import and run the training script
    print("\n=== Starting Training ===")
    try:
        # Import the training script
        sys.path.append(str(project_root / 'train'))
        from bilstm import main as train_main
        
        # Override sys.argv to pass arguments to the training script
        original_argv = sys.argv.copy()
        sys.argv = [
            'bilstm.py',
            '--data_path', training_args.data_path,
            '--output_dir', training_args.output_dir,
            '--batch_size', str(training_args.batch_size),
            '--num_epochs', str(training_args.num_epochs),
            '--lstm_hidden_dim', str(training_args.lstm_hidden_dim),
            '--lstm_layers', str(training_args.lstm_layers),
            '--latent_dim', str(training_args.latent_dim),
            '--vae_checkpoint', training_args.vae_checkpoint,
            '--attention_dim', str(training_args.attention_dim),
            '--loss_type', training_args.loss_type
        ]
        
        if training_args.use_attention:
            sys.argv.append('--use_attention')
        if training_args.visualize_attention:
            sys.argv.append('--visualize_attention')
        
        # Run training
        train_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print("\n=== Training Completed Successfully ===")
        
    except Exception as e:
        print(f"\n=== Training Failed ===")
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
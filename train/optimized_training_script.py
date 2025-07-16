#!/usr/bin/env python3
"""
Complete Optimized Training Script for BiLSTM Blood Pressure Prediction

This script incorporates all performance optimizations and best practices.
Run this instead of your original bilstm.py for maximum performance.

Usage:
    python optimized_training_script.py --data_path /path/to/data.h5 --vae_checkpoint /path/to/vae.pt
"""

import torch
import os
import sys
import argparse
import time
import psutil
import numpy as np
from pathlib import Path
import multiprocessing as mp

# Import the optimized modules
# Make sure these files are in the same directory or adjust the import paths
try:
    from completed_data_utils import PviDataset, PviBatchServer, optimize_system_settings, print_system_info
    from optimized_bilstm_complete import (
        VAE, VAEBiLSTMWithAttention, train_enhanced_bilstm, 
        evaluate_enhanced_bilstm, BPLossFunction
    )
    print("✓ Successfully imported optimized modules")
except ImportError as e:
    print(f"❌ Failed to import optimized modules: {e}")
    print("Make sure the optimized files are in the same directory")
    sys.exit(1)

import wandb


def determine_optimal_batch_size(device):
    """Determine optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return 4  # Conservative for CPU
    
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Batch size recommendations based on GPU memory
        if gpu_memory_gb >= 24:    # RTX 4090, A100, etc.
            return 32
        elif gpu_memory_gb >= 16:  # RTX 4080, V100, etc.
            return 24
        elif gpu_memory_gb >= 12:  # RTX 4070 Ti, RTX 3080 Ti, etc.
            return 16
        elif gpu_memory_gb >= 8:   # RTX 4070, RTX 3070, etc.
            return 12
        elif gpu_memory_gb >= 6:   # RTX 4060, RTX 3060, etc.
            return 8
        else:                      # Lower-end GPUs
            return 4
            
    except Exception:
        return 8  # Default fallback


def determine_optimal_workers():
    """Determine optimal number of workers based on system specs"""
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Conservative worker allocation to avoid memory issues
    if memory_gb >= 64:
        return min(cpu_count, 12)
    elif memory_gb >= 32:
        return min(cpu_count, 8)
    elif memory_gb >= 16:
        return min(cpu_count, 6)
    elif memory_gb >= 8:
        return min(cpu_count, 4)
    else:
        return min(cpu_count, 2)


def setup_mixed_precision_training():
    """Setup mixed precision training if available"""
    if torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast'):
        print("✓ Mixed precision training available")
        return True
    else:
        print("⚠ Mixed precision training not available")
        return False


def monitor_system_resources():
    """Monitor and log system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"System Status: CPU {cpu_percent:.1f}%, RAM {memory.percent:.1f}% ({memory.available/(1024**3):.1f}GB free)")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        print(f"GPU Memory: {gpu_memory:.1f}% allocated")
        
        # Clear cache if memory usage is high
        if gpu_memory > 90:
            torch.cuda.empty_cache()
            print("🧹 Cleared GPU cache due to high memory usage")


def create_optimized_model(vae_checkpoint_path, device, config):
    """Create and optimize the model"""
    print("Creating optimized model...")
    
    # Load VAE
    vae_model = VAE(latent_dim=config['latent_dim']).to(device)
    
    if os.path.exists(vae_checkpoint_path):
        try:
            checkpoint = torch.load(vae_checkpoint_path, map_location=device)
            vae_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ VAE loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        except Exception as e:
            print(f"⚠ Error loading VAE: {e}")
            print("Continuing with randomly initialized VAE")
    else:
        print(f"⚠ VAE checkpoint not found at {vae_checkpoint_path}")
        print("Continuing with randomly initialized VAE")
    
    vae_model.eval()
    
    # Create optimized BiLSTM model
    model = VAEBiLSTMWithAttention(
        vae_model=vae_model,
        input_dim=config['latent_dim'],
        hidden_dim=config['lstm_hidden_dim'],
        num_layers=config['lstm_layers'],
        output_dim=50,  # BP signal length
        dropout=config['dropout'],
        use_attention=config['use_attention'],
        attention_dim=config['attention_dim'],
        vae_batch_size=config['vae_batch_size']
    ).to(device)
    
    # Compile model if available (PyTorch 2.0+) - but keep as conditional
    # Note: torch.compile can cause issues with model methods, so we disable it for now
    if hasattr(torch, 'compile') and False:  # Disabled for compatibility
        try:
            model = torch.compile(model)
            print("✓ Model compiled for faster execution")
        except Exception as e:
            print(f"⚠ Model compilation failed: {e}")
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    return model


def run_performance_benchmark(dataset, config):
    """Run a quick performance benchmark"""
    print("\n" + "="*60)
    print("RUNNING PERFORMANCE BENCHMARK")
    print("="*60)
    
    from optimized_data_utils_complete import benchmark_loading_performance
    
    # Test different configurations
    batch_sizes = [config['batch_size']//2, config['batch_size'], config['batch_size']*2]
    num_workers_list = [0, config['num_workers']//2, config['num_workers']]
    
    # Filter out invalid values
    batch_sizes = [bs for bs in batch_sizes if bs > 0]
    num_workers_list = [nw for nw in num_workers_list if nw >= 0]
    
    results = benchmark_loading_performance(dataset, batch_sizes, num_workers_list)
    
    return results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Optimized BiLSTM Training Script')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the H5 data file')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to VAE checkpoint')
    parser.add_argument('--output_dir', type=str, default='optimized_output',
                        help='Directory for saving output files')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='VAE latent dimension')
    parser.add_argument('--lstm_hidden_dim', type=int, default=256,
                        help='LSTM hidden dimension')
    parser.add_argument('--lstm_layers', type=int, default=3,
                        help='Number of LSTM layers')
    parser.add_argument('--attention_dim', type=int, default=128,
                        help='Attention dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (auto-determined if not specified)')
    parser.add_argument('--vae_batch_size', type=int, default=64,
                        help='VAE processing batch size')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers (auto-determined if not specified)')
    parser.add_argument('--loss_type', type=str, default='composite',
                        choices=['mse', 'systolic_distance', 'diastolic_distance', 'composite'],
                        help='Loss function type')
    
    # Optimization arguments
    parser.add_argument('--disable_caching', action='store_true',
                        help='Disable sequence caching')
    parser.add_argument('--disable_attention', action='store_true',
                        help='Disable attention mechanism')
    parser.add_argument('--disable_mixed_precision', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark before training')
    
    # Monitoring arguments
    parser.add_argument('--wandb_project', type=str, default='optimized-bilstm-bp',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--wandb_mode', type=str, default='offline',
                        choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode')
    
    args = parser.parse_args()
    
    # Print system information
    print_system_info()
    
    # Apply system optimizations
    optimize_system_settings()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")
    
    # Determine optimal parameters if not specified
    if args.batch_size is None:
        args.batch_size = determine_optimal_batch_size(device)
        print(f"✓ Auto-determined batch size: {args.batch_size}")
    
    if args.num_workers is None:
        args.num_workers = determine_optimal_workers()
        print(f"✓ Auto-determined number of workers: {args.num_workers}")
    
    # Setup mixed precision
    use_mixed_precision = not args.disable_mixed_precision and setup_mixed_precision_training()
    
    # Configuration
    config = {
        'latent_dim': args.latent_dim,
        'lstm_hidden_dim': args.lstm_hidden_dim,
        'lstm_layers': args.lstm_layers,
        'attention_dim': args.attention_dim,
        'dropout': args.dropout,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'vae_batch_size': args.vae_batch_size,
        'num_workers': args.num_workers,
        'loss_type': args.loss_type,
        'use_attention': not args.disable_attention,
        'enable_caching': not args.disable_caching,
        'use_mixed_precision': use_mixed_precision,
        'pattern_offsets': [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],
        'bp_norm_params': (40, 200)
    }
    
    # Initialize wandb
    if args.wandb_mode != 'disabled':
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config,
            mode=args.wandb_mode
        )
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)
    (output_dir / 'cache').mkdir(exist_ok=True)
    
    print(f"✓ Output directory: {output_dir}")
    
    # Load dataset with optimizations
    print(f"\n📂 Loading dataset from: {args.data_path}")
    
    try:
        dataset = PviDataset(
            args.data_path,
            device=device,
            cache_dir=str(output_dir / 'cache'),
            preload_to_memory=(device.type == 'cuda' and torch.cuda.get_device_properties(0).total_memory > 8*1024**3),
            use_mmap=True
        )
        print(f"✓ Dataset loaded with {len(dataset)} samples")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_results = run_performance_benchmark(dataset, config)
    
    # Create optimized batch server
    print(f"\n⚙️ Setting up data loaders...")
    
    batch_server = PviBatchServer(
        dataset=dataset,
        input_type="img",
        output_type="full",
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    
    # Split data: 70% train, 15% val, 15% test
    batch_server.set_loader_params(
        batch_size=args.batch_size, 
        test_size=0.3
    )
    
    # Get data loaders
    train_loader, test_val_loader = batch_server.get_loaders()
    
    # Split test_val into validation and test
    test_val_batches = list(test_val_loader)
    val_size = len(test_val_batches) // 2  # 50-50 split
    val_batches = test_val_batches[:val_size]
    test_batches = test_val_batches[val_size:]
    
    print(f"✓ Data split - Train: {len(train_loader)} batches, Val: {len(val_batches)} batches, Test: {len(test_batches)} batches")
    
    # Create model
    model = create_optimized_model(args.vae_checkpoint, device, config)
    
    # Monitor initial system state
    monitor_system_resources()
    
    # Training
    print(f"\n🚂 Starting optimized training for {args.num_epochs} epochs...")
    
    training_start_time = time.time()
    
    try:
        best_epoch, best_val_loss = train_enhanced_bilstm(
            model=model,
            train_loader=train_loader,
            val_batches=val_batches,
            device=device,
            num_epochs=args.num_epochs,
            pattern_offsets=config['pattern_offsets'],
            bp_norm_params=config['bp_norm_params'],
            loss_type=args.loss_type,
            output_dir=str(output_dir),
            enable_caching=config['enable_caching'],
            cache_dir=str(output_dir / 'cache')
        )
        
        training_time = time.time() - training_start_time
        print(f"\n🏁 Training completed in {training_time/60:.1f} minutes!")
        print(f"Best model: Epoch {best_epoch}, Validation Loss: {best_val_loss:.6f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluation
    print(f"\n📊 Evaluating model...")
    
    # Create test loader
    class BatchListDataLoader:
        def __init__(self, batch_list):
            self.batch_list = batch_list
        def __iter__(self):
            return iter(self.batch_list)
        def __len__(self):
            return len(self.batch_list)
    
    test_loader = BatchListDataLoader(test_batches)
    
    # Load best model
    best_checkpoint_path = output_dir / 'checkpoints' / 'enhanced_bilstm_best.pt'
    if best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print("✓ Loaded best model for evaluation")
    
    # Run evaluation
    try:
        eval_results = evaluate_enhanced_bilstm(
            model=model,
            test_loader=test_loader,
            device=device,
            pattern_offsets=config['pattern_offsets'],
            bp_norm_params=config['bp_norm_params'],
            output_dir=str(output_dir),
            visualize_attention=config['use_attention']
        )
        
        test_loss, mse, mae, r2, sys_mae, dias_mae, sys_error, dias_error = eval_results
        
        if mse is not None:
            print(f"\n📈 Final Results:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R²:  {r2:.6f}")
            print(f"  Systolic MAE: {sys_mae:.2f} mmHg")
            print(f"  Diastolic MAE: {dias_mae:.2f} mmHg")
            
            # Log final results to wandb
            if args.wandb_mode != 'disabled':
                wandb.run.summary.update({
                    "final_mse": mse,
                    "final_mae": mae,
                    "final_r2": r2,
                    "systolic_mae": sys_mae,
                    "diastolic_mae": dias_mae,
                    "training_time_minutes": training_time/60
                })
        
    except Exception as e:
        print(f"⚠ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Final system monitoring
    print(f"\n💻 Final system status:")
    monitor_system_resources()
    
    # Save final configuration and results
    results_summary = {
        'config': config,
        'training_time_minutes': training_time/60,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'system_info': {
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': mp.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
    }
    
    if torch.cuda.is_available():
        results_summary['system_info']['gpu_name'] = torch.cuda.get_device_name(0)
        results_summary['system_info']['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Save summary
    import json
    with open(output_dir / 'results' / 'training_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"📊 Results summary: {output_dir / 'results' / 'training_summary.json'}")
    
    # Finish wandb
    if args.wandb_mode != 'disabled':
        wandb.finish()


if __name__ == "__main__":
    main()

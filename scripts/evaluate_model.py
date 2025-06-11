#!/usr/bin/env python3
"""
Evaluation script for blood pressure prediction models.
Loads trained models and computes comprehensive evaluation metrics.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config import create_config_from_yaml
from src.data.loaders import create_data_loaders
from src.models import create_model
from src.training.metrics import MetricsCalculator
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate BP prediction model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str,
                       help="Path to configuration file (if not in checkpoint)")
    parser.add_argument("--data-dir", type=str,
                       help="Override data directory")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                       help="Data split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--save-predictions", action="store_true",
                       help="Save predictions to file")
    parser.add_argument("--create-plots", action="store_true",
                       help="Create evaluation plots")
    
    return parser.parse_args()


def load_model_and_config(checkpoint_path: Path, config_path: Path = None):
    """Load model and configuration from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load configuration
    if config_path:
        configs = create_config_from_yaml(config_path)
    elif 'config' in checkpoint:
        # Try to load from checkpoint
        configs = checkpoint['config']
    else:
        raise ValueError("No configuration found. Please provide --config argument.")
    
    data_config, model_config, training_config = configs
    
    # Create model
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Best metric: {checkpoint.get('best_metric', 'unknown')}")
    
    return model, data_config, model_config, training_config


def evaluate_model(model, data_loader, device, metrics_calculator):
    """Evaluate model on data loader."""
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    all_sample_ids = []
    
    logger.info("Running evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Handle different model types
            if 'sequence' in batch:
                # BiLSTM input
                x = batch['sequence'].to(device)
                y = batch['target'].to(device)
            else:
                # VAE input
                x = batch['pviHP'].to(device)
                y = batch['minmax'].to(device)
            
            # Forward pass
            if hasattr(model, 'forward'):
                if 'VAEBiLSTM' in str(type(model)):
                    output, _ = model(x)  # VAE-BiLSTM returns (output, attention)
                elif 'VAE' in str(type(model)):
                    recon_x, _, _ = model(x)  # VAE returns (recon, mu, logvar)
                    output = recon_x
                else:
                    output = model(x)
            else:
                output = model(x)
            
            # Store results
            all_predictions.append(output.cpu())
            all_targets.append(y.cpu())
            
            if 'sample_id' in batch:
                all_sample_ids.extend(batch['sample_id'])
    
    # Concatenate all results
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = metrics_calculator.compute_comprehensive_metrics(predictions, targets)
    error_stats = metrics_calculator.compute_error_statistics(predictions, targets)
    
    return {
        'predictions': predictions.numpy(),
        'targets': targets.numpy(),
        'sample_ids': all_sample_ids,
        'metrics': metrics,
        'error_statistics': error_stats
    }


def create_evaluation_plots(results, output_dir: Path):
    """Create evaluation plots."""
    predictions = results['predictions']
    targets = results['targets']
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Prediction vs Target scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Flatten for overall comparison
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Overall scatter
    axes[0, 0].scatter(target_flat, pred_flat, alpha=0.5, s=1)
    axes[0, 0].plot([target_flat.min(), target_flat.max()], 
                    [target_flat.min(), target_flat.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Target BP (mmHg)')
    axes[0, 0].set_ylabel('Predicted BP (mmHg)')
    axes[0, 0].set_title('Overall Prediction vs Target')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Systolic vs Diastolic
    pred_sys = np.max(predictions, axis=1)
    pred_dia = np.min(predictions, axis=1)
    target_sys = np.max(targets, axis=1)
    target_dia = np.min(targets, axis=1)
    
    axes[0, 1].scatter(target_sys, pred_sys, alpha=0.7, label='Systolic', s=20)
    axes[0, 1].scatter(target_dia, pred_dia, alpha=0.7, label='Diastolic', s=20)
    axes[0, 1].plot([40, 200], [40, 200], 'r--', lw=2)
    axes[0, 1].set_xlabel('Target BP (mmHg)')
    axes[0, 1].set_ylabel('Predicted BP (mmHg)')
    axes[0, 1].set_title('Systolic/Diastolic Prediction')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    errors = pred_flat - target_flat
    axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Prediction Error (mmHg)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Example waveforms
    n_examples = min(5, predictions.shape[0])
    for i in range(n_examples):
        axes[1, 1].plot(targets[i], label=f'Target {i+1}', linestyle='-')
        axes[1, 1].plot(predictions[i], label=f'Pred {i+1}', linestyle='--')
    
    axes[1, 1].set_xlabel('Time Points')
    axes[1, 1].set_ylabel('BP (mmHg)')
    axes[1, 1].set_title('Example Waveforms')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bland-Altman plot for clinical analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Systolic Bland-Altman
    sys_mean = (pred_sys + target_sys) / 2
    sys_diff = pred_sys - target_sys
    sys_bias = np.mean(sys_diff)
    sys_std = np.std(sys_diff)
    
    ax1.scatter(sys_mean, sys_diff, alpha=0.6)
    ax1.axhline(sys_bias, color='red', linestyle='-', label=f'Bias: {sys_bias:.2f}')
    ax1.axhline(sys_bias + 1.96*sys_std, color='red', linestyle='--', 
                label=f'+1.96SD: {sys_bias + 1.96*sys_std:.2f}')
    ax1.axhline(sys_bias - 1.96*sys_std, color='red', linestyle='--',
                label=f'-1.96SD: {sys_bias - 1.96*sys_std:.2f}')
    ax1.set_xlabel('Mean Systolic BP (mmHg)')
    ax1.set_ylabel('Difference (Pred - Target)')
    ax1.set_title('Bland-Altman Plot - Systolic')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Diastolic Bland-Altman
    dia_mean = (pred_dia + target_dia) / 2
    dia_diff = pred_dia - target_dia
    dia_bias = np.mean(dia_diff)
    dia_std = np.std(dia_diff)
    
    ax2.scatter(dia_mean, dia_diff, alpha=0.6)
    ax2.axhline(dia_bias, color='red', linestyle='-', label=f'Bias: {dia_bias:.2f}')
    ax2.axhline(dia_bias + 1.96*dia_std, color='red', linestyle='--',
                label=f'+1.96SD: {dia_bias + 1.96*dia_std:.2f}')
    ax2.axhline(dia_bias - 1.96*dia_std, color='red', linestyle='--',
                label=f'-1.96SD: {dia_bias - 1.96*dia_std:.2f}')
    ax2.set_xlabel('Mean Diastolic BP (mmHg)')
    ax2.set_ylabel('Difference (Pred - Target)')
    ax2.set_title('Bland-Altman Plot - Diastolic')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bland_altman_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}")


def save_results(results, output_dir: Path):
    """Save evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as YAML
    metrics_file = output_dir / 'metrics.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump({
            'metrics': results['metrics'],
            'error_statistics': results['error_statistics']
        }, f, default_flow_style=False)
    
    # Save predictions if requested
    if args.save_predictions:
        np.savez(
            output_dir / 'predictions.npz',
            predictions=results['predictions'],
            targets=results['targets'],
            sample_ids=results['sample_ids']
        )
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main evaluation function."""
    global args
    args = parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger.info("Starting model evaluation")
    
    try:
        # Load model and configuration
        checkpoint_path = Path(args.checkpoint)
        config_path = Path(args.config) if args.config else None
        
        model, data_config, model_config, training_config = load_model_and_config(
            checkpoint_path, config_path
        )
        
        # Override data directory if specified
        if args.data_dir:
            data_config.data_root = Path(args.data_dir)
        
        # Setup device
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        
        logger.info(f"Using device: {device}")
        
        # Create data loaders
        data_loaders, data_manager = create_data_loaders(
            data_config=data_config,
            splits=[args.split],
            verbose=True
        )
        
        # Override batch size
        data_loaders[args.split].batch_size = args.batch_size
        
        # Setup metrics calculator
        metrics_calculator = MetricsCalculator(
            bp_norm_range=data_config.bp_normalization
        )
        
        # Run evaluation
        results = evaluate_model(
            model, data_loaders[args.split], device, metrics_calculator
        )
        
        # Print key metrics
        logger.info("=== Evaluation Results ===")
        for key, value in results['metrics'].items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        # Save results
        output_dir = Path(args.output_dir)
        save_results(results, output_dir)
        
        # Create plots if requested
        if args.create_plots:
            create_evaluation_plots(results, output_dir)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
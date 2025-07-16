#!/usr/bin/env python3
"""
Visualization script for the improved BP predictor model.
Generates prediction examples and attention maps.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import utilities
from utils.data_utils import PviDataset, DataPathManager

# Import model classes from the training script
from scripts.train_improved_bp_predictor import ImprovedBPDataset, ImprovedBPPredictor, VAE, load_config_file


def load_trained_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load the trained improved BP predictor model"""
    
    # Load VAE
    vae_config = config.get('model_config', {}).get('vae_config', {})
    vae = VAE(latent_dim=vae_config.get('latent_dim', 64))
    
    vae_checkpoint_path = vae_config.get('vae_checkpoint_path')
    if vae_checkpoint_path and os.path.exists(vae_checkpoint_path):
        vae_checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in vae_checkpoint:
            vae.load_state_dict(vae_checkpoint['model_state_dict'])
        else:
            vae.load_state_dict(vae_checkpoint)
    
    vae.to(device)
    vae.eval()
    
    # Create BP predictor model
    data_config = config.get('data_config', {})
    model_config = config.get('model_config', {})
    bilstm_config = model_config.get('bilstm_config', {})
    attention_config = model_config.get('attention_config', {})
    
    model = ImprovedBPPredictor(
        vae_model=vae,
        latent_dim=vae_config.get('latent_dim', 64),
        hidden_dim=bilstm_config.get('hidden_dim', 128),
        num_layers=bilstm_config.get('num_layers', 2),
        num_heads=attention_config.get('num_attention_heads', 4),
        dropout=bilstm_config.get('dropout_rate', 0.4),
        use_attention=attention_config.get('use_attention', True),
        pattern_offsets=data_config.get('pattern_offsets', [-4, -3, -2, -1, 0, 1, 2]),
        current_frame_bias=attention_config.get('current_frame_bias', 1.5)
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vae


def extract_bp_values(waveform):
    """Extract systolic and diastolic values from BP waveform"""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    
    if len(waveform.shape) > 1:
        waveform = waveform.flatten()
    
    # Systolic (maximum)
    systolic = np.max(waveform)
    sys_idx = np.argmax(waveform)
    
    # Diastolic (minimum after systolic peak)
    search_start = max(sys_idx + 1, len(waveform) // 2)
    
    if search_start < len(waveform):
        post_systolic = waveform[search_start:]
        diastolic = np.min(post_systolic)
    else:
        diastolic = np.min(waveform)
    
    return systolic, diastolic


def plot_prediction_examples(model, dataset, device, num_examples=5, save_path=None):
    """Generate and plot prediction examples"""
    
    model.eval()
    examples = []
    
    # Select random examples
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            sequences = sample['sequences'].unsqueeze(0).to(device)  # Add batch dimension
            targets = sample['targets'].unsqueeze(0).to(device)
            
            # Get prediction with attention
            outputs = model(sequences, return_attention=True)
            pred_waveform = outputs['waveform'].cpu().numpy()[0]
            target_waveform = targets.cpu().numpy()[0]
            
            # Extract BP values
            pred_sys, pred_dias = extract_bp_values(pred_waveform)
            target_sys, target_dias = extract_bp_values(target_waveform)
            
            # Get attention weights if available
            attention_weights = None
            if 'attention_weights' in outputs:
                attention_weights = outputs['attention_weights'].cpu().numpy()[0]
            
            examples.append({
                'idx': idx,
                'pred_waveform': pred_waveform,
                'target_waveform': target_waveform,
                'pred_sys': pred_sys,
                'pred_dias': pred_dias,
                'target_sys': target_sys,
                'target_dias': target_dias,
                'attention_weights': attention_weights,
                'sequences': sequences.cpu().numpy()[0]
            })
    
    # Create visualization
    fig, axes = plt.subplots(num_examples, 2, figsize=(15, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i, example in enumerate(examples):
        # Plot waveform comparison
        ax1 = axes[i, 0]
        time_points = np.linspace(0, 1, len(example['target_waveform']))
        
        ax1.plot(time_points, example['target_waveform'], 'b-', label='Ground Truth', linewidth=2)
        ax1.plot(time_points, example['pred_waveform'], 'r--', label='Prediction', linewidth=2)
        
        # Add BP value annotations
        ax1.axhline(y=example['target_sys'], color='blue', linestyle=':', alpha=0.7, 
                   label=f'True SBP: {example["target_sys"]:.1f}')
        ax1.axhline(y=example['target_dias'], color='blue', linestyle=':', alpha=0.7,
                   label=f'True DBP: {example["target_dias"]:.1f}')
        ax1.axhline(y=example['pred_sys'], color='red', linestyle=':', alpha=0.7,
                   label=f'Pred SBP: {example["pred_sys"]:.1f}')
        ax1.axhline(y=example['pred_dias'], color='red', linestyle=':', alpha=0.7,
                   label=f'Pred DBP: {example["pred_dias"]:.1f}')
        
        ax1.set_xlabel('Normalized Time')
        ax1.set_ylabel('Blood Pressure (mmHg)')
        ax1.set_title(f'Example {i+1}: BP Waveform Prediction\n'
                     f'MAE: {np.mean(np.abs(example["pred_waveform"] - example["target_waveform"])):.2f} mmHg')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot attention heatmap
        ax2 = axes[i, 1]
        if example['attention_weights'] is not None:
            pattern_offsets = [-4, -3, -2, -1, 0, 1, 2]  # From config
            
            # Create heatmap
            sns.heatmap(example['attention_weights'], 
                       xticklabels=[f't{offset:+d}' for offset in pattern_offsets],
                       yticklabels=[f't{offset:+d}' for offset in pattern_offsets],
                       annot=True, fmt='.3f', cmap='Blues', ax=ax2)
            ax2.set_title(f'Attention Weights\n(Focus on current frame t+0)')
            ax2.set_xlabel('Key Frames')
            ax2.set_ylabel('Query Frames')
        else:
            ax2.text(0.5, 0.5, 'No Attention\nWeights Available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Attention Weights')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction examples saved to: {save_path}")
    
    return fig, examples


def plot_attention_analysis(examples, save_path=None):
    """Create detailed attention analysis plots"""
    
    # Filter examples with attention weights
    attention_examples = [ex for ex in examples if ex['attention_weights'] is not None]
    
    if not attention_examples:
        print("No attention weights available for analysis")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Average attention pattern
    ax1 = axes[0, 0]
    avg_attention = np.mean([ex['attention_weights'] for ex in attention_examples], axis=0)
    pattern_offsets = [-4, -3, -2, -1, 0, 1, 2]
    
    sns.heatmap(avg_attention, 
               xticklabels=[f't{offset:+d}' for offset in pattern_offsets],
               yticklabels=[f't{offset:+d}' for offset in pattern_offsets],
               annot=True, fmt='.3f', cmap='Blues', ax=ax1)
    ax1.set_title('Average Attention Pattern')
    ax1.set_xlabel('Key Frames')
    ax1.set_ylabel('Query Frames')
    
    # 2. Attention to current frame (t+0)
    ax2 = axes[0, 1]
    current_frame_idx = pattern_offsets.index(0)  # Index of t+0
    current_frame_attention = [ex['attention_weights'][:, current_frame_idx] 
                             for ex in attention_examples]
    
    attention_matrix = np.array(current_frame_attention).T
    sns.heatmap(attention_matrix, 
               yticklabels=[f't{offset:+d}' for offset in pattern_offsets],
               cmap='Blues', ax=ax2)
    ax2.set_title('Attention TO Current Frame (t+0)')
    ax2.set_xlabel('Example')
    ax2.set_ylabel('Query Frames')
    
    # 3. Attention FROM current frame
    ax3 = axes[1, 0]
    from_current_attention = [ex['attention_weights'][current_frame_idx, :] 
                            for ex in attention_examples]
    
    attention_matrix = np.array(from_current_attention).T
    sns.heatmap(attention_matrix, 
               yticklabels=[f't{offset:+d}' for offset in pattern_offsets],
               cmap='Blues', ax=ax3)
    ax3.set_title('Attention FROM Current Frame (t+0)')
    ax3.set_xlabel('Example')
    ax3.set_ylabel('Key Frames')
    
    # 4. Temporal attention distribution
    ax4 = axes[1, 1]
    temporal_attention = np.mean([ex['attention_weights'] for ex in attention_examples], axis=(0, 1))
    
    ax4.bar(range(len(pattern_offsets)), temporal_attention, 
           color='skyblue', alpha=0.7)
    ax4.set_xticks(range(len(pattern_offsets)))
    ax4.set_xticklabels([f't{offset:+d}' for offset in pattern_offsets])
    ax4.set_title('Average Temporal Attention Distribution')
    ax4.set_xlabel('Frame Offset')
    ax4.set_ylabel('Average Attention Weight')
    ax4.grid(True, alpha=0.3)
    
    # Highlight current frame
    ax4.bar(current_frame_idx, temporal_attention[current_frame_idx], 
           color='red', alpha=0.7, label='Current Frame (t+0)')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention analysis saved to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize Improved BP Predictor')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--checkpoint', type=str, 
                       default='./experiments/improved_bp_predictor/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, 
                       default='./experiments/improved_bp_predictor/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of prediction examples to generate')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = load_config_file(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("IMPROVED BP PREDICTOR VISUALIZATION")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # Load trained model
    print("\nLoading trained model...")
    model, vae = load_trained_model(args.checkpoint, config, device)
    print("Model loaded successfully!")
    
    # Create dataset
    print("\nCreating dataset...")
    data_config = config.get('data_config', {})
    data_root = data_config.get('root_path', '/home/lucas_takanori/phd/data')
    path_manager = DataPathManager(
        subject="subject001",
        session="baseline",
        root=data_root
    )
    h5_file_path = str(path_manager._h5_path)
    
    dataset = ImprovedBPDataset(
        data_root=h5_file_path,
        pattern_offsets=data_config.get('pattern_offsets', [-4, -3, -2, -1, 0, 1, 2]),
        max_samples_per_subject=data_config.get('max_samples_per_subject', 100),
        sequence_step_size=data_config.get('sequence_step_size', 10),
        use_augmentation=False,  # No augmentation for visualization
        noise_level=0.0
    )
    print(f"Dataset created with {len(dataset)} samples")
    
    # Generate prediction examples
    print(f"\nGenerating {args.num_examples} prediction examples...")
    examples_path = os.path.join(args.output_dir, 'prediction_examples.png')
    fig_examples, examples = plot_prediction_examples(
        model, dataset, device, args.num_examples, examples_path
    )
    
    # Generate attention analysis
    print("\nGenerating attention analysis...")
    attention_path = os.path.join(args.output_dir, 'attention_analysis.png')
    fig_attention = plot_attention_analysis(examples, attention_path)
    
    # Save summary statistics
    print("\nComputing performance statistics...")
    stats = {
        'num_examples': len(examples),
        'avg_mae': np.mean([np.mean(np.abs(ex['pred_waveform'] - ex['target_waveform'])) 
                           for ex in examples]),
        'avg_sys_error': np.mean([abs(ex['pred_sys'] - ex['target_sys']) for ex in examples]),
        'avg_dias_error': np.mean([abs(ex['pred_dias'] - ex['target_dias']) for ex in examples]),
        'sys_correlation': np.corrcoef([ex['pred_sys'] for ex in examples],
                                     [ex['target_sys'] for ex in examples])[0, 1],
        'dias_correlation': np.corrcoef([ex['pred_dias'] for ex in examples],
                                      [ex['target_dias'] for ex in examples])[0, 1]
    }
    
    # Save statistics
    stats_path = os.path.join(args.output_dir, 'visualization_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("IMPROVED BP PREDICTOR VISUALIZATION STATISTICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Number of examples: {stats['num_examples']}\n")
        f.write(f"Average MAE: {stats['avg_mae']:.2f} mmHg\n")
        f.write(f"Average Systolic Error: {stats['avg_sys_error']:.2f} mmHg\n")
        f.write(f"Average Diastolic Error: {stats['avg_dias_error']:.2f} mmHg\n")
        f.write(f"Systolic Correlation: {stats['sys_correlation']:.3f}\n")
        f.write(f"Diastolic Correlation: {stats['dias_correlation']:.3f}\n")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETED")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    print(f"- Prediction examples: {examples_path}")
    print(f"- Attention analysis: {attention_path}")
    print(f"- Statistics: {stats_path}")
    print(f"\nPerformance Summary:")
    print(f"- Average MAE: {stats['avg_mae']:.2f} mmHg")
    print(f"- Systolic Error: {stats['avg_sys_error']:.2f} mmHg")
    print(f"- Diastolic Error: {stats['avg_dias_error']:.2f} mmHg")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Evaluation Script for Improved BP Predictor (Continuous Version)

Evaluates the ImprovedBPPredictor model trained with sophisticated_bp_predictor_improved.yaml
Model path: /home/lucas_takanori/phd/bp_prediction/experiments/improved_bp_predictor_continuous/checkpoints/best_model.pt
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import yaml
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utilities and model classes
from utils.data_utils import PviDataset, DataPathManager
from scripts.train_improved_bp_predictor import (
    VAE, 
    ImprovedBPPredictor, 
    ImprovedBPDataset,
    ImprovedMultiHeadAttention,
    ImprovedBPLoss,
    extract_bp_values_numpy
)


class ImprovedBPEvaluator:
    """Evaluation class for the Improved BP Predictor model"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        self.config = self._load_config()
        
        # Load model
        self.model = self._load_model()
        
        print(f"✅ ImprovedBPPredictor loaded successfully!")
        print(f"📊 Device: {self.device}")
        print(f"🔧 Configuration loaded from: {config_path}")
    
    def _load_config(self):
        """Load YAML configuration file and replace dynamic placeholders"""
        import datetime
        
        with open(self.config_path, 'r') as file:
            config_text = file.read()
        
        # Replace timestamp placeholder with current datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_text = config_text.replace('{timestamp}', timestamp)
        
        # Parse the modified YAML
        config = yaml.safe_load(config_text)
        return config
    
    def _load_model(self):
        """Load the trained ImprovedBPPredictor model"""
        print(f"🔄 Loading model from: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract configuration
        data_config = self.config.get('data_config', {})
        model_config = self.config.get('model_config', {})
        vae_config = model_config.get('vae_config', {})
        bilstm_config = model_config.get('bilstm_config', {})
        attention_config = model_config.get('attention_config', {})
        
        # Load VAE first
        vae = VAE(latent_dim=vae_config.get('latent_dim', 128))
        vae_checkpoint_path = vae_config.get('vae_checkpoint_path')
        
        if vae_checkpoint_path and os.path.exists(vae_checkpoint_path):
            vae_checkpoint = torch.load(vae_checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in vae_checkpoint:
                vae.load_state_dict(vae_checkpoint['model_state_dict'])
            else:
                vae.load_state_dict(vae_checkpoint)
            print(f"✅ VAE loaded from: {vae_checkpoint_path}")
        else:
            print(f"⚠️  Warning: VAE checkpoint not found at {vae_checkpoint_path}")
        
        vae.to(self.device)
        vae.eval()
        
        # Create ImprovedBPPredictor
        model = ImprovedBPPredictor(
            vae_model=vae,
            latent_dim=vae_config.get('latent_dim', 128),
            hidden_dim=bilstm_config.get('hidden_dim', 128),
            num_layers=bilstm_config.get('num_layers', 2),
            num_heads=attention_config.get('num_attention_heads', 4),
            dropout=bilstm_config.get('dropout_rate', 0.3),
            use_attention=attention_config.get('use_attention', True),
            pattern_offsets=data_config.get('pattern_offsets', [-4, -3, -2, -1, 0, 1, 2]),
            current_frame_bias=attention_config.get('current_frame_bias', 1.5),
            use_physiological_features=attention_config.get('use_physiological_features', True)
        )
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate_model(self, max_samples: int = 500):
        """Evaluate the model on the dataset"""
        print(f"🚀 Starting evaluation with max {max_samples} samples...")
        
        # Get data configuration
        data_config = self.config.get('data_config', {})
        data_root = data_config.get('root_path', '/home/lucas_takanori/phd/data')
        
        # Setup data path
        path_manager = DataPathManager(
            subject="subject001",
            session="baseline", 
            root=data_root
        )
        h5_file_path = str(path_manager._h5_path)
        
        print(f"📁 Data file: {h5_file_path}")
        
        # Create dataset matching EXACTLY the training configuration
        dataset = ImprovedBPDataset(
            data_root=h5_file_path,
            pattern_offsets=data_config.get('pattern_offsets', [-4, -3, -2, -1, 0, 1, 2]),
            max_samples_per_subject=data_config.get('max_samples_per_subject', 150),  # MATCH training config
            sequence_step_size=data_config.get('sequence_step_size', 15),
            use_augmentation=data_config.get('use_augmentation', False),
            noise_level=data_config.get('noise_level', 0.0)
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=False, num_workers=2
        )
        
        # Collect predictions
        all_predictions = []
        all_targets = []
        
        print("🔍 Running model inference...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                sequences = batch['sequences'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Model prediction
                outputs = self.model(sequences)
                pred_waveforms = outputs['waveform']
                pred_systolic = outputs['systolic'].squeeze() if 'systolic' in outputs else None
                pred_diastolic = outputs['diastolic'].squeeze() if 'diastolic' in outputs else None
                
                # Extract physiological values from targets using SAME method as training
                true_sys, true_dias = self._extract_bp_values_training_compatible(targets)
                
                # Store results
                for i in range(len(targets)):
                    pred_dict = {
                        'waveform': pred_waveforms[i].cpu().numpy()
                    }
                    
                    target_dict = {
                        'waveform': targets[i].cpu().numpy(),
                        'systolic': true_sys[i] if isinstance(true_sys, np.ndarray) else true_sys,
                        'diastolic': true_dias[i] if isinstance(true_dias, np.ndarray) else true_dias
                    }
                    
                    # Add direct predictions if available
                    if pred_systolic is not None:
                        pred_dict['systolic'] = pred_systolic[i].cpu().numpy()
                    else:
                        # Use same method as training for consistency
                        pred_sys, _ = self._extract_bp_values_training_compatible_single(pred_waveforms[i:i+1])
                        pred_dict['systolic'] = pred_sys[0] if isinstance(pred_sys, np.ndarray) else pred_sys
                    
                    if pred_diastolic is not None:
                        pred_dict['diastolic'] = pred_diastolic[i].cpu().numpy()
                    else:
                        # Use same method as training for consistency
                        _, pred_dias = self._extract_bp_values_training_compatible_single(pred_waveforms[i:i+1])
                        pred_dict['diastolic'] = pred_dias[0] if isinstance(pred_dias, np.ndarray) else pred_dias
                    
                    all_predictions.append(pred_dict)
                    all_targets.append(target_dict)
        
        print(f"✅ Collected {len(all_predictions)} predictions")
        
        return {
            'predictions': all_predictions,
            'targets': all_targets,
            'config': self.config
        }
    
    def _extract_bp_values_training_compatible(self, waveform):
        """Extract BP values using EXACTLY the same method as training (ImprovedBPLoss.extract_bp_values_improved)"""
        if isinstance(waveform, torch.Tensor):
            batch_size, signal_length = waveform.shape
            device = waveform.device
            
            systolic_values = torch.zeros(batch_size, device=device)
            diastolic_values = torch.zeros(batch_size, device=device)
            
            for i in range(batch_size):
                signal = waveform[i]
                
                # Apply stronger smoothing for continuous extraction (SAME as training)
                if signal_length > 7:
                    # Use larger kernel for smoother extraction (EXACTLY as in training)
                    kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device)
                    signal_smooth = torch.nn.functional.conv1d(
                        signal.unsqueeze(0).unsqueeze(0), 
                        kernel.unsqueeze(0).unsqueeze(0), 
                        padding=2
                    ).squeeze()
                else:
                    signal_smooth = signal
                
                # Systolic: Use parabolic interpolation around maximum for continuity
                sys_val, sys_idx = torch.max(signal_smooth, dim=0)
                sys_idx = sys_idx.item()
                
                # Parabolic interpolation for sub-sample precision
                if 1 <= sys_idx <= signal_length - 2:
                    y1, y2, y3 = signal_smooth[sys_idx-1], signal_smooth[sys_idx], signal_smooth[sys_idx+1]
                    # Parabolic interpolation formula
                    a = (y1 + y3 - 2*y2) / 2
                    if abs(a) > 1e-6:  # Avoid division by zero
                        offset = (y1 - y3) / (4 * a)
                        sys_val = y2 + a * offset * offset
                        
                # Diastolic: Use smooth minimum finding with parabolic interpolation
                search_start = max(sys_idx + 1, int(signal_length * 0.6))
                search_end = min(signal_length, int(signal_length * 0.95))
                
                if search_start < search_end:
                    diastolic_window = signal_smooth[search_start:search_end]
                    dias_val_rel, local_min_idx = torch.min(diastolic_window, dim=0)
                    global_min_idx = search_start + local_min_idx.item()
                    
                    # Parabolic interpolation for diastolic minimum
                    if search_start + 1 <= global_min_idx <= search_end - 2:
                        y1 = signal_smooth[global_min_idx-1]
                        y2 = signal_smooth[global_min_idx] 
                        y3 = signal_smooth[global_min_idx+1]
                        a = (y1 + y3 - 2*y2) / 2
                        if abs(a) > 1e-6:
                            offset = (y1 - y3) / (4 * a)
                            dias_val = y2 + a * offset * offset
                        else:
                            dias_val = dias_val_rel
                    else:
                        dias_val = dias_val_rel
                else:
                    # Fallback: use global minimum with smoothing
                    dias_val = torch.min(signal_smooth)
                    
                # Ensure physiological constraint with smooth enforcement
                pulse_pressure = sys_val - dias_val
                min_pulse_pressure = 15.0  # Minimum realistic pulse pressure
                
                if pulse_pressure < min_pulse_pressure:
                    # Smoothly adjust to maintain minimum pulse pressure
                    center_pressure = (sys_val + dias_val) / 2
                    sys_val = center_pressure + min_pulse_pressure / 2
                    dias_val = center_pressure - min_pulse_pressure / 2
                    
                systolic_values[i] = sys_val
                diastolic_values[i] = dias_val
            
            return systolic_values.cpu().numpy(), diastolic_values.cpu().numpy()
        else:
            # Convert numpy to torch if needed
            waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
            return self._extract_bp_values_training_compatible(waveform_tensor)
    
    def _extract_bp_values_training_compatible_single(self, waveform):
        """Helper for single waveform extraction"""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return self._extract_bp_values_training_compatible(waveform)
    
    def _extract_bp_values_improved(self, waveform):
        """Extract systolic and diastolic values with improved method"""
        if isinstance(waveform, torch.Tensor):
            if waveform.dim() == 1:
                signal = waveform.detach().cpu().numpy()
                sys_val = np.max(signal)
                sys_idx = np.argmax(signal)
                
                search_start = max(sys_idx + 1, int(len(signal) * 0.6))
                search_end = min(len(signal), int(len(signal) * 0.95))
                
                if search_start < search_end:
                    diastolic_window = signal[search_start:search_end]
                    dias_val = np.min(diastolic_window)
                else:
                    dias_val = np.min(signal)
                
                if dias_val >= sys_val:
                    dias_val = sys_val - 10.0
                    
                return sys_val, dias_val
            else:
                batch_size = waveform.shape[0]
                systolic_values = []
                diastolic_values = []
                
                for i in range(batch_size):
                    sys_val, dias_val = self._extract_bp_values_improved(waveform[i])
                    systolic_values.append(sys_val)
                    diastolic_values.append(dias_val)
                
                return np.array(systolic_values), np.array(diastolic_values)
    
    def calculate_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Extract arrays
        pred_sys = np.array([float(p['systolic']) for p in predictions])
        pred_dias = np.array([float(p['diastolic']) for p in predictions])
        pred_waveforms = np.array([p['waveform'] for p in predictions])
        
        true_sys = np.array([float(t['systolic']) for t in targets])
        true_dias = np.array([float(t['diastolic']) for t in targets])
        true_waveforms = np.array([t['waveform'] for t in targets])
        
        metrics = {}
        
        # Systolic metrics
        metrics['systolic'] = {
            'r2': r2_score(true_sys, pred_sys),
            'mae': mean_absolute_error(true_sys, pred_sys),
            'rmse': np.sqrt(mean_squared_error(true_sys, pred_sys)),
            'mean_diff': np.mean(pred_sys - true_sys),
            'std_diff': np.std(pred_sys - true_sys),
            'pearson_r': stats.pearsonr(true_sys, pred_sys)[0],
            'pearson_p': stats.pearsonr(true_sys, pred_sys)[1]
        }
        
        # Diastolic metrics
        metrics['diastolic'] = {
            'r2': r2_score(true_dias, pred_dias),
            'mae': mean_absolute_error(true_dias, pred_dias),
            'rmse': np.sqrt(mean_squared_error(true_dias, pred_dias)),
            'mean_diff': np.mean(pred_dias - true_dias),
            'std_diff': np.std(pred_dias - true_dias),
            'pearson_r': stats.pearsonr(true_dias, pred_dias)[0],
            'pearson_p': stats.pearsonr(true_dias, pred_dias)[1]
        }
        
        # Waveform metrics
        waveform_r2_scores = []
        waveform_mae_scores = []
        for i in range(len(pred_waveforms)):
            waveform_r2_scores.append(r2_score(true_waveforms[i], pred_waveforms[i]))
            waveform_mae_scores.append(mean_absolute_error(true_waveforms[i], pred_waveforms[i]))
        
        metrics['waveform'] = {
            'r2': np.mean(waveform_r2_scores),
            'mae': np.mean(waveform_mae_scores),
            'rmse': np.sqrt(mean_squared_error(true_waveforms.flatten(), pred_waveforms.flatten())),
            'mean_diff': np.mean(pred_waveforms - true_waveforms),
            'std_diff': np.std(pred_waveforms - true_waveforms),
        }
        
        # Clinical accuracy metrics
        metrics['clinical'] = {
            'systolic_5mmhg': np.mean(np.abs(pred_sys - true_sys) <= 5) * 100,
            'systolic_10mmhg': np.mean(np.abs(pred_sys - true_sys) <= 10) * 100,
            'systolic_15mmhg': np.mean(np.abs(pred_sys - true_sys) <= 15) * 100,
            'diastolic_5mmhg': np.mean(np.abs(pred_dias - true_dias) <= 5) * 100,
            'diastolic_10mmhg': np.mean(np.abs(pred_dias - true_dias) <= 10) * 100,
            'diastolic_15mmhg': np.mean(np.abs(pred_dias - true_dias) <= 15) * 100,
        }
        
        return metrics
    
    def create_evaluation_plots(self, predictions: List[Dict], targets: List[Dict], 
                              output_dir: str, model_name: str = "Improved BP Predictor"):
        """Create comprehensive evaluation plots"""
        
        # Extract data
        pred_sys = np.array([float(p['systolic']) for p in predictions])
        pred_dias = np.array([float(p['diastolic']) for p in predictions])
        pred_waveforms = np.array([p['waveform'] for p in predictions])
        
        true_sys = np.array([float(t['systolic']) for t in targets])
        true_dias = np.array([float(t['diastolic']) for t in targets])
        true_waveforms = np.array([t['waveform'] for t in targets])
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, targets)
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        
        # Helper function for correlation plot
        def plot_correlation(ax, true_vals, pred_vals, title, metrics_dict):
            ax.scatter(true_vals, pred_vals, alpha=0.6, s=20)
            
            # Perfect correlation line
            min_val, max_val = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect')
            
            # Regression line
            z = np.polyfit(true_vals, pred_vals, 1)
            p = np.poly1d(z)
            ax.plot(true_vals, p(true_vals), "r-", alpha=0.8, label='Fit')
            
            ax.set_xlabel(f'True {title} (mmHg)')
            ax.set_ylabel(f'Predicted {title} (mmHg)')
            ax.set_title(f'{title} Correlation')
            
            # Add metrics text
            r2 = metrics_dict.get('r2', 0)
            p_val = metrics_dict.get('pearson_p', 1)
            ax.text(0.05, 0.95, f'R²={r2:.4f}\np={p_val:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Helper function for Bland-Altman plot
        def plot_bland_altman(ax, true_vals, pred_vals, title, metrics_dict):
            diff = pred_vals - true_vals
            mean_vals = (pred_vals + true_vals) / 2
            
            ax.scatter(mean_vals, diff, alpha=0.6, s=20)
            
            # Mean difference line
            mean_diff = np.mean(diff)
            ax.axhline(mean_diff, color='blue', linestyle='-', alpha=0.8, label=f'Mean: {mean_diff:.2f}')
            
            # 95% limits of agreement
            std_diff = np.std(diff)
            upper_loa = mean_diff + 1.96 * std_diff
            lower_loa = mean_diff - 1.96 * std_diff
            
            ax.axhline(upper_loa, color='red', linestyle='--', alpha=0.8, label=f'+1.96SD: {upper_loa:.2f}')
            ax.axhline(lower_loa, color='red', linestyle='--', alpha=0.8, label=f'-1.96SD: {lower_loa:.2f}')
            
            ax.set_xlabel('Mean (mmHg)')
            ax.set_ylabel('Difference (mmHg)')
            ax.set_title(f'{title} Bland-Altman')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Helper function for error histogram (matching example format)
        def plot_error_histogram(ax, true_vals, pred_vals, title, metrics_dict):
            errors = np.abs(pred_vals - true_vals)
            
            # Create histogram with bins up to 40 mmHg with more bars
            bins = np.arange(0, 41, 1)  # 0, 1, 2, 3, ..., 40 (1 mmHg intervals)
            counts, _, _ = ax.hist(errors, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add vertical lines for clinical thresholds
            ax.axvline(5, color='green', linestyle='--', alpha=0.8, linewidth=2)
            ax.axvline(10, color='orange', linestyle='--', alpha=0.8, linewidth=2)
            ax.axvline(15, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Absolute error (mmHg)')
            ax.set_ylabel('Occurrences')
            ax.set_title(title)
            
            # Calculate statistics matching the example format
            mae = np.mean(errors)  # Mean absolute error
            std_error = np.std(errors)  # Standard deviation of errors
            acc_5 = np.mean(errors <= 5) * 100   # 5-tolerance
            acc_10 = np.mean(errors <= 10) * 100 # 10-tolerance
            acc_15 = np.mean(errors <= 15) * 100 # 15-tolerance
            
            # Create stats text box matching example format
            stats_text = f'mean: {mae:.2f}\nstd: {std_error:.2f}\n5-tol: {acc_5:.2f} %\n10-tol: {acc_10:.2f} %\n15-tol: {acc_15:.2f} %'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5),
                   fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 40)  # Set x-axis limit to match example
        
        # Create subplots (3x3 grid)
        # Row 1: Full waveform
        ax1 = plt.subplot(3, 3, 1)
        plot_correlation(ax1, true_waveforms.flatten(), pred_waveforms.flatten(), 
                        'Waveform', metrics['waveform'])
        
        ax2 = plt.subplot(3, 3, 2)
        plot_bland_altman(ax2, true_waveforms.flatten(), pred_waveforms.flatten(), 
                         'Waveform', metrics['waveform'])
        
        ax3 = plt.subplot(3, 3, 3)
        plot_error_histogram(ax3, true_waveforms.flatten(), pred_waveforms.flatten(), 
                           'FULL', metrics['waveform'])
        
        # Row 2: Systolic BP
        ax4 = plt.subplot(3, 3, 4)
        plot_correlation(ax4, true_sys, pred_sys, 'Systolic', metrics['systolic'])
        
        ax5 = plt.subplot(3, 3, 5)
        plot_bland_altman(ax5, true_sys, pred_sys, 'Systolic', metrics['systolic'])
        
        ax6 = plt.subplot(3, 3, 6)
        plot_error_histogram(ax6, true_sys, pred_sys, 'SBP', metrics['systolic'])
        
        # Row 3: Diastolic BP
        ax7 = plt.subplot(3, 3, 7)
        plot_correlation(ax7, true_dias, pred_dias, 'Diastolic', metrics['diastolic'])
        
        ax8 = plt.subplot(3, 3, 8)
        plot_bland_altman(ax8, true_dias, pred_dias, 'Diastolic', metrics['diastolic'])
        
        ax9 = plt.subplot(3, 3, 9)
        plot_error_histogram(ax9, true_dias, pred_dias, 'DBP', metrics['diastolic'])
        
        # Add main title
        n_samples = len(predictions)
        fig.suptitle(f'{model_name} Evaluation | {n_samples} samples', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'improved_bp_evaluation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Evaluation plot saved: {plot_path}")
        
        return metrics
    
    def save_results(self, metrics: Dict, output_dir: str):
        """Save evaluation results to files"""
        
        # Save metrics to text file
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("IMPROVED BP PREDICTOR EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for category in ['systolic', 'diastolic', 'waveform']:
                if category in metrics:
                    f.write(f"{category.upper()} METRICS:\n")
                    for metric, value in metrics[category].items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n")
            
            if 'clinical' in metrics:
                f.write("CLINICAL ACCURACY:\n")
                for metric, value in metrics['clinical'].items():
                    f.write(f"  {metric}: {value:.2f}%\n")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save metrics to JSON for programmatic access
        import json
        json_path = os.path.join(output_dir, 'evaluation_metrics.json')
        json_compatible_metrics = convert_numpy_types(metrics)
        with open(json_path, 'w') as f:
            json.dump(json_compatible_metrics, f, indent=2)
        
        print(f"📄 Metrics saved: {metrics_path}")
        print(f"📄 JSON metrics saved: {json_path}")


def main():
    """Main evaluation function"""
    
    # Default paths
    MODEL_PATH = "/home/lucas_takanori/phd/bp_prediction/experiments/improved_bp_predictor_continuous/checkpoints/best_model.pt"
    CONFIG_PATH = "/home/lucas_takanori/phd/bp_prediction/configs/sophisticated_bp_predictor_improved.yaml"
    OUTPUT_DIR = "./evaluation_results_improved_bp"
    
    parser = argparse.ArgumentParser(description='Evaluate Improved BP Predictor Model')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=CONFIG_PATH,
                       help='Path to YAML configuration file')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🩺 IMPROVED BP PREDICTOR EVALUATION")
    print("=" * 80)
    print(f"📁 Model: {args.model_path}")
    print(f"⚙️  Config: {args.config_path}")
    print(f"📂 Output: {args.output_dir}")
    print(f"🔢 Max samples: {args.max_samples}")
    print("=" * 80)
    
    # Verify files exist
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.config_path):
        print(f"❌ Config file not found: {args.config_path}")
        return
    
    try:
        # Initialize evaluator
        evaluator = ImprovedBPEvaluator(args.model_path, args.config_path)
        
        # Run evaluation
        results = evaluator.evaluate_model(args.max_samples)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(results['predictions'], results['targets'])
        
        # Create plots
        evaluator.create_evaluation_plots(
            results['predictions'], 
            results['targets'], 
            args.output_dir, 
            "Improved BP Predictor"
        )
        
        # Save results
        evaluator.save_results(metrics, args.output_dir)
        
        # Print summary
        print("\n" + "=" * 80)
        print("🎉 EVALUATION COMPLETED!")
        print("=" * 80)
        print(f"📊 Samples evaluated: {len(results['predictions'])}")
        print("\n📈 KEY METRICS:")
        print("-" * 40)
        print(f"Systolic  MAE: {metrics['systolic']['mae']:.2f} mmHg (R²={metrics['systolic']['r2']:.3f})")
        print(f"Diastolic MAE: {metrics['diastolic']['mae']:.2f} mmHg (R²={metrics['diastolic']['r2']:.3f})")
        print(f"Waveform  MAE: {metrics['waveform']['mae']:.2f} mmHg (R²={metrics['waveform']['r2']:.3f})")
        
        print("\n🎯 CLINICAL ACCURACY:")
        print(f"Systolic  ≤5mmHg:  {metrics['clinical']['systolic_5mmhg']:.1f}%")
        print(f"Systolic  ≤10mmHg: {metrics['clinical']['systolic_10mmhg']:.1f}%")
        print(f"Diastolic ≤5mmHg:  {metrics['clinical']['diastolic_5mmhg']:.1f}%")
        print(f"Diastolic ≤10mmHg: {metrics['clinical']['diastolic_10mmhg']:.1f}%")
        
        print(f"\n📁 Results saved to: {args.output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 
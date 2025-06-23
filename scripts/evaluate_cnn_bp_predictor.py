#!/usr/bin/env python3
"""
CNN-Based BP Predictor Evaluation Script

Evaluates the CNN-based BP predictor model and generates comprehensive metrics
and visualizations for comparison with VAE-based approach.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import utilities and models
from utils.data_utils import PviDataset, DataPathManager

# Import model classes from FIXED training script
from scripts.train_cnn_bp_predictor_fixed import CNNBPDataset, CNNBPPredictor, PVIFeatureExtractor, MultiHeadAttention


class CNNBPEvaluator:
    """Evaluator for CNN-based BP predictor"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config and model
        self.config = self._load_config()
        self.model = self._load_model()
        
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
        """Load the trained CNN-based BP predictor model"""
        print(f"🔄 Loading model from: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract configuration
        data_config = self.config.get('data_config', {})
        model_config = self.config.get('model_config', {})
        cnn_config = model_config.get('cnn_config', {})
        bilstm_config = model_config.get('bilstm_config', {})
        attention_config = model_config.get('attention_config', {})
        
        # Create model with correct parameters (matching the CNNBPPredictor constructor)
        model = CNNBPPredictor(
            cnn_config=cnn_config,
            bilstm_config=bilstm_config,
            attention_config=attention_config,
            pattern_offsets=data_config.get('pattern_offsets', [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]),
            auxiliary_heads=model_config.get('auxiliary_heads', {})
        )
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"✅ CNN model loaded successfully")
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
        
        # Create dataset matching training configuration
        dataset = CNNBPDataset(
            data_root=h5_file_path,
            pattern_offsets=data_config.get('pattern_offsets', [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]),
            max_samples_per_subject=min(max_samples, data_config.get('max_samples_per_subject', 1000)),
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
                
                # Denormalize predictions from [0,1] to [40,200] mmHg
                pred_waveforms = pred_waveforms * (200.0 - 40.0) + 40.0
                if pred_systolic is not None:
                    pred_systolic = pred_systolic * (200.0 - 40.0) + 40.0
                if pred_diastolic is not None:
                    pred_diastolic = pred_diastolic * (200.0 - 40.0) + 40.0
                
                # Denormalize targets from [0,1] to [40,200] mmHg
                targets_denorm = targets * (200.0 - 40.0) + 40.0
                
                # Extract physiological values from denormalized targets
                true_sys, true_dias = self._extract_bp_values(targets_denorm)
                
                # Store results
                for i in range(len(targets)):
                    pred_dict = {
                        'waveform': pred_waveforms[i].cpu().numpy()
                    }
                    
                    target_dict = {
                        'waveform': targets_denorm[i].cpu().numpy(),
                        'systolic': true_sys[i] if isinstance(true_sys, np.ndarray) else true_sys,
                        'diastolic': true_dias[i] if isinstance(true_dias, np.ndarray) else true_dias
                    }
                    
                    # Add direct predictions if available
                    if pred_systolic is not None:
                        pred_dict['systolic'] = pred_systolic[i].cpu().numpy()
                    else:
                        pred_sys, _ = self._extract_bp_values_single(pred_waveforms[i:i+1])
                        pred_dict['systolic'] = pred_sys[0] if isinstance(pred_sys, np.ndarray) else pred_sys
                    
                    if pred_diastolic is not None:
                        pred_dict['diastolic'] = pred_diastolic[i].cpu().numpy()
                    else:
                        _, pred_dias = self._extract_bp_values_single(pred_waveforms[i:i+1])
                        pred_dict['diastolic'] = pred_dias[0] if isinstance(pred_dias, np.ndarray) else pred_dias
                    
                    all_predictions.append(pred_dict)
                    all_targets.append(target_dict)
        
        print(f"✅ Collected {len(all_predictions)} predictions")
        
        return {
            'predictions': all_predictions,
            'targets': all_targets,
            'config': self.config
        }
    
    def _extract_bp_values(self, waveform):
        """Extract BP values from waveform (batch)"""
        if isinstance(waveform, torch.Tensor):
            batch_size = waveform.shape[0]
            device = waveform.device
            
            systolic_values = torch.zeros(batch_size, device=device)
            diastolic_values = torch.zeros(batch_size, device=device)
            
            for i in range(batch_size):
                signal = waveform[i]
                
                # Systolic: maximum value
                sys_val, sys_idx = torch.max(signal, dim=0)
                systolic_values[i] = sys_val
                
                # Diastolic: minimum in second half
                search_start = max(sys_idx.item() + 1, len(signal) // 2)
                if search_start < len(signal):
                    dias_val, _ = torch.min(signal[search_start:], dim=0)
                    diastolic_values[i] = dias_val
                else:
                    diastolic_values[i] = torch.min(signal)
            
            return systolic_values.cpu().numpy(), diastolic_values.cpu().numpy()
        else:
            # Handle numpy arrays
            systolic_values = []
            diastolic_values = []
            
            for signal in waveform:
                sys_val = np.max(signal)
                sys_idx = np.argmax(signal)
                
                search_start = max(sys_idx + 1, len(signal) // 2)
                if search_start < len(signal):
                    dias_val = np.min(signal[search_start:])
                else:
                    dias_val = np.min(signal)
                
                systolic_values.append(sys_val)
                diastolic_values.append(dias_val)
            
            return np.array(systolic_values), np.array(diastolic_values)
    
    def _extract_bp_values_single(self, waveform):
        """Extract BP values from single waveform"""
        return self._extract_bp_values(waveform)
    
    def calculate_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        print("📊 Calculating metrics...")
        
        # Extract arrays
        pred_waveforms = np.array([p['waveform'] for p in predictions])
        true_waveforms = np.array([t['waveform'] for t in targets])
        
        pred_sys = np.array([p['systolic'] for p in predictions])
        true_sys = np.array([t['systolic'] for t in targets])
        
        pred_dias = np.array([p['diastolic'] for p in predictions])
        true_dias = np.array([t['diastolic'] for t in targets])
        
        metrics = {}
        
        # Waveform metrics
        waveform_mae = np.mean(np.abs(pred_waveforms - true_waveforms))
        waveform_mse = np.mean((pred_waveforms - true_waveforms) ** 2)
        waveform_rmse = np.sqrt(waveform_mse)
        
        # Correlation
        waveform_corr = np.corrcoef(pred_waveforms.flatten(), true_waveforms.flatten())[0, 1]
        
        # R-squared
        ss_res = np.sum((true_waveforms.flatten() - pred_waveforms.flatten()) ** 2)
        ss_tot = np.sum((true_waveforms.flatten() - np.mean(true_waveforms.flatten())) ** 2)
        waveform_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics['waveform'] = {
            'mae': waveform_mae,
            'mse': waveform_mse,
            'rmse': waveform_rmse,
            'correlation': waveform_corr,
            'r2': waveform_r2
        }
        
        # Systolic metrics
        sys_mae = np.mean(np.abs(pred_sys - true_sys))
        sys_mse = np.mean((pred_sys - true_sys) ** 2)
        sys_rmse = np.sqrt(sys_mse)
        sys_corr = np.corrcoef(pred_sys, true_sys)[0, 1]
        
        ss_res_sys = np.sum((true_sys - pred_sys) ** 2)
        ss_tot_sys = np.sum((true_sys - np.mean(true_sys)) ** 2)
        sys_r2 = 1 - (ss_res_sys / ss_tot_sys) if ss_tot_sys != 0 else 0
        
        metrics['systolic'] = {
            'mae': sys_mae,
            'mse': sys_mse,
            'rmse': sys_rmse,
            'correlation': sys_corr,
            'r2': sys_r2
        }
        
        # Diastolic metrics
        dias_mae = np.mean(np.abs(pred_dias - true_dias))
        dias_mse = np.mean((pred_dias - true_dias) ** 2)
        dias_rmse = np.sqrt(dias_mse)
        dias_corr = np.corrcoef(pred_dias, true_dias)[0, 1]
        
        ss_res_dias = np.sum((true_dias - pred_dias) ** 2)
        ss_tot_dias = np.sum((true_dias - np.mean(true_dias)) ** 2)
        dias_r2 = 1 - (ss_res_dias / ss_tot_dias) if ss_tot_dias != 0 else 0
        
        metrics['diastolic'] = {
            'mae': dias_mae,
            'mse': dias_mse,
            'rmse': dias_rmse,
            'correlation': dias_corr,
            'r2': dias_r2
        }
        
        # Clinical accuracy metrics
        sys_errors = np.abs(pred_sys - true_sys)
        dias_errors = np.abs(pred_dias - true_dias)
        
        metrics['clinical'] = {
            'systolic_5mmhg': np.mean(sys_errors <= 5) * 100,
            'systolic_10mmhg': np.mean(sys_errors <= 10) * 100,
            'systolic_15mmhg': np.mean(sys_errors <= 15) * 100,
            'diastolic_5mmhg': np.mean(dias_errors <= 5) * 100,
            'diastolic_10mmhg': np.mean(dias_errors <= 10) * 100,
            'diastolic_15mmhg': np.mean(dias_errors <= 15) * 100
        }
        
        return metrics
    
    def create_evaluation_plots(self, predictions: List[Dict], targets: List[Dict], 
                              output_dir: str, model_name: str = "CNN BP Predictor"):
        """Create comprehensive evaluation plots"""
        print("📈 Creating evaluation plots...")
        
        # Extract data
        pred_waveforms = np.array([p['waveform'] for p in predictions])
        true_waveforms = np.array([t['waveform'] for t in targets])
        
        pred_sys = np.array([p['systolic'] for p in predictions])
        true_sys = np.array([t['systolic'] for t in targets])
        
        pred_dias = np.array([p['diastolic'] for p in predictions])
        true_dias = np.array([t['diastolic'] for t in targets])
        
        # Calculate metrics for plots
        metrics = self.calculate_metrics(predictions, targets)
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        
        def plot_correlation(ax, true_vals, pred_vals, title, metrics_dict):
            ax.scatter(true_vals, pred_vals, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Stats
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{title} Correlation\nR²={metrics_dict["r2"]:.3f}, r={metrics_dict["correlation"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        def plot_bland_altman(ax, true_vals, pred_vals, title, metrics_dict):
            mean_vals = (true_vals + pred_vals) / 2
            diff_vals = pred_vals - true_vals
            
            ax.scatter(mean_vals, diff_vals, alpha=0.6, s=20)
            
            # Mean difference line
            mean_diff = np.mean(diff_vals)
            ax.axhline(mean_diff, color='red', linestyle='-', alpha=0.8)
            
            # Limits of agreement
            std_diff = np.std(diff_vals)
            ax.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', alpha=0.6)
            ax.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', alpha=0.6)
            
            ax.set_xlabel('Mean of True and Predicted')
            ax.set_ylabel('Predicted - True')
            ax.set_title(f'{title} Bland-Altman\nBias={mean_diff:.2f}±{1.96*std_diff:.2f}')
            ax.grid(True, alpha=0.3)
        
        def plot_error_histogram(ax, true_vals, pred_vals, title, metrics_dict):
            errors = np.abs(pred_vals - true_vals)
            ax.hist(errors, bins=30, alpha=0.7, edgecolor='black')
            
            # Statistics
            mae = np.mean(errors)
            std_error = np.std(errors)
            acc_5 = np.mean(errors <= 5) * 100
            acc_10 = np.mean(errors <= 10) * 100
            acc_15 = np.mean(errors <= 15) * 100
            
            # Stats text
            stats_text = f'MAE: {mae:.2f}\nStd: {std_error:.2f}\n≤5mmHg: {acc_5:.1f}%\n≤10mmHg: {acc_10:.1f}%\n≤15mmHg: {acc_15:.1f}%'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
            
            ax.set_xlabel('Absolute Error (mmHg)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{title} Error Distribution')
            ax.grid(True, alpha=0.3)
        
        # Create subplots (3x3 grid)
        # Row 1: Waveform
        ax1 = plt.subplot(3, 3, 1)
        plot_correlation(ax1, true_waveforms.flatten(), pred_waveforms.flatten(),
                        'Waveform', metrics['waveform'])
        
        ax2 = plt.subplot(3, 3, 2)
        plot_bland_altman(ax2, true_waveforms.flatten(), pred_waveforms.flatten(),
                         'Waveform', metrics['waveform'])
        
        ax3 = plt.subplot(3, 3, 3)
        plot_error_histogram(ax3, true_waveforms.flatten(), pred_waveforms.flatten(),
                           'Waveform', metrics['waveform'])
        
        # Row 2: Systolic
        ax4 = plt.subplot(3, 3, 4)
        plot_correlation(ax4, true_sys, pred_sys, 'Systolic', metrics['systolic'])
        
        ax5 = plt.subplot(3, 3, 5)
        plot_bland_altman(ax5, true_sys, pred_sys, 'Systolic', metrics['systolic'])
        
        ax6 = plt.subplot(3, 3, 6)
        plot_error_histogram(ax6, true_sys, pred_sys, 'Systolic', metrics['systolic'])
        
        # Row 3: Diastolic
        ax7 = plt.subplot(3, 3, 7)
        plot_correlation(ax7, true_dias, pred_dias, 'Diastolic', metrics['diastolic'])
        
        ax8 = plt.subplot(3, 3, 8)
        plot_bland_altman(ax8, true_dias, pred_dias, 'Diastolic', metrics['diastolic'])
        
        ax9 = plt.subplot(3, 3, 9)
        plot_error_histogram(ax9, true_dias, pred_dias, 'Diastolic', metrics['diastolic'])
        
        # Add main title
        n_samples = len(predictions)
        fig.suptitle(f'{model_name} Evaluation | {n_samples} samples', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'cnn_bp_evaluation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Evaluation plot saved: {plot_path}")
        
        return metrics
    
    def save_results(self, metrics: Dict, output_dir: str):
        """Save evaluation results to files"""
        
        # Save metrics to text file
        metrics_path = os.path.join(output_dir, 'cnn_evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("CNN-BASED BP PREDICTOR EVALUATION RESULTS\n")
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
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
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
        
        # Save metrics to JSON
        import json
        json_path = os.path.join(output_dir, 'cnn_evaluation_metrics.json')
        json_compatible_metrics = convert_numpy_types(metrics)
        with open(json_path, 'w') as f:
            json.dump(json_compatible_metrics, f, indent=2)
        
        print(f"📄 Metrics saved: {metrics_path}")
        print(f"📄 JSON metrics saved: {json_path}")


def main():
    """Main evaluation function"""
    
    # Default paths - update timestamp to match your training run
    MODEL_PATH = "./experiments/cnn_bp_predictor_{timestamp}/checkpoints/best_model.pt"  # Replace {timestamp} with your actual training timestamp
    CONFIG_PATH = "configs/cnn_bp_predictor.yaml"
    OUTPUT_DIR = "./evaluation_results_cnn_bp"
    
    parser = argparse.ArgumentParser(description='Evaluate CNN-Based BP Predictor Model')
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
    print("🧠 CNN-BASED BP PREDICTOR EVALUATION")
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
        evaluator = CNNBPEvaluator(args.model_path, args.config_path)
        
        # Run evaluation
        results = evaluator.evaluate_model(args.max_samples)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(results['predictions'], results['targets'])
        
        # Create plots
        evaluator.create_evaluation_plots(
            results['predictions'],
            results['targets'],
            args.output_dir,
            "CNN BP Predictor"
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
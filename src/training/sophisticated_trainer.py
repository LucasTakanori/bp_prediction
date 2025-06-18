"""
Sophisticated BP Predictor Training and Evaluation Module

This module provides comprehensive training, evaluation, and visualization capabilities 
for the sophisticated blood pressure prediction system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
from tqdm import tqdm
import wandb
import warnings
warnings.filterwarnings('ignore')

# Import our models and utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.models.sophisticated_bp_predictor import (
    SophisticatedBPPredictor, 
    EnhancedVAE,
    ComprehensiveBPLoss,
    ModelConfig,
    create_sliding_window_sequences,
    normalize_bp_data,
    denormalize_bp_data,
    extract_physiological_bp_values,
    load_pretrained_vae
)


class BPMetricsCalculator:
    """Calculator for comprehensive BP prediction metrics"""
    
    @staticmethod
    def compute_waveform_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute waveform-level metrics"""
        # Flatten arrays for computation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        
        if len(pred_flat) == 0:
            return {
                'mse': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'correlation': 0.0
            }
        
        # Compute metrics
        mse = mean_squared_error(target_flat, pred_flat)
        mae = mean_absolute_error(target_flat, pred_flat)
        rmse = np.sqrt(mse)
        r2 = r2_score(target_flat, pred_flat)
        
        # Correlation coefficient
        if np.std(pred_flat) > 0 and np.std(target_flat) > 0:
            correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        else:
            correlation = 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation
        }
    
    @staticmethod
    def compute_bp_component_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute systolic and diastolic specific metrics"""
        # Convert to torch tensors for consistent processing
        pred_tensor = torch.tensor(predictions, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        
        # Extract physiological values
        pred_systolic, pred_diastolic = extract_physiological_bp_values(pred_tensor)
        target_systolic, target_diastolic = extract_physiological_bp_values(target_tensor)
        
        # Convert back to numpy
        pred_sys = pred_systolic.numpy()
        pred_dias = pred_diastolic.numpy()
        target_sys = target_systolic.numpy()
        target_dias = target_diastolic.numpy()
        
        # Compute systolic metrics
        systolic_metrics = {
            'mae': mean_absolute_error(target_sys, pred_sys),
            'mse': mean_squared_error(target_sys, pred_sys),
            'rmse': np.sqrt(mean_squared_error(target_sys, pred_sys)),
            'bias': np.mean(pred_sys - target_sys),
            'std_error': np.std(pred_sys - target_sys),
            'correlation': np.corrcoef(pred_sys, target_sys)[0, 1] if np.std(pred_sys) > 0 and np.std(target_sys) > 0 else 0.0
        }
        
        # Compute diastolic metrics
        diastolic_metrics = {
            'mae': mean_absolute_error(target_dias, pred_dias),
            'mse': mean_squared_error(target_dias, pred_dias),
            'rmse': np.sqrt(mean_squared_error(target_dias, pred_dias)),
            'bias': np.mean(pred_dias - target_dias),
            'std_error': np.std(pred_dias - target_dias),
            'correlation': np.corrcoef(pred_dias, target_dias)[0, 1] if np.std(pred_dias) > 0 and np.std(target_dias) > 0 else 0.0
        }
        
        return {
            'systolic': systolic_metrics,
            'diastolic': diastolic_metrics
        }
    
    @staticmethod
    def compute_comprehensive_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Compute all metrics for comprehensive evaluation"""
        waveform_metrics = BPMetricsCalculator.compute_waveform_metrics(predictions, targets)
        component_metrics = BPMetricsCalculator.compute_bp_component_metrics(predictions, targets)
        
        return {
            'waveform': waveform_metrics,
            'components': component_metrics
        }


class BPVisualizationTools:
    """Tools for visualizing BP prediction results"""
    
    @staticmethod
    def plot_prediction_examples(predictions: np.ndarray, targets: np.ndarray, 
                                output_dir: str, num_examples: int = 6, 
                                title_prefix: str = "BP Prediction"):
        """Plot example predictions vs targets"""
        plt.style.use('seaborn-v0_8')
        
        num_examples = min(num_examples, len(predictions))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(num_examples):
            ax = axes[i]
            
            # Plot target and prediction
            ax.plot(targets[i], label='Ground Truth', linewidth=2, color='blue', alpha=0.8)
            ax.plot(predictions[i], label='Prediction', linewidth=2, color='red', alpha=0.8)
            
            # Calculate metrics for this example
            mse = mean_squared_error(targets[i], predictions[i])
            mae = mean_absolute_error(targets[i], predictions[i])
            
            # Extract systolic and diastolic values
            pred_tensor = torch.tensor(predictions[i:i+1], dtype=torch.float32)
            target_tensor = torch.tensor(targets[i:i+1], dtype=torch.float32)
            
            pred_sys, pred_dias = extract_physiological_bp_values(pred_tensor)
            target_sys, target_dias = extract_physiological_bp_values(target_tensor)
            
            ax.set_title(f'Example {i+1}\nMSE: {mse:.2f}, MAE: {mae:.2f}\n'
                        f'Sys: {target_sys.item():.1f}→{pred_sys.item():.1f}, '
                        f'Dias: {target_dias.item():.1f}→{pred_dias.item():.1f}', 
                        fontsize=10)
            ax.set_xlabel('Time Points')
            ax.set_ylabel('BP (mmHg)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}_examples.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_bland_altman(predictions: np.ndarray, targets: np.ndarray, 
                         output_dir: str, component: str = 'systolic'):
        """Create Bland-Altman plots for clinical assessment"""
        # Extract component values
        pred_tensor = torch.tensor(predictions, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        
        pred_sys, pred_dias = extract_physiological_bp_values(pred_tensor)
        target_sys, target_dias = extract_physiological_bp_values(target_tensor)
        
        if component == 'systolic':
            pred_vals = pred_sys.numpy()
            target_vals = target_sys.numpy()
            title = 'Systolic BP'
            unit = 'mmHg'
        else:
            pred_vals = pred_dias.numpy()
            target_vals = target_dias.numpy()
            title = 'Diastolic BP'
            unit = 'mmHg'
        
        # Calculate differences and means
        differences = pred_vals - target_vals
        means = (pred_vals + target_vals) / 2
        
        # Calculate bias and limits of agreement
        bias = np.mean(differences)
        std_diff = np.std(differences)
        upper_loa = bias + 1.96 * std_diff
        lower_loa = bias - 1.96 * std_diff
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(means, differences, alpha=0.6, s=30)
        
        # Add reference lines
        plt.axhline(bias, color='red', linestyle='-', linewidth=2, 
                   label=f'Bias: {bias:.2f} {unit}')
        plt.axhline(upper_loa, color='red', linestyle='--', linewidth=1, 
                   label=f'Upper LoA: {upper_loa:.2f} {unit}')
        plt.axhline(lower_loa, color='red', linestyle='--', linewidth=1, 
                   label=f'Lower LoA: {lower_loa:.2f} {unit}')
        plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        plt.xlabel(f'Mean {title} ({unit})')
        plt.ylabel(f'Difference (Predicted - Actual) ({unit})')
        plt.title(f'Bland-Altman Plot: {title}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'n = {len(means)}\nBias = {bias:.2f} ± {std_diff:.2f} {unit}\n95% LoA: {lower_loa:.2f} to {upper_loa:.2f} {unit}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bland_altman_{component}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_scatter_correlation(predictions: np.ndarray, targets: np.ndarray, 
                               output_dir: str, component: str = 'systolic'):
        """Create scatter plots showing correlation"""
        # Extract component values
        pred_tensor = torch.tensor(predictions, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        
        pred_sys, pred_dias = extract_physiological_bp_values(pred_tensor)
        target_sys, target_dias = extract_physiological_bp_values(target_tensor)
        
        if component == 'systolic':
            pred_vals = pred_sys.numpy()
            target_vals = target_sys.numpy()
            title = 'Systolic BP'
        else:
            pred_vals = pred_dias.numpy()
            target_vals = target_dias.numpy()
            title = 'Diastolic BP'
        
        # Calculate correlation and regression line
        correlation = np.corrcoef(pred_vals, target_vals)[0, 1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(target_vals, pred_vals)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(target_vals, pred_vals, alpha=0.6, s=30)
        
        # Add perfect prediction line
        min_val = min(np.min(target_vals), np.min(pred_vals))
        max_val = max(np.max(target_vals), np.max(pred_vals))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect Prediction')
        
        # Add regression line
        regression_line = slope * target_vals + intercept
        plt.plot(target_vals, regression_line, 'b-', linewidth=2, alpha=0.8,
                label=f'Regression Line (R²={r_value**2:.3f})')
        
        plt.xlabel(f'Actual {title} (mmHg)')
        plt.ylabel(f'Predicted {title} (mmHg)')
        plt.title(f'{title} Correlation Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mae = mean_absolute_error(target_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(target_vals, pred_vals))
        stats_text = f'r = {correlation:.3f}\nMAE = {mae:.2f} mmHg\nRMSE = {rmse:.2f} mmHg'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'correlation_{component}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_attention_heatmap(attention_weights: np.ndarray, pattern_offsets: List[int], 
                             output_dir: str, num_examples: int = 5):
        """Visualize attention patterns as heatmaps"""
        if attention_weights is None or len(attention_weights) == 0:
            print("No attention weights available for visualization")
            return
        
        # Average attention weights across examples
        avg_attention = np.mean(attention_weights[:num_examples], axis=0)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Create labels for frame offsets
        frame_labels = [f't{offset:+d}' if offset != 0 else 't' for offset in pattern_offsets]
        
        sns.heatmap(avg_attention, 
                   xticklabels=frame_labels,
                   yticklabels=frame_labels,
                   cmap='Blues',
                   annot=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title('Average Attention Patterns\n(Temporal Frame Dependencies)')
        plt.xlabel('Key Frames')
        plt.ylabel('Query Frames')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'attention_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def create_comprehensive_report(metrics: Dict[str, Any], output_dir: str):
        """Create a comprehensive evaluation report"""
        report_path = os.path.join(output_dir, 'evaluation_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Sophisticated BP Predictor - Evaluation Report\n\n")
            
            # Waveform metrics
            f.write("## Overall Waveform Metrics\n\n")
            waveform_metrics = metrics['waveform']
            f.write(f"- **MSE**: {waveform_metrics['mse']:.4f}\n")
            f.write(f"- **MAE**: {waveform_metrics['mae']:.4f}\n")
            f.write(f"- **RMSE**: {waveform_metrics['rmse']:.4f}\n")
            f.write(f"- **R²**: {waveform_metrics['r2']:.4f}\n")
            f.write(f"- **Correlation**: {waveform_metrics['correlation']:.4f}\n\n")
            
            # Component metrics
            f.write("## Systolic BP Metrics\n\n")
            sys_metrics = metrics['components']['systolic']
            f.write(f"- **MAE**: {sys_metrics['mae']:.2f} mmHg\n")
            f.write(f"- **RMSE**: {sys_metrics['rmse']:.2f} mmHg\n")
            f.write(f"- **Bias**: {sys_metrics['bias']:.2f} mmHg\n")
            f.write(f"- **Std Error**: {sys_metrics['std_error']:.2f} mmHg\n")
            f.write(f"- **Correlation**: {sys_metrics['correlation']:.4f}\n\n")
            
            f.write("## Diastolic BP Metrics\n\n")
            dias_metrics = metrics['components']['diastolic']
            f.write(f"- **MAE**: {dias_metrics['mae']:.2f} mmHg\n")
            f.write(f"- **RMSE**: {dias_metrics['rmse']:.2f} mmHg\n")
            f.write(f"- **Bias**: {dias_metrics['bias']:.2f} mmHg\n")
            f.write(f"- **Std Error**: {dias_metrics['std_error']:.2f} mmHg\n")
            f.write(f"- **Correlation**: {dias_metrics['correlation']:.4f}\n\n")
            
            # Clinical interpretation
            f.write("## Clinical Interpretation\n\n")
            
            # Systolic assessment
            sys_mae = sys_metrics['mae']
            if sys_mae < 5:
                sys_assessment = "Excellent"
            elif sys_mae < 10:
                sys_assessment = "Good"
            elif sys_mae < 15:
                sys_assessment = "Acceptable"
            else:
                sys_assessment = "Needs Improvement"
            
            # Diastolic assessment
            dias_mae = dias_metrics['mae']
            if dias_mae < 5:
                dias_assessment = "Excellent"
            elif dias_mae < 10:
                dias_assessment = "Good"
            elif dias_mae < 15:
                dias_assessment = "Acceptable"
            else:
                dias_assessment = "Needs Improvement"
            
            f.write(f"- **Systolic Accuracy**: {sys_assessment} (MAE: {sys_mae:.1f} mmHg)\n")
            f.write(f"- **Diastolic Accuracy**: {dias_assessment} (MAE: {dias_mae:.1f} mmHg)\n\n")
            
            f.write("### Clinical Standards\n")
            f.write("- **Excellent**: MAE < 5 mmHg\n")
            f.write("- **Good**: MAE 5-10 mmHg\n")
            f.write("- **Acceptable**: MAE 10-15 mmHg\n")
            f.write("- **Needs Improvement**: MAE > 15 mmHg\n\n")
        
        print(f"Comprehensive evaluation report saved to {report_path}")


class SophisticatedBPTrainer:
    """Comprehensive trainer for sophisticated BP prediction"""
    
    def __init__(self, model: SophisticatedBPPredictor, config: ModelConfig,
                 train_loader: DataLoader, val_loader: DataLoader,
                 loss_type: str = 'composite', device: torch.device = None):
        
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = ComprehensiveBPLoss(loss_type=loss_type)
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,  # Will be updated with actual num_epochs
            eta_min=1e-6
        )
        
        # Initialize metrics calculator
        self.metrics_calculator = BPMetricsCalculator()
        self.visualization_tools = BPVisualizationTools()
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_waveform_loss': [],
            'val_waveform_loss': [],
            'train_systolic_loss': [],
            'val_systolic_loss': [],
            'train_diastolic_loss': [],
            'val_diastolic_loss': [],
            'learning_rates': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'waveform_loss': 0.0,
            'systolic_loss': 0.0,
            'diastolic_loss': 0.0
        }
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_data in progress_bar:
            try:
                # Extract sequences and targets
                sequences = batch_data['sequences'].to(self.device)
                targets = batch_data['targets'].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Calculate loss
                loss_dict = self.criterion(outputs, targets)
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                
                # Update metrics
                batch_size = sequences.size(0)
                for key, value in loss_dict.items():
                    if key in epoch_losses:
                        epoch_losses[key] += value.item() * batch_size
                
                num_batches += batch_size
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = {
            'total_loss': 0.0,
            'waveform_loss': 0.0,
            'systolic_loss': 0.0,
            'diastolic_loss': 0.0
        }
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        all_attention_weights = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch_data in progress_bar:
                try:
                    sequences = batch_data['sequences'].to(self.device)
                    targets = batch_data['targets'].to(self.device)
                    
                    # Forward pass with attention
                    outputs = self.model(sequences, return_attention=True)
                    
                    # Calculate loss
                    loss_dict = self.criterion(outputs, targets)
                    
                    # Store predictions and targets
                    predictions = outputs['waveform'].cpu().numpy()
                    targets_np = targets.cpu().numpy()
                    
                    all_predictions.append(predictions)
                    all_targets.append(targets_np)
                    
                    # Store attention weights if available
                    if 'attention_weights' in outputs:
                        all_attention_weights.append(outputs['attention_weights'].cpu().numpy())
                    
                    # Update metrics
                    batch_size = sequences.size(0)
                    for key, value in loss_dict.items():
                        if key in epoch_losses:
                            epoch_losses[key] += value.item() * batch_size
                    
                    num_batches += batch_size
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        # Compute comprehensive metrics
        if all_predictions and all_targets:
            all_predictions = np.vstack(all_predictions)
            all_targets = np.vstack(all_targets)
            comprehensive_metrics = self.metrics_calculator.compute_comprehensive_metrics(
                all_predictions, all_targets
            )
            
            # Add attention weights to metrics
            if all_attention_weights:
                comprehensive_metrics['attention_weights'] = np.concatenate(all_attention_weights, axis=0)
        else:
            comprehensive_metrics = {}
        
        return epoch_losses, comprehensive_metrics
    
    def train(self, num_epochs: int, output_dir: str, 
              save_every: int = 5, visualize_attention: bool = True,
              use_wandb: bool = False, wandb_project: str = "sophisticated-bp-predictor"):
        """Complete training loop"""
        
        # Update scheduler
        self.scheduler.T_max = num_epochs
        
        # Initialize W&B if requested
        if use_wandb:
            wandb.init(project=wandb_project, config={
                'model': 'SophisticatedBPPredictor',
                'num_epochs': num_epochs,
                'batch_size': self.train_loader.batch_size,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'loss_type': self.criterion.loss_type,
                **self.config.__dict__
            })
        
        # Create output directories
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        results_dir = os.path.join(output_dir, 'results')
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate epoch
            val_losses, val_metrics = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Store training history
            self.training_history['train_loss'].append(train_losses['total_loss'])
            self.training_history['val_loss'].append(val_losses['total_loss'])
            self.training_history['train_waveform_loss'].append(train_losses['waveform_loss'])
            self.training_history['val_waveform_loss'].append(val_losses['waveform_loss'])
            self.training_history['train_systolic_loss'].append(train_losses['systolic_loss'])
            self.training_history['val_systolic_loss'].append(val_losses['systolic_loss'])
            self.training_history['train_diastolic_loss'].append(train_losses['diastolic_loss'])
            self.training_history['val_diastolic_loss'].append(val_losses['diastolic_loss'])
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"Train Loss: {train_losses['total_loss']:.4f} | Val Loss: {val_losses['total_loss']:.4f}")
            if val_metrics and 'waveform' in val_metrics:
                wf_metrics = val_metrics['waveform']
                print(f"Val R²: {wf_metrics['r2']:.4f} | Val MAE: {wf_metrics['mae']:.4f}")
            
            # Log to W&B
            if use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/total_loss': train_losses['total_loss'],
                    'train/waveform_loss': train_losses['waveform_loss'],
                    'train/systolic_loss': train_losses['systolic_loss'],
                    'train/diastolic_loss': train_losses['diastolic_loss'],
                    'val/total_loss': val_losses['total_loss'],
                    'val/waveform_loss': val_losses['waveform_loss'],
                    'val/systolic_loss': val_losses['systolic_loss'],
                    'val/diastolic_loss': val_losses['diastolic_loss'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                if val_metrics and 'waveform' in val_metrics:
                    wf_metrics = val_metrics['waveform']
                    log_dict.update({
                        'val/r2': wf_metrics['r2'],
                        'val/mae': wf_metrics['mae'],
                        'val/correlation': wf_metrics['correlation']
                    })
                
                if val_metrics and 'components' in val_metrics:
                    sys_metrics = val_metrics['components']['systolic']
                    dias_metrics = val_metrics['components']['diastolic']
                    log_dict.update({
                        'val/systolic_mae': sys_metrics['mae'],
                        'val/diastolic_mae': dias_metrics['mae'],
                        'val/systolic_correlation': sys_metrics['correlation'],
                        'val/diastolic_correlation': dias_metrics['correlation']
                    })
                
                wandb.log(log_dict)
            
            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.best_epoch = epoch
                
                # Save best checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'config': self.config,
                    'training_history': self.training_history
                }
                
                torch.save(checkpoint, os.path.join(checkpoints_dir, 'best_model.pt'))
                print(f"New best model saved! Val Loss: {self.best_val_loss:.4f}")
            
            # Save periodic checkpoints
            if epoch % save_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_losses['total_loss'],
                    'config': self.config,
                    'training_history': self.training_history
                }
                
                torch.save(checkpoint, os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # Plot training curves
        self.plot_training_curves(results_dir)
        
        # Final evaluation with visualizations
        print("\nPerforming final evaluation...")
        final_metrics = self.evaluate_model(results_dir, visualize_attention=visualize_attention)
        
        if use_wandb:
            wandb.finish()
        
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        return final_metrics
    
    def evaluate_model(self, output_dir: str, visualize_attention: bool = True) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_attention_weights = []
        
        print("Collecting predictions for evaluation...")
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Evaluating"):
                try:
                    sequences = batch_data['sequences'].to(self.device)
                    targets = batch_data['targets'].to(self.device)
                    
                    outputs = self.model(sequences, return_attention=visualize_attention)
                    
                    predictions = outputs['waveform'].cpu().numpy()
                    targets_np = targets.cpu().numpy()
                    
                    all_predictions.append(predictions)
                    all_targets.append(targets_np)
                    
                    if visualize_attention and 'attention_weights' in outputs:
                        all_attention_weights.append(outputs['attention_weights'].cpu().numpy())
                        
                except Exception as e:
                    print(f"Error in evaluation batch: {e}")
                    continue
        
        if not all_predictions:
            print("No valid predictions collected!")
            return {}
        
        # Combine all predictions and targets
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        print(f"Evaluating {len(all_predictions)} samples...")
        
        # Compute comprehensive metrics
        metrics = self.metrics_calculator.compute_comprehensive_metrics(all_predictions, all_targets)
        
        # Create visualizations
        print("Creating visualizations...")
        
        # Prediction examples
        self.visualization_tools.plot_prediction_examples(
            all_predictions, all_targets, output_dir, num_examples=6
        )
        
        # Scatter plots
        self.visualization_tools.plot_scatter_correlation(
            all_predictions, all_targets, output_dir, component='systolic'
        )
        self.visualization_tools.plot_scatter_correlation(
            all_predictions, all_targets, output_dir, component='diastolic'
        )
        
        # Bland-Altman plots
        self.visualization_tools.plot_bland_altman(
            all_predictions, all_targets, output_dir, component='systolic'
        )
        self.visualization_tools.plot_bland_altman(
            all_predictions, all_targets, output_dir, component='diastolic'
        )
        
        # Attention visualization
        if visualize_attention and all_attention_weights:
            attention_weights = np.concatenate(all_attention_weights, axis=0)
            self.visualization_tools.plot_attention_heatmap(
                attention_weights, self.config.pattern_offsets, output_dir
            )
        
        # Create comprehensive report
        self.visualization_tools.create_comprehensive_report(metrics, output_dir)
        
        return metrics
    
    def plot_training_curves(self, output_dir: str):
        """Plot training curves"""
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total loss
        axes[0, 0].plot(epochs, self.training_history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.training_history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Waveform loss
        axes[0, 1].plot(epochs, self.training_history['train_waveform_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.training_history['val_waveform_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Waveform Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Systolic loss
        axes[1, 0].plot(epochs, self.training_history['train_systolic_loss'], label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.training_history['val_systolic_loss'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Systolic Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Diastolic loss
        axes[1, 1].plot(epochs, self.training_history['train_diastolic_loss'], label='Train', linewidth=2)
        axes[1, 1].plot(epochs, self.training_history['val_diastolic_loss'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Diastolic Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Learning rate curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.training_history['learning_rates'], linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('epoch', 0) 
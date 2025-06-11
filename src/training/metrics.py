"""
Metrics for blood pressure prediction evaluation.
Includes clinical metrics and standard ML metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator for various BP prediction metrics."""
    
    def __init__(self, bp_norm_range: Tuple[float, float] = (40.0, 200.0)):
        """
        Initialize metrics calculator.
        
        Args:
            bp_norm_range: (min, max) values used for BP normalization
        """
        self.bp_min, self.bp_max = bp_norm_range
    
    def denormalize_bp(self, normalized_bp: torch.Tensor) -> torch.Tensor:
        """Denormalize BP values back to mmHg."""
        return normalized_bp * (self.bp_max - self.bp_min) + self.bp_min
    
    def compute_bp_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute basic BP prediction metrics.
        
        Args:
            predictions: Predicted BP values [batch, bp_length]
            targets: Target BP values [batch, bp_length]
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy for easier computation
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Denormalize if needed (assume inputs are normalized)
        if pred_np.max() <= 1.0 and pred_np.min() >= 0.0:
            pred_np = pred_np * (self.bp_max - self.bp_min) + self.bp_min
            target_np = target_np * (self.bp_max - self.bp_min) + self.bp_min
        
        metrics = {}
        
        # Basic regression metrics
        pred_flat = pred_np.flatten()
        target_flat = target_np.flatten()
        
        metrics['mae'] = mean_absolute_error(target_flat, pred_flat)
        metrics['mse'] = mean_squared_error(target_flat, pred_flat)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # R-squared
        try:
            metrics['r2'] = r2_score(target_flat, pred_flat)
        except:
            metrics['r2'] = 0.0
        
        # Correlation
        try:
            corr, _ = pearsonr(target_flat, pred_flat)
            metrics['correlation'] = corr if not np.isnan(corr) else 0.0
        except:
            metrics['correlation'] = 0.0
        
        return metrics
    
    def compute_clinical_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute clinical BP metrics (systolic, diastolic, pulse pressure).
        
        Args:
            predictions: Predicted BP waveforms [batch, bp_length]
            targets: Target BP waveforms [batch, bp_length]
            
        Returns:
            Dictionary of clinical metrics
        """
        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Denormalize if needed
        if pred_np.max() <= 1.0 and pred_np.min() >= 0.0:
            pred_np = pred_np * (self.bp_max - self.bp_min) + self.bp_min
            target_np = target_np * (self.bp_max - self.bp_min) + self.bp_min
        
        metrics = {}
        
        # Extract systolic and diastolic values
        pred_systolic = np.max(pred_np, axis=1)
        pred_diastolic = np.min(pred_np, axis=1)
        target_systolic = np.max(target_np, axis=1)
        target_diastolic = np.min(target_np, axis=1)
        
        # Systolic metrics
        metrics['systolic_mae'] = mean_absolute_error(target_systolic, pred_systolic)
        metrics['systolic_rmse'] = np.sqrt(mean_squared_error(target_systolic, pred_systolic))
        
        # Diastolic metrics
        metrics['diastolic_mae'] = mean_absolute_error(target_diastolic, pred_diastolic)
        metrics['diastolic_rmse'] = np.sqrt(mean_squared_error(target_diastolic, pred_diastolic))
        
        # Pulse pressure metrics
        pred_pp = pred_systolic - pred_diastolic
        target_pp = target_systolic - target_diastolic
        metrics['pulse_pressure_mae'] = mean_absolute_error(target_pp, pred_pp)
        metrics['pulse_pressure_rmse'] = np.sqrt(mean_squared_error(target_pp, pred_pp))
        
        # Mean arterial pressure (approximation)
        pred_map = (pred_systolic + 2 * pred_diastolic) / 3
        target_map = (target_systolic + 2 * target_diastolic) / 3
        metrics['map_mae'] = mean_absolute_error(target_map, pred_map)
        metrics['map_rmse'] = np.sqrt(mean_squared_error(target_map, pred_map))
        
        # Clinical accuracy thresholds
        # AAMI standard: ±5 mmHg for 68% of readings, ±15 mmHg for 95% of readings
        systolic_errors = np.abs(pred_systolic - target_systolic)
        diastolic_errors = np.abs(pred_diastolic - target_diastolic)
        
        metrics['systolic_accuracy_5mmhg'] = np.mean(systolic_errors <= 5.0) * 100
        metrics['systolic_accuracy_15mmhg'] = np.mean(systolic_errors <= 15.0) * 100
        metrics['diastolic_accuracy_5mmhg'] = np.mean(diastolic_errors <= 5.0) * 100
        metrics['diastolic_accuracy_15mmhg'] = np.mean(diastolic_errors <= 15.0) * 100
        
        # BHS grading criteria
        metrics['bhs_systolic'] = self._compute_bhs_grade(systolic_errors)
        metrics['bhs_diastolic'] = self._compute_bhs_grade(diastolic_errors)
        
        return metrics
    
    def _compute_bhs_grade(self, errors: np.ndarray) -> str:
        """Compute British Hypertension Society grade."""
        pct_5 = np.mean(errors <= 5.0) * 100
        pct_10 = np.mean(errors <= 10.0) * 100
        pct_15 = np.mean(errors <= 15.0) * 100
        
        if pct_5 >= 60 and pct_10 >= 85 and pct_15 >= 95:
            return 'A'
        elif pct_5 >= 50 and pct_10 >= 75 and pct_15 >= 90:
            return 'B'
        elif pct_5 >= 40 and pct_10 >= 65 and pct_15 >= 85:
            return 'C'
        else:
            return 'D'
    
    def compute_waveform_similarity_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute waveform shape similarity metrics.
        
        Args:
            predictions: Predicted BP waveforms [batch, bp_length]
            targets: Target BP waveforms [batch, bp_length]
            
        Returns:
            Dictionary of similarity metrics
        """
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        metrics = {}
        batch_size = pred_np.shape[0]
        
        # Compute per-sample similarities
        correlations = []
        dtw_distances = []
        
        for i in range(batch_size):
            pred_signal = pred_np[i]
            target_signal = target_np[i]
            
            # Correlation
            try:
                corr, _ = pearsonr(pred_signal, target_signal)
                correlations.append(corr if not np.isnan(corr) else 0.0)
            except:
                correlations.append(0.0)
            
            # Simple DTW approximation (Euclidean distance)
            dtw_dist = np.linalg.norm(pred_signal - target_signal)
            dtw_distances.append(dtw_dist)
        
        metrics['waveform_correlation'] = np.mean(correlations)
        metrics['waveform_dtw_distance'] = np.mean(dtw_distances)
        
        # Frequency domain analysis (simplified)
        try:
            # Compute power spectral density similarity
            from scipy.signal import welch
            
            psd_similarities = []
            for i in range(min(batch_size, 10)):  # Limit for computational efficiency
                pred_signal = pred_np[i]
                target_signal = target_np[i]
                
                # Compute PSDs
                _, pred_psd = welch(pred_signal, nperseg=min(len(pred_signal)//2, 16))
                _, target_psd = welch(target_signal, nperseg=min(len(target_signal)//2, 16))
                
                # Normalize PSDs
                pred_psd = pred_psd / np.sum(pred_psd)
                target_psd = target_psd / np.sum(target_psd)
                
                # Compute similarity (1 - Jensen-Shannon divergence)
                try:
                    from scipy.spatial.distance import jensenshannon
                    js_div = jensenshannon(pred_psd, target_psd)
                    psd_similarities.append(1 - js_div)
                except:
                    # Fallback to correlation
                    corr, _ = pearsonr(pred_psd, target_psd)
                    psd_similarities.append(corr if not np.isnan(corr) else 0.0)
            
            metrics['spectral_similarity'] = np.mean(psd_similarities)
        
        except ImportError:
            # If scipy.signal is not available, skip spectral analysis
            metrics['spectral_similarity'] = 0.0
        
        return metrics
    
    def compute_comprehensive_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            predictions: Predicted BP values
            targets: Target BP values
            
        Returns:
            Dictionary containing all metrics
        """
        all_metrics = {}
        
        # Basic metrics
        all_metrics.update(self.compute_bp_metrics(predictions, targets))
        
        # Clinical metrics
        all_metrics.update(self.compute_clinical_metrics(predictions, targets))
        
        # Waveform similarity metrics
        all_metrics.update(self.compute_waveform_similarity_metrics(predictions, targets))
        
        return all_metrics
    
    def compute_error_statistics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute detailed error statistics.
        
        Args:
            predictions: Predicted BP values
            targets: Target BP values
            
        Returns:
            Dictionary of error statistics
        """
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Denormalize if needed
        if pred_np.max() <= 1.0 and pred_np.min() >= 0.0:
            pred_np = pred_np * (self.bp_max - self.bp_min) + self.bp_min
            target_np = target_np * (self.bp_max - self.bp_min) + self.bp_min
        
        errors = pred_np - target_np
        
        return {
            'error_mean': np.mean(errors),
            'error_std': np.std(errors),
            'error_median': np.median(errors),
            'error_q25': np.percentile(errors, 25),
            'error_q75': np.percentile(errors, 75),
            'error_min': np.min(errors),
            'error_max': np.max(errors),
            'bias': np.mean(errors),  # Same as error_mean but clinically relevant
            'precision': np.std(errors)  # Same as error_std but clinically relevant
        }


def compute_batch_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    metrics_calculator: MetricsCalculator = None
) -> Dict[str, float]:
    """
    Convenience function to compute metrics for a batch.
    
    Args:
        predictions: Predicted values
        targets: Target values
        metrics_calculator: Optional metrics calculator instance
        
    Returns:
        Dictionary of computed metrics
    """
    if metrics_calculator is None:
        metrics_calculator = MetricsCalculator()
    
    return metrics_calculator.compute_bp_metrics(predictions, targets)


def format_metrics_for_logging(metrics: Dict[str, float], prefix: str = "") -> Dict[str, float]:
    """
    Format metrics dictionary for logging with optional prefix.
    
    Args:
        metrics: Metrics dictionary
        prefix: Optional prefix for metric names
        
    Returns:
        Formatted metrics dictionary
    """
    if not prefix:
        return metrics
    
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def create_metrics_summary(metrics_history: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Create summary statistics from metrics history.
    
    Args:
        metrics_history: List of metrics dictionaries
        
    Returns:
        Summary statistics (mean, std, min, max) for each metric
    """
    if not metrics_history:
        return {}
    
    # Get all metric names
    all_metrics = set()
    for metrics in metrics_history:
        all_metrics.update(metrics.keys())
    
    summary = {}
    
    for metric_name in all_metrics:
        values = [m.get(metric_name, 0.0) for m in metrics_history if metric_name in m]
        
        if values:
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    return summary 
#!/usr/bin/env python3
"""
Training-Matched Evaluation Script

This script evaluates the model EXACTLY the same way as during training:
- Same data split (150 samples max)
- Same train/validation split (70/15/15)
- Same BP extraction method
- Same data processing pipeline
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.evaluate_improved_bp_predictor import ImprovedBPEvaluator

def training_matched_evaluation():
    """Run evaluation matching EXACTLY the training setup"""
    
    MODEL_PATH = "/home/lucas_takanori/phd/bp_prediction/experiments/improved_bp_predictor_continuous/checkpoints/best_model.pt"
    CONFIG_PATH = "/home/lucas_takanori/phd/bp_prediction/configs/sophisticated_bp_predictor_improved.yaml"
    OUTPUT_DIR = "./evaluation_results_training_matched"
    
    print("=" * 80)
    print("🎯 TRAINING-MATCHED BP PREDICTOR EVALUATION")
    print("=" * 80)
    print("This evaluation matches EXACTLY the training setup:")
    print("- Max 150 samples (as in config)")
    print("- 70/15/15 train/val/test split")
    print("- Same BP extraction method")
    print("- Same data processing")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ImprovedBPEvaluator(MODEL_PATH, CONFIG_PATH)
    
    # Get configuration
    data_config = evaluator.config.get('data_config', {})
    
    # Create dataset EXACTLY as in training
    from utils.data_utils import DataPathManager
    from scripts.train_improved_bp_predictor import ImprovedBPDataset
    
    data_root = data_config.get('root_path', '/home/lucas_takanori/phd/data')
    path_manager = DataPathManager(
        subject="subject001",
        session="baseline",
        root=data_root
    )
    h5_file_path = str(path_manager._h5_path)
    
    print(f"📁 Data file: {h5_file_path}")
    
    # Create dataset EXACTLY as in training
    full_dataset = ImprovedBPDataset(
        data_root=h5_file_path,
        pattern_offsets=data_config.get('pattern_offsets', [-4, -3, -2, -1, 0, 1, 2]),
        max_samples_per_subject=data_config.get('max_samples_per_subject', 150),  # EXACTLY 150
        sequence_step_size=data_config.get('sequence_step_size', 15),
        use_augmentation=data_config.get('use_augmentation', False),
        noise_level=data_config.get('noise_level', 0.0)
    )
    
    print(f"📊 Total dataset size: {len(full_dataset)} sequences")
    
    # Apply EXACT same split as training
    train_split = data_config.get('train_split', 0.7)
    val_split = data_config.get('val_split', 0.15)
    test_split = data_config.get('test_split', 0.15)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"📋 Data splits:")
    print(f"  - Training:   {train_size} sequences ({train_split:.1%})")
    print(f"  - Validation: {val_size} sequences ({val_split:.1%})")
    print(f"  - Test:       {test_size} sequences ({test_split:.1%})")
    
    # Use same random seed as training for consistent splits
    torch.manual_seed(42)  # Same seed as in config
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Evaluate on VALIDATION set (same as training used for "best model")
    print(f"\n🎯 Evaluating on VALIDATION set ({len(val_dataset)} samples)...")
    val_results = evaluate_dataset(evaluator, val_dataset, "Validation")
    
    # Evaluate on TEST set (truly unseen data)
    print(f"\n🧪 Evaluating on TEST set ({len(test_dataset)} samples)...")
    test_results = evaluate_dataset(evaluator, test_dataset, "Test")
    
    # Create comparison
    print("\n" + "=" * 80)
    print("📊 TRAINING-MATCHED COMPARISON")
    print("=" * 80)
    
    print(f"🎯 VALIDATION SET RESULTS (used during training):")
    print_metrics(val_results['metrics'])
    
    print(f"\n🧪 TEST SET RESULTS (truly unseen):")
    print_metrics(test_results['metrics'])
    
    print(f"\n📈 TRAINING REPORTED RESULTS:")
    print(f"  - Systolic MAE:  3.34 mmHg")
    print(f"  - Diastolic MAE: 3.62 mmHg") 
    print(f"  - Waveform MAE:  3.18 mmHg")
    
    # Calculate differences
    val_metrics = val_results['metrics']
    print(f"\n🔍 VALIDATION vs TRAINING DIFFERENCE:")
    print(f"  - Systolic MAE:  {val_metrics['systolic']['mae']:.2f} vs 3.34 = {val_metrics['systolic']['mae'] - 3.34:+.2f}")
    print(f"  - Diastolic MAE: {val_metrics['diastolic']['mae']:.2f} vs 3.62 = {val_metrics['diastolic']['mae'] - 3.62:+.2f}")
    print(f"  - Waveform MAE:  {val_metrics['waveform']['mae']:.2f} vs 3.18 = {val_metrics['waveform']['mae'] - 3.18:+.2f}")
    
    # Save results
    save_results(val_results, test_results, OUTPUT_DIR)
    
    return val_results, test_results

def evaluate_dataset(evaluator, dataset, name):
    """Evaluate model on a specific dataset split"""
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=2
    )
    
    # Collect predictions
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequences'].to(evaluator.device)
            targets = batch['targets'].to(evaluator.device)
            
            # Model prediction
            outputs = evaluator.model(sequences)
            pred_waveforms = outputs['waveform']
            pred_systolic = outputs['systolic'].squeeze() if 'systolic' in outputs else None
            pred_diastolic = outputs['diastolic'].squeeze() if 'diastolic' in outputs else None
            
            # Extract physiological values using training-compatible method
            true_sys, true_dias = evaluator._extract_bp_values_training_compatible(targets)
            
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
                    pred_sys, _ = evaluator._extract_bp_values_training_compatible_single(pred_waveforms[i:i+1])
                    pred_dict['systolic'] = pred_sys[0] if isinstance(pred_sys, np.ndarray) else pred_sys
                
                if pred_diastolic is not None:
                    pred_dict['diastolic'] = pred_diastolic[i].cpu().numpy()
                else:
                    _, pred_dias = evaluator._extract_bp_values_training_compatible_single(pred_waveforms[i:i+1])
                    pred_dict['diastolic'] = pred_dias[0] if isinstance(pred_dias, np.ndarray) else pred_dias
                
                all_predictions.append(pred_dict)
                all_targets.append(target_dict)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(all_predictions, all_targets)
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'metrics': metrics,
        'name': name
    }

def print_metrics(metrics):
    """Print metrics in a nice format"""
    print(f"  Systolic  MAE: {metrics['systolic']['mae']:.2f} mmHg (R²={metrics['systolic']['r2']:.3f})")
    print(f"  Diastolic MAE: {metrics['diastolic']['mae']:.2f} mmHg (R²={metrics['diastolic']['r2']:.3f})")
    print(f"  Waveform  MAE: {metrics['waveform']['mae']:.2f} mmHg (R²={metrics['waveform']['r2']:.3f})")
    print(f"  Clinical Accuracy:")
    print(f"    Systolic  ≤5mmHg:  {metrics['clinical']['systolic_5mmhg']:.1f}%")
    print(f"    Systolic  ≤10mmHg: {metrics['clinical']['systolic_10mmhg']:.1f}%")
    print(f"    Diastolic ≤5mmHg:  {metrics['clinical']['diastolic_5mmhg']:.1f}%")
    print(f"    Diastolic ≤10mmHg: {metrics['clinical']['diastolic_10mmhg']:.1f}%")

def save_results(val_results, test_results, output_dir):
    """Save comparison results"""
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    datasets = [val_results, test_results]
    titles = ['Validation Set', 'Test Set']
    
    for row, (data, title) in enumerate(zip(datasets, titles)):
        predictions = data['predictions']
        targets = data['targets']
        
        # Extract data
        pred_sys = np.array([float(p['systolic']) for p in predictions])
        pred_dias = np.array([float(p['diastolic']) for p in predictions])
        pred_waveforms = np.array([p['waveform'] for p in predictions])
        
        true_sys = np.array([float(t['systolic']) for t in targets])
        true_dias = np.array([float(t['diastolic']) for t in targets])
        true_waveforms = np.array([t['waveform'] for t in targets])
        
        # Waveform correlation
        ax = axes[row, 0]
        ax.scatter(true_waveforms.flatten(), pred_waveforms.flatten(), alpha=0.5, s=1)
        min_val = min(true_waveforms.min(), pred_waveforms.min())
        max_val = max(true_waveforms.max(), pred_waveforms.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax.set_xlabel('True Waveform (mmHg)')
        ax.set_ylabel('Pred Waveform (mmHg)')
        ax.set_title(f'{title}\nWaveform (R²={data["metrics"]["waveform"]["r2"]:.3f})')
        ax.grid(True, alpha=0.3)
        
        # Systolic correlation
        ax = axes[row, 1]
        ax.scatter(true_sys, pred_sys, alpha=0.6, s=20)
        min_val = min(true_sys.min(), pred_sys.min())
        max_val = max(true_sys.max(), pred_sys.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax.set_xlabel('True Systolic (mmHg)')
        ax.set_ylabel('Pred Systolic (mmHg)')
        ax.set_title(f'Systolic (R²={data["metrics"]["systolic"]["r2"]:.3f})')
        ax.grid(True, alpha=0.3)
        
        # Diastolic correlation
        ax = axes[row, 2]
        ax.scatter(true_dias, pred_dias, alpha=0.6, s=20)
        min_val = min(true_dias.min(), pred_dias.min())
        max_val = max(true_dias.max(), pred_dias.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax.set_xlabel('True Diastolic (mmHg)')
        ax.set_ylabel('Pred Diastolic (mmHg)')
        ax.set_title(f'Diastolic (R²={data["metrics"]["diastolic"]["r2"]:.3f})')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_matched_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics
    import json
    results_summary = {
        'validation': val_results['metrics'],
        'test': test_results['metrics'],
        'training_reported': {
            'systolic_mae': 3.34,
            'diastolic_mae': 3.62,
            'waveform_mae': 3.18
        }
    }
    
    with open(os.path.join(output_dir, 'training_matched_results.json'), 'w') as f:
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
        
        json.dump(convert_numpy_types(results_summary), f, indent=2)
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📊 Plot: training_matched_comparison.png")
    print(f"📄 Data: training_matched_results.json")

if __name__ == '__main__':
    training_matched_evaluation() 
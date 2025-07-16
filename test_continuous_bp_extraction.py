#!/usr/bin/env python3
"""
Test Script for Continuous BP Extraction
Verifies that the improved extraction methods produce smooth, continuous values
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import ndimage
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_root)

def extract_bp_values_old(waveform):
    """Old discrete extraction method"""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    
    if len(waveform.shape) > 1:
        waveform = waveform.flatten()
    
    # Simple max/min extraction
    systolic = np.max(waveform)
    sys_idx = np.argmax(waveform)
    
    search_start = max(sys_idx + 1, len(waveform) // 2)
    if search_start < len(waveform):
        post_systolic = waveform[search_start:]
        diastolic = np.min(post_systolic)
    else:
        diastolic = np.min(waveform)
    
    return systolic, diastolic

def extract_bp_values_continuous(waveform):
    """New continuous extraction method"""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    
    if len(waveform.shape) > 1:
        waveform = waveform.flatten()
    
    # Apply smoothing for continuous extraction
    if len(waveform) > 7:
        # Gaussian smoothing for continuous values
        smoothed = ndimage.gaussian_filter1d(waveform, sigma=1.0)
    else:
        smoothed = waveform
    
    # Systolic with parabolic interpolation
    sys_idx = np.argmax(smoothed)
    systolic = smoothed[sys_idx]
    
    # Parabolic interpolation for sub-sample precision
    if 1 <= sys_idx <= len(smoothed) - 2:
        y1, y2, y3 = smoothed[sys_idx-1], smoothed[sys_idx], smoothed[sys_idx+1]
        a = (y1 + y3 - 2*y2) / 2
        if abs(a) > 1e-6:
            offset = (y1 - y3) / (4 * a)
            systolic = y2 + a * offset * offset
    
    # Diastolic with smooth minimum finding
    search_start = max(sys_idx + 1, int(len(smoothed) * 0.6))
    search_end = min(len(smoothed), int(len(smoothed) * 0.95))
    
    if search_start < search_end:
        post_systolic = smoothed[search_start:search_end]
        local_min_idx = np.argmin(post_systolic)
        global_min_idx = search_start + local_min_idx
        diastolic = smoothed[global_min_idx]
        
        # Parabolic interpolation for diastolic
        if search_start + 1 <= global_min_idx <= search_end - 2:
            y1 = smoothed[global_min_idx-1]
            y2 = smoothed[global_min_idx]
            y3 = smoothed[global_min_idx+1]
            a = (y1 + y3 - 2*y2) / 2
            if abs(a) > 1e-6:
                offset = (y1 - y3) / (4 * a)
                diastolic = y2 + a * offset * offset
    else:
        diastolic = np.min(smoothed)
    
    # Ensure physiological constraint with smooth enforcement
    pulse_pressure = systolic - diastolic
    min_pulse_pressure = 15.0
    
    if pulse_pressure < min_pulse_pressure:
        center_pressure = (systolic + diastolic) / 2
        systolic = center_pressure + min_pulse_pressure / 2
        diastolic = center_pressure - min_pulse_pressure / 2
    
    return systolic, diastolic

def generate_synthetic_bp_waveforms(n_samples=1000, noise_level=0.05):
    """Generate synthetic BP waveforms with realistic variations"""
    waveforms = []
    true_systolic = []
    true_diastolic = []
    
    for i in range(n_samples):
        # Generate realistic BP parameters
        sys_pressure = np.random.normal(120, 15)  # mmHg
        dias_pressure = np.random.normal(80, 10)  # mmHg
        
        # Ensure physiological constraints
        if dias_pressure >= sys_pressure - 10:
            dias_pressure = sys_pressure - 15
        
        true_systolic.append(sys_pressure)
        true_diastolic.append(dias_pressure)
        
        # Generate synthetic cardiac cycle
        t = np.linspace(0, 2*np.pi, 50)
        
        # Realistic BP waveform shape
        systolic_phase = np.exp(-((t - 0.8) / 0.4)**2) * (sys_pressure - dias_pressure)
        diastolic_baseline = dias_pressure
        
        # Add secondary wave (dicrotic notch)
        dicrotic_wave = 0.1 * (sys_pressure - dias_pressure) * np.exp(-((t - 4.0) / 0.6)**2)
        
        waveform = diastolic_baseline + systolic_phase + dicrotic_wave
        
        # Add realistic noise
        noise = np.random.normal(0, noise_level * (sys_pressure - dias_pressure), len(waveform))
        waveform += noise
        
        waveforms.append(waveform)
    
    return np.array(waveforms), np.array(true_systolic), np.array(true_diastolic)

def test_extraction_methods():
    """Test and compare old vs new extraction methods"""
    print("Generating synthetic BP waveforms...")
    waveforms, true_sys, true_dias = generate_synthetic_bp_waveforms(n_samples=500)
    
    # Extract using both methods
    old_sys = []
    old_dias = []
    new_sys = []
    new_dias = []
    
    print("Extracting BP values using both methods...")
    for waveform in waveforms:
        # Old method
        sys_old, dias_old = extract_bp_values_old(waveform)
        old_sys.append(sys_old)
        old_dias.append(dias_old)
        
        # New method
        sys_new, dias_new = extract_bp_values_continuous(waveform)
        new_sys.append(sys_new)
        new_dias.append(dias_new)
    
    old_sys = np.array(old_sys)
    old_dias = np.array(old_dias)
    new_sys = np.array(new_sys)
    new_dias = np.array(new_dias)
    
    # Calculate metrics
    print("\nMETHOD COMPARISON RESULTS:")
    print("=" * 50)
    
    # R² scores
    old_sys_r2 = np.corrcoef(true_sys, old_sys)[0,1]**2
    old_dias_r2 = np.corrcoef(true_dias, old_dias)[0,1]**2
    new_sys_r2 = np.corrcoef(true_sys, new_sys)[0,1]**2
    new_dias_r2 = np.corrcoef(true_dias, new_dias)[0,1]**2
    
    print(f"SYSTOLIC R²:")
    print(f"  Old method: {old_sys_r2:.4f}")
    print(f"  New method: {new_sys_r2:.4f}")
    print(f"  Improvement: {new_sys_r2 - old_sys_r2:.4f}")
    
    print(f"\nDIASTOLIC R²:")
    print(f"  Old method: {old_dias_r2:.4f}")
    print(f"  New method: {new_dias_r2:.4f}")
    print(f"  Improvement: {new_dias_r2 - old_dias_r2:.4f}")
    
    # MAE
    old_sys_mae = np.mean(np.abs(true_sys - old_sys))
    old_dias_mae = np.mean(np.abs(true_dias - old_dias))
    new_sys_mae = np.mean(np.abs(true_sys - new_sys))
    new_dias_mae = np.mean(np.abs(true_dias - new_dias))
    
    print(f"\nSYSTOLIC MAE:")
    print(f"  Old method: {old_sys_mae:.2f} mmHg")
    print(f"  New method: {new_sys_mae:.2f} mmHg")
    print(f"  Improvement: {old_sys_mae - new_sys_mae:.2f} mmHg")
    
    print(f"\nDIASTOLIC MAE:")
    print(f"  Old method: {old_dias_mae:.2f} mmHg")
    print(f"  New method: {new_dias_mae:.2f} mmHg")
    print(f"  Improvement: {old_dias_mae - new_dias_mae:.2f} mmHg")
    
    # Check for discrete patterns
    def check_discretization(values, name):
        unique_vals = len(np.unique(np.round(values, 1)))
        total_vals = len(values)
        discretization_ratio = unique_vals / total_vals
        print(f"{name} discretization: {unique_vals}/{total_vals} = {discretization_ratio:.3f}")
        return discretization_ratio
    
    print(f"\nDISCRETIZATION ANALYSIS:")
    print("(Higher ratio = more continuous, lower ratio = more discrete)")
    old_sys_disc = check_discretization(old_sys, "Old Systolic")
    new_sys_disc = check_discretization(new_sys, "New Systolic")
    old_dias_disc = check_discretization(old_dias, "Old Diastolic")
    new_dias_disc = check_discretization(new_dias, "New Diastolic")
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Systolic comparisons
    axes[0,0].scatter(true_sys, old_sys, alpha=0.6, s=20)
    axes[0,0].plot([true_sys.min(), true_sys.max()], [true_sys.min(), true_sys.max()], 'k--')
    axes[0,0].set_xlabel('True Systolic')
    axes[0,0].set_ylabel('Predicted Systolic')
    axes[0,0].set_title(f'Old Method\nR²={old_sys_r2:.3f}')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].scatter(true_sys, new_sys, alpha=0.6, s=20)
    axes[0,1].plot([true_sys.min(), true_sys.max()], [true_sys.min(), true_sys.max()], 'k--')
    axes[0,1].set_xlabel('True Systolic')
    axes[0,1].set_ylabel('Predicted Systolic')
    axes[0,1].set_title(f'New Method\nR²={new_sys_r2:.3f}')
    axes[0,1].grid(True, alpha=0.3)
    
    # Bland-Altman for systolic
    mean_old_sys = (true_sys + old_sys) / 2
    diff_old_sys = old_sys - true_sys
    mean_new_sys = (true_sys + new_sys) / 2
    diff_new_sys = new_sys - true_sys
    
    axes[0,2].scatter(mean_old_sys, diff_old_sys, alpha=0.6, s=20)
    axes[0,2].axhline(0, color='k', linestyle='-', alpha=0.8)
    axes[0,2].axhline(np.mean(diff_old_sys), color='b', linestyle='--', alpha=0.8)
    axes[0,2].set_xlabel('Mean Systolic')
    axes[0,2].set_ylabel('Difference')
    axes[0,2].set_title('Old Method\nBland-Altman')
    axes[0,2].grid(True, alpha=0.3)
    
    axes[0,3].scatter(mean_new_sys, diff_new_sys, alpha=0.6, s=20)
    axes[0,3].axhline(0, color='k', linestyle='-', alpha=0.8)
    axes[0,3].axhline(np.mean(diff_new_sys), color='b', linestyle='--', alpha=0.8)
    axes[0,3].set_xlabel('Mean Systolic')
    axes[0,3].set_ylabel('Difference')
    axes[0,3].set_title('New Method\nBland-Altman')
    axes[0,3].grid(True, alpha=0.3)
    
    # Diastolic comparisons
    axes[1,0].scatter(true_dias, old_dias, alpha=0.6, s=20)
    axes[1,0].plot([true_dias.min(), true_dias.max()], [true_dias.min(), true_dias.max()], 'k--')
    axes[1,0].set_xlabel('True Diastolic')
    axes[1,0].set_ylabel('Predicted Diastolic')
    axes[1,0].set_title(f'Old Method\nR²={old_dias_r2:.3f}')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].scatter(true_dias, new_dias, alpha=0.6, s=20)
    axes[1,1].plot([true_dias.min(), true_dias.max()], [true_dias.min(), true_dias.max()], 'k--')
    axes[1,1].set_xlabel('True Diastolic')
    axes[1,1].set_ylabel('Predicted Diastolic')
    axes[1,1].set_title(f'New Method\nR²={new_dias_r2:.3f}')
    axes[1,1].grid(True, alpha=0.3)
    
    # Bland-Altman for diastolic
    mean_old_dias = (true_dias + old_dias) / 2
    diff_old_dias = old_dias - true_dias
    mean_new_dias = (true_dias + new_dias) / 2
    diff_new_dias = new_dias - true_dias
    
    axes[1,2].scatter(mean_old_dias, diff_old_dias, alpha=0.6, s=20)
    axes[1,2].axhline(0, color='k', linestyle='-', alpha=0.8)
    axes[1,2].axhline(np.mean(diff_old_dias), color='b', linestyle='--', alpha=0.8)
    axes[1,2].set_xlabel('Mean Diastolic')
    axes[1,2].set_ylabel('Difference')
    axes[1,2].set_title('Old Method\nBland-Altman')
    axes[1,2].grid(True, alpha=0.3)
    
    axes[1,3].scatter(mean_new_dias, diff_new_dias, alpha=0.6, s=20)
    axes[1,3].axhline(0, color='k', linestyle='-', alpha=0.8)
    axes[1,3].axhline(np.mean(diff_new_dias), color='b', linestyle='--', alpha=0.8)
    axes[1,3].set_xlabel('Mean Diastolic')
    axes[1,3].set_ylabel('Difference')
    axes[1,3].set_title('New Method\nBland-Altman')
    axes[1,3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bp_extraction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSUMMARY:")
    print("=" * 50)
    if new_sys_r2 > old_sys_r2 and new_dias_r2 > old_dias_r2:
        print("✅ NEW METHOD SHOWS IMPROVED CORRELATIONS")
    else:
        print("❌ NEW METHOD NEEDS FURTHER IMPROVEMENT")
        
    if new_sys_disc > old_sys_disc and new_dias_disc > old_dias_disc:
        print("✅ NEW METHOD PRODUCES MORE CONTINUOUS VALUES")
    else:
        print("❌ NEW METHOD STILL HAS DISCRETIZATION ISSUES")
        
    if new_sys_mae < old_sys_mae and new_dias_mae < old_dias_mae:
        print("✅ NEW METHOD HAS LOWER ERRORS")
    else:
        print("❌ NEW METHOD HAS HIGHER ERRORS")

def show_sample_waveforms():
    """Show sample waveforms and extraction points"""
    print("\nGenerating sample waveform comparison...")
    waveforms, true_sys, true_dias = generate_synthetic_bp_waveforms(n_samples=5)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i in range(5):
        waveform = waveforms[i]
        
        # Extract using both methods
        sys_old, dias_old = extract_bp_values_old(waveform)
        sys_new, dias_new = extract_bp_values_continuous(waveform)
        
        # Plot waveform
        t = np.linspace(0, 1, len(waveform))
        axes[i].plot(t, waveform, 'b-', linewidth=2, label='Original')
        
        # Apply smoothing for visualization
        smoothed = ndimage.gaussian_filter1d(waveform, sigma=1.0)
        axes[i].plot(t, smoothed, 'g--', alpha=0.7, label='Smoothed')
        
        # Mark extraction points
        axes[i].axhline(y=true_sys[i], color='red', linestyle=':', alpha=0.8, label=f'True SBP: {true_sys[i]:.1f}')
        axes[i].axhline(y=true_dias[i], color='red', linestyle=':', alpha=0.8, label=f'True DBP: {true_dias[i]:.1f}')
        
        axes[i].axhline(y=sys_old, color='orange', linestyle='--', alpha=0.8, label=f'Old SBP: {sys_old:.1f}')
        axes[i].axhline(y=dias_old, color='orange', linestyle='--', alpha=0.8, label=f'Old DBP: {dias_old:.1f}')
        
        axes[i].axhline(y=sys_new, color='purple', linestyle='-', alpha=0.8, label=f'New SBP: {sys_new:.1f}')
        axes[i].axhline(y=dias_new, color='purple', linestyle='-', alpha=0.8, label=f'New DBP: {dias_new:.1f}')
        
        axes[i].set_title(f'Sample {i+1}')
        axes[i].set_xlabel('Normalized Time')
        axes[i].set_ylabel('BP (mmHg)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('sample_waveforms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("CONTINUOUS BP EXTRACTION TEST")
    print("=" * 60)
    print("Testing improved BP extraction methods to prevent block patterns...")
    
    test_extraction_methods()
    show_sample_waveforms()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("Check the generated plots:")
    print("- bp_extraction_comparison.png: Method comparison")
    print("- sample_waveforms_comparison.png: Sample extractions")
    print("=" * 60) 
#!/usr/bin/env python3
"""
Multi-Signal Comparison Plotter

This script loads and plots BP, ECG, PVI HP, and PVI LP signals side by side
to visualize their relationships across multiple time frames.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_all_signals(data_path):
    """Load all physiological signals from HDF5 file"""
    print(f"📂 Loading all signals from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # Load all signals
        bp_signal = f['data']['bp']['signal'][:]     # (50, 867)
        ecg_signal = f['data']['ecg']['signal'][:]   # (50, 867)  
        pvi_hp_signal = f['data']['pviHP']['signal'][:]  # (50, 867)
        pvi_lp_signal = f['data']['pviLP']['signal'][:]  # (50, 867)
        
        print(f"✅ BP signal loaded: {bp_signal.shape}")
        print(f"✅ ECG signal loaded: {ecg_signal.shape}")
        print(f"✅ PVI HP signal loaded: {pvi_hp_signal.shape}")
        print(f"✅ PVI LP signal loaded: {pvi_lp_signal.shape}")
        
        return bp_signal, ecg_signal, pvi_hp_signal, pvi_lp_signal

def analyze_signal_characteristics(bp_signal, ecg_signal, pvi_hp_signal, pvi_lp_signal):
    """Analyze characteristics of all signals"""
    print(f"\n📊 SIGNAL CHARACTERISTICS ANALYSIS:")
    print("-"*45)
    
    signals = {
        'BP': bp_signal,
        'ECG': ecg_signal,
        'PVI HP': pvi_hp_signal,
        'PVI LP': pvi_lp_signal
    }
    
    for name, signal in signals.items():
        mean_val = np.nanmean(signal)
        std_val = np.nanstd(signal)
        min_val = np.nanmin(signal)
        max_val = np.nanmax(signal)
        nan_pct = np.mean(np.isnan(signal)) * 100
        
        print(f"• {name} Signal:")
        print(f"  - Mean: {mean_val:.4f}")
        print(f"  - Std: {std_val:.4f}")
        print(f"  - Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  - NaN %: {nan_pct:.2f}%")
        print()

def create_multi_signal_comparison(bp_signal, ecg_signal, pvi_hp_signal, pvi_lp_signal, 
                                 batch_idx=0, save_dir='multi_signal_analysis'):
    """Create comprehensive multi-signal comparison plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select signals from specified batch
    bp = bp_signal[batch_idx]
    ecg = ecg_signal[batch_idx]
    pvi_hp = pvi_hp_signal[batch_idx]
    pvi_lp = pvi_lp_signal[batch_idx]
    
    print(f"\n🎨 Creating multi-signal comparison for batch {batch_idx}...")
    
    # 1. Full signals overview
    fig, axes = plt.subplots(4, 1, figsize=(20, 12))
    time_axis = np.arange(len(bp))
    
    # BP Signal
    axes[0].plot(time_axis, bp, 'b-', linewidth=1, alpha=0.8, label='BP Signal')
    axes[0].set_title('Blood Pressure Signal', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('BP (mmHg)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # ECG Signal
    axes[1].plot(time_axis, ecg, 'r-', linewidth=1, alpha=0.8, label='ECG Signal')
    axes[1].set_title('Electrocardiogram (ECG) Signal', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('ECG (mV)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # PVI HP Signal
    axes[2].plot(time_axis, pvi_hp, 'g-', linewidth=1, alpha=0.8, label='PVI HP Signal')
    axes[2].set_title('Photoplethysmography Variability Index - High Pass', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('PVI HP')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # PVI LP Signal  
    axes[3].plot(time_axis, pvi_lp, 'm-', linewidth=1, alpha=0.8, label='PVI LP Signal')
    axes[3].set_title('Photoplethysmography Variability Index - Low Pass', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('PVI LP')
    axes[3].set_xlabel('Time (frames)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    plt.suptitle(f'Complete Physiological Signals Overview - Batch {batch_idx}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'complete_signals_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Zoomed comparison of multiple time windows
    windows = [
        (0, 200),      # Early
        (200, 400),    # Early-mid
        (400, 600),    # Mid
        (600, 800),    # Late
    ]
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    for row, (start, end) in enumerate(windows):
        time_window = np.arange(start, end)
        bp_window = bp[start:end]
        ecg_window = ecg[start:end]
        pvi_hp_window = pvi_hp[start:end]
        pvi_lp_window = pvi_lp[start:end]
        
        # BP Signal
        axes[row, 0].plot(time_window, bp_window, 'b-', linewidth=2)
        axes[row, 0].set_title(f'BP Signal\nFrames {start}-{end}')
        axes[row, 0].set_ylabel('BP (mmHg)')
        axes[row, 0].grid(True, alpha=0.3)
        
        # ECG Signal
        axes[row, 1].plot(time_window, ecg_window, 'r-', linewidth=2)
        axes[row, 1].set_title(f'ECG Signal\nFrames {start}-{end}')
        axes[row, 1].set_ylabel('ECG (mV)')
        axes[row, 1].grid(True, alpha=0.3)
        
        # PVI HP Signal
        axes[row, 2].plot(time_window, pvi_hp_window, 'g-', linewidth=2)
        axes[row, 2].set_title(f'PVI HP Signal\nFrames {start}-{end}')
        axes[row, 2].set_ylabel('PVI HP')
        axes[row, 2].grid(True, alpha=0.3)
        
        # PVI LP Signal
        axes[row, 3].plot(time_window, pvi_lp_window, 'm-', linewidth=2)
        axes[row, 3].set_title(f'PVI LP Signal\nFrames {start}-{end}')
        axes[row, 3].set_ylabel('PVI LP')
        axes[row, 3].grid(True, alpha=0.3)
        
        # Add x-label only to bottom row
        if row == len(windows) - 1:
            for col in range(4):
                axes[row, col].set_xlabel('Time (frames)')
    
    plt.suptitle(f'Multi-Signal Time Window Comparison - Batch {batch_idx}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multi_signal_windows.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Overlay comparison for specific cardiac cycles
    # Find BP peaks to identify cardiac cycles
    peaks, _ = find_peaks(bp, height=np.percentile(bp, 70), distance=50)
    
    if len(peaks) >= 3:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Select 3 cardiac cycles
        cycle_starts = peaks[:3]
        cycle_lengths = [80, 80, 80]  # Fixed length for comparison
        
        for i, (start, length) in enumerate(zip(cycle_starts, cycle_lengths)):
            if i >= 3:  # Only plot first 3 cycles
                break
                
            end = min(start + length, len(bp))
            cycle_time = np.arange(end - start)
            
            # Normalize signals for overlay comparison
            bp_cycle = bp[start:end]
            ecg_cycle = ecg[start:end]
            pvi_hp_cycle = pvi_hp[start:end]
            pvi_lp_cycle = pvi_lp[start:end]
            
            # Plot in first subplot
            if i == 0:
                axes[0, 0].plot(cycle_time, bp_cycle, 'b-', linewidth=2, label='BP', alpha=0.8)
                axes[0, 0].plot(cycle_time, (ecg_cycle - np.mean(ecg_cycle))*10 + np.mean(bp_cycle), 
                               'r-', linewidth=2, label='ECG (scaled)', alpha=0.8)
                axes[0, 0].set_title(f'BP vs ECG - Cycle {i+1}')
                axes[0, 0].set_ylabel('Amplitude')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                axes[0, 1].plot(cycle_time, bp_cycle, 'b-', linewidth=2, label='BP', alpha=0.8)
                axes[0, 1].plot(cycle_time, (pvi_hp_cycle - np.mean(pvi_hp_cycle))*50 + np.mean(bp_cycle), 
                               'g-', linewidth=2, label='PVI HP (scaled)', alpha=0.8)
                axes[0, 1].set_title(f'BP vs PVI HP - Cycle {i+1}')
                axes[0, 1].set_ylabel('Amplitude')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                axes[1, 0].plot(cycle_time, bp_cycle, 'b-', linewidth=2, label='BP', alpha=0.8)
                axes[1, 0].plot(cycle_time, (pvi_lp_cycle - np.mean(pvi_lp_cycle))*50 + np.mean(bp_cycle), 
                               'm-', linewidth=2, label='PVI LP (scaled)', alpha=0.8)
                axes[1, 0].set_title(f'BP vs PVI LP - Cycle {i+1}')
                axes[1, 0].set_ylabel('Amplitude')
                axes[1, 0].set_xlabel('Time (frames)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # Cross-correlation analysis
        if len(bp) > 100:
            # Calculate cross-correlations
            bp_clean = bp[~np.isnan(bp)]
            ecg_clean = ecg[~np.isnan(ecg)]
            pvi_hp_clean = pvi_hp[~np.isnan(pvi_hp)]
            pvi_lp_clean = pvi_lp[~np.isnan(pvi_lp)]
            
            min_len = min(len(bp_clean), len(ecg_clean), len(pvi_hp_clean), len(pvi_lp_clean))
            
            if min_len > 100:
                bp_subset = bp_clean[:min_len]
                ecg_subset = ecg_clean[:min_len]
                pvi_hp_subset = pvi_hp_clean[:min_len]
                pvi_lp_subset = pvi_lp_clean[:min_len]
                
                # Compute correlations
                corr_bp_ecg = np.corrcoef(bp_subset, ecg_subset)[0, 1]
                corr_bp_pvi_hp = np.corrcoef(bp_subset, pvi_hp_subset)[0, 1]
                corr_bp_pvi_lp = np.corrcoef(bp_subset, pvi_lp_subset)[0, 1]
                corr_pvi_hp_lp = np.corrcoef(pvi_hp_subset, pvi_lp_subset)[0, 1]
                
                corr_text = [
                    "Signal Correlations:",
                    f"BP ↔ ECG: {corr_bp_ecg:.3f}",
                    f"BP ↔ PVI HP: {corr_bp_pvi_hp:.3f}", 
                    f"BP ↔ PVI LP: {corr_bp_pvi_lp:.3f}",
                    f"PVI HP ↔ PVI LP: {corr_pvi_hp_lp:.3f}",
                    "",
                    "Interpretation:",
                    f"Strongest BP correlation:",
                    f"{'ECG' if abs(corr_bp_ecg) > max(abs(corr_bp_pvi_hp), abs(corr_bp_pvi_lp)) else 'PVI HP' if abs(corr_bp_pvi_hp) > abs(corr_bp_pvi_lp) else 'PVI LP'}"
                ]
                
                axes[1, 1].text(0.1, 0.9, '\n'.join(corr_text), 
                               transform=axes[1, 1].transAxes, fontsize=11,
                               verticalalignment='top', 
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                axes[1, 1].set_title('Signal Correlation Analysis')
                axes[1, 1].axis('off')
        
        plt.suptitle(f'Cardiac Cycle Signal Comparison - Batch {batch_idx}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cardiac_cycle_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Statistical comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Signal distributions
    signals = {'BP': bp, 'ECG': ecg, 'PVI HP': pvi_hp, 'PVI LP': pvi_lp}
    colors = ['blue', 'red', 'green', 'magenta']
    
    for i, (name, signal) in enumerate(signals.items()):
        clean_signal = signal[~np.isnan(signal)]
        axes[0, 0].hist(clean_signal, bins=50, alpha=0.6, label=name, color=colors[i])
    
    axes[0, 0].set_title('Signal Value Distributions')
    axes[0, 0].set_xlabel('Amplitude')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot comparison
    clean_signals = []
    labels = []
    for name, signal in signals.items():
        clean_signal = signal[~np.isnan(signal)]
        if len(clean_signal) > 0:
            clean_signals.append(clean_signal)
            labels.append(name)
    
    axes[0, 1].boxplot(clean_signals, tick_labels=labels)
    axes[0, 1].set_title('Signal Statistics Comparison')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Power spectral density (if scipy is available)
    try:
        from scipy import signal as scipy_signal
        
        for i, (name, sig) in enumerate(signals.items()):
            clean_sig = sig[~np.isnan(sig)]
            if len(clean_sig) > 256:
                f, psd = scipy_signal.welch(clean_sig, fs=1000, nperseg=256)
                axes[1, 0].semilogy(f[:50], psd[:50], label=name, color=colors[i], alpha=0.8)
        
        axes[1, 0].set_title('Power Spectral Density')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('PSD')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
    except ImportError:
        axes[1, 0].text(0.5, 0.5, 'PSD Analysis\nRequires scipy', 
                       transform=axes[1, 0].transAxes, ha='center', va='center')
        axes[1, 0].set_title('Power Spectral Density')
        
    # Signal quality metrics
    quality_metrics = {}
    for name, signal in signals.items():
        nan_pct = np.mean(np.isnan(signal)) * 100
        if nan_pct < 100:
            clean_signal = signal[~np.isnan(signal)]
            snr = np.mean(clean_signal) / np.std(clean_signal) if np.std(clean_signal) > 0 else 0
            quality_metrics[name] = {'NaN %': nan_pct, 'SNR': snr}
    
    quality_text = ["Signal Quality Metrics:"]
    for name, metrics in quality_metrics.items():
        quality_text.append(f"{name}:")
        quality_text.append(f"  NaN: {metrics['NaN %']:.1f}%")
        quality_text.append(f"  SNR: {metrics['SNR']:.2f}")
        quality_text.append("")
    
    axes[1, 1].text(0.1, 0.9, '\n'.join(quality_text), 
                   transform=axes[1, 1].transAxes, fontsize=10,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_title('Signal Quality Assessment')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'signal_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Multi-signal comparison plots saved to: {save_dir}")
    return save_dir

def main():
    # Configuration
    data_path = "/home/lucas_takanori/phd/data/subject001_baseline_masked.h5"
    save_dir = "multi_signal_analysis"
    batch_idx = 0  # Which batch to analyze
    
    print("🔬 MULTI-SIGNAL COMPARISON ANALYSIS")
    print("="*45)
    
    try:
        # Load all signals
        bp_signal, ecg_signal, pvi_hp_signal, pvi_lp_signal = load_all_signals(data_path)
        
        # Analyze signal characteristics
        analyze_signal_characteristics(bp_signal, ecg_signal, pvi_hp_signal, pvi_lp_signal)
        
        # Create comprehensive comparison plots
        print(f"\n🎨 GENERATING MULTI-SIGNAL VISUALIZATIONS:")
        print("-"*45)
        save_dir = create_multi_signal_comparison(bp_signal, ecg_signal, pvi_hp_signal, pvi_lp_signal, 
                                                batch_idx=batch_idx, save_dir=save_dir)
        
        # Final summary
        print(f"\n📁 GENERATED VISUALIZATION FILES:")
        print("-"*35)
        viz_files = [
            'complete_signals_overview.png',
            'multi_signal_windows.png',
            'cardiac_cycle_comparison.png',
            'signal_statistics.png'
        ]
        
        for viz_file in viz_files:
            filepath = os.path.join(save_dir, viz_file)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  ✅ {viz_file}: {size_mb:.1f} MB")
            else:
                print(f"  ❌ {viz_file}: Not generated")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print("-"*15)
        print("• Complete overview of all physiological signals")
        print("• Time window comparisons for detailed analysis")
        print("• Cardiac cycle alignment across all signals")
        print("• Statistical comparison and correlation analysis")
        print("• Signal quality assessment and metrics")
        
        print(f"\n✅ Multi-signal analysis complete!")
        print(f"Results saved to: {save_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
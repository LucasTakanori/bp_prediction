#!/usr/bin/env python3
"""Multi-Signal Comparison Plotter"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def load_signals(data_path):
    """Load all physiological signals"""
    print(f"Loading signals from: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        bp_signal = f['data']['bp']['signal'][:]
        ecg_signal = f['data']['ecg']['signal'][:]  
        pvi_hp_signal = f['data']['pviHP']['signal'][:]
        pvi_lp_signal = f['data']['pviLP']['signal'][:]
        
        print(f"BP: {bp_signal.shape}, ECG: {ecg_signal.shape}")
        print(f"PVI HP: {pvi_hp_signal.shape}, PVI LP: {pvi_lp_signal.shape}")
        
        return bp_signal, ecg_signal, pvi_hp_signal, pvi_lp_signal

def plot_signals(bp, ecg, pvi_hp, pvi_lp, batch_idx=0, save_dir='signal_plots'):
    """Create signal comparison plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select batch
    bp_data = bp[batch_idx]
    ecg_data = ecg[batch_idx]
    pvi_hp_data = pvi_hp[batch_idx]
    pvi_lp_data = pvi_lp[batch_idx]
    
    print(f"Plotting signals for batch {batch_idx}")
    
    # Full overview
    fig, axes = plt.subplots(4, 1, figsize=(20, 12))
    time_axis = np.arange(len(bp_data))
    
    axes[0].plot(time_axis, bp_data, 'b-', linewidth=1, label='BP Signal')
    axes[0].set_title('Blood Pressure Signal')
    axes[0].set_ylabel('BP (mmHg)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(time_axis, ecg_data, 'r-', linewidth=1, label='ECG Signal')
    axes[1].set_title('ECG Signal')
    axes[1].set_ylabel('ECG (mV)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(time_axis, pvi_hp_data, 'g-', linewidth=1, label='PVI HP Signal')
    axes[2].set_title('PVI High Pass Signal')
    axes[2].set_ylabel('PVI HP')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    axes[3].plot(time_axis, pvi_lp_data, 'm-', linewidth=1, label='PVI LP Signal')
    axes[3].set_title('PVI Low Pass Signal')
    axes[3].set_ylabel('PVI LP')
    axes[3].set_xlabel('Time (frames)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    plt.suptitle(f'Multi-Signal Comparison - Batch {batch_idx}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_signals_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Windowed comparison
    windows = [(0, 200), (200, 400), (400, 600), (600, 800)]
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    for row, (start, end) in enumerate(windows):
        time_window = np.arange(start, end)
        
        axes[row, 0].plot(time_window, bp_data[start:end], 'b-', linewidth=2)
        axes[row, 0].set_title(f'BP: Frames {start}-{end}')
        axes[row, 0].set_ylabel('BP (mmHg)')
        axes[row, 0].grid(True, alpha=0.3)
        
        axes[row, 1].plot(time_window, ecg_data[start:end], 'r-', linewidth=2)
        axes[row, 1].set_title(f'ECG: Frames {start}-{end}')
        axes[row, 1].set_ylabel('ECG (mV)')
        axes[row, 1].grid(True, alpha=0.3)
        
        axes[row, 2].plot(time_window, pvi_hp_data[start:end], 'g-', linewidth=2)
        axes[row, 2].set_title(f'PVI HP: Frames {start}-{end}')
        axes[row, 2].set_ylabel('PVI HP')
        axes[row, 2].grid(True, alpha=0.3)
        
        axes[row, 3].plot(time_window, pvi_lp_data[start:end], 'm-', linewidth=2)
        axes[row, 3].set_title(f'PVI LP: Frames {start}-{end}')
        axes[row, 3].set_ylabel('PVI LP')
        axes[row, 3].grid(True, alpha=0.3)
        
        if row == len(windows) - 1:
            for col in range(4):
                axes[row, col].set_xlabel('Time (frames)')
    
    plt.suptitle(f'Signal Windows Comparison - Batch {batch_idx}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'signal_windows.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {save_dir}")

def main():
    data_path = "/home/lucas_takanori/phd/data/subject002_baseline_masked.h5"
    
    print("Multi-Signal Analysis")
    print("=" * 30)
    
    try:
        bp, ecg, pvi_hp, pvi_lp = load_signals(data_path)
        plot_signals(bp, ecg, pvi_hp, pvi_lp)
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main() 
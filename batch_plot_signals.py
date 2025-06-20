#!/usr/bin/env python3
"""
Batch Multi-Signal Comparison Plotter

This script processes all H5 files in a folder and generates signal comparison plots
for each file, organizing outputs into separate folders under signal_analysis/
"""

import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path
import time

def load_signals(data_path):
    """Load all physiological signals from H5 file"""
    print(f"  📂 Loading: {os.path.basename(data_path)}")
    
    try:
        with h5py.File(data_path, 'r') as f:
            bp_signal = f['data']['bp']['signal'][:]
            ecg_signal = f['data']['ecg']['signal'][:]  
            pvi_hp_signal = f['data']['pviHP']['signal'][:]
            pvi_lp_signal = f['data']['pviLP']['signal'][:]
            
            print(f"    ✅ BP: {bp_signal.shape}, ECG: {ecg_signal.shape}")
            print(f"    ✅ PVI HP: {pvi_hp_signal.shape}, PVI LP: {pvi_lp_signal.shape}")
            
            return bp_signal, ecg_signal, pvi_hp_signal, pvi_lp_signal, True
            
    except Exception as e:
        print(f"    ❌ Error loading {data_path}: {e}")
        return None, None, None, None, False

def analyze_signal_quality(bp, ecg, pvi_hp, pvi_lp):
    """Quick signal quality analysis"""
    signals = {'BP': bp, 'ECG': ecg, 'PVI_HP': pvi_hp, 'PVI_LP': pvi_lp}
    quality_report = {}
    
    for name, signal in signals.items():
        if signal is not None:
            nan_pct = np.mean(np.isnan(signal)) * 100
            mean_val = np.nanmean(signal)
            std_val = np.nanstd(signal)
            
            quality_report[name] = {
                'nan_percent': nan_pct,
                'mean': mean_val,
                'std': std_val,
                'quality': 'Good' if nan_pct < 50 else 'Poor'
            }
        else:
            quality_report[name] = {'quality': 'Failed'}
    
    return quality_report

def plot_signals(bp, ecg, pvi_hp, pvi_lp, output_dir, filename_base, batch_idx=0):
    """Create signal comparison plots for a single file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select batch
    bp_data = bp[batch_idx] if batch_idx < len(bp) else bp[0]
    ecg_data = ecg[batch_idx] if batch_idx < len(ecg) else ecg[0]
    pvi_hp_data = pvi_hp[batch_idx] if batch_idx < len(pvi_hp) else pvi_hp[0]
    pvi_lp_data = pvi_lp[batch_idx] if batch_idx < len(pvi_lp) else pvi_lp[0]
    
    print(f"    🎨 Creating plots for batch {batch_idx}")
    
    try:
        # 1. Full overview plot
        fig, axes = plt.subplots(4, 1, figsize=(20, 12))
        time_axis = np.arange(len(bp_data))
        
        # BP Signal
        axes[0].plot(time_axis, bp_data, 'b-', linewidth=1, label='BP Signal', alpha=0.8)
        axes[0].set_title('Blood Pressure Signal', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('BP (mmHg)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Add basic stats
        bp_mean = np.nanmean(bp_data)
        bp_std = np.nanstd(bp_data)
        axes[0].text(0.02, 0.95, f'Mean: {bp_mean:.1f} ± {bp_std:.1f} mmHg', 
                    transform=axes[0].transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # ECG Signal
        axes[1].plot(time_axis, ecg_data, 'r-', linewidth=1, label='ECG Signal', alpha=0.8)
        axes[1].set_title('Electrocardiogram (ECG) Signal', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('ECG (mV)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # PVI HP Signal
        axes[2].plot(time_axis, pvi_hp_data, 'g-', linewidth=1, label='PVI HP Signal', alpha=0.8)
        axes[2].set_title('PVI High Pass Signal', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('PVI HP')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # PVI LP Signal
        axes[3].plot(time_axis, pvi_lp_data, 'm-', linewidth=1, label='PVI LP Signal', alpha=0.8)
        axes[3].set_title('PVI Low Pass Signal', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('PVI LP')
        axes[3].set_xlabel('Time (frames)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        plt.suptitle(f'Multi-Signal Overview: {filename_base} (Batch {batch_idx})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_signals_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Windowed comparison
        windows = [(0, 200), (200, 400), (400, 600), (600, 800)]
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        
        for row, (start, end) in enumerate(windows):
            # Ensure we don't go beyond signal length
            actual_end = min(end, len(bp_data))
            if start >= len(bp_data):
                continue
                
            time_window = np.arange(start, actual_end)
            
            # BP Signal
            axes[row, 0].plot(time_window, bp_data[start:actual_end], 'b-', linewidth=2)
            axes[row, 0].set_title(f'BP: Frames {start}-{actual_end}')
            axes[row, 0].set_ylabel('BP (mmHg)')
            axes[row, 0].grid(True, alpha=0.3)
            
            # ECG Signal
            axes[row, 1].plot(time_window, ecg_data[start:actual_end], 'r-', linewidth=2)
            axes[row, 1].set_title(f'ECG: Frames {start}-{actual_end}')
            axes[row, 1].set_ylabel('ECG (mV)')
            axes[row, 1].grid(True, alpha=0.3)
            
            # PVI HP Signal
            axes[row, 2].plot(time_window, pvi_hp_data[start:actual_end], 'g-', linewidth=2)
            axes[row, 2].set_title(f'PVI HP: Frames {start}-{actual_end}')
            axes[row, 2].set_ylabel('PVI HP')
            axes[row, 2].grid(True, alpha=0.3)
            
            # PVI LP Signal
            axes[row, 3].plot(time_window, pvi_lp_data[start:actual_end], 'm-', linewidth=2)
            axes[row, 3].set_title(f'PVI LP: Frames {start}-{actual_end}')
            axes[row, 3].set_ylabel('PVI LP')
            axes[row, 3].grid(True, alpha=0.3)
            
            # Add x-label to bottom row
            if row == len(windows) - 1:
                for col in range(4):
                    axes[row, col].set_xlabel('Time (frames)')
        
        plt.suptitle(f'Signal Windows: {filename_base} (Batch {batch_idx})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'signal_windows.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Plots saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"    ❌ Error creating plots: {e}")
        return False

def create_summary_report(processed_files, output_base_dir):
    """Create a summary report of all processed files"""
    summary_path = os.path.join(output_base_dir, 'processing_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("BATCH SIGNAL ANALYSIS - PROCESSING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files processed: {len(processed_files)}\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        successful = sum(1 for pf in processed_files if pf['success'])
        failed = len(processed_files) - successful
        
        f.write(f"✅ Successful: {successful}\n")
        f.write(f"❌ Failed: {failed}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 20 + "\n")
        
        for pf in processed_files:
            f.write(f"\n📁 {pf['filename']}:\n")
            f.write(f"   Status: {'✅ Success' if pf['success'] else '❌ Failed'}\n")
            if pf['success']:
                f.write(f"   Output: {pf['output_dir']}\n")
                if 'quality' in pf:
                    f.write(f"   Signal Quality:\n")
                    for signal, quality in pf['quality'].items():
                        if 'nan_percent' in quality:
                            f.write(f"     {signal}: {quality['quality']} (NaN: {quality['nan_percent']:.1f}%)\n")
                        else:
                            f.write(f"     {signal}: {quality['quality']}\n")
            else:
                f.write(f"   Error: {pf.get('error', 'Unknown error')}\n")
    
    print(f"📄 Summary report saved: {summary_path}")

def main():
    # Configuration
    data_folder = "/home/lucas_takanori/phd/data"  # Folder containing H5 files
    output_base_dir = "signal_analysis"  # Main output directory
    batch_idx = 0  # Which batch to analyze (usually 0)
    
    print("🔬 BATCH MULTI-SIGNAL ANALYSIS")
    print("=" * 45)
    print(f"📂 Source folder: {data_folder}")
    print(f"📁 Output folder: {output_base_dir}")
    print(f"🎯 Processing batch: {batch_idx}")
    print()
    
    # Create main output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find all H5 files
    h5_pattern = os.path.join(data_folder, "*.h5")
    h5_files = glob.glob(h5_pattern)
    
    if not h5_files:
        print(f"❌ No H5 files found in {data_folder}")
        return
    
    print(f"🔍 Found {len(h5_files)} H5 files:")
    for f in h5_files:
        print(f"  • {os.path.basename(f)}")
    print()
    
    # Process each file
    processed_files = []
    
    for i, h5_file in enumerate(h5_files, 1):
        filename_base = Path(h5_file).stem  # Remove .h5 extension
        output_dir = os.path.join(output_base_dir, filename_base)
        
        print(f"📊 Processing file {i}/{len(h5_files)}: {filename_base}")
        
        # Load signals
        bp, ecg, pvi_hp, pvi_lp, load_success = load_signals(h5_file)
        
        if not load_success:
            processed_files.append({
                'filename': filename_base,
                'success': False,
                'error': 'Failed to load signals'
            })
            continue
        
        # Analyze signal quality
        quality_report = analyze_signal_quality(bp, ecg, pvi_hp, pvi_lp)
        print(f"    📈 Signal quality check completed")
        
        # Create plots
        plot_success = plot_signals(bp, ecg, pvi_hp, pvi_lp, output_dir, filename_base, batch_idx)
        
        # Record results
        file_result = {
            'filename': filename_base,
            'success': plot_success,
            'output_dir': output_dir,
            'quality': quality_report
        }
        
        if not plot_success:
            file_result['error'] = 'Failed to create plots'
        
        processed_files.append(file_result)
        print(f"    {'✅ Completed' if plot_success else '❌ Failed'}")
        print()
    
    # Create summary report
    create_summary_report(processed_files, output_base_dir)
    
    # Final summary
    successful = sum(1 for pf in processed_files if pf['success'])
    failed = len(processed_files) - successful
    
    print("🏁 BATCH PROCESSING COMPLETE!")
    print("=" * 35)
    print(f"📊 Total files: {len(processed_files)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved in: {output_base_dir}/")
    print()
    
    if successful > 0:
        print("📂 Generated folder structure:")
        print(f"{output_base_dir}/")
        for pf in processed_files:
            if pf['success']:
                print(f"├── {pf['filename']}/")
                print(f"│   ├── all_signals_overview.png")
                print(f"│   └── signal_windows.png")
        print(f"└── processing_summary.txt")
    
    print(f"\n🎯 Next steps:")
    print("• Review the generated plots for each subject")
    print("• Check processing_summary.txt for quality assessment") 
    print("• Use insights for model training and validation")

if __name__ == '__main__':
    main() 
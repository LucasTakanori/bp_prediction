#!/usr/bin/env python3
"""
Compare Masked vs Whole Data Samples
Shows exactly what data is being filtered out by the masking approach
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import both dataset approaches
from utils.data_utils import PviDataset, DataPathManager  # Whole data
from utils.data_utils_enhanced import EnhancedPviDataset, load_dataset_with_best_mask  # Masked data


def compare_single_subject(data_root: str, subject: str, session: str = "baseline", 
                          mask_type: str = "auto") -> Dict:
    """Compare masked vs whole data for a single subject"""
    
    print(f"\n🔍 Comparing {subject}...")
    print("=" * 50)
    
    # Create file path
    data_manager = DataPathManager(subject=subject, session=session, root=data_root)
    file_path = data_manager._h5_path
    
    if not file_path.exists():
        print(f"❌ Data file not found: {file_path}")
        return None
    
    results = {
        'subject': subject,
        'file_path': str(file_path),
        'file_size_mb': file_path.stat().st_size / (1024 * 1024)
    }
    
    try:
        # Load with WHOLE DATA (no masking)
        print("📊 Loading with WHOLE DATA (no masking)...")
        whole_dataset = PviDataset(str(file_path))
        
        results['whole_data_samples'] = len(whole_dataset)
        results['whole_data_status'] = 'success'
        
        # Get sample structure
        if len(whole_dataset) > 0:
            sample = whole_dataset[0]
            results['sample_keys'] = list(sample.keys())
            
            # Check PVI data dimensions
            if 'pviHP' in sample and 'img' in sample['pviHP']:
                pvi_shape = sample['pviHP']['img'].shape
                results['pvi_shape'] = pvi_shape
                results['pvi_frames'] = pvi_shape[-1] if len(pvi_shape) > 2 else 1
        
        print(f"   ✅ Whole data samples: {results['whole_data_samples']}")
        if 'pvi_shape' in results:
            print(f"   📐 PVI shape: {results['pvi_shape']}")
        
    except Exception as e:
        print(f"   ❌ Error loading whole data: {e}")
        results['whole_data_samples'] = 0
        results['whole_data_status'] = f'error: {e}'
    
    try:
        # Load with MASKING
        print(f"🎭 Loading with MASKING (mask_type={mask_type})...")
        masked_dataset = load_dataset_with_best_mask(str(file_path), mask_type)
        
        results['masked_samples'] = len(masked_dataset)
        results['masked_status'] = 'success'
        
        # Get mask info
        mask_info = masked_dataset.get_mask_info()
        results['available_masks'] = mask_info.get('available_masks', [])
        results['used_mask'] = mask_info.get('recommended_mask', 'unknown')
        
        print(f"   ✅ Masked samples: {results['masked_samples']}")
        print(f"   🎯 Available masks: {results['available_masks']}")
        print(f"   🎯 Used mask: {results['used_mask']}")
        
    except Exception as e:
        print(f"   ❌ Error loading masked data: {e}")
        results['masked_samples'] = 0
        results['masked_status'] = f'error: {e}'
        results['available_masks'] = []
        results['used_mask'] = 'error'
    
    # Calculate differences
    if results['whole_data_samples'] > 0 and results['masked_samples'] > 0:
        results['filtered_out_samples'] = results['whole_data_samples'] - results['masked_samples']
        results['filtering_ratio'] = results['masked_samples'] / results['whole_data_samples']
        results['data_kept_percent'] = results['filtering_ratio'] * 100
        
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"   🔢 Whole data samples: {results['whole_data_samples']}")
        print(f"   🎭 Masked samples: {results['masked_samples']}")
        print(f"   🗑️  Filtered out: {results['filtered_out_samples']}")
        print(f"   📈 Data kept: {results['data_kept_percent']:.1f}%")
        
        if results['filtered_out_samples'] > 0:
            print(f"   ⚠️  MASKING REMOVES {results['filtered_out_samples']} samples!")
        else:
            print(f"   ✅ No data filtered out")
    
    return results


def compare_multiple_subjects(data_root: str, subjects: List[str], 
                            session: str = "baseline", mask_type: str = "auto") -> pd.DataFrame:
    """Compare multiple subjects and return summary DataFrame"""
    
    print(f"\n🚀 Comparing {len(subjects)} subjects...")
    print(f"📂 Data root: {data_root}")
    print(f"🎭 Mask type: {mask_type}")
    print("=" * 70)
    
    results = []
    
    for subject in subjects:
        result = compare_single_subject(data_root, subject, session, mask_type)
        if result:
            results.append(result)
    
    # Create summary DataFrame
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        return pd.DataFrame()


def visualize_comparison(df: pd.DataFrame, output_dir: str = "comparison_results"):
    """Create visualization of the comparison results"""
    
    if df.empty:
        print("No data to visualize")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Filter successful comparisons
    valid_df = df[(df['whole_data_samples'] > 0) & (df['masked_samples'] > 0)].copy()
    
    if valid_df.empty:
        print("No valid comparisons to visualize")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Masked vs Whole Data Comparison', fontsize=16)
    
    # Plot 1: Sample counts comparison
    subjects = valid_df['subject']
    whole_counts = valid_df['whole_data_samples']
    masked_counts = valid_df['masked_samples']
    
    x = np.arange(len(subjects))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, whole_counts, width, label='Whole Data', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x + width/2, masked_counts, width, label='Masked Data', alpha=0.8, color='orange')
    axes[0, 0].set_xlabel('Subjects')
    axes[0, 0].set_ylabel('Sample Count')
    axes[0, 0].set_title('Sample Counts: Whole vs Masked')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(subjects, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Data kept percentage
    axes[0, 1].bar(subjects, valid_df['data_kept_percent'], color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Subjects')
    axes[0, 1].set_ylabel('Data Kept (%)')
    axes[0, 1].set_title('Percentage of Data Kept After Masking')
    axes[0, 1].set_xticklabels(subjects, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Filtered out samples
    axes[1, 0].bar(subjects, valid_df['filtered_out_samples'], color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Subjects')
    axes[1, 0].set_ylabel('Filtered Out Samples')
    axes[1, 0].set_title('Samples Removed by Masking')
    axes[1, 0].set_xticklabels(subjects, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: File sizes
    axes[1, 1].bar(subjects, valid_df['file_size_mb'], color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Subjects')
    axes[1, 1].set_ylabel('File Size (MB)')
    axes[1, 1].set_title('H5 File Sizes')
    axes[1, 1].set_xticklabels(subjects, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "mask_vs_whole_data_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved to: {output_path / 'mask_vs_whole_data_comparison.png'}")


def save_detailed_results(df: pd.DataFrame, output_dir: str = "comparison_results"):
    """Save detailed comparison results to CSV"""
    
    if df.empty:
        print("No data to save")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save full results
    csv_path = output_path / "detailed_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Detailed results saved to: {csv_path}")
    
    # Create summary report
    summary_path = output_path / "comparison_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("MASK vs WHOLE DATA COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        total_subjects = len(df)
        successful_comparisons = len(df[(df['whole_data_samples'] > 0) & (df['masked_samples'] > 0)])
        
        f.write(f"Total subjects analyzed: {total_subjects}\n")
        f.write(f"Successful comparisons: {successful_comparisons}\n\n")
        
        if successful_comparisons > 0:
            valid_df = df[(df['whole_data_samples'] > 0) & (df['masked_samples'] > 0)]
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"Total whole data samples: {valid_df['whole_data_samples'].sum()}\n")
            f.write(f"Total masked samples: {valid_df['masked_samples'].sum()}\n")
            f.write(f"Total filtered out: {valid_df['filtered_out_samples'].sum()}\n")
            f.write(f"Average data kept: {valid_df['data_kept_percent'].mean():.1f}%\n\n")
            
            f.write("PER-SUBJECT BREAKDOWN:\n")
            for _, row in valid_df.iterrows():
                f.write(f"\n{row['subject']}:\n")
                f.write(f"  Whole data: {row['whole_data_samples']} samples\n")
                f.write(f"  Masked: {row['masked_samples']} samples\n")
                f.write(f"  Filtered: {row['filtered_out_samples']} samples\n")
                f.write(f"  Data kept: {row['data_kept_percent']:.1f}%\n")
                f.write(f"  Mask used: {row['used_mask']}\n")
                f.write(f"  Available masks: {row['available_masks']}\n")
    
    print(f"📋 Summary report saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare masked vs whole data samples")
    parser.add_argument("--data-root", type=str, 
                       default=os.getenv('BP_DATA_ROOT', '/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data'),
                       help="Data root directory")
    parser.add_argument("--subjects", nargs="+", default=["subject001", "subject002"],
                       help="Subjects to compare")
    parser.add_argument("--session", type=str, default="baseline",
                       help="Session name")
    parser.add_argument("--mask-type", type=str, default="auto",
                       choices=["auto", "metadata", "mask01", "mask05", "mask10", "mask15"],
                       help="Mask type to use")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                       help="Output directory for results")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip creating plots")
    
    args = parser.parse_args()
    
    # Run comparison
    df = compare_multiple_subjects(
        data_root=args.data_root,
        subjects=args.subjects,
        session=args.session,
        mask_type=args.mask_type
    )
    
    if not df.empty:
        # Save results
        save_detailed_results(df, args.output_dir)
        
        # Create visualizations
        if not args.no_plots:
            try:
                visualize_comparison(df, args.output_dir)
            except Exception as e:
                print(f"Warning: Could not create plots: {e}")
        
        # Print summary
        print(f"\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        
        valid_df = df[(df['whole_data_samples'] > 0) & (df['masked_samples'] > 0)]
        if not valid_df.empty:
            total_whole = valid_df['whole_data_samples'].sum()
            total_masked = valid_df['masked_samples'].sum()
            total_filtered = valid_df['filtered_out_samples'].sum()
            
            print(f"🔢 Total samples with WHOLE DATA: {total_whole:,}")
            print(f"🎭 Total samples with MASKING: {total_masked:,}")
            print(f"🗑️  Total samples FILTERED OUT: {total_filtered:,}")
            print(f"📈 Overall data kept: {(total_masked/total_whole)*100:.1f}%")
            print(f"\n⚠️  MASKING removes {total_filtered:,} samples ({((total_filtered/total_whole)*100):.1f}% of data)!")
            
            if total_filtered > 0:
                print(f"\n💡 Use the WHOLE DATA approach to train on {total_filtered:,} more samples!")
        else:
            print("❌ No valid comparisons completed")
    else:
        print("❌ No data to compare")


if __name__ == "__main__":
    main() 
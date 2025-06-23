#!/usr/bin/env python3
"""
Dataset Examination Script
Loads all available subjects and examines data structure, masks, and content
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import your utilities
from utils.data_utils import DataPathManager, PviDataset


def get_available_subjects(data_root: str) -> List[str]:
    """Get list of available subjects from data directory"""
    data_path = Path(data_root)
    subjects = []
    
    print(f"🔍 Scanning for subjects in: {data_path}")
    for file_path in data_path.glob("subject*_baseline_masked.h5"):
        subject = file_path.stem.split('_')[0]
        subjects.append(subject)
        print(f"   Found: {file_path.name}")
    
    return sorted(subjects)


def examine_h5_file_structure(file_path: Path) -> Dict:
    """Examine the raw HDF5 file structure"""
    print(f"\n📁 Examining HDF5 structure: {file_path.name}")
    
    structure = {}
    
    def visit_func(name, obj):
        if isinstance(obj, h5py.Group):
            structure[name] = {"type": "Group", "keys": list(obj.keys())}
        elif isinstance(obj, h5py.Dataset):
            structure[name] = {
                "type": "Dataset", 
                "shape": obj.shape, 
                "dtype": obj.dtype,
                "size": obj.size
            }
    
    try:
        with h5py.File(file_path, 'r') as h5f:
            h5f.visititems(visit_func)
            
            # Get top-level groups
            top_level = list(h5f.keys())
            print(f"   Top-level groups: {top_level}")
            
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return {}
    
    return structure


def examine_dataset_masks(dataset: PviDataset) -> Dict:
    """Examine the masks in the dataset"""
    print(f"\n🎭 Examining masks for dataset: {dataset.file_name}")
    
    mask_info = {
        "has_masks": False,
        "num_masks": 0,
        "mask_ranges": [],
        "mask_details": {}
    }
    
    # Check metadata for masks
    if hasattr(dataset, '_h5meta') and 'mask' in dataset._h5meta:
        masks = dataset._h5meta['mask']
        if masks is not None:
            mask_info["has_masks"] = True
            mask_info["num_masks"] = len(masks)
            mask_info["mask_ranges"] = masks
            
            print(f"   ✅ Masks found: {len(masks)} total")
            print(f"   📊 Mask ranges:")
            for i, mask_range in enumerate(masks[:5]):  # Show first 5
                print(f"      {i}: {mask_range}")
            if len(masks) > 5:
                print(f"      ... and {len(masks) - 5} more")
                
            # Analyze mask statistics
            mask_lengths = [end - start for start, end in masks]
            mask_info["mask_details"] = {
                "min_length": min(mask_lengths),
                "max_length": max(mask_lengths),
                "mean_length": np.mean(mask_lengths),
                "std_length": np.std(mask_lengths)
            }
            
            print(f"   📈 Mask length statistics:")
            print(f"      Min: {mask_info['mask_details']['min_length']}")
            print(f"      Max: {mask_info['mask_details']['max_length']}")
            print(f"      Mean: {mask_info['mask_details']['mean_length']:.2f}")
            print(f"      Std: {mask_info['mask_details']['std_length']:.2f}")
        else:
            print(f"   ⚠️  Mask key exists but is None")
    else:
        print(f"   ❌ No masks found in metadata")
    
    return mask_info


def examine_sample_structure(dataset: PviDataset, sample_idx: int = 0) -> Dict:
    """Examine the structure of a single sample"""
    print(f"\n🔬 Examining sample structure (sample {sample_idx}):")
    
    if len(dataset) == 0:
        print("   ❌ Dataset is empty")
        return {}
    
    sample = dataset[sample_idx]
    sample_info = {}
    
    print(f"   📊 Sample keys: {list(sample.keys())}")
    
    for key in sample.keys():
        print(f"\n   🔑 Key: '{key}'")
        if isinstance(sample[key], dict):
            sample_info[key] = {}
            for subkey in sample[key].keys():
                data = sample[key][subkey]
                if isinstance(data, torch.Tensor):
                    shape = tuple(data.shape)
                    dtype = data.dtype
                    sample_info[key][subkey] = {
                        "shape": shape,
                        "dtype": str(dtype),
                        "min": float(data.min()),
                        "max": float(data.max()),
                        "mean": float(data.mean()),
                        "std": float(data.std())
                    }
                    print(f"      {subkey}: shape={shape}, dtype={dtype}")
                    print(f"                 range=[{data.min():.4f}, {data.max():.4f}]")
                    print(f"                 mean={data.mean():.4f}, std={data.std():.4f}")
                else:
                    sample_info[key][subkey] = {"type": str(type(data))}
                    print(f"      {subkey}: {type(data)}")
        else:
            if isinstance(sample[key], torch.Tensor):
                shape = tuple(sample[key].shape)
                dtype = sample[key].dtype
                sample_info[key] = {
                    "shape": shape,
                    "dtype": str(dtype),
                    "min": float(sample[key].min()),
                    "max": float(sample[key].max()),
                    "mean": float(sample[key].mean()),
                    "std": float(sample[key].std())
                }
                print(f"   Direct tensor: shape={shape}, dtype={dtype}")
                print(f"                  range=[{sample[key].min():.4f}, {sample[key].max():.4f}]")
            else:
                sample_info[key] = {"type": str(type(sample[key]))}
                print(f"   Direct data: {type(sample[key])}")
    
    return sample_info


def visualize_masks(dataset: PviDataset, max_masks: int = 5):
    """Visualize the first few masks if available"""
    print(f"\n🎨 Visualizing masks (first {max_masks}):")
    
    if not hasattr(dataset, '_h5meta') or 'mask' not in dataset._h5meta or dataset._h5meta['mask'] is None:
        print("   ⚠️  No masks available for visualization")
        return
    
    masks = dataset._h5meta['mask']
    num_to_plot = min(len(masks), max_masks)
    
    if num_to_plot == 0:
        print("   ⚠️  No masks to visualize")
        return
    
    try:
        # Create a figure showing mask ranges
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot mask ranges as horizontal bars
        for i, (start, end) in enumerate(masks[:num_to_plot]):
            ax.barh(i, end - start, left=start, height=0.8, alpha=0.7, label=f'Mask {i}')
            ax.text(start + (end - start) / 2, i, f'{start}-{end}', 
                   ha='center', va='center', fontweight='bold')
        
        ax.set_xlabel('Period Index')
        ax.set_ylabel('Mask Index')
        ax.set_title(f'First {num_to_plot} Mask Ranges')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mask_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Mask visualization saved as 'mask_visualization.png'")
        
    except Exception as e:
        print(f"   ❌ Error creating visualization: {e}")


def visualize_sample_data(dataset: PviDataset, sample_idx: int = 0):
    """Visualize data from a sample"""
    print(f"\n📊 Visualizing sample data (sample {sample_idx}):")
    
    if len(dataset) == 0:
        print("   ❌ Dataset is empty")
        return
    
    sample = dataset[sample_idx]
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot signal data if available
        for key in sample.keys():
            if isinstance(sample[key], dict):
                for subkey in sample[key].keys():
                    if 'signal' in subkey.lower() and plot_idx < 4:
                        data = sample[key][subkey]
                        if isinstance(data, torch.Tensor) and data.numel() > 1:
                            axes[plot_idx].plot(data.numpy())
                            axes[plot_idx].set_title(f'{key}.{subkey}')
                            axes[plot_idx].grid(True, alpha=0.3)
                            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('sample_data_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Sample data visualization saved as 'sample_data_visualization.png'")
        
    except Exception as e:
        print(f"   ❌ Error creating sample visualization: {e}")


def main():
    """Main examination function"""
    print("🚀 Dataset Examination Script")
    print("=" * 60)
    
    # Setup data root
    data_root = os.getenv('BP_DATA_ROOT', '/home/lucas_takanori/phd/data')
    print(f"📁 Data root: {data_root}")
    
    # Get available subjects
    subjects = get_available_subjects(data_root)
    
    if not subjects:
        print("❌ No subjects found!")
        return
    
    print(f"\n✅ Found {len(subjects)} subjects: {subjects}")
    
    # Examine each subject
    all_results = {}
    
    for i, subject in enumerate(subjects):
        print(f"\n{'='*60}")
        print(f"📋 Subject {i+1}/{len(subjects)}: {subject}")
        print(f"{'='*60}")
        
        try:
            # Create data manager
            data_manager = DataPathManager(
                subject=subject,
                session="baseline",
                root=data_root
            )
            
            # Check if file exists
            if not data_manager._h5_path.exists():
                print(f"⚠️  Data file not found: {data_manager._h5_path}")
                continue
            
            # Examine raw HDF5 structure
            h5_structure = examine_h5_file_structure(data_manager._h5_path)
            
            # Load dataset
            print(f"\n🔄 Loading PviDataset...")
            dataset = PviDataset(str(data_manager._h5_path))
            
            # Examine masks
            mask_info = examine_dataset_masks(dataset)
            
            # Examine sample structure
            sample_info = examine_sample_structure(dataset)
            
            # Store results
            all_results[subject] = {
                "file_path": str(data_manager._h5_path),
                "h5_structure": h5_structure,
                "mask_info": mask_info,
                "sample_info": sample_info,
                "num_samples": len(dataset)
            }
            
            # Visualize for first subject only
            if i == 0:
                visualize_masks(dataset)
                visualize_sample_data(dataset)
            
            print(f"✅ Successfully examined {subject}")
            
        except Exception as e:
            print(f"❌ Error examining {subject}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    total_samples = sum(result["num_samples"] for result in all_results.values())
    subjects_with_masks = sum(1 for result in all_results.values() if result["mask_info"]["has_masks"])
    
    print(f"📈 Total subjects examined: {len(all_results)}")
    print(f"📈 Total samples across all subjects: {total_samples}")
    print(f"📈 Subjects with masks: {subjects_with_masks}/{len(all_results)}")
    
    if all_results:
        print(f"\n📋 Per-subject breakdown:")
        for subject, result in all_results.items():
            mask_status = "✅" if result["mask_info"]["has_masks"] else "❌"
            print(f"   {subject}: {result['num_samples']} samples, masks: {mask_status}")
    
    print(f"\n🎯 Examination complete! Check 'mask_visualization.png' and 'sample_data_visualization.png' for plots.")


if __name__ == "__main__":
    main() 
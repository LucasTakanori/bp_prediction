"""
Memory-Optimized Data Utils
Reduces memory usage during multi-subject loading
"""

import h5py
import numpy as np
import gc
import time
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from utils.data_utils_enhanced import EnhancedPviDataset, load_dataset_with_best_mask


class MemoryOptimizedPviDataset(EnhancedPviDataset):
    """Memory-optimized version that reduces RAM usage"""
    
    def __init__(self, file_path: str, mask_type: str = "auto", 
                 device: torch.device = torch.device('cpu'),
                 max_samples: Optional[int] = None) -> None:
        """
        Args:
            file_path: Path to HDF5 file
            mask_type: Type of mask to use
            device: torch device
            max_samples: Maximum number of samples to load (for memory control)
        """
        self.max_samples = max_samples
        super().__init__(file_path, mask_type, device)
    
    def _stack_samples(self) -> List[Dict]:
        """Memory-optimized sample stacking with garbage collection"""
        samples = []
        total_masks = len(self._h5meta['mask'])
        
        # Limit samples if requested
        if self.max_samples and self.max_samples < total_masks:
            num_samples = self.max_samples
            print(f"🔄 Limiting to {num_samples} samples (out of {total_masks} available)")
        else:
            num_samples = total_masks
        
        print(f'Stacking samples ({num_samples} total):')
        t1 = time.time()
        
        for k in range(num_samples):
            sample = {}
            
            # for NOVA tensors, extract a single period per sample
            idx = range(*self._h5meta['mask'][k])[-1]
            for signal_key in self._h5meta['nova']['signals']:
                sample[signal_key] = {}
                for signal_field in self._h5data[signal_key].keys():
                    tensor = self._h5data[signal_key][signal_field][idx]
                    sample[signal_key][signal_field] = tensor

            # for PVI tensors, stack multiple periods per sample
            sl = slice(*self._h5meta['mask'][k])
            
            for signal_key in self._h5meta['pvi']['signals']:
                sample[signal_key] = {}
                for signal_field in self._h5data[signal_key].keys():
                    tup = self._h5data[signal_key][signal_field][sl]
                    sample[signal_key][signal_field] = torch.cat(tup, dim=-1)
            
            sample['stats'] = {}
            for stat_key in self._h5meta['pvi']['stats']:
                sample['stats'][stat_key] = self._h5data['stats'][stat_key][sl]
                    
            samples.append(sample)
            
            # Garbage collection every 50 samples to free memory
            if k % 50 == 0:
                gc.collect()
            
            if (not (k+1) % 100) or (k+1) == num_samples:
                print(f"\t ...{k+1}/{num_samples} samples")
        
        # Final garbage collection
        gc.collect()
        
        dt = time.time() - t1
        print(f'\t ...Done! ({dt:.2f} seconds)')
        print(f"Number of samples: {num_samples}")
        return samples


def load_dataset_memory_optimized(file_path: str, mask_type: str = "auto", 
                                 max_samples: Optional[int] = None) -> MemoryOptimizedPviDataset:
    """
    Load dataset with memory optimization
    
    Args:
        file_path: Path to HDF5 file
        mask_type: Mask type to use
        max_samples: Maximum samples to load per subject
    """
    print(f"🚀 Loading memory-optimized dataset (max_samples={max_samples})")
    dataset = MemoryOptimizedPviDataset(file_path, mask_type=mask_type, max_samples=max_samples)
    
    # Print mask information
    mask_info = dataset.get_mask_info()
    print(f"📊 Available masks: {mask_info.get('available_masks', [])}")
    print(f"🎯 Used mask: {mask_info.get('recommended_mask', 'unknown')}")
    
    return dataset


def estimate_memory_usage(data_root: str, subjects: List[str], mask_type: str = "mask10") -> Dict:
    """Estimate memory usage for loading all subjects"""
    print("🧮 Estimating memory usage...")
    
    estimates = {}
    total_samples = 0
    
    for subject in subjects[:5]:  # Sample first 5 subjects
        try:
            file_path = Path(data_root) / f"{subject}_baseline_masked.h5"
            if not file_path.exists():
                continue
                
            # Quick analysis without loading full data
            with h5py.File(file_path, 'r') as h5f:
                if 'masks' in h5f and mask_type in h5f['masks']:
                    mask_count = h5f['masks'][mask_type].shape[1]
                elif 'metadata' in h5f and 'mask' in h5f['metadata']:
                    mask_count = h5f['metadata']['mask'].shape[1]
                else:
                    mask_count = h5f['data']['bp']['signal'].shape[1]  # Fallback
                
                estimates[subject] = mask_count
                total_samples += mask_count
                
        except Exception as e:
            print(f"⚠️  Error estimating {subject}: {e}")
            estimates[subject] = 0
    
    avg_samples = total_samples / len(estimates) if estimates else 0
    estimated_total = avg_samples * len(subjects)
    
    # Rough memory estimate (each sample ~1-5MB depending on data size)
    estimated_memory_gb = (estimated_total * 3) / 1000  # Conservative estimate
    
    print(f"📊 Memory estimates:")
    print(f"   Average samples per subject: {avg_samples:.0f}")
    print(f"   Estimated total samples: {estimated_total:.0f}")
    print(f"   Estimated memory usage: {estimated_memory_gb:.1f} GB")
    
    return {
        'estimates': estimates,
        'avg_samples': avg_samples,
        'estimated_total': estimated_total,
        'estimated_memory_gb': estimated_memory_gb
    }


def create_memory_efficient_multisubject_dataset(data_root: str, subjects: List[str], 
                                                mask_type: str = "auto", 
                                                max_samples_per_subject: Optional[int] = None,
                                                session: str = "baseline"):
    """Create multi-subject dataset with memory optimization"""
    from torch.utils.data import ConcatDataset
    
    print(f"Creating memory-efficient multi-subject dataset...")
    print(f"Max samples per subject: {max_samples_per_subject}")
    
    datasets = []
    successful_subjects = []
    failed_subjects = []
    total_samples = 0
    
    for i, subject in enumerate(subjects):
        try:
            print(f"\n📊 Loading subject {i+1}/{len(subjects)}: {subject}")
            
            file_path = Path(data_root) / f"{subject}_{session}_masked.h5"
            
            if not file_path.exists():
                print(f"⚠️  Data file not found for {subject}, skipping")
                failed_subjects.append(subject)
                continue
            
            # Load with memory optimization
            dataset = load_dataset_memory_optimized(
                str(file_path), 
                mask_type, 
                max_samples_per_subject
            )
            
            datasets.append(dataset)
            successful_subjects.append(subject)
            total_samples += len(dataset)
            
            print(f"✅ Loaded {subject}: {len(dataset)} samples")
            print(f"📈 Running total: {total_samples} samples")
            
            # Force garbage collection after each subject
            gc.collect()
            
        except Exception as e:
            print(f"❌ Failed to load {subject}: {e}")
            failed_subjects.append(subject)
            continue
    
    if not datasets:
        raise ValueError("No valid datasets were loaded!")
    
    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)
    
    print(f"\n🎯 Memory-efficient multi-subject dataset ready:")
    print(f"   📊 Successful subjects: {len(successful_subjects)}")
    print(f"   📊 Failed subjects: {len(failed_subjects)}")
    print(f"   📈 Total samples: {len(combined_dataset)}")
    print(f"   📝 Average per successful subject: {len(combined_dataset) / len(successful_subjects):.1f}")
    
    return combined_dataset, successful_subjects, failed_subjects 
"""
Enhanced Data Utils with Multi-Format Mask Support
Handles both metadata/mask and masks/* group formats
"""

import h5py
import numpy as np
import copy
import time
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union, KeysView
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class EnhancedDataPathManager:
    """Enhanced version that handles multiple data formats"""
    
    class __DefaultPaths:
        root: str = os.getenv('BP_DATA_ROOT', '/home/lucas_takanori/phd/data')
        subject: str = "subject001"
        session: str = "baseline"

    def __init__(self, subject=None, session=None, root=None) -> None:
        defaults = self.__DefaultPaths()
        self._root = Path(root or defaults.root)
        self._subject = subject or defaults.subject
        self._session = session or defaults.session
        
        # Check if we're in the new directory structure (direct h5 files)
        self._h5_path = self._root / f"{self._subject}_{self._session}_masked.h5"
        
        self.figures_dir = self._make_figures_dir()
        self.results_dir = self._make_results_dir()
        
        print(f"Looking for data file at: {self._h5_path}")
        print("EnhancedDataPathManager successfully initiated!")
    
    def _make_results_dir(self):
        output_dir = self._root / "_results_ml"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _make_figures_dir(self):
        figure_dir = self._root / "_figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        return figure_dir


class EnhancedPviDataset(Dataset):
    """Enhanced PVI Dataset with multi-format mask support"""
    
    def __init__(self, file_path: str, mask_type: str = "auto", device: torch.device = torch.device('cpu')) -> None:
        """
        Args:
            file_path: Path to HDF5 file
            mask_type: "auto", "metadata", "masks_group", or specific mask like "mask10"
            device: torch device
        """
        self._file_path = Path(file_path)
        self.file_name = self._file_path.name
        self.parent_dir = self._file_path.parent
        self.device = device
        self.mask_type = mask_type
        self.num_periods = 0
        
        print(f'Using Torch version: {torch.__version__}')
        print(f'Data directory set to:\n\t{self.parent_dir}')
        
        if self._check_file():
            print(f"Found HDF5 data: '{self.file_name}'")
            self._h5meta = self._load_metadata()
            print("EnhancedPviDataset successfully initiated!")
        else:
            raise FileNotFoundError(f"Data file not found at: {self._file_path}")
        
        self._h5data = self._load_raw_data()
        self._determine_and_load_masks()
        self.samples = self._stack_samples()
        
        print("Finish loading EnhancedPviDataset!")
    
    def _check_file(self) -> bool:
        """Check if h5 file exists and is readable"""
        try:
            with h5py.File(self._file_path, 'r') as _:
                return True
        except (OSError, IOError) as e:
            print(f"Error accessing file: {e}")
            return False

    def _load_metadata(self) -> Dict:
        """Load metadata with enhanced mask detection"""
        print('Loading metadata...')
        metadata = {}
        
        with h5py.File(self._file_path, 'r') as h5f:
            h5meta = h5f['metadata']
            
            # Load standard metadata
            for key in h5meta.keys():
                if isinstance(h5meta[key], h5py.Group):
                    metadata[key] = {}
                    for subkey in h5meta[key].keys():
                        metadata[key][subkey] = [s.item().decode() for s in h5meta[key][subkey][()]]
                else:
                    metadata[key] = []
            
            # Handle date
            if 'date' in h5meta:
                metadata['date'] = h5meta['date'][()].item().decode()
            else:
                metadata['date'] = 'Unknown'
            
            # Enhanced mask detection
            metadata['mask_info'] = self._detect_mask_formats(h5f)
            
            # Load specific mask based on mask_type
            metadata['mask'] = self._load_appropriate_mask(h5f, metadata['mask_info'])
        
        print('\t ...Done!')
        return metadata
    
    def _detect_mask_formats(self, h5f) -> Dict:
        """Detect all available mask formats"""
        mask_info = {
            'has_metadata_mask': False,
            'has_masks_group': False,
            'available_masks': [],
            'recommended_mask': None
        }
        
        # Check for metadata/mask
        if 'mask' in h5f['metadata']:
            mask_info['has_metadata_mask'] = True
            mask_info['available_masks'].append('metadata')
            mask_info['recommended_mask'] = 'metadata'
            print("   ✅ Found masks in metadata/mask")
        
        # Check for masks group
        if 'masks' in h5f:
            mask_info['has_masks_group'] = True
            available_group_masks = list(h5f['masks'].keys())
            mask_info['available_masks'].extend(available_group_masks)
            
            print(f"   ✅ Found masks group with: {available_group_masks}")
            
            # If no metadata mask, recommend mask10 or first available
            if not mask_info['has_metadata_mask']:
                if 'mask10' in available_group_masks:
                    mask_info['recommended_mask'] = 'mask10'
                else:
                    mask_info['recommended_mask'] = available_group_masks[0]
        
        if not mask_info['available_masks']:
            print("   ⚠️  No masks found - will generate default")
            mask_info['recommended_mask'] = 'generate'
        
        return mask_info
    
    def _load_appropriate_mask(self, h5f, mask_info: Dict):
        """Load the appropriate mask based on mask_type setting"""
        
        if self.mask_type == "auto":
            # Use recommended mask
            selected_mask = mask_info['recommended_mask']
        else:
            # Use specified mask type
            selected_mask = self.mask_type
            if selected_mask not in mask_info['available_masks'] and selected_mask != 'generate':
                print(f"   ⚠️  Requested mask '{selected_mask}' not found, using recommended: '{mask_info['recommended_mask']}'")
                selected_mask = mask_info['recommended_mask']
        
        print(f"   🎯 Using mask: {selected_mask}")
        
        # Load the selected mask
        if selected_mask == 'metadata':
            arr = h5f['metadata']['mask'][()].astype(np.int32)
            arr[0] = arr[0] - 1  # for slicing in python
            return tuple(map(tuple, arr.T))
        
        elif selected_mask in h5f.get('masks', {}):
            arr = h5f['masks'][selected_mask][()].astype(np.int32)
            arr[0] = arr[0] - 1  # for slicing in python
            return tuple(map(tuple, arr.T))
        
        else:
            # Generate default or return None
            return None
    
    def _load_raw_data(self) -> Dict:
        """Load raw data - same as original"""
        print('Loading raw data...')
        t1 = time.time()
        
        with h5py.File(self._file_path, 'r') as h5f:
            data = {}
            h5data = h5f['data']
            
            for category in ['nova', 'pvi']:
                for signal_key in self._h5meta[category]['signals']:
                    data[signal_key] = {}
                    for signal_field in self._h5meta[category]['fields']:
                        tensor = torch.FloatTensor(h5data[signal_key][signal_field][()].T)
                        data[signal_key][signal_field] = tuple(tensor[n] for n in range(tensor.shape[0]))
            
            stats = h5f['stats']['pviHP']
            data['stats'] = {}
            for key in self._h5meta['pvi']['stats']:
                data['stats'][key] = torch.FloatTensor(stats[key][()].squeeze())

        dt = time.time() - t1
        print(f'\t ...Done! ({dt:.2f} seconds)')
        
        if 'tensor' in locals():
            self.num_periods = tensor.shape[0]
            print(f"Number of periods: {self.num_periods}")
        else:
            self.num_periods = 0
            print(f"Number of periods: 0")
        
        return data
    
    def _determine_and_load_masks(self):
        """Determine final masks to use"""
        if self._h5meta.get('mask') is None:
            if self.num_periods > 0:
                print(f"Generating default mask from {self.num_periods} periods.")
                self._h5meta['mask'] = tuple((i, i + 1) for i in range(self.num_periods))
            else:
                self._h5meta['mask'] = []
        
        print(f"Final mask count: {len(self._h5meta['mask'])} samples")
    
    def _stack_samples(self) -> List[Dict]:
        """Stack samples - same as original"""
        samples = []
        num_samples = len(self._h5meta['mask'])
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
            
            if (not (k+1) % 100) or (k+1) == num_samples:
                print(f"\t ...{k+1}/{num_samples} samples")
        
        dt = time.time() - t1
        print(f'\t ...Done! ({dt:.2f} seconds)')
        print(f"Number of samples: {num_samples}")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
    
    def get_mask_info(self) -> Dict:
        """Get information about available masks"""
        return self._h5meta.get('mask_info', {})


def load_dataset_with_best_mask(file_path: str, preferred_mask: str = "auto") -> EnhancedPviDataset:
    """
    Convenience function to load dataset with best available mask
    
    Args:
        file_path: Path to HDF5 file
        preferred_mask: "auto", "metadata", "mask10", "mask05", etc.
    
    Returns:
        EnhancedPviDataset instance
    """
    print(f"🚀 Loading dataset with preferred mask: {preferred_mask}")
    dataset = EnhancedPviDataset(file_path, mask_type=preferred_mask)
    
    # Print mask information
    mask_info = dataset.get_mask_info()
    print(f"📊 Available masks: {mask_info.get('available_masks', [])}")
    print(f"🎯 Used mask: {mask_info.get('recommended_mask', 'unknown')}")
    
    return dataset


def analyze_all_mask_formats(data_root: str) -> Dict:
    """Analyze mask formats across all subjects"""
    data_path = Path(data_root)
    subjects = []
    
    for file_path in data_path.glob("subject*_baseline_masked.h5"):
        subject = file_path.stem.split('_')[0]
        subjects.append((subject, file_path))
    
    analysis = {}
    
    for subject, file_path in subjects:
        print(f"\n📋 Analyzing {subject}...")
        try:
            with h5py.File(file_path, 'r') as h5f:
                info = {
                    'has_metadata_mask': 'mask' in h5f['metadata'],
                    'has_masks_group': 'masks' in h5f,
                    'masks_group_keys': list(h5f['masks'].keys()) if 'masks' in h5f else [],
                    'data_periods': h5f['data']['bp']['signal'].shape[1] if 'data' in h5f else 0
                }
                
                if info['has_metadata_mask']:
                    info['metadata_mask_shape'] = h5f['metadata']['mask'].shape
                    info['metadata_mask_count'] = h5f['metadata']['mask'].shape[1]
                
                analysis[subject] = info
                
        except Exception as e:
            print(f"   ❌ Error analyzing {subject}: {e}")
            analysis[subject] = {'error': str(e)}
    
    return analysis 
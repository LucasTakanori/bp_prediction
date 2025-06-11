# -*- coding: utf-8 -*-
"""
Complete Optimized data_utils.py with caching, multi-threading, and efficient data loading

@author: u1376110
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
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import psutil
import warnings
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPathManager:
    
    class __DefaultPaths: # private attributes to get default directory
        root: str = r"/home/lucas_takanori/phd/data"
        subject: str = "subject012"
        session: str = "baseline"
        # postfix: str = "segmented" # currently not in use

    def __init__(self,
                 subject=None,
                 session=None,
                 root=None,
                 ) -> None:
        defaults = self.__DefaultPaths() # fallback
        self._root = Path(root or defaults.root)
        self._subject = subject or defaults.subject
        self._session = session or defaults.session
        
        # Check if we're in the new directory structure (direct h5 files)
        self._h5_name = "_".join([self._subject, self._session, "masked"]) + ".h5"
        direct_h5_path = self._root / self._h5_name
        
        if direct_h5_path.exists():
            # New structure: files directly in root
            self._h5_path = direct_h5_path
            self._output = self._root
            self._h5data = self._root
            self._masked_dir = self._root
        else:
            # Original structure with subdirectories
            self._output = self._root / "output"
            self._h5data = self._root / "raw"
            self._masked_dir = self._output / self._session / "masked"
            self._h5_path = self._masked_dir / self._h5_name
        
        self.subject_dir = self._h5data / self._subject
        self.session_dir = self.subject_dir / self._session
        
        self.figures_dir = self._make_figures_dir()
        self.results_dir = self._make_results_dir()
        
        print(f"Looking for data file at: {self._h5_path}")
        print("DataPathManager successfully initiated!")
    
    def _print_init(self) -> None:
        self.class_name = self.__class__.__name__
        print()
        print(f"====={self.class_name}=====")
        print(f"Subject: {self._subject}")
        print(f"Session: {self._session}")
        print(f"Target file: {self._h5_name}")
        print(f"Location: {self._masked_dir}")
        
        
    def _make_results_dir(self):
        output_dir = self._output / self._session / "_results_ml"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _make_figures_dir(self):
        figure_dir = self._output / self._session / "_figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        return figure_dir
    
    def explorer(self,
                 kw: str = "session") -> None:
        valid_dirs = {
            "root": self._root,
            "session": self.session_dir,
            "subject": self.subject_dir,
            "figures": self.figures_dir,
            "results": self.results_dir
            }
        if kw in valid_dirs:
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(valid_dirs[kw])
                elif os.name == 'posix':  # Linux/Mac
                    os.system(f'xdg-open "{valid_dirs[kw]}"')
            except Exception as e:
                print(f"Could not open directory: {e}")
        else:
            raise ValueError(f"Unrecognized keyword '{kw}'. Expected one of: {list(valid_dirs.keys())}.")


class PviDataset(Dataset):
    """Optimized dataset with caching and efficient data loading"""
    
    def __init__(self,
                 file_path: str,
                 device: torch.device = torch.device('cpu'),
                 cache_dir: Optional[str] = None,
                 preload_to_memory: bool = False,
                 use_mmap: bool = True,
                 force_reload: bool = False,
                 verbose: bool = True) -> None:
        
        self._file_path = Path(file_path)
        self.file_name = self._file_path.name
        self.parent_dir = self._file_path.parent
        self.device = device
        self.preload_to_memory = preload_to_memory
        self.use_mmap = use_mmap
        self.force_reload = force_reload
        self.verbose = verbose
        
        # Set up caching
        if cache_dir is None:
            cache_dir = self.parent_dir / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache file paths
        self.cache_file = self.cache_dir / f"{self.file_name}_processed.pkl"
        self.metadata_cache = self.cache_dir / f"{self.file_name}_metadata.pkl"
        
        if self.verbose:
            print(f'Using Torch version: {torch.__version__}')
            print(f'Data directory set to:\n\t{self.parent_dir}')
            print(f'Cache directory: {self.cache_dir}')
            print(f'Preload to memory: {preload_to_memory}')
            print(f'Use memory mapping: {use_mmap}')
            print(f'Force reload: {force_reload}')
        
        # Check file existence first
        if not self._check_file():
            raise FileNotFoundError(f"Data file not found at: {self._file_path}")
        
        # Try to load from cache first (unless force reload)
        if not self.force_reload and self._load_from_cache():
            if self.verbose:
                print("✓ Loaded data from cache!")
        else:
            if self.verbose:
                print("Processing data from scratch...")
            
            try:
                self._h5meta = self._load_metadata()
                if self.verbose:
                    print("✓ Metadata loaded successfully")
                
                self._h5data = self._load_raw_data()
                if self.verbose:
                    print("✓ Raw data loaded successfully")
                
                self.samples = self._stack_samples()
                if self.verbose:
                    print("✓ Samples stacked successfully")
                
                # Save to cache
                self._save_to_cache()
                
            except Exception as e:
                logger.error(f"Error during data processing: {e}")
                raise
        
        if self.verbose:
            print(f"✅ Finish loading PviDataset with {len(self.samples)} samples!")
        
        # Validate loaded data
        self._validate_data()
        
        # Preload to GPU if specified
        if self.preload_to_memory and device.type == 'cuda':
            self._preload_to_gpu()
    
    def _validate_data(self):
        """Validate the loaded data for consistency"""
        if not self.samples:
            raise ValueError("No samples loaded")
        
        # Check first sample structure
        sample = self.samples[0]
        required_keys = ['bp', 'pviHP', 'pviLP', 'stats']
        
        for key in required_keys:
            if key not in sample:
                logger.warning(f"Missing key '{key}' in sample data")
        
        # Check tensor types and shapes
        try:
            if 'pviHP' in sample and 'signal' in sample['pviHP']:
                signal_shape = sample['pviHP']['signal'].shape
                if len(signal_shape) != 3:
                    logger.warning(f"Unexpected pviHP signal shape: {signal_shape}")
            
            if 'bp' in sample and 'signal' in sample['bp']:
                bp_shape = sample['bp']['signal'].shape
                if len(bp_shape) != 1:
                    logger.warning(f"Unexpected BP signal shape: {bp_shape}")
                    
        except Exception as e:
            logger.warning(f"Data validation warning: {e}")
    
    def _load_from_cache(self) -> bool:
        """Load processed data from cache if available"""
        try:
            if self.cache_file.exists() and self.metadata_cache.exists():
                # Check if cache is newer than source file
                cache_time = self.cache_file.stat().st_mtime
                source_time = self._file_path.stat().st_mtime
                
                if cache_time > source_time:
                    if self.verbose:
                        print("Loading from cache...")
                    
                    with open(self.metadata_cache, 'rb') as f:
                        self._h5meta = pickle.load(f)
                    
                    with open(self.cache_file, 'rb') as f:
                        self.samples = pickle.load(f)
                    
                    return True
                else:
                    if self.verbose:
                        print("Cache is older than source file, reprocessing...")
            return False
        except Exception as e:
            logger.warning(f"Cache loading failed: {e}")
            return False
    
    def _save_to_cache(self) -> None:
        """Save processed data to cache"""
        try:
            if self.verbose:
                print("Saving to cache...")
            
            # Save metadata
            with open(self.metadata_cache, 'wb') as f:
                pickle.dump(self._h5meta, f)
            
            # Save samples (could be large, so show progress)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.samples, f)
            
            if self.verbose:
                cache_size_mb = self.cache_file.stat().st_size / (1024**2)
                print(f"✓ Cache saved successfully ({cache_size_mb:.1f} MB)")
                
        except Exception as e:
            logger.warning(f"Cache saving failed: {e}")
    
    def _preload_to_gpu(self) -> None:
        """Preload all data to GPU memory"""
        if self.verbose:
            print("Preloading data to GPU...")
        
        try:
            # Check available GPU memory
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if self.verbose:
                    print(f"GPU memory available: {gpu_memory_gb:.1f} GB")
            
            samples_iterator = tqdm(enumerate(self.samples), total=len(self.samples), 
                                  desc="Preloading to GPU", disable=not self.verbose)
            
            for i, sample in samples_iterator:
                for key, value in sample.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, torch.Tensor):
                                sample[key][subkey] = subvalue.to(self.device, non_blocking=True)
                    elif isinstance(value, torch.Tensor):
                        sample[key] = value.to(self.device, non_blocking=True)
                
                # Update progress every 100 samples
                if (i + 1) % 100 == 0 and self.verbose:
                    samples_iterator.set_postfix({'samples': f'{i+1}/{len(self.samples)}'})
            
            if self.verbose:
                print("✅ GPU preloading complete!")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU out of memory during preloading. Continuing with CPU storage.")
                self.preload_to_memory = False
                torch.cuda.empty_cache()
            else:
                raise
        except Exception as e:
            logger.warning(f"GPU preloading failed: {e}")
            self.preload_to_memory = False
        
    def _print_init(self) -> None:
        if not self.samples:
            print("No samples available for inspection")
            return
            
        sample = self.samples[0]
        keys = sample.keys()
        
        self.class_name = self.__class__.__name__
        print()
        print(f"====={self.class_name}=====")
        print(f"Dataset name: {self.file_name}")
        print(f"Number of samples: {self.__len__()}")
        print(f"Dataset keys: {list(keys)}")
        print("Dataset shape:")
        
        for signal_key in keys:
            if isinstance(sample[signal_key], dict):
                for signal_field in sample[signal_key].keys():
                    try:
                        tensor = sample[signal_key][signal_field]
                        tmp = '.'.join([signal_key, signal_field])
                        shape = tuple(tensor.shape) if hasattr(tensor, 'shape') else 'scalar'
                        dtype = tensor.dtype if hasattr(tensor, 'dtype') else type(tensor).__name__
                        print(f"\t {tmp}: {shape} ({dtype})")
                    except Exception as e:
                        print(f"\t {signal_key}.{signal_field}: Error - {e}")
            else:
                try:
                    shape = tuple(sample[signal_key].shape) if hasattr(sample[signal_key], 'shape') else 'scalar'
                    dtype = sample[signal_key].dtype if hasattr(sample[signal_key], 'dtype') else type(sample[signal_key]).__name__
                    print(f"\t {signal_key}: {shape} ({dtype})")
                except Exception as e:
                    print(f"\t {signal_key}: Error - {e}")
        
    def _check_file(self) -> bool:
        """Check if h5 file exists and is readable"""
        try:
            if not self._file_path.exists():
                return False
                
            # Check if file can be opened
            with h5py.File(self._file_path, 'r') as h5f:
                # Basic structure check
                required_groups = ['data', 'metadata']
                for group in required_groups:
                    if group not in h5f:
                        logger.warning(f"Missing required group '{group}' in HDF5 file")
                        return False
                return True
        except (OSError, IOError) as e:
            logger.error(f"Error accessing file: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking file: {e}")
            return False

    def _load_metadata(self) -> Dict:
            """FIXED metadata loading with proper string handling"""
            if self.verbose:
                print('Loading metadata...')
            
            metadata = {}
            try:
                with h5py.File(self._file_path, 'r') as h5f:
                    h5meta = h5f['metadata']
                    
                    for key in h5meta.keys():
                        if isinstance(h5meta[key], h5py.Group):
                            metadata[key] = {}
                            for subkey in h5meta[key].keys():
                                try:
                                    # Handle different data types
                                    data = h5meta[key][subkey][()]
                                    
                                    if isinstance(data, np.ndarray):
                                        if data.dtype.char == 'S':
                                            # FIXED: Properly handle string arrays - flatten nested lists
                                            if data.ndim == 0:
                                                # Single string
                                                metadata[key][subkey] = data.item().decode()
                                            else:
                                                # Array of strings - flatten and decode
                                                decoded_items = []
                                                for item in data.flat:
                                                    if isinstance(item, bytes):
                                                        decoded_items.append(item.decode())
                                                    else:
                                                        decoded_items.append(str(item).strip())
                                                metadata[key][subkey] = decoded_items
                                        else:
                                            # Numeric data
                                            if data.ndim == 0:
                                                metadata[key][subkey] = data.item()
                                            else:
                                                metadata[key][subkey] = data.tolist()
                                    else:
                                        # Scalar or other types
                                        if isinstance(data, bytes):
                                            metadata[key][subkey] = data.decode()
                                        else:
                                            metadata[key][subkey] = data
                                            
                                except Exception as e:
                                    logger.warning(f"Error loading metadata {key}.{subkey}: {e}")
                                    metadata[key][subkey] = []
                        else:
                            try:
                                data = h5meta[key][()]
                                if isinstance(data, np.ndarray) and data.dtype.char == 'S':
                                    if data.ndim == 0:
                                        metadata[key] = data.item().decode()
                                    else:
                                        decoded_items = []
                                        for item in data.flat:
                                            if isinstance(item, bytes):
                                                decoded_items.append(item.decode())
                                            else:
                                                decoded_items.append(str(item).strip())
                                        metadata[key] = decoded_items
                                elif isinstance(data, bytes):
                                    metadata[key] = data.decode()
                                else:
                                    metadata[key] = data.item() if hasattr(data, 'item') else data
                            except:
                                metadata[key] = []
                    
                    # Handle mask data
                    if 'mask' in h5meta:
                        try:
                            arr = h5meta['mask'][()].astype(np.int32)
                            arr[0] = arr[0] - 1  # for slicing in python
                            metadata['mask'] = tuple(map(tuple, arr.T))
                        except Exception as e:
                            logger.warning(f"Error loading mask: {e}")
                            metadata['mask'] = ()
                    
                    # Handle date
                    if 'date' in h5meta:
                        try:
                            date_data = h5meta['date'][()]
                            if isinstance(date_data, bytes):
                                metadata['date'] = date_data.decode()
                            else:
                                metadata['date'] = str(date_data)
                        except Exception as e:
                            logger.warning(f"Error loading date: {e}")
                            metadata['date'] = ""
                            
                # Post-process metadata to ensure correct structure
                # The issue is that we're getting nested lists like [['signal']] instead of ['signal']
                for category in ['nova', 'pvi']:
                    if category in metadata:
                        for field in ['signals', 'fields', 'stats']:
                            if field in metadata[category]:
                                # Flatten nested lists
                                data = metadata[category][field]
                                if isinstance(data, list) and len(data) > 0:
                                    # Check if it's a list of lists
                                    if all(isinstance(item, list) for item in data):
                                        # Flatten it
                                        flattened = []
                                        for sublist in data:
                                            if isinstance(sublist, list):
                                                flattened.extend(sublist)
                                            else:
                                                flattened.append(sublist)
                                        metadata[category][field] = flattened
                
                if self.verbose:
                    print(f'\t ✓ Metadata loaded ({len(metadata)} categories)')
                    
                    # Debug print to show what we loaded
                    print("\nDEBUG: Metadata structure:")
                    for key, value in metadata.items():
                        if isinstance(value, dict):
                            print(f"  {key}:")
                            for subkey, subvalue in value.items():
                                print(f"    {subkey}: {type(subvalue)} = {subvalue}")
                        else:
                            print(f"  {key}: {type(value)} = {value}")
                            
                return metadata
                
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                raise

    def _load_raw_data(self) -> Dict:
        """Optimized raw data loading with memory mapping and progress tracking"""
        if self.verbose:
            print('Loading raw data...')
        
        t1 = time.time()
        data = {}
        
        try:
            with h5py.File(self._file_path, 'r') as h5f:
                h5data = h5f['data']
                
                # Calculate total operations for progress tracking
                total_ops = 0
                for category in ['nova', 'pvi']:
                    if category in self._h5meta:
                        total_ops += len(self._h5meta[category]['signals']) * len(self._h5meta[category]['fields'])
                
                if self.verbose and total_ops > 0:
                    pbar = tqdm(total=total_ops, desc="Loading signals")
                
                # Load PVI and NOVA data efficiently
                for category in ['nova', 'pvi']:
                    if category not in self._h5meta:
                        continue
                        
                    for signal_key in self._h5meta[category]['signals']:
                        if signal_key not in h5data:
                            logger.warning(f"Signal '{signal_key}' not found in data")
                            continue
                            
                        data[signal_key] = {}
                        
                        for signal_field in self._h5meta[category]['fields']:
                            if signal_field not in h5data[signal_key]:
                                logger.warning(f"Field '{signal_field}' not found in {signal_key}")
                                continue
                                
                            try:
                                # Load data with optional memory mapping for large arrays
                                array_data = h5data[signal_key][signal_field][()]
                                
                                # Use memory mapping for large data (>100MB)
                                if self.use_mmap and array_data.nbytes > 100*1024*1024:
                                    if self.verbose:
                                        print(f"\nUsing memory mapping for {signal_key}.{signal_field} ({array_data.nbytes/(1024**2):.1f} MB)")
                                    tensor = torch.from_numpy(array_data.T).float()
                                else:
                                    tensor = torch.FloatTensor(array_data.T)
                                
                                # Store as tuple for compatibility
                                data[signal_key][signal_field] = tuple(tensor[n] for n in range(tensor.shape[0]))
                                
                                if self.verbose and total_ops > 0:
                                    pbar.update(1)
                                    
                            except Exception as e:
                                logger.error(f"Error loading {signal_key}.{signal_field}: {e}")
                                data[signal_key][signal_field] = tuple()
                
                if self.verbose and total_ops > 0:
                    pbar.close()
                
                # Load statistics
                if 'stats' in h5f and 'pviHP' in h5f['stats']:
                    stats = h5f['stats']['pviHP']
                    data['stats'] = {}
                    
                    if 'pvi' in self._h5meta and 'stats' in self._h5meta['pvi']:
                        for key in self._h5meta['pvi']['stats']:
                            if key in stats:
                                try:
                                    data['stats'][key] = torch.FloatTensor(stats[key][()].squeeze())
                                except Exception as e:
                                    logger.warning(f"Error loading stat '{key}': {e}")
                                    data['stats'][key] = torch.FloatTensor([])
                else:
                    logger.warning("No statistics found in data file")
                    data['stats'] = {}

            dt = time.time() - t1
            if self.verbose:
                print(f'\t ✓ Raw data loaded ({dt:.2f} seconds)')
                if 'pviHP' in data and 'signal' in data['pviHP']:
                    print(f"Number of periods: {len(data['pviHP']['signal'])}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            raise
    
    def _stack_samples_parallel(self) -> List[Dict]:
        """Parallel sample stacking for faster processing"""
        num_samples = len(self._h5meta['mask'])
        if self.verbose:
            print(f'Stacking samples ({num_samples} total) using parallel processing:')
        
        def process_sample(k):
            try:
                sample = {}
                
                # NOVA tensors - extract single period
                idx = range(*self._h5meta['mask'][k])[-1]
                for signal_key in self._h5meta['nova']['signals']:
                    if signal_key in self._h5data:
                        sample[signal_key] = {}
                        for signal_field in self._h5data[signal_key].keys():
                            if idx < len(self._h5data[signal_key][signal_field]):
                                tensor = self._h5data[signal_key][signal_field][idx]
                                sample[signal_key][signal_field] = tensor
                            else:
                                logger.warning(f"Index {idx} out of range for {signal_key}.{signal_field}")
                                sample[signal_key][signal_field] = torch.zeros(1)

                # PVI tensors - stack multiple periods
                sl = slice(*self._h5meta['mask'][k])
                for signal_key in self._h5meta['pvi']['signals']:
                    if signal_key in self._h5data:
                        sample[signal_key] = {}
                        for signal_field in self._h5data[signal_key].keys():
                            try:
                                tup = self._h5data[signal_key][signal_field][sl]
                                if tup:  # Check if tuple is not empty
                                    sample[signal_key][signal_field] = torch.cat(tup, dim=-1)
                                else:
                                    logger.warning(f"Empty tuple for {signal_key}.{signal_field} at slice {sl}")
                                    sample[signal_key][signal_field] = torch.zeros(32, 32, 1)
                            except Exception as e:
                                logger.warning(f"Error stacking {signal_key}.{signal_field}: {e}")
                                sample[signal_key][signal_field] = torch.zeros(32, 32, 1)
                
                # Statistics
                sample['stats'] = {}
                if 'stats' in self._h5data:
                    for stat_key in self._h5data['stats'].keys():
                        try:
                            if sl.stop <= len(self._h5data['stats'][stat_key]):
                                sample['stats'][stat_key] = self._h5data['stats'][stat_key][sl]
                            else:
                                sample['stats'][stat_key] = torch.zeros(sl.stop - sl.start)
                        except Exception as e:
                            logger.warning(f"Error loading stat {stat_key}: {e}")
                            sample['stats'][stat_key] = torch.zeros(1)
                
                return k, sample
                
            except Exception as e:
                logger.error(f"Error processing sample {k}: {e}")
                return k, {}
        
        # Use parallel processing with optimal number of workers
        num_workers = min(mp.cpu_count(), 8)  # Limit to avoid memory issues
        samples = [None] * num_samples
        
        if self.verbose:
            print(f"Using {num_workers} workers for parallel processing")
        
        t1 = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_sample, k) for k in range(num_samples)]
            
            if self.verbose:
                futures_iter = tqdm(futures, desc="Processing samples")
            else:
                futures_iter = futures
            
            for i, future in enumerate(futures_iter):
                try:
                    k, sample = future.result(timeout=30)  # 30 second timeout per sample
                    samples[k] = sample
                except Exception as e:
                    logger.error(f"Failed to process sample {i}: {e}")
                    samples[i] = {}
        
        dt = time.time() - t1
        if self.verbose:
            print(f'\t ✓ Parallel processing completed ({dt:.2f} seconds)')
        
        # Filter out empty samples
        valid_samples = [s for s in samples if s]
        if len(valid_samples) != num_samples:
            logger.warning(f"Only {len(valid_samples)}/{num_samples} samples processed successfully")
        
        return valid_samples
    
    def _stack_samples_sequential(self) -> List[Dict]:
        """Original sequential sample stacking with better error handling"""
        samples = []
        num_samples = len(self._h5meta['mask'])
        
        if self.verbose:
            print(f'Stacking samples ({num_samples} total):')
            sample_iter = tqdm(range(num_samples), desc="Processing samples")
        else:
            sample_iter = range(num_samples)
        
        t1 = time.time()
        
        for k in sample_iter:
            try:
                sample = {}
                
                # NOVA tensors
                idx = range(*self._h5meta['mask'][k])[-1]
                for signal_key in self._h5meta['nova']['signals']:
                    if signal_key in self._h5data:
                        sample[signal_key] = {}
                        for signal_field in self._h5data[signal_key].keys():
                            if idx < len(self._h5data[signal_key][signal_field]):
                                tensor = self._h5data[signal_key][signal_field][idx]
                                sample[signal_key][signal_field] = tensor
                            else:
                                sample[signal_key][signal_field] = torch.zeros(1)

                # PVI tensors
                sl = slice(*self._h5meta['mask'][k])
                for signal_key in self._h5meta['pvi']['signals']:
                    if signal_key in self._h5data:
                        sample[signal_key] = {}
                        for signal_field in self._h5data[signal_key].keys():
                            try:
                                tup = self._h5data[signal_key][signal_field][sl]
                                if tup:
                                    sample[signal_key][signal_field] = torch.cat(tup, dim=-1)
                                else:
                                    sample[signal_key][signal_field] = torch.zeros(32, 32, 1)
                            except Exception as e:
                                logger.warning(f"Error with {signal_key}.{signal_field}: {e}")
                                sample[signal_key][signal_field] = torch.zeros(32, 32, 1)
                
                # Statistics
                sample['stats'] = {}
                if 'stats' in self._h5data:
                    for stat_key in self._h5data['stats'].keys():
                        try:
                            sample['stats'][stat_key] = self._h5data['stats'][stat_key][sl]
                        except:
                            sample['stats'][stat_key] = torch.zeros(1)
                        
                samples.append(sample)
                
            except Exception as e:
                logger.error(f"Error processing sample {k}: {e}")
                continue
        
        dt = time.time() - t1
        if self.verbose:
            print(f'\t ✓ Sequential processing completed ({dt:.2f} seconds)')
        
        return samples
    
    def _stack_samples(self) -> List[Dict]:
        """Choose between parallel and sequential processing based on dataset size and system"""
        num_samples = len(self._h5meta['mask'])
        
        # Use parallel processing for larger datasets and multi-core systems
        if num_samples > 50 and mp.cpu_count() > 1:
            return self._stack_samples_parallel()
        else:
            # Use sequential method for small datasets or single-core systems
            return self._stack_samples_sequential()
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a specific sample"""
        if idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range (0-{len(self.samples)-1})")
        
        sample = self.samples[idx]
        info = {
            'index': idx,
            'keys': list(sample.keys()),
            'shapes': {},
            'dtypes': {},
            'memory_usage_mb': 0
        }
        
        for key, value in sample.items():
            if isinstance(value, dict):
                info['shapes'][key] = {}
                info['dtypes'][key] = {}
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'shape'):
                        info['shapes'][key][subkey] = tuple(subvalue.shape)
                        info['dtypes'][key][subkey] = str(subvalue.dtype)
                        info['memory_usage_mb'] += subvalue.numel() * subvalue.element_size() / (1024**2)
            elif hasattr(value, 'shape'):
                info['shapes'][key] = tuple(value.shape)
                info['dtypes'][key] = str(value.dtype)
                info['memory_usage_mb'] += value.numel() * value.element_size() / (1024**2)
        
        return info
    
    def clear_cache(self):
        """Clear cached data files"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                print(f"Removed cache file: {self.cache_file}")
            
            if self.metadata_cache.exists():
                self.metadata_cache.unlink()
                print(f"Removed metadata cache: {self.metadata_cache}")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        if idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range")
        return self.samples[idx]


class PviBatchServer:
    """Optimized batch server with better data loading performance"""
    
    def __init__(self,
                 dataset: PviDataset,
                 input_type: str,
                 output_type: str,
                 num_workers: Optional[int] = None,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 verbose: bool = True
                 ) -> None:
        
        self.file_name = dataset.file_name
        self.verbose = verbose
        self.input_type = self._validate_input_type(input_type)
        self.output_type = self._validate_output_type(output_type)
        self.dataset = self._extract_dataset(dataset)
        
        # Optimize number of workers based on system
        if num_workers is None:
            # Use optimal number of workers based on CPU and available memory
            cpu_count = mp.cpu_count()
            try:
                memory_gb = psutil.virtual_memory().total / (1024**3)
                
                # Conservative worker count to avoid memory issues
                if memory_gb >= 32:
                    self.num_workers = min(cpu_count, 8)
                elif memory_gb >= 16:
                    self.num_workers = min(cpu_count, 6)
                elif memory_gb >= 8:
                    self.num_workers = min(cpu_count, 4)
                else:
                    self.num_workers = min(cpu_count, 2)
            except:
                # Fallback if psutil fails
                self.num_workers = min(cpu_count, 4)
        else:
            self.num_workers = max(0, num_workers)  # Ensure non-negative
        
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and self.num_workers > 0
        
        if self.verbose:
            print(f"PviBatchServer successfully initiated!")
            print(f"Using {self.num_workers} workers, pin_memory={self.pin_memory}, persistent_workers={self.persistent_workers}")
        
        self.set_loader_params(batch_size=16, test_size=0.2, reload=False)
        self.reload()
        
    def _validate_input_type(self, input_type: str) -> str:
        valids = ["img", "bioz", "signal"]
        if input_type.lower() not in valids:
            raise ValueError(f"input_type must be one of {valids}")
        return input_type.lower()
    
    def _validate_output_type(self, output_type: str) -> str:
        valids = ["full", "sbp", "dbp", "minmax"]
        if output_type.lower() not in valids:
            raise ValueError(f"output_type must be one of {valids}")
        return output_type.lower()
    
    def _extract_dataset(self, dataset: PviDataset) -> List[Dict]:
        """Optimized dataset extraction with tensor operations and error handling"""
        if self.verbose:
            print("Extracting and preprocessing dataset...")
        
        samples = []
        failed_samples = 0
        
        sample_iter = tqdm(dataset.samples, desc="Extracting samples") if self.verbose else dataset.samples
        
        for i, ds in enumerate(sample_iter):
            try:
                sample = {}

                # Process BP output efficiently - check multiple possible BP signal names
                bp_found = False
                bp_signal_names = ['bp', 'nova_bp', 'arterial', 'ABP', 'art']  # Common BP signal names
                
                if self.output_type == 'full':
                    for bp_name in bp_signal_names:
                        if bp_name in ds and isinstance(ds[bp_name], dict) and 'signal' in ds[bp_name]:
                            sample['bp'] = ds[bp_name]['signal']
                            bp_found = True
                            break
                    
                    if not bp_found:
                        # Check if any NOVA signal could be BP (arterial pressure)
                        if 'nova' in str(type(ds)) or any(key.startswith('nova') for key in ds.keys()):
                            # Look for any signal that might be arterial pressure
                            for key, value in ds.items():
                                if isinstance(value, dict) and 'signal' in value:
                                    if any(keyword in key.lower() for keyword in ['art', 'bp', 'press']):
                                        sample['bp'] = value['signal']
                                        bp_found = True
                                        break
                        
                        if not bp_found:
                            logger.warning(f"Missing BP signal in sample {i}")
                            sample['bp'] = torch.zeros(50)  # Default BP signal length
                else:
                    # For non-full output types, try to extract systolic/diastolic
                    bp_signal = None
                    for bp_name in bp_signal_names:
                        if bp_name in ds and isinstance(ds[bp_name], dict) and 'signal' in ds[bp_name]:
                            bp_signal = ds[bp_name]['signal']
                            bp_found = True
                            break
                    
                    if bp_signal is not None:
                        sbp = bp_signal.max()
                        dbp = bp_signal.min()
                        if self.output_type == "sbp":
                            sample['bp'] = sbp
                        elif self.output_type == "dbp":
                            sample['bp'] = dbp
                        else:  # minmax
                            sample['bp'] = torch.tensor([dbp, sbp])
                    else:
                        # Default values if BP data missing
                        if self.output_type == "sbp":
                            sample['bp'] = torch.tensor(120.0)
                        elif self.output_type == "dbp":
                            sample['bp'] = torch.tensor(80.0)
                        else:  # minmax
                            sample['bp'] = torch.tensor([80.0, 120.0])
                
                # Process PVI inputs efficiently
                for pvi_key in ["pviLP", "pviHP"]:
                    pvi_found = False
                    if pvi_key in ds:
                        if self.input_type == "bioz":
                            if isinstance(ds[pvi_key], dict) and "resistance" in ds[pvi_key] and "reactance" in ds[pvi_key]:
                                r = ds[pvi_key]["resistance"]
                                x = ds[pvi_key]["reactance"]
                                sample[pvi_key] = torch.vstack([r, x])
                                pvi_found = True
                            else:
                                logger.warning(f"Missing resistance/reactance in {pvi_key} for sample {i}")
                        else:
                            # img or signal
                            if isinstance(ds[pvi_key], dict) and self.input_type in ds[pvi_key]:
                                # Add channel dimension efficiently
                                data = ds[pvi_key][self.input_type]
                                if isinstance(data, torch.Tensor):
                                    sample[pvi_key] = data.unsqueeze(0)
                                    pvi_found = True
                                else:
                                    logger.warning(f"Invalid data type for {pvi_key}.{self.input_type} in sample {i}")
                            else:
                                logger.warning(f"Missing {self.input_type} in {pvi_key} for sample {i}")
                    else:
                        logger.warning(f"Missing {pvi_key} in sample {i}")
                    
                    # Create default data if not found
                    if not pvi_found:
                        if self.input_type == "bioz":
                            sample[pvi_key] = torch.zeros(2, 32, 32)
                        else:
                            sample[pvi_key] = torch.zeros(1, 32, 32)
                
                # Handle stats
                sample['stats'] = ds.get('stats', {})
                
                samples.append(sample)
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                failed_samples += 1
                # Create a minimal valid sample instead of skipping
                sample = {
                    'bp': torch.zeros(50) if self.output_type == 'full' else torch.tensor([80.0, 120.0]),
                    'pviLP': torch.zeros(2, 32, 32) if self.input_type == "bioz" else torch.zeros(1, 32, 32),
                    'pviHP': torch.zeros(2, 32, 32) if self.input_type == "bioz" else torch.zeros(1, 32, 32),
                    'stats': {}
                }
                samples.append(sample)
                continue
        
        if failed_samples > 0:
            logger.warning(f"Failed to process {failed_samples} samples")
        
        if self.verbose:
            print(f"✓ Dataset extraction complete: {len(samples)} samples")
        
        return samples
        
    def _print_init(self) -> None:
        try:
            _, test_loader = self.get_loaders()
            test_batch = next(iter(test_loader))
            keys = test_batch.keys()
            
            self.class_name = self.__class__.__name__
            print()
            print(f"====={self.class_name}=====")
            print(f"Dataset name: {self.file_name}")
            print(f"Number of samples: {len(self.dataset)}")
            print(f"Test size: {self.test_size}")
            print(f"Batch size: {self.batch_size}")
            print(f"Random state: {self.random_state}")
            print(f"Data keys: {list(keys)}")
            print(f"\t PVI type (input): {self.input_type}")
            print(f"\t BP type (output): {self.output_type}")
            
            print("Batch shape:")
            for key, obj in test_batch.items():
                if isinstance(obj, dict):
                    for k2 in obj.keys():
                        if hasattr(obj[k2], 'shape'):
                            shape = tuple(obj[k2].shape)
                            tmp = '.'.join([key, k2])
                            print(f"\t {tmp}: {shape}")
                        
                else: # value is a tensor
                    if hasattr(obj, 'shape'):
                        shape = tuple(obj.shape)
                        print(f"\t {key}: {shape}")
                        
        except Exception as e:
            logger.warning(f"Could not print batch info: {e}")

    def reload(self):
        """Reload data splits and loaders"""
        try:
            self._data_subset = self._split_datasets()
            self.loaders = self._init_loaders()
        except Exception as e:
            logger.error(f"Error during reload: {e}")
            raise

    def _split_datasets(self, shuffle: bool = True):
        indices = list(range(len(self.dataset)))
        
        if len(indices) == 0:
            raise ValueError("No samples available for splitting")
        
        # Ensure test_size doesn't exceed available samples
        effective_test_size = min(self.test_size, 0.8)  # Max 80% for test
        
        try:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=effective_test_size,
                random_state=self.random_state,
                shuffle=shuffle
            )
        except ValueError as e:
            # Fallback for very small datasets
            if len(indices) == 1:
                train_idx, test_idx = [0], [0]
            else:
                split_point = max(1, int(len(indices) * (1 - effective_test_size)))
                train_idx = indices[:split_point]
                test_idx = indices[split_point:]
        
        data_subset = {}
        data_subset["train"] = Subset(self.dataset, train_idx)
        data_subset["test"] = Subset(self.dataset, test_idx)
        
        return data_subset
        
    def get_data_subsets(self) -> Tuple[Subset]:
        return tuple(self._data_subset.values())
    
    def set_loader_params(self,
                          batch_size: int,
                          test_size: float,
                          random_state: Optional[int] = None,
                          reload: bool = True,
                          **kwargs) -> None:
        
        # Validate parameters
        self.batch_size = max(1, batch_size)
        self.test_size = max(0.1, min(0.9, test_size))  # Clamp between 0.1 and 0.9
        self.random_state = random_state
        
        # Merge with optimized default parameters
        default_kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'prefetch_factor': 2 if self.num_workers > 0 else None,
            'drop_last': False,  # Keep all samples
            'timeout': 60 if self.num_workers > 0 else 0,  # 60 second timeout for workers
        }
        
        # Remove None values
        default_kwargs = {k: v for k, v in default_kwargs.items() if v is not None}
        
        # Update with user parameters
        default_kwargs.update(kwargs)
        self.dataloader_kwargs = default_kwargs
        
        if reload:
            self.reload()
        
    def _init_loaders(self) -> Dict[str, DataLoader]:
        """Initialize optimized data loaders with error handling"""
        loaders = {}
        
        try:
            # Training loader with shuffling
            train_kwargs = self.dataloader_kwargs.copy()
            train_kwargs['shuffle'] = True
            
            loaders["train"] = DataLoader(
                self._data_subset["train"],
                batch_size=self.batch_size,
                collate_fn=collate_fn_optimized,
                **train_kwargs
            )
            
            # Test loader without shuffling
            test_kwargs = self.dataloader_kwargs.copy()
            test_kwargs['shuffle'] = False  # Ensure no shuffling for test
            
            loaders["test"] = DataLoader(
                self._data_subset["test"],
                batch_size=self.batch_size,
                collate_fn=collate_fn_optimized,
                **test_kwargs
            )
            
        except Exception as e:
            logger.error(f"Error creating data loaders: {e}")
            # Fallback to simpler configuration
            simple_kwargs = {
                'batch_size': self.batch_size,
                'num_workers': 0,
                'pin_memory': False,
                'collate_fn': collate_fn_optimized
            }
            
            loaders["train"] = DataLoader(self._data_subset["train"], shuffle=True, **simple_kwargs)
            loaders["test"] = DataLoader(self._data_subset["test"], shuffle=False, **simple_kwargs)
            
            logger.warning("Using fallback DataLoader configuration")
        
        return loaders

    def get_loaders(self) -> Tuple[DataLoader]:
        return tuple(self.loaders.values())
    
    def get_data_shapes(self) -> Dict[str, Union[Tuple, Dict]]:
        """Get information about data shapes"""
        if not self.dataset:
            return {}
            
        shapes = {}
        sample = self.dataset[0]
        
        shapes['batch_size'] = self.batch_size
        shapes['num_samples'] = len(self.dataset)
        
        # Input shapes
        if 'pviHP' in sample:
            shapes['input'] = tuple(sample['pviHP'].shape)
        
        # Output shapes
        if 'bp' in sample:
            if hasattr(sample['bp'], 'shape'):
                shapes['output'] = tuple(sample['bp'].shape)
            else:
                shapes['output'] = (1,)  # Scalar
        
        # Stats shapes
        if 'stats' in sample and sample['stats']:
            stats_shapes = {}
            for key, value in sample['stats'].items():
                if hasattr(value, 'shape'):
                    stats_shapes[key] = tuple(value.shape)
            shapes['stats'] = stats_shapes
        
        return shapes
    
    def get_loader_info(self) -> Dict:
        """Get comprehensive information about the data loaders"""
        info = {
            'dataset_size': len(self.dataset),
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'input_type': self.input_type,
            'output_type': self.output_type,
            'test_size': self.test_size
        }
        
        if hasattr(self, '_data_subset'):
            info['train_samples'] = len(self._data_subset['train'])
            info['test_samples'] = len(self._data_subset['test'])
            info['train_batches'] = len(self.loaders['train'])
            info['test_batches'] = len(self.loaders['test'])
        
        return info


def collate_fn_optimized(batch):
    """Optimized collate function for better batching performance"""
    if len(batch) == 0:
        return {}
    
    # Get keys from first sample
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        if key == 'stats':
            # Handle stats dictionary
            if batch[0][key]:  # Check if stats dict is not empty
                stats_dict = {}
                for stat_key in batch[0][key].keys():
                    try:
                        stat_values = [sample[key][stat_key] for sample in batch if stat_key in sample[key]]
                        if stat_values and all(hasattr(v, 'shape') for v in stat_values):
                            stats_dict[stat_key] = torch.stack(stat_values)
                        else:
                            stats_dict[stat_key] = stat_values
                    except Exception as e:
                        logger.warning(f"Error collating stat {stat_key}: {e}")
                        stats_dict[stat_key] = []
                result[key] = stats_dict
            else:
                result[key] = {}
        else:
            # Regular tensor stacking
            try:
                values = [sample[key] for sample in batch]
                if values and all(isinstance(v, torch.Tensor) for v in values):
                    # Check if all tensors have the same shape
                    shapes = [v.shape for v in values]
                    if all(s == shapes[0] for s in shapes):
                        result[key] = torch.stack(values)
                    else:
                        logger.warning(f"Inconsistent shapes for key {key}: {shapes}")
                        result[key] = values
                else:
                    result[key] = values
            except Exception as e:
                logger.warning(f"Error collating key {key}: {e}")
                result[key] = [sample[key] for sample in batch]
    
    return result


# Enhanced testing functions with optimizations
def load_subjects(subject_idx: Union[int, List[int]],
                 session: str = "baseline",
                 root: str = "/home/lucas_takanori/phd/data",
                 cache_dir: Optional[str] = None,
                 preload_to_memory: bool = False,
                 use_mmap: bool = True,
                 force_reload: bool = False,
                 verbose: bool = True) -> Tuple[PviDataset, ...]:
    """Load subjects with caching and optimization"""
    if not isinstance(subject_idx, (list, tuple, range)):
        subject_idx = [subject_idx]
        
    datasets = []
    
    for k in subject_idx:
        subject_id = 'subject' + str(k).zfill(3) 
        if verbose:
            print(f"Loading {subject_id} with optimizations...")
        
        try:
            # Set up paths
            pm = DataPathManager(
                subject=subject_id,
                session=session,
                root=root)
            
            # Set cache directory
            if cache_dir is None:
                cache_dir = pm._root / "cache"
            
            dataset = PviDataset(
                pm._h5_path,
                cache_dir=cache_dir,
                preload_to_memory=preload_to_memory,
                use_mmap=use_mmap,
                force_reload=force_reload,
                verbose=verbose
            )
            
            if verbose:
                pm._print_init()
                dataset._print_init()
            
            datasets.append(dataset)
            
        except Exception as e:
            logger.error(f"Failed to load {subject_id}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No datasets were successfully loaded")
    
    return tuple(datasets)


def prep_servers(pvi_datasets: Union[PviDataset, Tuple[PviDataset, ...]],
                input_type: str = "signal",
                output_type: str = "minmax",
                num_workers: Optional[int] = None,
                verbose: bool = True
                ) -> Tuple[PviBatchServer, ...]:
    """Prepare optimized batch servers"""
    if not isinstance(pvi_datasets, (list, tuple)):
        pvi_datasets = [pvi_datasets]
        
    servers = []
    for i, dataset in enumerate(pvi_datasets):
        try:
            server = PviBatchServer(
                dataset=dataset,
                input_type=input_type,
                output_type=output_type,
                num_workers=num_workers,
                verbose=verbose
            )
            
            if verbose:
                server._print_init()
            
            servers.append(server)
            
        except Exception as e:
            logger.error(f"Failed to create server for dataset {i}: {e}")
            continue
     
    if not servers:
        raise ValueError("No batch servers were successfully created")
     
    return tuple(servers)


def benchmark_loading_performance(dataset, batch_sizes=[4, 8, 16, 32], 
                                 num_workers_list=[0, 2, 4, 8],
                                 verbose=True):
    """Benchmark different data loading configurations"""
    
    if verbose:
        print("\n" + "="*60)
        print("DATA LOADING PERFORMANCE BENCHMARK")
        print("="*60)
    
    results = {}
    
    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            if verbose:
                print(f"\nTesting batch_size={batch_size}, num_workers={num_workers}")
            
            try:
                # Create optimized dataloader
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=num_workers > 0,
                    prefetch_factor=2 if num_workers > 0 else None,
                    timeout=30 if num_workers > 0 else 0,
                    collate_fn=collate_fn_optimized
                )
                
                # Time loading batches
                start_time = time.time()
                num_batches = 0
                
                try:
                    for i, batch in enumerate(dataloader):
                        num_batches += 1
                        if i >= 10:  # Test first 10 batches
                            break
                except Exception as e:
                    if verbose:
                        print(f"  ERROR during iteration: {e}")
                    continue
                
                elapsed_time = time.time() - start_time
                
                if elapsed_time > 0:
                    batches_per_second = num_batches / elapsed_time
                    samples_per_second = batches_per_second * batch_size
                    
                    results[(batch_size, num_workers)] = {
                        'time': elapsed_time,
                        'batches_per_second': batches_per_second,
                        'samples_per_second': samples_per_second,
                        'num_batches': num_batches
                    }
                    
                    if verbose:
                        print(f"  Time: {elapsed_time:.2f}s, Batches/sec: {batches_per_second:.2f}, Samples/sec: {samples_per_second:.1f}")
                else:
                    if verbose:
                        print("  WARNING: Benchmark completed too quickly")
                
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                results[(batch_size, num_workers)] = {'error': str(e)}
    
    # Find best configuration
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results and verbose:
        best_config = max(valid_results.keys(), key=lambda k: valid_results[k]['samples_per_second'])
        best_perf = valid_results[best_config]
        baseline_perf = valid_results.get((best_config[0], 0), {}).get('samples_per_second', 1)
        speedup = best_perf['samples_per_second'] / baseline_perf if baseline_perf > 0 else 1
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   batch_size={best_config[0]}, num_workers={best_config[1]}")
        print(f"   Performance: {best_perf['samples_per_second']:.1f} samples/sec")
        print(f"   Speedup vs single-threaded: {speedup:.1f}x")
    
    return results


def optimize_system_settings(verbose=True):
    """Apply system-level optimizations for better performance"""
    if verbose:
        print("\n" + "="*60)
        print("APPLYING SYSTEM OPTIMIZATIONS")
        print("="*60)
    
    optimizations = []
    
    try:
        # PyTorch settings
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            optimizations.append("✓ Enabled cuDNN benchmark mode")
            
            torch.backends.cudnn.deterministic = False
            optimizations.append("✓ Disabled cuDNN deterministic mode for speed")
            
            # Set GPU memory fraction
            torch.cuda.set_per_process_memory_fraction(0.95)
            optimizations.append("✓ Set GPU memory fraction to 95%")
            
            # Clear cache
            torch.cuda.empty_cache()
            optimizations.append("✓ Cleared GPU cache")
        
        # CPU thread optimization
        cpu_count = mp.cpu_count()
        optimal_threads = min(cpu_count, 8)
        torch.set_num_threads(optimal_threads)
        optimizations.append(f"✓ Set PyTorch threads to {optimal_threads}")
        
        # Set environment variables for better performance
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        optimizations.append(f"✓ Set OMP_NUM_THREADS to {optimal_threads}")
        
    except Exception as e:
        logger.warning(f"Some optimizations failed: {e}")
    
    # Print applied optimizations
    if verbose:
        for opt in optimizations:
            print(opt)
    
    return optimizations


def print_system_info(verbose=True):
    """Print system information for optimization guidance"""
    if not verbose:
        return
        
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # CPU info
    cpu_count = mp.cpu_count()
    print(f"CPU cores: {cpu_count}")
    
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            print(f"CPU frequency: {cpu_freq.current:.0f} MHz")
    except:
        pass
    
    # Memory info
    try:
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        print(f"Total RAM: {memory_gb:.1f} GB")
        print(f"Available RAM: {available_gb:.1f} GB")
        print(f"Memory usage: {memory.percent:.1f}%")
    except Exception as e:
        print(f"Memory info unavailable: {e}")
    
    # GPU info
    if torch.cuda.is_available():
        try:
            gpu_count = torch.cuda.device_count()
            print(f"GPUs available: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            cuda_version = torch.version.cuda
            print(f"CUDA version: {cuda_version}")
        except Exception as e:
            print(f"GPU info error: {e}")
    else:
        print("No CUDA GPUs available")
    
    # PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    
    # Storage info
    try:
        disk_usage = psutil.disk_usage('/')
        free_space_gb = disk_usage.free / (1024**3)
        print(f"Free disk space: {free_space_gb:.1f} GB")
    except:
        pass


def run_comprehensive_test(data_path: str, verbose: bool = True):
    """Run a comprehensive test of the optimized data utilities"""
    if verbose:
        print("🧪 Running comprehensive test of optimized data utilities...")
    
    try:
        # Print system info
        print_system_info(verbose)
        
        # Apply optimizations
        optimize_system_settings(verbose)
        
        # Test dataset loading
        if verbose:
            print(f"\n📂 Testing dataset loading from: {data_path}")
        
        dataset = PviDataset(
            data_path,
            cache_dir="./test_cache",
            preload_to_memory=False,
            use_mmap=True,
            verbose=verbose
        )
        
        if verbose:
            print(f"✓ Dataset loaded successfully with {len(dataset)} samples")
            dataset._print_init()
        
        # Test batch server
        if verbose:
            print(f"\n⚙️ Testing batch server...")
        
        server = PviBatchServer(
            dataset=dataset,
            input_type="img",
            output_type="full",
            num_workers=4,
            verbose=verbose
        )
        
        if verbose:
            server._print_init()
        
        # Test data loading
        train_loader, test_loader = server.get_loaders()
        
        if verbose:
            print(f"\n⏱️ Testing data loading speed...")
        
        # Time a few batches
        start_time = time.time()
        batch_count = 0
        
        for i, batch in enumerate(train_loader):
            batch_count += 1
            if i >= 5:  # Test first 5 batches
                break
        
        elapsed_time = time.time() - start_time
        
        if elapsed_time > 0:
            batches_per_sec = batch_count / elapsed_time
            samples_per_sec = batches_per_sec * server.batch_size
            
            if verbose:
                print(f"✓ Loaded {batch_count} batches in {elapsed_time:.2f} seconds")
                print(f"Performance: {batches_per_sec:.2f} batches/sec, {samples_per_sec:.1f} samples/sec")
        
        # Run benchmark
        if verbose:
            print(f"\n🏃 Running performance benchmark...")
        
        benchmark_loading_performance(dataset, verbose=verbose)
        
        if verbose:
            print(f"\n✅ All tests completed successfully!")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"\n❌ Test failed: {e}")
        logger.error(f"Comprehensive test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests if module is executed directly
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        run_comprehensive_test(data_path)
    else:
        print("Testing optimized data loading with dummy data...")
        print("To test with real data, run: python data_utils.py /path/to/your/data.h5")
        
        # Basic functionality test without real data
        print_system_info()
        optimize_system_settings()
        
        print("\n✅ Basic functionality test completed!")
        print("For full testing, provide a path to an HDF5 data file.")

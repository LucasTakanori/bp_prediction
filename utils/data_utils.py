# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:33:34 2024

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

class DataPathManager:
    
    class __DefaultPaths: # private attributes to get default directory
        root: str = os.getenv('BP_DATA_ROOT', '/home/lucas_takanori/phd/data')
        subject: str = "subject001"
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
            "figures": self.figure_dir,
            "results": self.result_dir
            }
        if kw in valid_dirs:
            os.startfile(valid_dirs[kw])
        else:
            raise ValueError(f"Unrecognized keyword '{kw}'. Expected one of: {list(valid_dirs.keys())}.")
            
class PviDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 device: torch.device = torch.device('cpu')) -> None:
        
        self._file_path = Path(file_path)
        self.file_name = self._file_path.name
        self.parent_dir = self._file_path.parent
        self.device = device
        self.num_periods = 0
        
        print(f'Using Torch version: {torch.__version__}')
        print(f'Data directory set to:\n\t{self.parent_dir}')
        
        if self._check_file():
            print(f"Found HDF5 data: '{self.file_name}'")
            self._h5meta = self._load_metadata()
            print("PviDataset successfully initiated!")
            
        else:
            raise FileNotFoundError(f"Data file not found at: {self._file_path}")
        
        self._h5data = self._load_raw_data()

        if self._h5meta.get('mask') is None:
            if self.num_periods > 0:
                print(f"Generating default mask from {self.num_periods} periods.")
                self._h5meta['mask'] = tuple((i, i + 1) for i in range(self.num_periods))
            else:
                self._h5meta['mask'] = []

        self.samples = self._stack_samples()

        print("Finish loading PviDataset!")
    
    def _print_init(self) -> None:
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
            for signal_field in sample[signal_key].keys():
                tensor = sample[signal_key][signal_field]
                tmp = '.'.join([signal_key,signal_field])
                shape = tuple(tensor.shape)
                print(f"\t {tmp}: {shape}")
        
    def _check_file(self) -> bool:
        """Check if h5 file exists and is readable"""
        try:
            # Check if file exists and can be opened
            with h5py.File(self._file_path, 'r') as _:
                return True
        except (OSError, IOError) as e:
            print(f"Error accessing file: {e}")
            return False

    def _load_metadata(self) -> Dict:
        print('Loading metadata...')
        # t1 = time.time()
        metadata = {}
        with h5py.File(self._file_path, 'r') as h5f:
            h5meta = h5f['metadata']
            for key in h5meta.keys():
                if isinstance(h5meta[key],h5py.Group):
                    metadata[key] = {}
                    for subkey in h5meta[key].keys():
                        metadata[key][subkey] = [s.item().decode() for s in h5meta[key][subkey][()]]
                else:
                    metadata[key] = []
            
            if 'mask' in h5meta:
                arr = h5meta['mask'][()].astype(np.int32)
                arr[0] = arr[0] - 1 # for slicing in python
                metadata['mask'] = tuple(map(tuple, arr.T))
            else:
                print("⚠️  'mask' not found in metadata. Will attempt to generate one.")
                metadata['mask'] = None
            
            if 'date' in h5meta:
                metadata['date'] = h5meta['date'][()].item().decode()
            else:
                metadata['date'] = 'Unknown'

        # dt = time.time() - t1
        print('\t ...Done!')
        return metadata
    
    def _load_raw_data(self) -> Dict:
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
    
    def _stack_samples(self) -> Dict:
        samples = []
        num_samples = len(self._h5meta['mask'])
        print(f'Stacking samples ({num_samples} total):')
        t1 = time.time()
        for k in range(num_samples):
            sample = {}
            
            # for NOVA tensors,
            # we extract a single period per sample
            idx = range(*self._h5meta['mask'][k])[-1]
            for signal_key in self._h5meta['nova']['signals']:
                sample[signal_key] = {}
                for signal_field in self._h5data[signal_key].keys():
                    tensor = self._h5data[signal_key][signal_field][idx]
                    sample[signal_key][signal_field] = tensor

            # for PVI tensors,
            # we stack multiple periods per sample
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
            
            if (not (k+1)%100) or (k+1)==num_samples:
                print(f"\t ...{k+1}/{num_samples} samples")
        
        dt = time.time() - t1
        print(f'\t ...Done! ({dt:.2f} seconds)')  
        print(f"Number of samples: {num_samples}")
        return samples
            
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

class PviBatchServer:
    def __init__(self,
                 dataset: PviDataset,
                 input_type: str,
                 output_type: str,
                 ) -> None:
        
        # self.num_samples = len(dataset)
        self.dataset = dataset
        self.input_type = self._validate_input_type(input_type)
        self.output_type = self._validate_output_type(output_type)
        
        self._extracted_data = self._extract_dataset(dataset)
        self._print_init()
        
    def _validate_input_type(self, input_type: str) -> str:
        valid_types = ["signal", "image"]
        if input_type not in valid_types:
            raise ValueError(f"Invalid input_type '{input_type}'. Must be one of {valid_types}")
        return input_type
    
    def _validate_output_type(self, output_type: str) -> str:
        valid_types = ["minmax", "mean", "std"]
        if output_type not in valid_types:
            raise ValueError(f"Invalid output_type '{output_type}'. Must be one of {valid_types}")
        return output_type
            
    def _extract_dataset(self,
                         dataset: PviDataset) -> list[Dict]:
        extracted_data = []
        for sample in dataset:
            extracted_sample = {}
            
            # Extract input data
            if self.input_type == "signal":
                # For signal input, use PVI signal
                extracted_sample['input'] = sample['pviHP']['signal']
            else:  # image input
                # For image input, use PVI image data
                extracted_sample['input'] = sample['pviHP']['img']
            
            # Extract output data (simplified - using BP signal for now)
            if self.output_type == "minmax":
                # Use BP signal min/max as output
                bp_signal = sample['bp']['signal']
                extracted_sample['output'] = torch.stack([
                    torch.min(bp_signal),
                    torch.max(bp_signal)
                ])
            elif self.output_type == "mean":
                # Use BP signal mean as output
                bp_signal = sample['bp']['signal']
                extracted_sample['output'] = torch.mean(bp_signal).unsqueeze(0)
            else:  # std output
                # Use BP signal std as output
                bp_signal = sample['bp']['signal']
                extracted_sample['output'] = torch.std(bp_signal).unsqueeze(0)
            
            extracted_data.append(extracted_sample)
        
        return extracted_data
        
    def _print_init(self) -> None:
        self.class_name = self.__class__.__name__
        print()
        print(f"====={self.class_name}=====")
        print(f"Input type: {self.input_type}")
        print(f"Output type: {self.output_type}")
        print(f"Number of samples: {len(self._extracted_data)}")
        
        # Print shapes
        sample = self._extracted_data[0]
        print("Data shapes:")
        print(f"\tInput: {sample['input'].shape}")
        print(f"\tOutput: {sample['output'].shape}")
        
    def reload(self):
        """Reload the dataset"""
        self._extracted_data = self._extract_dataset(self.dataset)

    def _split_datasets(self, shuffle: bool=True):
        """Split the dataset into train and test sets"""
        if shuffle:
            np.random.shuffle(self._extracted_data)
        
        # Split into train and test sets
        split_idx = int(len(self._extracted_data) * 0.8)  # 80% train, 20% test
        train_data = self._extracted_data[:split_idx]
        test_data = self._extracted_data[split_idx:]
        
        return train_data, test_data
        
    def get_data_subsets(self) -> Tuple[Subset]:
        """Get train and test subsets"""
        train_data, test_data = self._split_datasets()
        return Subset(self._extracted_data, range(len(train_data))), \
               Subset(self._extracted_data, range(len(train_data), len(self._extracted_data)))
    
    def set_loader_params(self,
                           batch_size: int,
                           test_size: float,
                           random_state: Optional[int] = None,
                           reload: bool = True,
                           **kwargs) -> None:
        """Set parameters for data loading"""
        if reload:
            self.reload()
        
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize data loaders
        self._init_loaders()
        
    def _init_loaders(self) -> Dict[str, DataLoader]:
        """Initialize data loaders"""
        train_subset, test_subset = self.get_data_subsets()
        
        self.train_loader = DataLoader(
            train_subset,
                            batch_size=self.batch_size,
                            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_subset,
                            batch_size=self.batch_size,
                            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return {
            'train': self.train_loader,
            'test': self.test_loader
        }

    def get_loaders(self) -> Tuple[DataLoader]:
        """Get train and test data loaders"""
        return self.train_loader, self.test_loader
    
    def get_data_shapes(self) -> Dict[str, Dict[str, Tuple]]:
        """Get shapes of input and output data"""
        sample = self._extracted_data[0]
        return {
            'input': sample['input'].shape,
            'output': sample['output'].shape
        }

def load_subjects(subject_idx: list[int],
                  session="baseline",
                  root=r"/home/lucas_takanori/phd/data",
                  ) -> Tuple:
    """Load multiple subjects' data"""
    datasets = []
    for idx in subject_idx:
        subject = f"subject{idx:03d}"
        path_manager = DataPathManager(subject=subject, session=session, root=root)
        dataset = PviDataset(path_manager._h5_path)
        datasets.append(dataset)
    return tuple(datasets)

def prep_servers(pvi_datasets: Tuple['PviDataset'],
                 input_type="signal",
                 output_type="minmax",
                 ) -> Tuple['PviBatchServer']:
    """Prepare batch servers for multiple datasets"""
    servers = []
    for dataset in pvi_datasets:
        server = PviBatchServer(dataset, input_type, output_type)
        servers.append(server)
    return tuple(servers)

if __name__ == "__main__":
    print("tmp...")
    
    datasets = load_subjects(range(21,23))
    feeders = prep_servers(datasets)
    
    train_loader, test_loader = feeders[0].get_loaders()
    
    test_batch = next(iter(test_loader))
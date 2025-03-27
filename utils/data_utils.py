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
        
        print(f'Using Torch version: {torch.__version__}')
        print(f'Data directory set to:\n\t{self.parent_dir}')
        
        if self._check_file():
            print(f"Found HDF5 data: '{self.file_name}'")
            self._h5meta = self._load_metadata()
            print("PviDataset successfully initiated!")
            
        else:
            raise FileNotFoundError(f"Data file not found at: {self._file_path}")
        
        self._h5data = self._load_raw_data()
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
            
            arr = h5meta['mask'][()].astype(np.int32)
            arr[0] = arr[0] - 1 # for slicing in python
            metadata['mask'] = tuple(map(tuple, arr.T))
            
            metadata['date'] = h5meta['date'][()].item().decode()
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
        print(f"Number of periods: {tensor.shape[0]}")
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
        self.file_name = dataset.file_name
        
        self.input_type = self._validate_input_type(input_type)
        self.output_type = self._validate_output_type(output_type)
        self.dataset = self._extract_dataset(dataset)
        print("PviBatchServer successfully initiated!")
        
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
            
    def _extract_dataset(self,
                         dataset: PviDataset) -> list[Dict]:
        # dataset_clone = copy.deepcopy(dataset)

        # # Intended shape exported from MATLAB: [N, H, W ,T] (Fortran layout)
        # # Actual shape read into from NumPy: [T, W, H ,N] (C layout)
        # # Desired shape in PyTorch: [N, C, H, W, T] (we will deal with this later)
        samples = []
        for ds in dataset.samples:
            sample = {}

            if self.output_type == 'full':
                sample['bp'] = ds['bp']['signal']
            else:
                sbp = ds['bp']['signal'].max()
                dbp = ds['bp']['signal'].min()
                if self.output_type == "sbp":
                    ds['bp'] = sbp
                elif self.output_type == "dbp":
                    sample['bp'] = dbp
                else:
                    sample['bp'] = torch.hstack([dbp,sbp])
            
            for pvi_key in ["pviLP", "pviHP"]:
                if self.input_type == "bioz":
                    r = ds[pvi_key]["resistance"]
                    x = ds[pvi_key]["reactance"]
                    sample[pvi_key] = torch.vstack([r,x])
                else:
                    # add channel dims
                    sample[pvi_key] = ds[pvi_key][self.input_type].unsqueeze(dim=0)
            
            sample['stats'] = ds['stats']
            
            samples.append(sample)
        
        # dataset_clone.samples = samples
        
        return samples
        
    def _print_init(self) -> None:
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
            if isinstance(obj,dict):
                for k2 in obj.keys():
                    shape = tuple(obj[k2].shape)
                    tmp = '.'.join([key,k2])
                    print(f"\t {tmp}: {shape}")
                    
            else: # value is a tensor
                shape = tuple(obj.shape)
                print(f"\t {key}: {shape}")
        
    def reload(self):
        self._data_subset = self._split_datasets()
        self.loaders = self._init_loaders()

    def _split_datasets(self, shuffle: bool=True):
        indices = list(range(len(self.dataset)))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=shuffle)
        
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
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.datasetloader_kwargs = kwargs
        
        if reload:
            self.reload()
        
    def _init_loaders(self) -> Dict[str, DataLoader]:
        loaders = {}
        loaders["train"] = DataLoader(self._data_subset["train"],
                            batch_size=self.batch_size,
                            shuffle=True,
                            **self.datasetloader_kwargs)
        
        loaders["test"] = DataLoader(self._data_subset["test"],
                            batch_size=self.batch_size,
                            shuffle=False,
                            **self.datasetloader_kwargs)
        return loaders

    def get_loaders(self) -> Tuple[DataLoader]:
        return tuple(self.loaders.values())
        
    # def get_train_loader(self) -> DataLoader:
    #     return self.loaders["train"]
    
    # def get_test_loader(self) -> DataLoader:
    #     return self.loaders["test"]
    
    def get_data_shapes(self) -> Dict[str, Dict[str, Tuple]]:
        shapes = {}
        sample = self.dataset[0]
        
        shapes['batch_size'] = self.batch_size
        shapes['input'] = tuple(sample['pviHP'].shape)
        shapes['output'] = tuple(sample['bp'].shape)
        
        if 'stats' in sample:
            num_stats = len(sample['stats'])
            stats_length = len(next(iter(sample['stats'].items()))[1])
            shapes['stats'] = (num_stats, stats_length)
        
        return shapes

# Testing ground:
def load_subjects(subject_idx: list[int],
                  session="baseline",
                  root=r"/home/lucas_takanori/phd/data",
                  ) -> Tuple:
    if type(subject_idx) not in [list, tuple, range]:
        subject_idx = [subject_idx]
        
    datasets = []
    
    for k in subject_idx:
        subject_id = 'subject' + str(k).zfill(3) 
        print(f"Testing with {subject_id}...")
        
        pm = DataPathManager(
            subject=subject_id,
            session=session,
            root=root)
        
        dataset = PviDataset(pm._h5_path)
        
        pm._print_init()
        dataset._print_init()
        
        datasets.append(dataset)
    
    return tuple(datasets)

def prep_servers(pvi_datasets: Tuple['PviDataset'],
                 input_type="signal",
                 output_type="minmax",
                 ) -> Tuple['PviBatchServer']:
    if type(pvi_datasets) not in [list, tuple, range]:
        pvi_datasets = [pvi_datasets]
        
    feeders = []
    for dataset in pvi_datasets:
        feeder = PviBatchServer(dataset=dataset,
                                input_type=input_type,
                                output_type=output_type)
        
        feeder._print_init()
        
        feeders.append(feeder)
     
    return tuple(feeders)

if __name__ == "__main__":
    print("tmp...")
    
    datasets = load_subjects(range(21,23))
    feeders = prep_servers(datasets)
    
    train_loader, test_loader = feeders[0].get_loaders()
    
    test_batch = next(iter(test_loader))
# Save this as debug_data_structure.py in your project root

import torch
import os
from pathlib import Path
import sys
import pprint

# Add project root to path to import your modules
sys.path.append('.')

from data_utils import PviDataset, PviBatchServer, DataPathManager

def inspect_dataset_structure(dataset_path):
    """Examine the structure of the dataset without training anything"""
    print(f"Loading dataset from: {dataset_path}")
    
    # Load dataset
    try:
        dataset = PviDataset(dataset_path)
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        
        # Inspect the first sample
        print("\n--- First Sample Structure ---")
        first_sample = dataset.samples[0]
        print_nested_structure(first_sample)
        
        # Create batch server and inspect a batch
        print("\n--- Batch Structure ---")
        batch_server = PviBatchServer(dataset, input_type="img", output_type="full")
        train_loader, test_loader = batch_server.get_loaders()
        
        # Get first batch from train loader
        first_batch = next(iter(train_loader))
        print_batch_structure(first_batch)
        
        # Try accessing the specific field that caused the error
        print("\n--- Attempting to access the specific field that caused the error ---")
        try:
            if 'pviHP' in first_batch and 'img' in first_batch['pviHP']:
                img_data = first_batch['pviHP']['img']
                print(f"Successfully accessed first_batch['pviHP']['img']")
                print(f"Type: {type(img_data)}")
                print(f"Shape: {img_data.shape if isinstance(img_data, torch.Tensor) else 'Not a tensor'}")
            else:
                print("Could not find first_batch['pviHP']['img']")
                if 'pviHP' in first_batch:
                    print(f"'pviHP' keys: {list(first_batch['pviHP'].keys())}")
                    print(f"'pviHP' type: {type(first_batch['pviHP'])}")
                else:
                    print("'pviHP' key not found in batch")
        except Exception as e:
            print(f"Error accessing data: {e}")
        
        # Print available top-level keys and their types
        print("\n--- Top Level Batch Keys and Types ---")
        for key in first_batch:
            value = first_batch[key]
            if isinstance(value, torch.Tensor):
                print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape}")
            elif isinstance(value, dict):
                print(f"Key: {key}, Type: dict with keys: {list(value.keys())}")
            else:
                print(f"Key: {key}, Type: {type(value)}")
        
    except Exception as e:
        print(f"Error during inspection: {e}")
        import traceback
        traceback.print_exc()

def print_nested_structure(data, indent=0):
    """Recursively print the structure of nested dictionaries and tensors"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}: dict")
                print_nested_structure(value, indent + 2)
            elif isinstance(value, torch.Tensor):
                print(" " * indent + f"{key}: tensor, shape={value.shape}, dtype={value.dtype}")
            else:
                print(" " * indent + f"{key}: {type(value)}")
    elif isinstance(data, torch.Tensor):
        print(" " * indent + f"tensor, shape={data.shape}, dtype={data.dtype}")
    else:
        print(" " * indent + f"{type(data)}")

def print_batch_structure(batch):
    """Print the structure of a batch"""
    if not isinstance(batch, dict):
        print(f"Batch is not a dictionary, type: {type(batch)}")
        return
    
    for key, value in batch.items():
        if isinstance(value, dict):
            print(f"Key: {key} (dict)")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    print(f"  {sub_key}: tensor, shape={sub_value.shape}, dtype={sub_value.dtype}")
                else:
                    print(f"  {sub_key}: {type(sub_value)}")
        elif isinstance(value, torch.Tensor):
            print(f"Key: {key}, tensor shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"Key: {key}, type={type(value)}")

if __name__ == "__main__":
    # Get the path to your H5 file - adjust this to your file location
    data_path = os.path.expanduser("~/phd/data/subject001_baseline_masked.h5")
    
    # Inspect the dataset
    inspect_dataset_structure(data_path)
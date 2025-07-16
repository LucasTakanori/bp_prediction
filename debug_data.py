#!/usr/bin/env python3
"""
Debug script to inspect data structure
"""

from utils.data_utils import PviDataset
import pprint

# Load dataset
dataset = PviDataset('/home/lucas_takanori/phd/data/subject001_baseline_masked.h5')

# Get first sample
sample = dataset[0]

print("Keys in sample:")
print(list(sample.keys()))

print("\nSample structure:")
for key in sample.keys():
    print(f"\n{key}:")
    if isinstance(sample[key], dict):
        for subkey in sample[key].keys():
            if hasattr(sample[key][subkey], 'shape'):
                print(f"  {subkey}: shape {sample[key][subkey].shape}")
            else:
                print(f"  {subkey}: {type(sample[key][subkey])}")
    else:
        if hasattr(sample[key], 'shape'):
            print(f"  shape: {sample[key].shape}")
        else:
            print(f"  type: {type(sample[key])}") 
#!/usr/bin/env python3
"""
Simple test script to verify the training pipeline works
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from utils.data_utils_memory_optimized import create_memory_efficient_multisubject_dataset
from src.utils.paths import setup_paths
from train.vae_clean import VAE

def test_data_loading():
    """Test data loading with memory optimization"""
    print("🧪 Testing data loading...")
    
    paths = setup_paths()
    subjects = ['subject001', 'subject002']
    
    try:
        combined_dataset, successful, failed = create_memory_efficient_multisubject_dataset(
            str(paths.data_root),
            subjects,
            mask_type='mask10',
            max_samples_per_subject=50,
            session="baseline"
        )
        
        print(f"✅ Dataset loaded: {len(combined_dataset)} samples")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        
        return combined_dataset
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        raise


def test_model_creation():
    """Test VAE model creation"""
    print("\n🧪 Testing model creation...")
    
    try:
        model = VAE(latent_dim=64)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"✅ Model created on device: {device}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        raise


def test_data_batch():
    """Test loading a single batch"""
    print("\n🧪 Testing data batch...")
    
    dataset = test_data_loading()
    
    try:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        print(f"✅ Batch loaded successfully")
        print(f"   Batch type: {type(batch)}")
        
        if isinstance(batch, dict):
            print(f"   Batch keys: {batch.keys()}")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"   {key} shape: {value.shape}")
                elif isinstance(value, dict):
                    print(f"   {key} (dict): {list(value.keys())}")
                    for subkey, subvalue in value.items():
                        if torch.is_tensor(subvalue):
                            print(f"     {subkey} shape: {subvalue.shape}")
        elif torch.is_tensor(batch):
            print(f"   Batch shape: {batch.shape}")
        
        return batch
        
    except Exception as e:
        print(f"❌ Batch loading failed: {e}")
        raise


def test_model_forward():
    """Test model forward pass"""
    print("\n🧪 Testing model forward pass...")
    
    model, device = test_model_creation()
    batch = test_data_batch()
    
    try:
        model.eval()
        
        # Extract the actual image data from the batch
        if isinstance(batch, dict):
            if 'pviHP' in batch:
                if isinstance(batch['pviHP'], dict) and 'img' in batch['pviHP']:
                    # Extract a single frame
                    img_data = batch['pviHP']['img'][:, :, :, 0].unsqueeze(1)  # [batch, 1, 32, 32]
                else:
                    img_data = batch['pviHP'][:, 0, :, :, 0].unsqueeze(1)
            elif 'input' in batch:
                img_data = batch['input'][:, :, :, 0].unsqueeze(1)
            else:
                print(f"❌ Unexpected batch structure: {batch.keys()}")
                return
        else:
            img_data = batch
        
        img_data = img_data.to(device)
        print(f"   Input shape: {img_data.shape}")
        
        with torch.no_grad():
            recon, mu, logvar = model(img_data)
            
        print(f"✅ Forward pass successful")
        print(f"   Reconstruction shape: {recon.shape}")
        print(f"   Mu shape: {mu.shape}")
        print(f"   Logvar shape: {logvar.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise


def main():
    """Run all tests"""
    print("🚀 Testing Simple Training Pipeline")
    print("=" * 50)
    
    try:
        # Test each component
        test_data_loading()
        test_model_creation()
        test_data_batch()
        test_model_forward()
        
        print("\n✅ All tests passed! Training pipeline is ready.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
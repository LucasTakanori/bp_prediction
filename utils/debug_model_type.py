#!/usr/bin/env python3
"""
Debug script to check model type and available methods
"""
import sys
import os
import torch

# Add the training directory to path
sys.path.append('research/scripts/Lucas/bp_prediction/train')

from optimized_bilstm_complete import VAE, VAEBiLSTMWithAttention

def debug_model_creation():
    """Debug the model creation process"""
    device = torch.device("cpu")  # Use CPU for simplicity
    
    print("=== Model Creation Debug ===")
    
    # Create VAE
    print("1. Creating VAE model...")
    vae_model = VAE(latent_dim=256)
    print(f"   VAE type: {type(vae_model)}")
    print(f"   VAE has train(): {hasattr(vae_model, 'train')}")
    
    # Create BiLSTM model
    print("2. Creating BiLSTM model...")
    model = VAEBiLSTMWithAttention(
        vae_model=vae_model,
        input_dim=256,
        hidden_dim=256,
        num_layers=2,
        output_dim=50,
        dropout=0.3,
        use_attention=True,
        attention_dim=128,
        vae_batch_size=64
    )
    
    print(f"   Model type: {type(model)}")
    print(f"   Model has train(): {hasattr(model, 'train')}")
    print(f"   Model class: {model.__class__}")
    print(f"   Model module: {model.__class__.__module__}")
    
    # Test calling train() method
    print("3. Testing train() method...")
    try:
        model.train()
        print("   ✓ model.train() succeeded")
    except Exception as e:
        print(f"   ❌ model.train() failed: {e}")
        print(f"   Error type: {type(e)}")
    
    # Check if torch.compile was somehow applied
    print("4. Checking for torch.compile...")
    print(f"   Is compiled: {'_orig_mod' in dir(model)}")
    if hasattr(torch, 'compile'):
        print(f"   torch.compile available: True")
    else:
        print(f"   torch.compile available: False")
    
    # Test with potential torch.compile
    print("5. Testing with torch.compile disabled...")
    if hasattr(torch, 'compile'):
        # Verify that torch.compile is disabled in the optimized training script
        try:
            # This should be disabled
            compiled_model = torch.compile(model) if False else model  # Simulating the disabled condition
            print(f"   Compiled model type: {type(compiled_model)}")
            print(f"   Compiled model has train(): {hasattr(compiled_model, 'train')}")
        except Exception as e:
            print(f"   Error during compile test: {e}")

if __name__ == "__main__":
    debug_model_creation() 
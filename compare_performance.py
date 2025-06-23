#!/usr/bin/env python3
"""
Performance Comparison Script
Shows how batch size affects training speed
"""

import time
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def simulate_training_speed(samples, batch_size, num_epochs=1):
    """Simulate training with different batch sizes"""
    print(f"\n📊 Testing with {samples} samples, batch_size={batch_size}")
    
    # Create dummy data
    X = torch.randn(samples, 1, 32, 32)
    y = torch.randn(samples, 128)  # latent dim
    dataset = TensorDataset(X, y)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"   Number of batches: {len(dataloader)}")
    
    # Simulate training time
    start_time = time.time()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            # Simulate forward pass + backward pass
            time.sleep(0.01)  # Simulate compute time per batch
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx+1}/{len(dataloader)}", end='\r')
    
    end_time = time.time()
    print(f"   Time per epoch: {end_time - start_time:.2f} seconds")
    print(f"   Time per batch: {(end_time - start_time) / len(dataloader):.3f} seconds")
    
    return end_time - start_time

def main():
    """Compare different configurations"""
    print("🚀 Performance Comparison: Batch Size Impact")
    print("=" * 60)
    
    # Single subject scenario (train_enhanced.py)
    print("\n🔹 SINGLE SUBJECT SCENARIO (train_enhanced.py)")
    print("   Samples: 400, Batch size: 64")
    single_time = simulate_training_speed(400, 64)
    
    # Multisubject scenario - BEFORE optimization
    print("\n🔹 MULTISUBJECT SCENARIO - BEFORE OPTIMIZATION")
    print("   Samples: 735, Batch size: 32")
    multi_before_time = simulate_training_speed(735, 32)
    
    # Multisubject scenario - AFTER optimization
    print("\n🔹 MULTISUBJECT SCENARIO - AFTER OPTIMIZATION")
    print("   Samples: 735, Batch size: 64")
    multi_after_time = simulate_training_speed(735, 64)
    
    # Performance comparison
    print("\n📈 PERFORMANCE COMPARISON:")
    print("=" * 60)
    print(f"Single subject (400 samples, batch=64):     {single_time:.2f}s")
    print(f"Multi-subject BEFORE (735 samples, batch=32): {multi_before_time:.2f}s")
    print(f"Multi-subject AFTER (735 samples, batch=64):  {multi_after_time:.2f}s")
    print(f"\nSpeedup from optimization: {multi_before_time/multi_after_time:.2f}x")
    print(f"Relative to single subject: {multi_after_time/single_time:.2f}x slower")
    
    print("\n🎯 EXPECTED RESULTS:")
    print("- Larger batch sizes = fewer batches = faster training")
    print("- Memory optimizations prevent OOM kills")
    print("- Multisubject will still be ~2x slower due to more data")

if __name__ == "__main__":
    main() 
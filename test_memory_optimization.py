#!/usr/bin/env python3
"""
Test script for memory optimization functionality
"""

import sys
from pathlib import Path
import gc
import psutil
import os

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from utils.data_utils_memory_optimized import (
    estimate_memory_usage,
    load_dataset_memory_optimized,
    create_memory_efficient_multisubject_dataset
)
from src.utils.paths import setup_paths


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_memory_estimation():
    """Test memory estimation functionality"""
    print("🧮 Testing memory estimation...")
    
    paths = setup_paths()
    subjects = ['subject001', 'subject002', 'subject003', 'subject014', 'subject015']
    
    memory_info = estimate_memory_usage(
        str(paths.data_root), 
        subjects, 
        mask_type='mask10'
    )
    
    print(f"✅ Memory estimation complete:")
    print(f"   Estimated memory: {memory_info['estimated_memory_gb']:.1f} GB")
    print(f"   Average samples: {memory_info['avg_samples']:.0f}")
    print(f"   Total estimated samples: {memory_info['estimated_total']:.0f}")


def test_single_subject_loading():
    """Test loading a single subject with memory optimization"""
    print("\n📊 Testing single subject loading...")
    
    paths = setup_paths()
    file_path = str(paths.data_root / "subject001_baseline_masked.h5")
    
    # Test without limit
    print("Loading without sample limit...")
    memory_before = get_memory_usage()
    
    try:
        dataset1 = load_dataset_memory_optimized(file_path, mask_type="auto")
        memory_after = get_memory_usage()
        print(f"✅ Loaded {len(dataset1)} samples")
        print(f"   Memory usage: {memory_after - memory_before:.1f} MB")
        
        # Cleanup
        del dataset1
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with limit
    print("\nLoading with sample limit (200 samples)...")
    memory_before = get_memory_usage()
    
    try:
        dataset2 = load_dataset_memory_optimized(file_path, mask_type="auto", max_samples=200)
        memory_after = get_memory_usage()
        print(f"✅ Loaded {len(dataset2)} samples")
        print(f"   Memory usage: {memory_after - memory_before:.1f} MB")
        
        # Cleanup
        del dataset2
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_multisubject_loading():
    """Test multi-subject loading with memory optimization"""
    print("\n👥 Testing multi-subject loading...")
    
    paths = setup_paths()
    subjects = ['subject001', 'subject002', 'subject014']
    
    print(f"Loading 3 subjects with max 150 samples each...")
    memory_before = get_memory_usage()
    
    try:
        combined_dataset, successful, failed = create_memory_efficient_multisubject_dataset(
            str(paths.data_root),
            subjects,
            mask_type='mask10',
            max_samples_per_subject=150,
            session="baseline"
        )
        
        memory_after = get_memory_usage()
        print(f"✅ Multi-subject dataset created:")
        print(f"   Total samples: {len(combined_dataset)}")
        print(f"   Successful subjects: {len(successful)}")
        print(f"   Failed subjects: {len(failed)}")
        print(f"   Memory usage: {memory_after - memory_before:.1f} MB")
        
        # Cleanup
        del combined_dataset
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all tests"""
    print("🚀 Testing Memory Optimization Features")
    print("=" * 50)
    
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    try:
        test_memory_estimation()
        test_single_subject_loading()
        test_multisubject_loading()
        
        print(f"\n✅ All tests completed!")
        print(f"Final memory usage: {get_memory_usage():.1f} MB")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 
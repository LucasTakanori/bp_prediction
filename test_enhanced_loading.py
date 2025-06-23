#!/usr/bin/env python3
"""
Test Enhanced Data Loading with Different Mask Formats
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.data_utils_enhanced import (
    EnhancedPviDataset, 
    load_dataset_with_best_mask, 
    analyze_all_mask_formats
)

def test_mask_analysis():
    """Test mask format analysis across all subjects"""
    print("🔍 ANALYZING ALL MASK FORMATS")
    print("=" * 60)
    
    data_root = os.getenv('BP_DATA_ROOT', '/home/lucas_takanori/phd/data')
    analysis = analyze_all_mask_formats(data_root)
    
    # Categorize subjects
    metadata_mask_subjects = []
    masks_group_subjects = []
    no_mask_subjects = []
    
    for subject, info in analysis.items():
        if 'error' in info:
            print(f"❌ {subject}: {info['error']}")
            continue
            
        if info['has_metadata_mask']:
            metadata_mask_subjects.append(subject)
        elif info['has_masks_group']:
            masks_group_subjects.append(subject)
        else:
            no_mask_subjects.append(subject)
        
        print(f"📊 {subject}:")
        print(f"   Data periods: {info['data_periods']}")
        print(f"   Metadata mask: {'✅' if info['has_metadata_mask'] else '❌'}")
        if info['has_metadata_mask']:
            print(f"   Metadata mask count: {info.get('metadata_mask_count', 'unknown')}")
        print(f"   Masks group: {'✅' if info['has_masks_group'] else '❌'}")
        if info['has_masks_group']:
            print(f"   Available masks: {info['masks_group_keys']}")
    
    print(f"\n📈 SUMMARY:")
    print(f"   Subjects with metadata masks: {len(metadata_mask_subjects)}")
    print(f"   Subjects with masks group: {len(masks_group_subjects)}")
    print(f"   Subjects with no masks: {len(no_mask_subjects)}")
    
    return analysis

def test_subject_loading():
    """Test loading different subjects with enhanced loader"""
    print("\n🧪 TESTING ENHANCED LOADING")
    print("=" * 60)
    
    data_root = os.getenv('BP_DATA_ROOT', '/home/lucas_takanori/phd/data')
    
    # Test subjects with different mask formats
    test_cases = [
        ('subject001', 'auto'),      # Should use metadata mask
        ('subject014', 'auto'),      # Should use mask10 from masks group
        ('subject014', 'mask05'),    # Force specific mask
        ('subject014', 'mask10'),    # Force different mask
    ]
    
    for subject, mask_type in test_cases:
        print(f"\n🔬 Testing {subject} with mask_type='{mask_type}'")
        print("-" * 50)
        
        try:
            file_path = Path(data_root) / f"{subject}_baseline_masked.h5"
            if not file_path.exists():
                print(f"⚠️  File not found: {file_path}")
                continue
                
            dataset = load_dataset_with_best_mask(str(file_path), mask_type)
            
            print(f"📊 Dataset loaded successfully:")
            print(f"   Samples: {len(dataset)}")
            
            # Get mask info
            mask_info = dataset.get_mask_info()
            print(f"   Available masks: {mask_info.get('available_masks', [])}")
            print(f"   Used mask: {mask_info.get('recommended_mask', 'unknown')}")
            
            # Sample structure
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"   Sample 0 keys: {list(sample.keys())}")
                for key, value in sample.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if hasattr(subvalue, 'shape'):
                                print(f"     {key}.{subkey}: {subvalue.shape}")
            
        except Exception as e:
            print(f"❌ Error loading {subject}: {e}")
            import traceback
            traceback.print_exc()

def compare_mask_results():
    """Compare results from different masks on the same subject"""
    print("\n⚖️  COMPARING MASK RESULTS")
    print("=" * 60)
    
    data_root = os.getenv('BP_DATA_ROOT', '/home/lucas_takanori/phd/data')
    subject = 'subject014'  # Subject with multiple mask options
    file_path = Path(data_root) / f"{subject}_baseline_masked.h5"
    
    if not file_path.exists():
        print(f"⚠️  Test subject {subject} not found")
        return
    
    mask_types = ['mask01', 'mask05', 'mask10', 'mask15']
    results = {}
    
    for mask_type in mask_types:
        print(f"\n🎯 Loading with {mask_type}...")
        try:
            dataset = EnhancedPviDataset(str(file_path), mask_type=mask_type)
            results[mask_type] = {
                'samples': len(dataset),
                'first_sample_shape': {}
            }
            
            if len(dataset) > 0:
                sample = dataset[0]
                for key, value in sample.items():
                    if isinstance(value, dict):
                        results[mask_type]['first_sample_shape'][key] = {}
                        for subkey, subvalue in value.items():
                            if hasattr(subvalue, 'shape'):
                                results[mask_type]['first_sample_shape'][key][subkey] = subvalue.shape
                            
            print(f"   ✅ {len(dataset)} samples loaded")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[mask_type] = {'error': str(e)}
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS:")
    print("-" * 50)
    for mask_type, result in results.items():
        if 'error' in result:
            print(f"{mask_type}: ERROR - {result['error']}")
        else:
            print(f"{mask_type}: {result['samples']} samples")
            # Show signal shapes
            if 'pviHP' in result['first_sample_shape']:
                signal_shape = result['first_sample_shape']['pviHP'].get('signal', 'N/A')
                print(f"         pviHP.signal shape: {signal_shape}")

def main():
    """Run all tests"""
    print("🚀 ENHANCED DATA LOADING TESTS")
    print("=" * 80)
    
    # 1. Analyze all mask formats
    analysis = test_mask_analysis()
    
    # 2. Test loading different subjects
    test_subject_loading()
    
    # 3. Compare different masks
    compare_mask_results()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main() 
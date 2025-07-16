#!/usr/bin/env python3
"""
Quick test to see the actual difference between PviDataset and EnhancedPviDataset
"""

import sys
from pathlib import Path
import h5py

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.data_utils import PviDataset, DataPathManager
from utils.data_utils_enhanced import load_dataset_with_best_mask

def quick_analysis(data_root: str, subject: str):
    """Quick analysis without loading full datasets"""
    
    print(f"\n🔍 Quick analysis for {subject}")
    print("=" * 40)
    
    # Get file path
    data_manager = DataPathManager(subject=subject, session="baseline", root=data_root)
    file_path = data_manager._h5_path
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Check file directly
    with h5py.File(file_path, 'r') as h5f:
        print(f"📁 File: {file_path.name}")
        print(f"📊 File size: {file_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Check data structure
        if 'data' in h5f:
            data_keys = list(h5f['data'].keys())
            print(f"🔑 Data keys: {data_keys}")
            
            # Check periods count
            if data_keys:
                first_key = data_keys[0]
                if 'signal' in h5f['data'][first_key]:
                    periods = h5f['data'][first_key]['signal'].shape[0]
                    print(f"📈 Total periods in data: {periods}")
        
        # Check metadata masks
        if 'metadata' in h5f:
            meta = h5f['metadata']
            print(f"🎭 Metadata keys: {list(meta.keys())}")
            
            if 'mask' in meta:
                mask_data = meta['mask'][()]
                print(f"🎯 Metadata mask shape: {mask_data.shape}")
                print(f"🎯 Metadata mask samples: {mask_data.shape[1] if len(mask_data.shape) > 1 else len(mask_data)}")
            else:
                print("❌ No metadata mask found")
        
        # Check masks group
        if 'masks' in h5f:
            masks_group = h5f['masks']
            available_masks = list(masks_group.keys())
            print(f"🎭 Available mask groups: {available_masks}")
            
            for mask_name in available_masks:
                mask_data = masks_group[mask_name][()]
                mask_samples = mask_data.shape[1] if len(mask_data.shape) > 1 else len(mask_data)
                print(f"   {mask_name}: {mask_samples} samples")
        else:
            print("❌ No masks group found")

def test_loading_difference():
    """Test the actual loading difference"""
    data_root = "/gpfs/projects/bsc88/speech/research/scripts/Lucas/bp_prediction/data"
    
    subjects = ["subject001", "subject002"]
    
    for subject in subjects:
        quick_analysis(data_root, subject)
    
    print(f"\n" + "=" * 60)
    print("💡 UNDERSTANDING THE DIFFERENCE:")
    print("=" * 60)
    print("🔸 PviDataset (whole data): Uses metadata mask if available, otherwise generates default")
    print("🔸 EnhancedPviDataset (masked): Can choose specific masks from masks group") 
    print("🔸 The REAL difference is when subjects have masks group with different mask options")
    print("")
    print("📝 From your training logs:")
    print("   subject001: metadata mask → 400 samples (same for both approaches)")
    print("   subject002: masks group (mask01, mask05, mask10, mask15) → different sample counts")
    print("")
    print("🎯 The 'whole data' approach means:")
    print("   - Use ALL periods if no mask exists") 
    print("   - Don't filter with specific masks from masks group")
    print("   - Let PviDataset generate default mask from all available periods")

if __name__ == "__main__":
    test_loading_difference() 
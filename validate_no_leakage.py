#!/usr/bin/env python3
"""
Validate No Data Leakage Script
Checks experiment metadata to ensure subject-level splits have no overlap
"""

import json
import sys
from pathlib import Path


def validate_experiment_metadata(metadata_file: str):
    """Validate that experiment has no data leakage"""
    
    print(f"🔍 Checking: {metadata_file}")
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check if this is a leakage-free experiment
        if 'leakage_prevention' in metadata:
            leakage_info = metadata['leakage_prevention']
            print(f"✅ Leakage prevention: {leakage_info.get('method', 'unknown')}")
            print(f"✅ Seed: {leakage_info.get('seed', 'unknown')}")
            print(f"✅ No overlap: {leakage_info.get('no_subject_overlap', False)}")
            
            if leakage_info.get('no_subject_overlap', False):
                print("🔒 NO DATA LEAKAGE CONFIRMED!")
                return True
            else:
                print("❌ DATA LEAKAGE DETECTED!")
                return False
                
        elif 'split_configuration' in metadata:
            # VAE experiment format
            split_config = metadata['split_configuration']
            train_subjects = set(split_config.get('train_subjects', []))
            val_subjects = set(split_config.get('val_subjects', []))
            test_subjects = set(split_config.get('test_subjects', []))
            
            # Check all pairwise overlaps
            all_sets = [('train', train_subjects), ('val', val_subjects), ('test', test_subjects)]
            
            leakage_found = False
            for i, (name1, set1) in enumerate(all_sets):
                for j, (name2, set2) in enumerate(all_sets[i+1:], i+1):
                    overlap = set1 & set2
                    if len(overlap) == 0:
                        print(f"✅ No {name1}/{name2} subject overlap")
                    else:
                        print(f"❌ LEAKAGE DETECTED: {name1}/{name2} overlap: {overlap}")
                        leakage_found = True
                        
            if not leakage_found:
                print("🔒 NO DATA LEAKAGE CONFIRMED!")
                print(f"✅ Train subjects: {len(train_subjects)}")
                print(f"✅ Val subjects: {len(val_subjects)}")
                print(f"✅ Test subjects: {len(test_subjects)}")
                print(f"✅ Seed: {split_config.get('seed', 'unknown')}")
                print(f"🧪 Test subjects: {list(test_subjects)}")
                return True
            else:
                print(f"❌ DATA LEAKAGE DETECTED!")
                return False
        else:
            print("⚠️  Old experiment format - cannot validate leakage prevention")
            return False
            
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return False


def main():
    """Main validation function"""
    
    if len(sys.argv) < 2:
        print("Usage: python validate_no_leakage.py <experiment_metadata.json>")
        print("   or: python validate_no_leakage.py <experiment_directory>")
        sys.exit(1)
    
    target = Path(sys.argv[1])
    
    if target.is_file() and target.name.endswith('.json'):
        # Direct metadata file
        metadata_files = [target]
    elif target.is_dir():
        # Experiment directory - find metadata files
        metadata_files = list(target.glob("**/experiment_metadata.json"))
        if not metadata_files:
            print(f"❌ No experiment_metadata.json found in {target}")
            sys.exit(1)
    else:
        print(f"❌ Invalid target: {target}")
        sys.exit(1)
    
    print("🔒 VALIDATING DATA LEAKAGE PREVENTION")
    print("=" * 50)
    
    all_valid = True
    for metadata_file in metadata_files:
        valid = validate_experiment_metadata(metadata_file)
        all_valid = all_valid and valid
        print("-" * 30)
    
    if all_valid:
        print("🎉 ALL EXPERIMENTS VALIDATED - NO DATA LEAKAGE!")
        sys.exit(0)
    else:
        print("💥 VALIDATION FAILED - DATA LEAKAGE DETECTED!")
        sys.exit(1)


if __name__ == "__main__":
    main() 
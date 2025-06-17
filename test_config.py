#!/usr/bin/env python3
"""
Configuration Testing Utility
============================

This script tests configuration loading and validation for debugging purposes.
It's useful for:
- Testing new config files
- Debugging config issues
- Verifying environment variable substitution
- Checking config validation logic

Usage:
    python3 test_config.py [config_file]
    
If no config file is provided, it tests the default configs.
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import create_config_from_yaml
from src.utils.config_validator import validate_config


def test_config_file(config_path: str):
    """Test a specific config file"""
    print(f"\n📋 Testing config file: {config_path}")
    print("=" * 50)
    
    try:
        print("🔍 Loading configuration...")
        data_config, model_config, training_config = create_config_from_yaml(config_path)
        
        print("✅ Config loading successful!")
        print(f"   📁 Data root: {data_config.root_path}")
        print(f"   🤖 Model type: {model_config.model_type}")
        print(f"   📊 Batch size: {training_config.batch_size}")
        print(f"   📈 Learning rate: {training_config.learning_rate} (type: {type(training_config.learning_rate).__name__})")
        
        if hasattr(model_config, 'vae_checkpoint_path') and model_config.vae_checkpoint_path:
            print(f"   🔗 VAE checkpoint: {model_config.vae_checkpoint_path}")
        
        print("\n🔍 Running validation...")
        is_valid = validate_config(
            data_config, 
            model_config, 
            training_config, 
            raise_on_error=False
        )
        
        if is_valid:
            print("✅ Validation successful!")
        else:
            print("⚠️  Validation had warnings (but no errors)")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test configuration files")
    parser.add_argument('config_file', nargs='?', help='Config file to test (optional)')
    parser.add_argument('--all', action='store_true', help='Test all config files')
    
    args = parser.parse_args()
    
    if args.config_file:
        # Test specific file
        success = test_config_file(args.config_file)
        sys.exit(0 if success else 1)
    
    elif args.all:
        # Test all config files
        config_dir = Path('configs')
        if not config_dir.exists():
            print("❌ configs/ directory not found")
            sys.exit(1)
        
        config_files = list(config_dir.glob('*.yaml')) + list(config_dir.glob('*.yml'))
        if not config_files:
            print("❌ No config files found in configs/")
            sys.exit(1)
        
        print(f"🧪 Testing {len(config_files)} config files...")
        
        success_count = 0
        for config_file in config_files:
            if test_config_file(str(config_file)):
                success_count += 1
        
        print(f"\n📊 Results: {success_count}/{len(config_files)} configs passed")
        sys.exit(0 if success_count == len(config_files) else 1)
    
    else:
        # Default: test the baseline configs
        print("🧪 Testing default configuration files...")
        
        configs_to_test = [
            'configs/vae_baseline.yaml',
            'configs/bilstm_baseline.yaml'
        ]
        
        success_count = 0
        for config_path in configs_to_test:
            if Path(config_path).exists():
                if test_config_file(config_path):
                    success_count += 1
            else:
                print(f"\n❌ Config file not found: {config_path}")
        
        print(f"\n📊 Results: {success_count}/{len(configs_to_test)} configs passed")
        return success_count == len(configs_to_test)


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1) 
#!/usr/bin/env python3
"""
Setup validation script for BP prediction project.
Tests path management, configuration loading, and data validation.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.paths import setup_paths, PathConfig
from src.utils.config import create_config_from_yaml
from src.utils.config_validator import validate_config, ValidationError
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def test_path_management():
    """Test path management system"""
    print("🧪 Testing Path Management...")
    
    try:
        # Test with environment variables
        path_manager = setup_paths()
        
        print(f"✅ Data root: {path_manager.data_root}")
        print(f"✅ Experiments root: {path_manager.experiments_root}")
        print(f"✅ Cache dir: {path_manager.cache_dir}")
        
        # Test data file validation
        subjects = path_manager.list_available_subjects()
        print(f"✅ Available subjects: {subjects}")
        
        if subjects:
            sessions = path_manager.list_available_sessions(subjects[0])
            print(f"✅ Available sessions for {subjects[0]}: {sessions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Path management test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading and validation"""
    print("\n🧪 Testing Configuration Loading...")
    
    config_files = [
        "configs/vae_baseline.yaml",
        "configs/bilstm_baseline.yaml"
    ]
    
    success = True
    
    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"⚠️  Config file not found: {config_file}")
            continue
            
        try:
            print(f"   Testing {config_file}...")
            data_config, model_config, training_config = create_config_from_yaml(config_file)
            
            # Test validation
            is_valid = validate_config(
                data_config, 
                model_config, 
                training_config, 
                raise_on_error=False
            )
            
            if is_valid:
                print(f"   ✅ {config_file} - Valid")
            else:
                print(f"   ⚠️  {config_file} - Has validation issues")
                success = False
                
        except Exception as e:
            print(f"   ❌ {config_file} - Failed: {e}")
            success = False
    
    return success


def test_data_availability():
    """Test data file availability"""
    print("\n🧪 Testing Data Availability...")
    
    try:
        path_manager = setup_paths()
        
        # Check if data directory exists
        if not path_manager.data_root.exists():
            print(f"❌ Data root not found: {path_manager.data_root}")
            return False
        
        # List available data files
        data_files = list(path_manager.data_root.glob("*.h5"))
        
        if not data_files:
            print("❌ No .h5 data files found")
            return False
        
        print(f"✅ Found {len(data_files)} data files:")
        for file in data_files[:5]:  # Show first 5
            print(f"   📄 {file.name}")
        
        if len(data_files) > 5:
            print(f"   ... and {len(data_files) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Data availability test failed: {e}")
        return False


def test_environment_variables():
    """Test environment variable setup"""
    print("\n🧪 Testing Environment Variables...")
    
    required_vars = [
        "BP_DATA_ROOT",
        "BP_EXPERIMENTS_ROOT", 
        "BP_CACHE_DIR",
        "WANDB_PROJECT"
    ]
    
    success = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚠️  {var}: Not set (will use defaults)")
    
    return success


def test_imports():
    """Test that all required modules can be imported"""
    print("\n🧪 Testing Module Imports...")
    
    modules_to_test = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("yaml", "PyYAML"),
        ("tqdm", "TQDM"),
        ("h5py", "HDF5"),
        ("wandb", "Weights & Biases")
    ]
    
    success = True
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - Not available")
            success = False
    
    return success


def main():
    """Run all validation tests"""
    setup_logging()
    
    print("🔬 BP Prediction Project - Setup Validation")
    print("=" * 50)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Module Imports", test_imports),
        ("Path Management", test_path_management),
        ("Data Availability", test_data_availability),
        ("Configuration Loading", test_config_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Validation Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validations passed! Your setup is ready.")
        return 0
    else:
        print("⚠️  Some validations failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    exit(main()) 
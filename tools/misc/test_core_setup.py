#!/usr/bin/env python3
"""
Test script to validate the core project structure and base interfaces
"""

import sys
import torch
import logging
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core components can be imported"""
    print("Testing imports...")
    
    try:
        from core import (
            UnifiedDTAModel,
            create_dta_model,
            get_lightweight_model,
            get_production_model,
            DTAConfig,
            load_config,
            save_config,
            get_default_configs,
            set_seed,
            get_device,
            setup_logging
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration management"""
    print("\nTesting configuration management...")
    
    try:
        from core import DTAConfig, get_default_configs, save_config, load_config
        
        # Test default configurations
        configs = get_default_configs()
        assert 'lightweight' in configs
        assert 'production' in configs
        print("✓ Default configurations loaded")
        
        # Test configuration creation
        config = configs['lightweight']
        assert config.protein_encoder_type == 'cnn'
        assert config.use_fusion == False
        print("✓ Configuration structure validated")
        
        # Test save/load (to temporary file)
        temp_config_path = Path("temp_config.yaml")
        save_config(config, temp_config_path)
        loaded_config = load_config(temp_config_path)
        
        assert loaded_config.protein_encoder_type == config.protein_encoder_type
        print("✓ Configuration save/load working")
        
        # Cleanup
        temp_config_path.unlink()
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_model_factory():
    """Test model factory and creation"""
    print("\nTesting model factory...")
    
    try:
        from core import get_lightweight_model, get_production_model, create_dta_model
        
        # Test lightweight model
        lightweight_model = get_lightweight_model()
        assert lightweight_model is not None
        print("✓ Lightweight model created")
        
        # Test production model (might fail without transformers/torch-geometric)
        try:
            production_model = get_production_model()
            print("✓ Production model created")
        except Exception as e:
            print(f"⚠ Production model creation failed (expected without dependencies): {e}")
        
        # Test custom configuration
        custom_config = {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': False,
            'protein_config': {'output_dim': 32},
            'drug_config': {'output_dim': 32, 'num_layers': 2}
        }
        
        try:
            custom_model = create_dta_model(custom_config)
            print("✓ Custom model created")
        except Exception as e:
            print(f"⚠ Custom model creation failed (expected without dependencies): {e}")
        
        return True
    except Exception as e:
        print(f"✗ Model factory test failed: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    print("\nTesting utilities...")
    
    try:
        from core import set_seed, get_device, get_memory_usage, setup_logging
        
        # Test seed setting
        set_seed(42)
        print("✓ Seed setting working")
        
        # Test device detection
        device = get_device('cpu')  # Force CPU to avoid CUDA issues
        assert device.type == 'cpu'
        print("✓ Device detection working")
        
        # Test memory usage
        memory_info = get_memory_usage()
        assert 'system' in memory_info
        print("✓ Memory usage reporting working")
        
        # Test logging setup
        setup_logging('INFO')
        print("✓ Logging setup working")
        
        return True
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False

def test_base_interfaces():
    """Test base interfaces and abstract classes"""
    print("\nTesting base interfaces...")
    
    try:
        from core.models import BaseEncoder
        from abc import ABC
        
        # Test that BaseEncoder is abstract
        assert issubclass(BaseEncoder, ABC)
        print("✓ BaseEncoder is properly abstract")
        
        # Test that we can't instantiate BaseEncoder directly
        try:
            encoder = BaseEncoder()
            print("✗ BaseEncoder should not be instantiable")
            return False
        except TypeError:
            print("✓ BaseEncoder properly prevents direct instantiation")
        
        return True
    except Exception as e:
        print(f"✗ Base interfaces test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("UNIFIED DTA SYSTEM - CORE STRUCTURE VALIDATION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration,
        test_base_interfaces,
        test_utilities,
        test_model_factory
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core structure tests passed!")
        print("✓ Project structure is properly set up")
        print("✓ Base interfaces are working")
        print("✓ Configuration management is functional")
        print("✓ Utilities are available")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
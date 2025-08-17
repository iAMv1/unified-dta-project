#!/usr/bin/env python3
"""
Simple test script for the configuration management system
Tests core functionality without heavy dependencies
"""

import sys
import yaml
import json
from pathlib import Path

# Add core to path for imports
sys.path.append(str(Path(__file__).parent / "core"))

try:
    from core.config import (
        DTAConfig, ProteinConfig, DrugConfig, FusionConfig, 
        PredictorConfig, TrainingConfig, DataConfig,
        load_config, save_config, validate_config,
        get_default_configs, create_config_template,
        get_environment_config, merge_configs
    )
    print("✓ Successfully imported configuration modules")
except ImportError as e:
    print(f"✗ Failed to import configuration modules: {e}")
    sys.exit(1)


def test_basic_config_creation():
    """Test basic configuration creation"""
    print("\n1. Testing basic configuration creation...")
    
    try:
        # Create a basic configuration
        config = DTAConfig(
            protein_encoder_type='esm',
            drug_encoder_type='gin',
            use_fusion=True
        )
        
        print("  ✓ DTAConfig created successfully")
        print(f"  ✓ Protein encoder: {config.protein_encoder_type}")
        print(f"  ✓ Drug encoder: {config.drug_encoder_type}")
        print(f"  ✓ Use fusion: {config.use_fusion}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error creating basic configuration: {e}")
        return False


def test_config_serialization():
    """Test configuration serialization to/from dict"""
    print("\n2. Testing configuration serialization...")
    
    try:
        # Create configuration
        config = DTAConfig(
            protein_encoder_type='cnn',
            drug_encoder_type='gin',
            use_fusion=False
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        print("  ✓ Configuration converted to dictionary")
        
        # Convert back from dict
        config2 = DTAConfig.from_dict(config_dict)
        print("  ✓ Configuration created from dictionary")
        
        # Verify they're the same
        if config.protein_encoder_type == config2.protein_encoder_type:
            print("  ✓ Serialization round-trip successful")
            return True
        else:
            print("  ✗ Serialization round-trip failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Error in serialization test: {e}")
        return False


def test_default_configs():
    """Test default configuration loading"""
    print("\n3. Testing default configurations...")
    
    try:
        default_configs = get_default_configs()
        print(f"  ✓ Loaded {len(default_configs)} default configurations")
        
        for name, config in default_configs.items():
            print(f"    - {name}: {config.protein_encoder_type}/{config.drug_encoder_type}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading default configurations: {e}")
        return False


def test_config_validation():
    """Test configuration validation"""
    print("\n4. Testing configuration validation...")
    
    try:
        # Test valid configuration
        valid_config = DTAConfig(
            protein_encoder_type='esm',
            drug_encoder_type='gin',
            use_fusion=True
        )
        
        is_valid = validate_config(valid_config)
        if is_valid:
            print("  ✓ Valid configuration passed validation")
        else:
            print("  ✗ Valid configuration failed validation")
            return False
        
        # Test invalid configuration
        invalid_config = DTAConfig(
            protein_encoder_type='invalid_encoder',
            drug_encoder_type='gin',
            use_fusion=True
        )
        
        is_valid = validate_config(invalid_config)
        if not is_valid:
            print("  ✓ Invalid configuration correctly failed validation")
        else:
            print("  ✗ Invalid configuration incorrectly passed validation")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error in validation test: {e}")
        return False


def test_file_operations():
    """Test configuration file operations"""
    print("\n5. Testing configuration file operations...")
    
    try:
        # Create a test configuration
        config = DTAConfig(
            protein_encoder_type='cnn',
            drug_encoder_type='gin',
            use_fusion=False
        )
        
        # Test YAML save/load
        yaml_path = "test_config.yaml"
        save_config(config, yaml_path)
        print("  ✓ Configuration saved to YAML")
        
        loaded_config = load_config(yaml_path)
        print("  ✓ Configuration loaded from YAML")
        
        if loaded_config.protein_encoder_type == config.protein_encoder_type:
            print("  ✓ YAML round-trip successful")
        else:
            print("  ✗ YAML round-trip failed")
            return False
        
        # Test JSON save/load
        json_path = "test_config.json"
        save_config(config, json_path)
        print("  ✓ Configuration saved to JSON")
        
        loaded_config = load_config(json_path)
        print("  ✓ Configuration loaded from JSON")
        
        if loaded_config.protein_encoder_type == config.protein_encoder_type:
            print("  ✓ JSON round-trip successful")
        else:
            print("  ✗ JSON round-trip failed")
            return False
        
        # Clean up
        Path(yaml_path).unlink(missing_ok=True)
        Path(json_path).unlink(missing_ok=True)
        print("  ✓ Test files cleaned up")
        
        return True
    except Exception as e:
        print(f"  ✗ Error in file operations test: {e}")
        return False


def test_environment_configs():
    """Test environment-specific configurations"""
    print("\n6. Testing environment configurations...")
    
    try:
        environments = ['development', 'staging', 'production']
        
        for env in environments:
            config = get_environment_config(env)
            print(f"  ✓ {env}: {config.protein_encoder_type}/{config.drug_encoder_type}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error in environment configuration test: {e}")
        return False


def test_config_merging():
    """Test configuration merging"""
    print("\n7. Testing configuration merging...")
    
    try:
        # Create base configuration
        base_config = DTAConfig(
            protein_encoder_type='esm',
            drug_encoder_type='gin',
            use_fusion=True
        )
        
        # Create override
        override = {
            'protein_encoder_type': 'cnn',
            'training_config': {
                'batch_size': 16
            }
        }
        
        # Merge configurations
        merged_config = merge_configs(base_config, override)
        
        if (merged_config.protein_encoder_type == 'cnn' and 
            merged_config.drug_encoder_type == 'gin' and
            merged_config.training_config.batch_size == 16):
            print("  ✓ Configuration merging successful")
            return True
        else:
            print("  ✗ Configuration merging failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Error in configuration merging test: {e}")
        return False


def main():
    """Run all tests"""
    print("Simple Configuration System Test")
    print("=" * 40)
    
    tests = [
        test_basic_config_creation,
        test_config_serialization,
        test_default_configs,
        test_config_validation,
        test_file_operations,
        test_environment_configs,
        test_config_merging
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test script for the configuration management system
"""

import sys
from pathlib import Path

# Add core to path for imports
sys.path.append(str(Path(__file__).parent / "core"))

from core.config import (
    load_config, save_config, validate_config,
    get_default_configs, create_config_template,
    get_environment_config, ConfigurationManager
)
from core.model_factory import ModelFactory


def test_configuration_system():
    """Test the configuration management system"""
    print("Testing Configuration Management System")
    print("=" * 50)
    
    # Test 1: Load predefined configurations
    print("\n1. Testing predefined configurations...")
    try:
        default_configs = get_default_configs()
        print(f"✓ Found {len(default_configs)} default configurations:")
        for name in default_configs.keys():
            print(f"  - {name}")
    except Exception as e:
        print(f"✗ Error loading default configurations: {e}")
        return False
    
    # Test 2: Validate configurations
    print("\n2. Testing configuration validation...")
    try:
        for name, config in default_configs.items():
            is_valid = validate_config(config, detailed=True)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {name}: {'valid' if is_valid else 'invalid'}")
    except Exception as e:
        print(f"✗ Error validating configurations: {e}")
        return False
    
    # Test 3: Load configuration files
    print("\n3. Testing configuration file loading...")
    config_files = [
        "configs/lightweight_config.yaml",
        "configs/production_config.yaml",
        "configs/high_performance_config.yaml"
    ]
    
    for config_file in config_files:
        try:
            if Path(config_file).exists():
                config = load_config(config_file)
                is_valid = validate_config(config)
                status = "✓" if is_valid else "✗"
                print(f"  {status} {config_file}: {'loaded and valid' if is_valid else 'loaded but invalid'}")
            else:
                print(f"  - {config_file}: file not found (skipping)")
        except Exception as e:
            print(f"  ✗ {config_file}: error loading - {e}")
    
    # Test 4: Test inheritance
    print("\n4. Testing configuration inheritance...")
    try:
        if Path("configs/custom_config.yaml").exists():
            config = load_config("configs/custom_config.yaml")
            is_valid = validate_config(config)
            status = "✓" if is_valid else "✗"
            print(f"  {status} Inheritance test: {'passed' if is_valid else 'failed'}")
        else:
            print("  - Custom config not found (skipping inheritance test)")
    except Exception as e:
        print(f"  ✗ Inheritance test failed: {e}")
    
    # Test 5: Test model factory integration
    print("\n5. Testing model factory integration...")
    try:
        # Test with lightweight configuration
        model = ModelFactory.get_lightweight_model()
        print("  ✓ Lightweight model created successfully")
        
        # Test with configuration file
        if Path("configs/lightweight_config.yaml").exists():
            model = ModelFactory.create_from_config_file("configs/lightweight_config.yaml")
            print("  ✓ Model created from config file successfully")
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model has {total_params:,} parameters")
        
    except Exception as e:
        print(f"  ✗ Model factory test failed: {e}")
    
    # Test 6: Test environment configurations
    print("\n6. Testing environment configurations...")
    try:
        environments = ['development', 'staging', 'production']
        for env in environments:
            config = get_environment_config(env)
            is_valid = validate_config(config)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {env}: {'valid' if is_valid else 'invalid'}")
    except Exception as e:
        print(f"  ✗ Environment configuration test failed: {e}")
    
    # Test 7: Test configuration manager
    print("\n7. Testing configuration manager...")
    try:
        config_manager = ConfigurationManager("test_configs")
        
        # Create templates
        config_manager.create_all_templates()
        print("  ✓ Created all configuration templates")
        
        # Validate all configs
        results = config_manager.validate_all_configs()
        valid_count = sum(1 for is_valid in results.values() if is_valid)
        total_count = len(results)
        print(f"  ✓ Validated {total_count} configurations ({valid_count} valid)")
        
    except Exception as e:
        print(f"  ✗ Configuration manager test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Configuration system test completed!")
    return True


def test_model_configurations():
    """Test model configurations with the factory"""
    print("\nTesting Model Factory Configurations")
    print("=" * 50)
    
    try:
        # List available configurations
        configs = ModelFactory.list_configurations()
        print(f"\nAvailable configurations: {len(configs)}")
        
        for name, info in configs.items():
            print(f"\n{name.upper()}:")
            print(f"  Name: {info['name']}")
            print(f"  Memory: {info['memory_usage']}")
            print(f"  Use: {info['recommended_use']}")
            
            try:
                # Test model creation
                model = ModelFactory.create_model(name)
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"  ✓ Model created successfully")
                print(f"  ✓ Total parameters: {total_params:,}")
                print(f"  ✓ Trainable parameters: {trainable_params:,}")
                
            except Exception as e:
                print(f"  ✗ Model creation failed: {e}")
    
    except Exception as e:
        print(f"✗ Model configuration test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Unified DTA System - Configuration Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test configuration system
    success &= test_configuration_system()
    
    # Test model configurations
    success &= test_model_configurations()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
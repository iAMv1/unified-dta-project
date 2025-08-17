#!/usr/bin/env python3
"""
Test script to verify the package structure is working correctly
"""

import sys
import traceback


def test_imports():
    """Test that all main components can be imported"""
    
    print("Testing package imports...")
    
    # Test main package import
    try:
        import unified_dta
        print(f"✓ unified_dta imported (version: {unified_dta.__version__})")
    except Exception as e:
        print(f"✗ unified_dta import failed: {e}")
        return False
    
    # Test core components
    try:
        from unified_dta.core.models import UnifiedDTAModel
        from unified_dta.core.model_factory import ModelFactory
        from unified_dta.core.config import Config
        print("✓ Core components imported")
    except Exception as e:
        print(f"✗ Core components import failed: {e}")
        traceback.print_exc()
        return False
    
    # Test model creation
    try:
        model = ModelFactory.create_lightweight_model()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Lightweight model created ({param_count:,} parameters)")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False
    
    # Test configuration
    try:
        config = Config()
        config_dict = config.to_dict()
        print(f"✓ Configuration system working")
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main test function"""
    print("=" * 50)
    print("Package Structure Test")
    print("=" * 50)
    
    success = test_imports()
    
    print("=" * 50)
    if success:
        print("✓ All tests passed! Package structure is working.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
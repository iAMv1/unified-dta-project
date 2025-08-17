#!/usr/bin/env python3
"""
Installation verification script for the Unified DTA System
"""

import sys
import importlib
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is not supported")
        print("  Unified DTA requires Python 3.8 or higher")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_package_import():
    """Check if the package can be imported"""
    try:
        import unified_dta
        print(f"✓ Package imported successfully")
        print(f"  Version: {unified_dta.__version__}")
        print(f"  Location: {Path(unified_dta.__file__).parent}")
        return True
    except ImportError as e:
        print(f"✗ Package import failed: {e}")
        return False


def check_core_components():
    """Check if core components can be imported"""
    components = [
        ('unified_dta.core.models', 'UnifiedDTAModel'),
        ('unified_dta.core.model_factory', 'ModelFactory'),
        ('unified_dta.core.config', 'Config'),
        ('unified_dta.core.training', 'DTATrainer'),
        ('unified_dta.core.evaluation', 'DTAEvaluator'),
        ('unified_dta.encoders.protein_encoders', 'ESMProteinEncoder'),
        ('unified_dta.encoders.drug_encoders', 'GINDrugEncoder'),
        ('unified_dta.data.datasets', 'DTADataset'),
    ]
    
    success = True
    for module_name, class_name in components:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except (ImportError, AttributeError) as e:
            print(f"✗ {module_name}.{class_name}: {e}")
            success = False
    
    return success


def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = [
        'torch',
        'torch_geometric', 
        'transformers',
        'rdkit',
        'numpy',
        'pandas',
        'scipy',
        'yaml'
    ]
    
    success = True
    for dep in dependencies:
        try:
            if dep == 'yaml':
                import yaml
            elif dep == 'torch_geometric':
                import torch_geometric
            elif dep == 'rdkit':
                from rdkit import Chem
            else:
                importlib.import_module(dep)
            print(f"✓ {dep}")
        except ImportError as e:
            print(f"✗ {dep}: {e}")
            success = False
    
    return success


def check_cli_commands():
    """Check if CLI commands are available"""
    commands = [
        'unified-dta',
        'dta-train',
        'dta-predict', 
        'dta-evaluate'
    ]
    
    success = True
    for cmd in commands:
        try:
            result = subprocess.run(
                f"{cmd} --help",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✓ {cmd}")
            else:
                print(f"✗ {cmd}: Command failed")
                success = False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"✗ {cmd}: {e}")
            success = False
    
    return success


def test_basic_functionality():
    """Test basic model creation and functionality"""
    try:
        from unified_dta.core.model_factory import ModelFactory
        
        # Test lightweight model creation
        model = ModelFactory.create_lightweight_model()
        print(f"✓ Lightweight model created")
        
        # Test model parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """Main verification function"""
    print("=" * 60)
    print("Unified DTA System - Installation Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Package Import", check_package_import),
        ("Core Components", check_core_components),
        ("Dependencies", check_dependencies),
        ("CLI Commands", check_cli_commands),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * len(check_name))
        result = check_func()
        results.append((check_name, result))
    
    print("\n" + "=" * 60)
    print("Verification Summary:")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All checks passed! Installation is working correctly.")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
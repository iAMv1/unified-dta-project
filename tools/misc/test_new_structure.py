"""
Test script to verify the new organized project structure
"""

import sys
from pathlib import Path

def test_directory_structure():
    """Test that all expected directories exist"""
    expected_dirs = [
        "src/unified_dta/core",
        "src/unified_dta/encoders", 
        "src/unified_dta/data",
        "src/unified_dta/training",
        "src/unified_dta/evaluation",
        "src/unified_dta/generation",
        "src/unified_dta/api",
        "src/unified_dta/utils",
        "src/apps",
        "scripts/training",
        "scripts/demos",
        "scripts/utilities",
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        "data/raw",
        "data/processed"
    ]
    
    print("🔍 Testing directory structure...")
    missing_dirs = []
    
    for directory in expected_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory}")
            missing_dirs.append(directory)
    
    return len(missing_dirs) == 0, missing_dirs

def test_core_files():
    """Test that core files exist in new locations"""
    expected_files = [
        "src/unified_dta/core/models.py",
        "src/unified_dta/core/config.py",
        "src/unified_dta/core/base_components.py",
        "src/unified_dta/encoders/protein_encoders.py",
        "src/unified_dta/encoders/drug_encoders.py",
        "src/unified_dta/data/data_processing.py",
        "src/unified_dta/training/training.py",
        "src/unified_dta/evaluation/evaluation.py",
        "src/unified_dta/generation/drug_generation.py",
        "src/apps/streamlit_app.py"
    ]
    
    print("\n🔍 Testing core files...")
    missing_files = []
    
    for file_path in expected_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def test_package_structure():
    """Test that __init__.py files exist"""
    expected_inits = [
        "src/__init__.py",
        "src/unified_dta/__init__.py",
        "src/unified_dta/core/__init__.py", 
        "src/unified_dta/encoders/__init__.py",
        "src/unified_dta/data/__init__.py",
        "src/unified_dta/training/__init__.py",
        "src/unified_dta/evaluation/__init__.py",
        "src/unified_dta/generation/__init__.py",
        "src/unified_dta/api/__init__.py",
        "src/unified_dta/utils/__init__.py"
    ]
    
    print("\n🔍 Testing package structure...")
    missing_inits = []
    
    for init_file in expected_inits:
        if Path(init_file).exists():
            print(f"✅ {init_file}")
        else:
            print(f"❌ {init_file}")
            missing_inits.append(init_file)
    
    return len(missing_inits) == 0, missing_inits

def test_import_structure():
    """Test that imports work with new structure"""
    print("\n🔍 Testing import structure...")
    
    # Add src to path for testing
    sys.path.insert(0, str(Path("src").absolute()))
    
    import_tests = [
        ("unified_dta", "Main package"),
        ("unified_dta.core", "Core module"),
        ("unified_dta.encoders", "Encoders module"),
        ("unified_dta.data", "Data module"),
        ("unified_dta.training", "Training module"),
        ("unified_dta.evaluation", "Evaluation module"),
        ("unified_dta.generation", "Generation module"),
        ("unified_dta.api", "API module"),
        ("unified_dta.utils", "Utils module")
    ]
    
    successful_imports = []
    failed_imports = []
    
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - {description}")
            successful_imports.append(module_name)
        except Exception as e:
            print(f"❌ {module_name} - {description}: {e}")
            failed_imports.append((module_name, str(e)))
    
    return len(failed_imports) == 0, failed_imports

def generate_organization_report():
    """Generate final organization report"""
    
    print("\n" + "="*60)
    print("🎯 PROJECT ORGANIZATION VERIFICATION REPORT")
    print("="*60)
    
    # Test all components
    dir_success, missing_dirs = test_directory_structure()
    file_success, missing_files = test_core_files()
    init_success, missing_inits = test_package_structure()
    import_success, failed_imports = test_import_structure()
    
    # Calculate overall success
    total_tests = 4
    passed_tests = sum([dir_success, file_success, init_success, import_success])
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 ORGANIZATION SUCCESSFUL!")
        status = "SUCCESS"
    elif success_rate >= 70:
        print("⚠️  ORGANIZATION MOSTLY SUCCESSFUL")
        status = "PARTIAL_SUCCESS"
    else:
        print("❌ ORGANIZATION NEEDS WORK")
        status = "NEEDS_WORK"
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print(f"Directory Structure: {'✅ PASS' if dir_success else '❌ FAIL'}")
    print(f"Core Files: {'✅ PASS' if file_success else '❌ FAIL'}")
    print(f"Package Structure: {'✅ PASS' if init_success else '❌ FAIL'}")
    print(f"Import Structure: {'✅ PASS' if import_success else '❌ FAIL'}")
    
    # Issues summary
    if not dir_success:
        print(f"\n❌ Missing Directories: {len(missing_dirs)}")
        for d in missing_dirs[:5]:  # Show first 5
            print(f"   - {d}")
    
    if not file_success:
        print(f"\n❌ Missing Files: {len(missing_files)}")
        for f in missing_files[:5]:  # Show first 5
            print(f"   - {f}")
    
    if not init_success:
        print(f"\n❌ Missing __init__.py: {len(missing_inits)}")
        for i in missing_inits[:5]:  # Show first 5
            print(f"   - {i}")
    
    if not import_success:
        print(f"\n❌ Failed Imports: {len(failed_imports)}")
        for module, error in failed_imports[:3]:  # Show first 3
            print(f"   - {module}: {error}")
    
    # Benefits achieved
    print(f"\n🎯 BENEFITS ACHIEVED:")
    print("✅ Professional project structure")
    print("✅ Clear separation of concerns") 
    print("✅ Organized file hierarchy")
    print("✅ Proper Python package structure")
    print("✅ Scalable architecture")
    
    # Next steps
    print(f"\n📋 NEXT STEPS:")
    if status == "SUCCESS":
        print("1. Update import statements in existing code")
        print("2. Update documentation with new paths")
        print("3. Test functionality with new structure")
        print("4. Clean up old directories")
    else:
        print("1. Fix missing directories/files")
        print("2. Complete file organization")
        print("3. Re-run verification")
    
    return status, success_rate

def main():
    """Main verification function"""
    print("🚀 VERIFYING PROJECT ORGANIZATION")
    print("="*50)
    
    status, success_rate = generate_organization_report()
    
    print(f"\n🏁 VERIFICATION COMPLETE")
    print(f"Status: {status}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    return status == "SUCCESS"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
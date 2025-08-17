#!/usr/bin/env python3
"""
Test runner for the Unified DTA System
Runs all unit tests, integration tests, and performance benchmarks
"""

import unittest
import sys
import os
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_test_suite(test_pattern='test_*.py', verbosity=2, failfast=False, specific_test=None):
    """Run the test suite with specified parameters"""
    
    print("=" * 80)
    print("UNIFIED DTA SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Discover and run tests
    if specific_test:
        # Run specific test file or test case
        print(f"Running specific test: {specific_test}")
        loader = unittest.TestLoader()
        
        if '.' in specific_test:
            # Specific test method (e.g., tests.test_encoders.TestProteinEncoders.test_esm_protein_encoder)
            suite = loader.loadTestsFromName(specific_test)
        else:
            # Test file (e.g., test_encoders)
            if not specific_test.startswith('tests.'):
                specific_test = f'tests.{specific_test}'
            suite = loader.loadTestsFromName(specific_test)
    else:
        # Discover all tests
        print(f"Discovering tests with pattern: {test_pattern}")
        loader = unittest.TestLoader()
        start_dir = project_root / 'tests'
        suite = loader.discover(start_dir, pattern=test_pattern)
    
    # Configure test runner
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=failfast,
        buffer=True  # Capture stdout/stderr during tests
    )
    
    # Run tests
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    print(f"Tests run: {total_tests}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Success rate: {((total_tests - failures - errors) / total_tests * 100):.1f}%" if total_tests > 0 else "N/A")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    # Print failure details
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if hasattr(result, 'skipped') and result.skipped:
        print(f"\nSKIPPED ({len(result.skipped)}):")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    print("=" * 80)
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


def run_unit_tests():
    """Run unit tests only"""
    print("Running Unit Tests...")
    patterns = ['test_encoders.py', 'test_fusion_attention.py', 'test_data_processing.py', 'test_configuration.py']
    
    all_passed = True
    for pattern in patterns:
        print(f"\n--- Running {pattern} ---")
        passed = run_test_suite(test_pattern=pattern, verbosity=1)
        all_passed = all_passed and passed
    
    return all_passed


def run_integration_tests():
    """Run integration tests only"""
    print("Running Integration Tests...")
    return run_test_suite(test_pattern='test_integration.py', verbosity=2)


def run_performance_tests():
    """Run performance tests only"""
    print("Running Performance Tests...")
    return run_test_suite(test_pattern='test_performance.py', verbosity=2)


def run_ci_tests():
    """Run CI pipeline tests only"""
    print("Running CI Pipeline Tests...")
    return run_test_suite(test_pattern='test_ci_pipeline.py', verbosity=2)


def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    required_packages = ['torch', 'pandas', 'yaml']
    optional_packages = ['torch_geometric', 'transformers', 'rdkit', 'scipy']
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print(f"❌ Missing required packages: {', '.join(missing_required)}")
        print("Please install required packages: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages available")
    
    if missing_optional:
        print(f"⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Some tests may be skipped. Install with: pip install torch-geometric transformers rdkit scipy")
    else:
        print("✅ All optional packages available")
    
    return True


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='Run Unified DTA System tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--ci', action='store_true', help='Run CI pipeline tests only')
    parser.add_argument('--test', type=str, help='Run specific test file or test case')
    parser.add_argument('--pattern', type=str, default='test_*.py', help='Test file pattern')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--failfast', '-f', action='store_true', help='Stop on first failure')
    parser.add_argument('--no-deps-check', action='store_true', help='Skip dependency check')
    
    args = parser.parse_args()
    
    # Check dependencies unless skipped
    if not args.no_deps_check:
        if not check_dependencies():
            print("Dependency check failed. Use --no-deps-check to skip.")
            return 1
        print()
    
    verbosity = 2 if args.verbose else 1
    
    try:
        if args.test:
            # Run specific test
            success = run_test_suite(
                specific_test=args.test,
                verbosity=verbosity,
                failfast=args.failfast
            )
        elif args.unit:
            success = run_unit_tests()
        elif args.integration:
            success = run_integration_tests()
        elif args.performance:
            success = run_performance_tests()
        elif args.ci:
            success = run_ci_tests()
        else:
            # Run all tests
            print("Running all tests...")
            success = run_test_suite(
                test_pattern=args.pattern,
                verbosity=verbosity,
                failfast=args.failfast
            )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nTest execution failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
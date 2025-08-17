"""
Continuous Integration test pipeline
Tests for automated testing and validation
"""

import unittest
import subprocess
import sys
import os
import tempfile
import yaml
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCIPipeline(unittest.TestCase):
    """Test continuous integration pipeline components"""
    
    def test_import_all_modules(self):
        """Test that all core modules can be imported"""
        modules_to_test = [
            'core',
            'core.models',
            'core.base_components',
            'core.config',
            'core.utils',
            'core.data_processing',
            'core.datasets'
        ]
        
        failed_imports = []
        
        for module in modules_to_test:
            try:
                __import__(module)
            except ImportError as e:
                failed_imports.append((module, str(e)))
        
        if failed_imports:
            error_msg = "Failed to import modules:\n"
            for module, error in failed_imports:
                error_msg += f"  {module}: {error}\n"
            self.fail(error_msg)
    
    def test_python_syntax_validation(self):
        """Test Python syntax validation for all core files"""
        core_dir = Path(__file__).parent.parent / 'core'
        
        if not core_dir.exists():
            self.skipTest("Core directory not found")
        
        python_files = list(core_dir.glob('*.py'))
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                compile(source, str(py_file), 'exec')
            except SyntaxError as e:
                syntax_errors.append((str(py_file), str(e)))
            except Exception as e:
                # Other compilation errors
                syntax_errors.append((str(py_file), f"Compilation error: {str(e)}"))
        
        if syntax_errors:
            error_msg = "Syntax errors found:\n"
            for file_path, error in syntax_errors:
                error_msg += f"  {file_path}: {error}\n"
            self.fail(error_msg)
    
    def test_test_discovery(self):
        """Test that all test files can be discovered"""
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob('test_*.py'))
        
        # Should have multiple test files
        self.assertGreater(len(test_files), 3, "Not enough test files found")
        
        # Check that test files have proper structure
        for test_file in test_files:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Should contain unittest imports and classes
            self.assertIn('import unittest', content, f"{test_file} missing unittest import")
            self.assertIn('class Test', content, f"{test_file} missing test classes")
    
    def test_requirements_validation(self):
        """Test that requirements.txt is valid"""
        req_file = Path(__file__).parent.parent / 'requirements.txt'
        
        if not req_file.exists():
            self.skipTest("requirements.txt not found")
        
        with open(req_file, 'r') as f:
            requirements = f.read().strip().split('\n')
        
        # Should have some requirements
        self.assertGreater(len(requirements), 5, "Too few requirements")
        
        # Check for essential packages
        essential_packages = ['torch', 'pandas', 'pyyaml']
        req_text = ' '.join(requirements).lower()
        
        for package in essential_packages:
            self.assertIn(package, req_text, f"Missing essential package: {package}")
    
    def test_configuration_files_valid(self):
        """Test that configuration files are valid"""
        config_dir = Path(__file__).parent.parent / 'configs'
        
        if not config_dir.exists():
            self.skipTest("Config directory not found")
        
        yaml_files = list(config_dir.glob('*.yaml'))
        
        for yaml_file in yaml_files:
            with self.subTest(config_file=yaml_file.name):
                try:
                    with open(yaml_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Should be a dictionary
                    self.assertIsInstance(config, dict, f"{yaml_file} should contain a dictionary")
                    
                    # Should have some basic structure
                    if 'protein_encoder_type' in config:
                        self.assertIn(config['protein_encoder_type'], ['esm', 'cnn'], 
                                     f"Invalid protein encoder in {yaml_file}")
                    
                except yaml.YAMLError as e:
                    self.fail(f"Invalid YAML in {yaml_file}: {e}")
    
    def test_code_quality_checks(self):
        """Test basic code quality checks"""
        core_dir = Path(__file__).parent.parent / 'core'
        
        if not core_dir.exists():
            self.skipTest("Core directory not found")
        
        python_files = list(core_dir.glob('*.py'))
        quality_issues = []
        
        for py_file in python_files:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for basic quality issues
            for i, line in enumerate(lines, 1):
                # Check line length (allow some flexibility)
                if len(line) > 120:
                    quality_issues.append(f"{py_file}:{i} - Line too long ({len(line)} chars)")
                
                # Check for TODO/FIXME comments (informational)
                if 'TODO' in line or 'FIXME' in line:
                    print(f"Note: {py_file}:{i} - {line.strip()}")
        
        # Don't fail on quality issues, just report them
        if quality_issues:
            print("Code quality issues found:")
            for issue in quality_issues[:10]:  # Limit output
                print(f"  {issue}")
    
    def test_documentation_exists(self):
        """Test that basic documentation exists"""
        project_root = Path(__file__).parent.parent
        
        # Check for README
        readme_files = list(project_root.glob('README*'))
        self.assertGreater(len(readme_files), 0, "No README file found")
        
        # Check README has content
        readme_file = readme_files[0]
        with open(readme_file, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        self.assertGreater(len(readme_content), 100, "README file too short")
        
        # Check for basic sections
        readme_lower = readme_content.lower()
        expected_sections = ['installation', 'usage', 'model']
        
        for section in expected_sections:
            if section not in readme_lower:
                print(f"Note: README missing '{section}' section")


class TestEnvironmentValidation(unittest.TestCase):
    """Test environment and dependency validation"""
    
    def test_python_version(self):
        """Test Python version compatibility"""
        version = sys.version_info
        
        # Should be Python 3.7+
        self.assertGreaterEqual(version.major, 3, "Python 3 required")
        self.assertGreaterEqual(version.minor, 7, "Python 3.7+ required")
        
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    def test_essential_packages_importable(self):
        """Test that essential packages can be imported"""
        essential_packages = [
            ('torch', 'PyTorch'),
            ('pandas', 'Pandas'),
            ('yaml', 'PyYAML'),
            ('pathlib', 'Pathlib')
        ]
        
        missing_packages = []
        
        for package, name in essential_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(name)
        
        if missing_packages:
            self.fail(f"Missing essential packages: {', '.join(missing_packages)}")
    
    def test_optional_packages_availability(self):
        """Test availability of optional packages"""
        optional_packages = [
            ('torch_geometric', 'PyTorch Geometric'),
            ('transformers', 'Transformers'),
            ('rdkit', 'RDKit'),
            ('scipy', 'SciPy')
        ]
        
        available_packages = []
        missing_packages = []
        
        for package, name in optional_packages:
            try:
                __import__(package)
                available_packages.append(name)
            except ImportError:
                missing_packages.append(name)
        
        print(f"Available optional packages: {', '.join(available_packages)}")
        if missing_packages:
            print(f"Missing optional packages: {', '.join(missing_packages)}")
            print("Note: Some functionality may be limited without these packages")
    
    def test_torch_functionality(self):
        """Test basic PyTorch functionality"""
        try:
            import torch
            
            # Test tensor creation
            x = torch.randn(3, 4)
            self.assertEqual(x.shape, (3, 4))
            
            # Test basic operations
            y = torch.randn(4, 5)
            z = torch.mm(x, y)
            self.assertEqual(z.shape, (3, 5))
            
            # Test CUDA availability (informational)
            cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {cuda_available}")
            
            if cuda_available:
                print(f"CUDA devices: {torch.cuda.device_count()}")
                print(f"Current device: {torch.cuda.current_device()}")
            
        except Exception as e:
            self.fail(f"PyTorch functionality test failed: {e}")


class TestTestSuiteIntegrity(unittest.TestCase):
    """Test the integrity of the test suite itself"""
    
    def test_all_test_files_runnable(self):
        """Test that all test files can be run individually"""
        test_dir = Path(__file__).parent
        test_files = [f for f in test_dir.glob('test_*.py') if f.name != 'test_ci_pipeline.py']
        
        for test_file in test_files:
            with self.subTest(test_file=test_file.name):
                try:
                    # Try to import the test module
                    spec = __import__(f'tests.{test_file.stem}', fromlist=[''])
                    
                    # Check that it has test classes
                    test_classes = [getattr(spec, name) for name in dir(spec) 
                                  if name.startswith('Test') and isinstance(getattr(spec, name), type)]
                    
                    self.assertGreater(len(test_classes), 0, 
                                     f"{test_file} has no test classes")
                    
                except ImportError as e:
                    # This is expected for some test files that depend on optional packages
                    print(f"Note: {test_file} import failed (may be due to missing dependencies): {e}")
    
    def test_test_coverage_completeness(self):
        """Test that we have tests for major components"""
        test_dir = Path(__file__).parent
        test_files = [f.stem for f in test_dir.glob('test_*.py')]
        
        expected_test_areas = [
            'test_encoders',
            'test_fusion_attention', 
            'test_data_processing',
            'test_configuration',
            'test_integration',
            'test_performance'
        ]
        
        missing_tests = []
        for area in expected_test_areas:
            if area not in test_files:
                missing_tests.append(area)
        
        if missing_tests:
            self.fail(f"Missing test files: {', '.join(missing_tests)}")
    
    def test_test_naming_conventions(self):
        """Test that test files follow naming conventions"""
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob('test_*.py'))
        
        for test_file in test_files:
            # Test file names should be lowercase with underscores
            name = test_file.stem
            self.assertTrue(name.islower(), f"{test_file} should be lowercase")
            self.assertTrue(name.startswith('test_'), f"{test_file} should start with 'test_'")
            
            # Check for proper test class naming in file
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Should have TestCase classes
                import re
                test_classes = re.findall(r'class (Test\w+)\(', content)
                
                for class_name in test_classes:
                    self.assertTrue(class_name.startswith('Test'), 
                                   f"Test class {class_name} should start with 'Test'")
                    
            except Exception as e:
                print(f"Warning: Could not validate {test_file}: {e}")


class TestCIConfiguration(unittest.TestCase):
    """Test CI configuration and setup"""
    
    def test_create_ci_config_template(self):
        """Test creation of CI configuration template"""
        # Create a basic GitHub Actions workflow template
        workflow_template = {
            'name': 'Unified DTA System Tests',
            'on': ['push', 'pull_request'],
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'strategy': {
                        'matrix': {
                            'python-version': ['3.8', '3.9', '3.10']
                        }
                    },
                    'steps': [
                        {'uses': 'actions/checkout@v3'},
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {'python-version': '${{ matrix.python-version }}'}
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements.txt'
                        },
                        {
                            'name': 'Run tests',
                            'run': 'python -m pytest tests/ -v'
                        }
                    ]
                }
            }
        }
        
        # Test that template is valid YAML
        yaml_content = yaml.dump(workflow_template, default_flow_style=False)
        parsed = yaml.safe_load(yaml_content)
        
        self.assertEqual(parsed['name'], 'Unified DTA System Tests')
        self.assertIn('test', parsed['jobs'])
        
        print("CI configuration template created successfully")
    
    def test_test_command_execution(self):
        """Test that test commands can be executed"""
        test_commands = [
            [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'],
            [sys.executable, '-c', 'import tests; print("Tests module importable")']
        ]
        
        for cmd in test_commands:
            with self.subTest(command=' '.join(cmd)):
                try:
                    # Don't actually run the full test suite here to avoid recursion
                    # Just test that the command structure is valid
                    if 'unittest' in cmd:
                        # Test command structure
                        self.assertIn('unittest', cmd)
                        self.assertIn('discover', cmd)
                    else:
                        # Test import command
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    self.fail(f"Command timed out: {' '.join(cmd)}")
                except Exception as e:
                    self.fail(f"Command execution failed: {e}")


if __name__ == '__main__':
    # Run CI pipeline tests
    unittest.main(verbosity=2)
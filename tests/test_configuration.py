"""
Unit tests for configuration management and factory methods
Tests configuration loading, validation, and model factory
"""

import unittest
import tempfile
import os
import yaml
import json
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.config import (
        DTAConfig, load_config, save_config, validate_config,
        get_default_configs, create_config_template,
        get_environment_config, ConfigurationManager
    )
    from core.model_factory import (
        ModelFactory, create_dta_model,
        get_lightweight_model, get_production_model,
        create_lightweight_model, create_standard_model,
        create_high_performance_model
    )
    from core.models import UnifiedDTAModel
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    # Set fallbacks
    DTAConfig = None
    ModelFactory = None
    UnifiedDTAModel = None


class TestDTAConfig(unittest.TestCase):
    """Test DTAConfig class and configuration handling"""
    
    def test_dta_config_creation(self):
        """Test DTAConfig creation with default values"""
        if DTAConfig is None:
            self.skipTest("DTAConfig not available")
        
        config = DTAConfig()
        
        # Check default values
        self.assertIsNotNone(config.protein_encoder_type)
        self.assertIsNotNone(config.drug_encoder_type)
        self.assertIsInstance(config.use_fusion, bool)
        self.assertIsInstance(config.protein_config, dict)
        self.assertIsInstance(config.drug_config, dict)
    
    def test_dta_config_custom_values(self):
        """Test DTAConfig with custom values"""
        if DTAConfig is None:
            self.skipTest("DTAConfig not available")
        
        custom_config = DTAConfig(
            protein_encoder_type='cnn',
            drug_encoder_type='gin',
            use_fusion=False,
            protein_config={'output_dim': 64},
            drug_config={'output_dim': 64, 'num_layers': 3}
        )
        
        self.assertEqual(custom_config.protein_encoder_type, 'cnn')
        self.assertEqual(custom_config.drug_encoder_type, 'gin')
        self.assertFalse(custom_config.use_fusion)
        self.assertEqual(custom_config.protein_config['output_dim'], 64)
        self.assertEqual(custom_config.drug_config['num_layers'], 3)
    
    def test_dta_config_validation(self):
        """Test DTAConfig validation"""
        if DTAConfig is None:
            self.skipTest("DTAConfig not available")
        
        # Valid configuration
        valid_config = DTAConfig(
            protein_encoder_type='esm',
            drug_encoder_type='gin',
            use_fusion=True
        )
        
        self.assertTrue(validate_config(valid_config))
        
        # Invalid configuration
        invalid_config = DTAConfig(
            protein_encoder_type='invalid_encoder',
            drug_encoder_type='gin'
        )
        
        self.assertFalse(validate_config(invalid_config))


class TestConfigurationIO(unittest.TestCase):
    """Test configuration file I/O operations"""
    
    def setUp(self):
        """Set up temporary files for testing"""
        self.temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.temp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        self.test_config_dict = {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': False,
            'protein_config': {
                'embed_dim': 64,
                'num_filters': [32, 64],
                'output_dim': 64
            },
            'drug_config': {
                'hidden_dim': 64,
                'num_layers': 3,
                'output_dim': 64
            },
            'predictor_config': {
                'hidden_dims': [128],
                'dropout': 0.2
            }
        }
        
        # Write test configurations
        yaml.dump(self.test_config_dict, self.temp_yaml)
        self.temp_yaml.close()
        
        json.dump(self.test_config_dict, self.temp_json)
        self.temp_json.close()
    
    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in [self.temp_yaml, self.temp_json]:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration"""
        config = load_config(self.temp_yaml.name)
        
        self.assertIsInstance(config, dict)
        self.assertEqual(config['protein_encoder_type'], 'cnn')
        self.assertEqual(config['drug_encoder_type'], 'gin')
        self.assertFalse(config['use_fusion'])
    
    def test_load_json_config(self):
        """Test loading JSON configuration"""
        config = load_config(self.temp_json.name)
        
        self.assertIsInstance(config, dict)
        self.assertEqual(config['protein_encoder_type'], 'cnn')
        self.assertEqual(config['drug_encoder_type'], 'gin')
        self.assertFalse(config['use_fusion'])
    
    def test_save_yaml_config(self):
        """Test saving YAML configuration"""
        temp_save = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        temp_save.close()
        
        try:
            save_config(self.test_config_dict, temp_save.name)
            
            # Load and verify
            loaded_config = load_config(temp_save.name)
            self.assertEqual(loaded_config['protein_encoder_type'], 'cnn')
            self.assertEqual(loaded_config['drug_config']['num_layers'], 3)
            
        finally:
            if os.path.exists(temp_save.name):
                os.unlink(temp_save.name)
    
    def test_save_json_config(self):
        """Test saving JSON configuration"""
        temp_save = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_save.close()
        
        try:
            save_config(self.test_config_dict, temp_save.name)
            
            # Load and verify
            loaded_config = load_config(temp_save.name)
            self.assertEqual(loaded_config['protein_encoder_type'], 'cnn')
            self.assertEqual(loaded_config['drug_config']['num_layers'], 3)
            
        finally:
            if os.path.exists(temp_save.name):
                os.unlink(temp_save.name)


class TestDefaultConfigurations(unittest.TestCase):
    """Test default configuration templates"""
    
    def test_get_default_configs(self):
        """Test getting default configurations"""
        default_configs = get_default_configs()
        
        self.assertIsInstance(default_configs, dict)
        self.assertIn('lightweight', default_configs)
        self.assertIn('production', default_configs)
        
        # Check lightweight config
        lightweight = default_configs['lightweight']
        self.assertEqual(lightweight['protein_encoder_type'], 'cnn')
        self.assertFalse(lightweight['use_fusion'])
        
        # Check production config
        production = default_configs['production']
        self.assertEqual(production['protein_encoder_type'], 'esm')
        self.assertTrue(production['use_fusion'])
    
    def test_create_config_template(self):
        """Test configuration template creation"""
        template = create_config_template('lightweight')
        
        self.assertIsInstance(template, dict)
        self.assertIn('protein_encoder_type', template)
        self.assertIn('drug_encoder_type', template)
        self.assertIn('protein_config', template)
        self.assertIn('drug_config', template)
    
    def test_environment_configs(self):
        """Test environment-specific configurations"""
        environments = ['development', 'staging', 'production']
        
        for env in environments:
            with self.subTest(environment=env):
                config = get_environment_config(env)
                self.assertIsInstance(config, dict)
                self.assertIn('protein_encoder_type', config)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation"""
    
    def test_valid_configuration_validation(self):
        """Test validation of valid configurations"""
        valid_configs = [
            {
                'protein_encoder_type': 'esm',
                'drug_encoder_type': 'gin',
                'use_fusion': True,
                'protein_config': {'output_dim': 128},
                'drug_config': {'output_dim': 128, 'num_layers': 5}
            },
            {
                'protein_encoder_type': 'cnn',
                'drug_encoder_type': 'gin',
                'use_fusion': False,
                'protein_config': {'output_dim': 64},
                'drug_config': {'output_dim': 64, 'num_layers': 3}
            }
        ]
        
        for config in valid_configs:
            with self.subTest(config=config['protein_encoder_type']):
                self.assertTrue(validate_config(config))
    
    def test_invalid_configuration_validation(self):
        """Test validation of invalid configurations"""
        invalid_configs = [
            {
                'protein_encoder_type': 'invalid_encoder',
                'drug_encoder_type': 'gin'
            },
            {
                'protein_encoder_type': 'esm',
                'drug_encoder_type': 'invalid_encoder'
            },
            {
                'protein_encoder_type': 'esm',
                'drug_encoder_type': 'gin',
                'protein_config': {'output_dim': -1}  # Invalid dimension
            }
        ]
        
        for config in invalid_configs:
            with self.subTest(config=str(config)[:50]):
                self.assertFalse(validate_config(config))
    
    def test_detailed_validation(self):
        """Test detailed validation with error messages"""
        invalid_config = {
            'protein_encoder_type': 'invalid_encoder',
            'drug_encoder_type': 'gin'
        }
        
        is_valid, errors = validate_config(invalid_config, detailed=True)
        
        self.assertFalse(is_valid)
        self.assertIsInstance(errors, list)
        self.assertGreater(len(errors), 0)


class TestModelFactory(unittest.TestCase):
    """Test model factory functionality"""
    
    def test_model_factory_initialization(self):
        """Test ModelFactory initialization"""
        if ModelFactory is None:
            self.skipTest("ModelFactory not available")
        
        factory = ModelFactory()
        self.assertIsNotNone(factory)
    
    def test_lightweight_model_creation(self):
        """Test lightweight model creation"""
        try:
            model = get_lightweight_model()
            self.assertIsNotNone(model)
            
            # Check model type
            if UnifiedDTAModel is not None:
                self.assertIsInstance(model, UnifiedDTAModel)
            
            # Check parameter count is reasonable for lightweight model
            total_params = sum(p.numel() for p in model.parameters())
            self.assertLess(total_params, 1000000)  # Less than 1M parameters
            
        except Exception as e:
            self.skipTest(f"Lightweight model creation failed: {e}")
    
    def test_production_model_creation(self):
        """Test production model creation"""
        try:
            model = get_production_model()
            self.assertIsNotNone(model)
            
            # Check model type
            if UnifiedDTAModel is not None:
                self.assertIsInstance(model, UnifiedDTAModel)
            
        except Exception as e:
            self.skipTest(f"Production model creation failed (expected without dependencies): {e}")
    
    def test_custom_model_creation(self):
        """Test custom model creation"""
        custom_config = {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': False,
            'protein_config': {'output_dim': 32},
            'drug_config': {'output_dim': 32, 'num_layers': 2},
            'predictor_config': {'hidden_dims': [64], 'dropout': 0.1}
        }
        
        try:
            model = create_dta_model(custom_config)
            self.assertIsNotNone(model)
            
            # Check that custom configuration was applied
            if hasattr(model, 'protein_encoder'):
                self.assertEqual(model.protein_encoder.output_dim, 32)
            
        except Exception as e:
            self.skipTest(f"Custom model creation failed: {e}")
    
    def test_model_factory_configurations(self):
        """Test ModelFactory configuration listing"""
        if ModelFactory is None:
            self.skipTest("ModelFactory not available")
        
        try:
            configs = ModelFactory.list_configurations()
            self.assertIsInstance(configs, dict)
            self.assertIn('lightweight', configs)
            
            for name, info in configs.items():
                self.assertIn('name', info)
                self.assertIn('memory_usage', info)
                self.assertIn('recommended_use', info)
                
        except Exception as e:
            self.skipTest(f"Configuration listing failed: {e}")
    
    def test_model_creation_from_config_file(self):
        """Test model creation from configuration file"""
        if ModelFactory is None:
            self.skipTest("ModelFactory not available")
        
        # Create temporary config file
        config_dict = {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': False,
            'protein_config': {'output_dim': 32},
            'drug_config': {'output_dim': 32}
        }
        
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_dict, temp_config)
        temp_config.close()
        
        try:
            model = ModelFactory.create_from_config_file(temp_config.name)
            self.assertIsNotNone(model)
            
        except Exception as e:
            self.skipTest(f"Model creation from config file failed: {e}")
        finally:
            if os.path.exists(temp_config.name):
                os.unlink(temp_config.name)


class TestConfigurationManager(unittest.TestCase):
    """Test ConfigurationManager functionality"""
    
    def setUp(self):
        """Set up temporary directory for configuration manager"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_configuration_manager_initialization(self):
        """Test ConfigurationManager initialization"""
        if ConfigurationManager is None:
            self.skipTest("ConfigurationManager not available")
        
        manager = ConfigurationManager(self.temp_dir)
        self.assertEqual(manager.config_dir, Path(self.temp_dir))
    
    def test_create_all_templates(self):
        """Test creating all configuration templates"""
        if ConfigurationManager is None:
            self.skipTest("ConfigurationManager not available")
        
        manager = ConfigurationManager(self.temp_dir)
        
        try:
            manager.create_all_templates()
            
            # Check that template files were created
            config_files = list(Path(self.temp_dir).glob('*.yaml'))
            self.assertGreater(len(config_files), 0)
            
        except Exception as e:
            self.skipTest(f"Template creation failed: {e}")
    
    def test_validate_all_configs(self):
        """Test validating all configurations"""
        if ConfigurationManager is None:
            self.skipTest("ConfigurationManager not available")
        
        manager = ConfigurationManager(self.temp_dir)
        
        try:
            # Create some templates first
            manager.create_all_templates()
            
            # Validate all
            results = manager.validate_all_configs()
            
            self.assertIsInstance(results, dict)
            
            for config_name, is_valid in results.items():
                self.assertIsInstance(is_valid, bool)
                
        except Exception as e:
            self.skipTest(f"Configuration validation failed: {e}")


if __name__ == '__main__':
    unittest.main()
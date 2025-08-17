"""
Model Factory for Unified DTA System
Provides predefined configurations and factory methods for creating models
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

from .models import UnifiedDTAModel
from .config import DTAConfig, ProteinConfig, DrugConfig, FusionConfig, PredictorConfig, TrainingConfig, DataConfig
from .protein_encoders import ESMProteinEncoder, EnhancedCNNProteinEncoder
from .drug_encoders import GINDrugEncoder, EnhancedGINDrugEncoder, LightweightGINEncoder
from .prediction_heads import MLPPredictionHead, PredictionHeadFactory

logger = logging.getLogger(__name__)


class ModelConfigurationError(Exception):
    """Exception raised for model configuration errors"""
    pass


class ModelFactory:
    """Factory class for creating DTA models with predefined configurations"""
    
    # Predefined configurations
    CONFIGURATIONS = {
        'lightweight': {
            'name': 'Lightweight Development Model',
            'description': 'Fast, memory-efficient model for development and testing',
            'memory_usage': '~100MB',
            'recommended_use': 'Development, testing, CI/CD',
            'config': {
                'protein_encoder_type': 'cnn',
                'drug_encoder_type': 'gin',
                'use_fusion': False,
                'protein_config': {
                    'output_dim': 64,
                    'embed_dim': 64,
                    'num_filters': [32, 64],
                    'kernel_sizes': [3, 5],
                    'max_length': 100
                },
                'drug_config': {
                    'output_dim': 64,
                    'hidden_dim': 64,
                    'num_layers': 3,
                    'dropout': 0.2
                },
                'predictor_config': {
                    'type': 'mlp',
                    'hidden_dims': [128],
                    'dropout': 0.2,
                    'activation': 'relu',
                    'use_batch_norm': True
                }
            }
        },
        
        'standard': {
            'name': 'Standard Production Model',
            'description': 'Balanced model with good performance and reasonable resource usage',
            'memory_usage': '~2GB',
            'recommended_use': 'Production, general research',
            'config': {
                'protein_encoder_type': 'esm',
                'drug_encoder_type': 'gin',
                'use_fusion': True,
                'protein_config': {
                    'output_dim': 128,
                    'max_length': 200,
                    'model_name': 'facebook/esm2_t6_8M_UR50D'
                },
                'drug_config': {
                    'output_dim': 128,
                    'hidden_dim': 128,
                    'num_layers': 5,
                    'dropout': 0.2,
                    'use_batch_norm': True
                },
                'fusion_config': {
                    'hidden_dim': 256,
                    'num_heads': 8
                },
                'predictor_config': {
                    'type': 'mlp',
                    'hidden_dims': [512, 256],
                    'dropout': 0.3,
                    'activation': 'relu',
                    'use_batch_norm': True
                }
            }
        },
        
        'high_performance': {
            'name': 'High-Performance Model',
            'description': 'Maximum performance model for large-scale research',
            'memory_usage': '~8GB',
            'recommended_use': 'Large datasets, high-accuracy requirements',
            'config': {
                'protein_encoder_type': 'esm',
                'drug_encoder_type': 'gin',
                'use_fusion': True,
                'protein_config': {
                    'output_dim': 256,
                    'max_length': 400,
                    'model_name': 'facebook/esm2_t12_35M_UR50D'  # Larger ESM model
                },
                'drug_config': {
                    'output_dim': 256,
                    'hidden_dim': 256,
                    'num_layers': 7,
                    'dropout': 0.3,
                    'use_batch_norm': True
                },
                'fusion_config': {
                    'hidden_dim': 512,
                    'num_heads': 16
                },
                'predictor_config': {
                    'type': 'mlp',
                    'hidden_dims': [1024, 512, 256],
                    'dropout': 0.4,
                    'activation': 'gelu',
                    'use_batch_norm': True
                }
            }
        },
        
        'memory_optimized': {
            'name': 'Memory-Optimized Model',
            'description': 'Optimized for limited memory environments',
            'memory_usage': '~500MB',
            'recommended_use': 'Resource-constrained environments',
            'config': {
                'protein_encoder_type': 'cnn',
                'drug_encoder_type': 'gin',
                'use_fusion': False,
                'protein_config': {
                    'output_dim': 96,
                    'embed_dim': 96,
                    'num_filters': [48, 96],
                    'kernel_sizes': [3, 5],
                    'max_length': 150
                },
                'drug_config': {
                    'output_dim': 96,
                    'hidden_dim': 96,
                    'num_layers': 4,
                    'dropout': 0.25
                },
                'predictor_config': {
                    'type': 'mlp',
                    'hidden_dims': [192, 96],
                    'dropout': 0.25,
                    'activation': 'relu',
                    'use_batch_norm': True
                }
            }
        },
        
        'research': {
            'name': 'Research Model',
            'description': 'Flexible model for research and experimentation',
            'memory_usage': '~4GB',
            'recommended_use': 'Research, experimentation, ablation studies',
            'config': {
                'protein_encoder_type': 'esm',
                'drug_encoder_type': 'gin',
                'use_fusion': True,
                'protein_config': {
                    'output_dim': 192,
                    'max_length': 300,
                    'model_name': 'facebook/esm2_t6_8M_UR50D'
                },
                'drug_config': {
                    'output_dim': 192,
                    'hidden_dim': 192,
                    'num_layers': 6,
                    'dropout': 0.3,
                    'use_batch_norm': True
                },
                'fusion_config': {
                    'hidden_dim': 384,
                    'num_heads': 12
                },
                'predictor_config': {
                    'type': 'mlp',
                    'hidden_dims': [768, 384, 192],
                    'dropout': 0.35,
                    'activation': 'gelu',
                    'use_batch_norm': True
                }
            }
        }
    }
    
    @classmethod
    def create_model(cls, 
                     config_name: str, 
                     custom_config: Optional[Dict[str, Any]] = None,
                     validate: bool = True) -> UnifiedDTAModel:
        """
        Create a model with predefined or custom configuration
        
        Args:
            config_name: Name of predefined configuration or 'custom'
            custom_config: Custom configuration dict (required if config_name='custom')
            validate: Whether to validate the configuration
            
        Returns:
            UnifiedDTAModel instance
            
        Raises:
            ModelConfigurationError: If configuration is invalid
        """
        
        if config_name == 'custom':
            if custom_config is None:
                raise ModelConfigurationError("custom_config is required when config_name='custom'")
            config_dict = custom_config
        elif config_name in cls.CONFIGURATIONS:
            config_dict = cls.CONFIGURATIONS[config_name]['config'].copy()
            
            # Apply custom overrides if provided
            if custom_config:
                config_dict = cls._merge_configs(config_dict, custom_config)
        else:
            available_configs = list(cls.CONFIGURATIONS.keys()) + ['custom']
            raise ModelConfigurationError(
                f"Unknown configuration '{config_name}'. "
                f"Available configurations: {available_configs}"
            )
        
        # Validate configuration if requested
        if validate:
            cls._validate_config(config_dict)
        
        # Create and return model
        try:
            model = UnifiedDTAModel(**config_dict)
            logger.info(f"Created model with configuration: {config_name}")
            return model
        except Exception as e:
            raise ModelConfigurationError(f"Failed to create model: {str(e)}")
    
    @classmethod
    def get_lightweight_model(cls, **kwargs) -> UnifiedDTAModel:
        """Create lightweight model for development"""
        return cls.create_model('lightweight', kwargs)
    
    @classmethod
    def get_standard_model(cls, **kwargs) -> UnifiedDTAModel:
        """Create standard production model"""
        return cls.create_model('standard', kwargs)
    
    @classmethod
    def get_high_performance_model(cls, **kwargs) -> UnifiedDTAModel:
        """Create high-performance model"""
        return cls.create_model('high_performance', kwargs)
    
    @classmethod
    def get_memory_optimized_model(cls, **kwargs) -> UnifiedDTAModel:
        """Create memory-optimized model"""
        return cls.create_model('memory_optimized', kwargs)
    
    @classmethod
    def get_research_model(cls, **kwargs) -> UnifiedDTAModel:
        """Create research model"""
        return cls.create_model('research', kwargs)
    
    @classmethod
    def list_configurations(cls) -> Dict[str, Dict[str, str]]:
        """List available predefined configurations"""
        return {
            name: {
                'name': config['name'],
                'description': config['description'],
                'memory_usage': config['memory_usage'],
                'recommended_use': config['recommended_use']
            }
            for name, config in cls.CONFIGURATIONS.items()
        }
    
    @classmethod
    def get_configuration_details(cls, config_name: str) -> Dict[str, Any]:
        """Get detailed configuration information"""
        if config_name not in cls.CONFIGURATIONS:
            raise ModelConfigurationError(f"Unknown configuration: {config_name}")
        
        return cls.CONFIGURATIONS[config_name].copy()
    
    @classmethod
    def create_from_config_file(cls, config_path: Union[str, Path]) -> UnifiedDTAModel:
        """Create model from configuration file"""
        from .config import load_config
        
        config = load_config(config_path)
        config_dict = config.to_dict()
        
        # Extract model-specific configuration
        model_config = {
            'protein_encoder_type': config_dict['protein_encoder_type'],
            'drug_encoder_type': config_dict['drug_encoder_type'],
            'use_fusion': config_dict['use_fusion'],
            'protein_config': config_dict['protein_config'],
            'drug_config': config_dict['drug_config'],
            'predictor_config': config_dict['predictor_config']
        }
        
        if config_dict['use_fusion']:
            model_config['fusion_config'] = config_dict['fusion_config']
        
        return cls.create_model('custom', model_config)
    
    @staticmethod
    def _merge_configs(base_config: Dict[str, Any], 
                      override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ModelFactory._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate model configuration"""
        errors = []
        
        # Required fields
        required_fields = ['protein_encoder_type', 'drug_encoder_type', 'use_fusion']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Valid encoder types
        valid_protein_encoders = ['esm', 'cnn']
        valid_drug_encoders = ['gin']
        
        if config.get('protein_encoder_type') not in valid_protein_encoders:
            errors.append(f"Invalid protein encoder. Valid options: {valid_protein_encoders}")
        
        if config.get('drug_encoder_type') not in valid_drug_encoders:
            errors.append(f"Invalid drug encoder. Valid options: {valid_drug_encoders}")
        
        # Validate sub-configurations
        if 'protein_config' in config:
            protein_config = config['protein_config']
            if 'output_dim' in protein_config and protein_config['output_dim'] <= 0:
                errors.append("Protein output_dim must be positive")
        
        if 'drug_config' in config:
            drug_config = config['drug_config']
            if 'output_dim' in drug_config and drug_config['output_dim'] <= 0:
                errors.append("Drug output_dim must be positive")
            if 'num_layers' in drug_config and drug_config['num_layers'] <= 0:
                errors.append("Drug num_layers must be positive")
        
        if 'predictor_config' in config:
            predictor_config = config['predictor_config']
            if 'hidden_dims' in predictor_config:
                hidden_dims = predictor_config['hidden_dims']
                if not isinstance(hidden_dims, list) or not all(dim > 0 for dim in hidden_dims):
                    errors.append("Predictor hidden_dims must be a list of positive integers")
        
        if errors:
            raise ModelConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")


# Convenience functions for backward compatibility
def create_lightweight_model(**kwargs) -> UnifiedDTAModel:
    """Create lightweight model - convenience function"""
    return ModelFactory.get_lightweight_model(**kwargs)


def create_standard_model(**kwargs) -> UnifiedDTAModel:
    """Create standard model - convenience function"""
    return ModelFactory.get_standard_model(**kwargs)


def create_high_performance_model(**kwargs) -> UnifiedDTAModel:
    """Create high-performance model - convenience function"""
    return ModelFactory.get_high_performance_model(**kwargs)


def create_model_from_config(config_name: str, **kwargs) -> UnifiedDTAModel:
    """Create model from configuration name - convenience function"""
    return ModelFactory.create_model(config_name, kwargs if kwargs else None)


# Legacy compatibility - maintain existing function names
def get_lightweight_model() -> UnifiedDTAModel:
    """Legacy function for backward compatibility"""
    return ModelFactory.get_lightweight_model()


def get_production_model() -> UnifiedDTAModel:
    """Legacy function for backward compatibility"""
    return ModelFactory.get_standard_model()


def create_dta_model(config: Dict[str, Any]) -> UnifiedDTAModel:
    """Legacy function for backward compatibility"""
    return ModelFactory.create_model('custom', config)


if __name__ == "__main__":
    # Example usage and testing
    print("Available Model Configurations:")
    print("=" * 50)
    
    configs = ModelFactory.list_configurations()
    for name, info in configs.items():
        print(f"\n{name.upper()}:")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Memory Usage: {info['memory_usage']}")
        print(f"  Recommended Use: {info['recommended_use']}")
    
    # Test model creation
    print(f"\n{'='*50}")
    print("Testing Model Creation:")
    
    try:
        # Test lightweight model
        model = ModelFactory.get_lightweight_model()
        print(f"✓ Lightweight model created successfully")
        
        # Test custom configuration
        custom_config = {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': False,
            'protein_config': {'output_dim': 64},
            'drug_config': {'output_dim': 64, 'num_layers': 3}
        }
        
        custom_model = ModelFactory.create_model('custom', custom_config)
        print(f"✓ Custom model created successfully")
        
    except Exception as e:
        print(f"✗ Error creating models: {e}")
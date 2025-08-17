"""
Model Factory for creating different DTA model configurations
"""

from typing import Dict, Any, Optional
from .models import UnifiedDTAModel
from .config import Config


class ModelFactory:
    """Factory class for creating DTA models with different configurations"""
    
    @staticmethod
    def create_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> UnifiedDTAModel:
        """
        Create a DTA model based on type and configuration
        
        Args:
            model_type: Type of model ('lightweight', 'production', 'custom')
            config: Optional configuration dictionary
            
        Returns:
            UnifiedDTAModel instance
        """
        if model_type == 'lightweight':
            return ModelFactory.create_lightweight_model(config)
        elif model_type == 'production':
            return ModelFactory.create_production_model(config)
        elif model_type == 'custom':
            if config is None:
                raise ValueError("Custom model requires configuration")
            return ModelFactory.create_custom_model(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_lightweight_model(config: Optional[Dict[str, Any]] = None) -> UnifiedDTAModel:
        """Create lightweight model for development and testing"""
        default_config = {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': False,
            'protein_config': {
                'embed_dim': 64,
                'num_filters': [32, 64],
                'kernel_sizes': [3, 5],
                'output_dim': 64
            },
            'drug_config': {
                'hidden_dim': 64,
                'num_layers': 3,
                'output_dim': 64
            },
            'predictor_config': {
                'hidden_dims': [128],
                'dropout': 0.2,
                'activation': 'relu'
            }
        }
        
        if config:
            default_config.update(config)
            
        return UnifiedDTAModel(**default_config)
    
    @staticmethod
    def create_production_model(config: Optional[Dict[str, Any]] = None) -> UnifiedDTAModel:
        """Create production model with full features"""
        default_config = {
            'protein_encoder_type': 'esm',
            'drug_encoder_type': 'gin',
            'use_fusion': True,
            'protein_config': {
                'output_dim': 128,
                'max_length': 200
            },
            'drug_config': {
                'hidden_dim': 128,
                'num_layers': 5,
                'output_dim': 128
            },
            'fusion_config': {
                'hidden_dim': 256,
                'num_heads': 8
            },
            'predictor_config': {
                'hidden_dims': [512, 256],
                'dropout': 0.3,
                'activation': 'gelu'
            }
        }
        
        if config:
            default_config.update(config)
            
        return UnifiedDTAModel(**default_config)
    
    @staticmethod
    def create_custom_model(config: Dict[str, Any]) -> UnifiedDTAModel:
        """Create custom model from configuration"""
        return UnifiedDTAModel(**config)


# Convenience functions for backward compatibility
def create_dta_model(config: Dict[str, Any]) -> UnifiedDTAModel:
    """Factory function to create DTA models with configuration"""
    return ModelFactory.create_custom_model(config)


def get_lightweight_model() -> UnifiedDTAModel:
    """Get lightweight model for testing"""
    return ModelFactory.create_lightweight_model()


def get_production_model() -> UnifiedDTAModel:
    """Get production model"""
    return ModelFactory.create_production_model()


def create_lightweight_model() -> UnifiedDTAModel:
    """Alias for get_lightweight_model"""
    return get_lightweight_model()


def create_standard_model() -> UnifiedDTAModel:
    """Create standard model (alias for production)"""
    return get_production_model()


def create_high_performance_model() -> UnifiedDTAModel:
    """Create high-performance model with enhanced settings"""
    config = {
        'protein_encoder_type': 'esm',
        'drug_encoder_type': 'gin',
        'use_fusion': True,
        'protein_config': {
            'output_dim': 256,
            'max_length': 400
        },
        'drug_config': {
            'hidden_dim': 256,
            'num_layers': 6,
            'output_dim': 256
        },
        'fusion_config': {
            'hidden_dim': 512,
            'num_heads': 16
        },
        'predictor_config': {
            'hidden_dims': [1024, 512, 256],
            'dropout': 0.3,
            'activation': 'gelu'
        }
    }
    return ModelFactory.create_custom_model(config)
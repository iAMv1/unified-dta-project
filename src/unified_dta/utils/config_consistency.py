"""
Configuration consistency utilities for ensuring model configurations
match between training and prediction
"""

import logging
from typing import Dict, Any, Optional
import json
import os

logger = logging.getLogger(__name__)


class ConfigConsistencyChecker:
    """Utility class for checking configuration consistency"""
    
    @staticmethod
    def save_model_config(config: Dict[str, Any], model_path: str) -> str:
        """Save model configuration alongside the model"""
        config_path = f"{model_path}.config.json"
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Model configuration saved to {config_path}")
            return config_path
        except Exception as e:
            logger.error(f"Failed to save model configuration: {e}")
            return None
    
    @staticmethod
    def load_model_config(model_path: str) -> Optional[Dict[str, Any]]:
        """Load model configuration from file"""
        config_path = f"{model_path}.config.json"
        if not os.path.exists(config_path):
            logger.warning(f"Model configuration file not found: {config_path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Model configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load model configuration: {e}")
            return None
    
    @staticmethod
    def validate_config_consistency(
        training_config: Dict[str, Any], 
        prediction_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate consistency between training and prediction configurations"""
        inconsistencies = []
        warnings = []
        
        # Check model architecture components
        for key in ['protein_encoder_type', 'drug_encoder_type', 'use_fusion']:
            if training_config.get(key) != prediction_config.get(key):
                inconsistencies.append(
                    f"Model architecture mismatch: {key} "
                    f"(training: {training_config.get(key)}, prediction: {prediction_config.get(key)})"
                )
        
        # Check encoder dimensions
        for encoder_type in ['protein_config', 'drug_config']:
            if encoder_type in training_config and encoder_type in prediction_config:
                train_enc_config = training_config[encoder_type]
                pred_enc_config = prediction_config[encoder_type]
                
                # Check output dimensions
                if train_enc_config.get('output_dim') != pred_enc_config.get('output_dim'):
                    inconsistencies.append(
                        f"{encoder_type} output dimension mismatch: "
                        f"(training: {train_enc_config.get('output_dim')}, "
                        f"prediction: {pred_enc_config.get('output_dim')})"
                    )
        
        # Check fusion configuration if used
        if training_config.get('use_fusion', False) and 'fusion_config' in training_config:
            if 'fusion_config' not in prediction_config:
                inconsistencies.append("Fusion configuration missing in prediction config")
            else:
                # Basic fusion config check
                train_fusion = training_config['fusion_config']
                pred_fusion = prediction_config['fusion_config']
                
                if train_fusion.get('hidden_dim') != pred_fusion.get('hidden_dim'):
                    warnings.append(
                        f"Fusion hidden dimension mismatch: "
                        f"(training: {train_fusion.get('hidden_dim')}, "
                        f"prediction: {pred_fusion.get('hidden_dim')})"
                    )
        
        return {
            'consistent': len(inconsistencies) == 0,
            'inconsistencies': inconsistencies,
            'warnings': warnings
        }
    
    @staticmethod
    def create_prediction_config_from_model(model) -> Dict[str, Any]:
        """Create a prediction configuration from a loaded model"""
        config = {
            'protein_encoder_type': getattr(model, 'protein_encoder_type', 'esm'),
            'drug_encoder_type': 'gin',  # Default assumption
            'use_fusion': hasattr(model, 'fusion') and model.fusion is not None,
        }
        
        # Extract encoder configurations if available
        if hasattr(model, 'protein_encoder'):
            config['protein_config'] = {
                'output_dim': getattr(model.protein_encoder, 'output_dim', 128)
            }
        
        if hasattr(model, 'drug_encoder'):
            config['drug_config'] = {
                'output_dim': getattr(model.drug_encoder, 'output_dim', 128)
            }
        
        if config['use_fusion'] and hasattr(model, 'fusion'):
            config['fusion_config'] = {
                'hidden_dim': getattr(model.fusion, 'output_dim', 256) // 2  # Adjust for concatenation
            }
        
        return config


# Global instance
config_checker = ConfigConsistencyChecker()


def ensure_config_consistency(model_path: str, prediction_config: Dict[str, Any]) -> bool:
    """Ensure configuration consistency for a model"""
    # Load training configuration
    training_config = config_checker.load_model_config(model_path)
    
    if training_config is None:
        logger.warning("No training configuration found, proceeding with prediction configuration")
        return True
    
    # Validate consistency
    validation_result = config_checker.validate_config_consistency(training_config, prediction_config)
    
    if not validation_result['consistent']:
        logger.error("Configuration inconsistencies detected:")
        for inconsistency in validation_result['inconsistencies']:
            logger.error(f"  - {inconsistency}")
        return False
    
    if validation_result['warnings']:
        logger.warning("Configuration warnings:")
        for warning in validation_result['warnings']:
            logger.warning(f"  - {warning}")
    
    logger.info("Configuration consistency check passed")
    return True
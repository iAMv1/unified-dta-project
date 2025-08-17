#!/usr/bin/env python3
"""
Custom Configuration Example for Unified DTA System

This example demonstrates how to create and use custom model configurations
for different use cases and requirements.
"""

import sys
import os
import yaml
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unified_dta import UnifiedDTAModel

def create_minimal_config() -> Dict[str, Any]:
    """Create a minimal configuration for testing."""
    return {
        'model': {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': False
        },
        'protein_config': {
            'vocab_size': 25,
            'embed_dim': 32,
            'output_dim': 32
        },
        'drug_config': {
            'input_dim': 78,
            'hidden_dim': 32,
            'num_layers': 2,
            'output_dim': 32
        },
        'predictor_config': {
            'hidden_dims': [64],
            'dropout': 0.2
        }
    }

def create_production_config() -> Dict[str, Any]:
    """Create a configuration optimized for production."""
    return {
        'model': {
            'protein_encoder_type': 'esm',
            'drug_encoder_type': 'gin',
            'use_fusion': True
        },
        'protein_config': {
            'model_name': 'facebook/esm2_t6_8M_UR50D',
            'output_dim': 128,
            'max_length': 200
        },
        'drug_config': {
            'input_dim': 78,
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
            'dropout': 0.3
        }
    }

def test_configuration(config: Dict[str, Any], config_name: str) -> None:
    """Test a configuration by creating a model."""
    print(f"\n--- Testing {config_name} Configuration ---")
    
    try:
        model = UnifiedDTAModel.from_config(config)
        total_params = sum(p.numel() for p in model.parameters())
        
        print("✓ Model created successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Estimated size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Test prediction
        test_smiles = "CCO"
        test_protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        prediction = model.predict(test_smiles, test_protein)
        print(f"   Test prediction: {prediction:.4f}")
        
    exces e:
 {e}")

def main():
    print("=")
    
    # Test different configurations
    configs = {
        'Minimal': create_mini
        'Pig()
    }
    
    for name, config in configms():
        test_configuration(cone)
    
    print()

if __name__ == "__main__":
    main()
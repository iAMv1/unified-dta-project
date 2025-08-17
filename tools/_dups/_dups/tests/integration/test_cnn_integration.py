#!/usr/bin/env python3
"""
Integration test for Enhanced CNN Protein Encoder with the Unified DTA Model
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.models import UnifiedDTAModel, create_dta_model
from core.protein_encoders import EnhancedCNNProteinEncoder


def create_mock_drug_data(batch_size=4):
    """Create mock drug data for testing"""
    from torch_geometric.data import Data, Batch
    
    # Create mock molecular graphs
    graphs = []
    for i in range(batch_size):
        num_nodes = np.random.randint(10, 30)
        num_edges = np.random.randint(15, 50)
        
        # Node features (78-dimensional as per standard)
        x = torch.randn(num_nodes, 78)
        
        # Random edge indices
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        graph = Data(x=x, edge_index=edge_index)
        graphs.append(graph)
    
    return Batch.from_data_list(graphs)


def test_enhanced_cnn_in_unified_model():
    """Test Enhanced CNN encoder integration with Unified DTA Model"""
    print("Testing Enhanced CNN Integration with Unified DTA Model...")
    
    # Create model with enhanced CNN encoder
    config = {
        'protein_encoder_type': 'cnn',
        'drug_encoder_type': 'gin',
        'use_fusion': True,
        'protein_config': {
            'vocab_size': 25,
            'embed_dim': 128,
            'num_filters': [64, 128, 256],
            'kernel_sizes': [3, 5, 7],
            'output_dim': 128,
            'max_length': 200,
            'dropout': 0.1,
            'use_se': True
        },
        'drug_config': {
            'output_dim': 128,
            'num_layers': 3
        },
        'fusion_config': {
            'hidden_dim': 256,
            'num_heads': 4
        },
        'predictor_config': {
            'hidden_dims': [256, 128],
            'dropout': 0.2
        }
    }
    
    model = create_dta_model(config)
    
    # Verify that the protein encoder is the enhanced CNN
    assert isinstance(model.protein_encoder, EnhancedCNNProteinEncoder), \
        f"Expected EnhancedCNNProteinEncoder, got {type(model.protein_encoder)}"
    
    # Create test data
    batch_size = 4
    seq_len = 150
    
    # Mock protein tokens
    protein_tokens = torch.randint(1, 26, (batch_size, seq_len))
    
    # Mock drug data
    drug_data = create_mock_drug_data(batch_size)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(drug_data, protein_tokens)
    
    # Check output shape
    expected_shape = (batch_size, 1)
    assert predictions.shape == expected_shape, \
        f"Model output shape mismatch: {predictions.shape} vs {expected_shape}"
    
    # Check that predictions are reasonable (not NaN or infinite)
    assert torch.isfinite(predictions).all(), "Model predictions contain NaN or infinite values"
    
    print("✓ Enhanced CNN Integration test passed")


def test_lightweight_vs_production_config():
    """Test both lightweight and production configurations with CNN encoder"""
    print("Testing Lightweight vs Production CNN Configurations...")
    
    # Lightweight configuration
    lightweight_config = {
        'protein_encoder_type': 'cnn',
        'drug_encoder_type': 'gin',
        'use_fusion': False,
        'protein_config': {
            'embed_dim': 64,
            'num_filters': [32, 64],
            'kernel_sizes': [3, 5],
            'output_dim': 64,
            'max_length': 100
        },
        'drug_config': {
            'output_dim': 64,
            'num_layers': 2
        },
        'predictor_config': {
            'hidden_dims': [128],
            'dropout': 0.1
        }
    }
    
    # Production configuration
    production_config = {
        'protein_encoder_type': 'cnn',
        'drug_encoder_type': 'gin',
        'use_fusion': True,
        'protein_config': {
            'embed_dim': 256,
            'num_filters': [128, 256, 512],
            'kernel_sizes': [3, 5, 7, 9],
            'output_dim': 256,
            'max_length': 200,
            'use_se': True,
            'use_positional_encoding': True
        },
        'drug_config': {
            'output_dim': 256,
            'num_layers': 5
        },
        'fusion_config': {
            'hidden_dim': 512,
            'num_heads': 8
        },
        'predictor_config': {
            'hidden_dims': [512, 256, 128],
            'dropout': 0.3
        }
    }
    
    # Test both configurations
    for config_name, config in [("Lightweight", lightweight_config), ("Production", production_config)]:
        print(f"  Testing {config_name} configuration...")
        
        model = create_dta_model(config)
        
        # Test data
        batch_size = 2
        seq_len = 80
        protein_tokens = torch.randint(1, 26, (batch_size, seq_len))
        drug_data = create_mock_drug_data(batch_size)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            predictions = model(drug_data, protein_tokens)
        
        assert predictions.shape == (batch_size, 1), \
            f"{config_name} model output shape mismatch: {predictions.shape}"
        
        assert torch.isfinite(predictions).all(), \
            f"{config_name} model predictions contain NaN or infinite values"
        
        print(f"    ✓ {config_name} configuration test passed")
    
    print("✓ Configuration comparison test passed")


def test_training_mode():
    """Test that the model works in training mode"""
    print("Testing Training Mode...")
    
    config = {
        'protein_encoder_type': 'cnn',
        'drug_encoder_type': 'gin',
        'use_fusion': True,
        'protein_config': {
            'num_filters': [64, 128],
            'kernel_sizes': [3, 5],
            'output_dim': 128
        },
        'drug_config': {'output_dim': 128},
        'fusion_config': {'hidden_dim': 256},
        'predictor_config': {'hidden_dims': [256]}
    }
    
    model = create_dta_model(config)
    model.train()
    
    # Test data
    batch_size = 2
    protein_tokens = torch.randint(1, 26, (batch_size, 100))
    drug_data = create_mock_drug_data(batch_size)
    
    # Forward pass with gradients
    predictions = model(drug_data, protein_tokens)
    
    # Test backward pass
    loss = predictions.sum()
    loss.backward()
    
    # Check that gradients were computed
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients were computed during backward pass"
    
    print("✓ Training mode test passed")


def run_integration_tests():
    """Run all integration tests"""
    print("Running Enhanced CNN Integration Tests...")
    print("=" * 50)
    
    try:
        test_enhanced_cnn_in_unified_model()
        test_lightweight_vs_production_config()
        test_training_mode()
        
        print("=" * 50)
        print("✅ All integration tests passed successfully!")
        print("\nEnhanced CNN Integration Verified:")
        print("- ✓ Works with Unified DTA Model")
        print("- ✓ Supports both lightweight and production configs")
        print("- ✓ Compatible with fusion mechanisms")
        print("- ✓ Proper gradient computation in training mode")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
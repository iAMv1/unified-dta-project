#!/usr/bin/env python3
"""
Integration test for Enhanced GIN Drug Encoder with the Unified DTA Model
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from torch_geometric.data import Data, Batch

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.models import UnifiedDTAModel, create_dta_model
from core.drug_encoders import EnhancedGINDrugEncoder, MultiScaleGINEncoder


def create_mock_data(batch_size=4):
    """Create mock data for testing"""
    # Mock molecular graphs
    graphs = []
    for i in range(batch_size):
        num_nodes = np.random.randint(10, 30)
        num_edges = np.random.randint(15, 50)
        
        # Node features (78-dimensional)
        x = torch.randn(num_nodes, 78)
        
        # Random edge indices
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        graph = Data(x=x, edge_index=edge_index)
        graphs.append(graph)
    
    drug_data = Batch.from_data_list(graphs)
    
    # Mock protein tokens
    seq_len = 150
    protein_tokens = torch.randint(1, 26, (batch_size, seq_len))
    
    return drug_data, protein_tokens


def test_enhanced_gin_in_unified_model():
    """Test Enhanced GIN encoder integration with Unified DTA Model"""
    print("Testing Enhanced GIN Integration with Unified DTA Model...")
    
    # Create model with enhanced GIN encoder
    config = {
        'protein_encoder_type': 'cnn',
        'drug_encoder_type': 'gin',
        'use_fusion': True,
        'protein_config': {
            'embed_dim': 128,
            'num_filters': [64, 128],
            'kernel_sizes': [3, 5],
            'output_dim': 128
        },
        'drug_config': {
            'node_features': 78,
            'hidden_dim': 128,
            'num_layers': 3,
            'output_dim': 128,
            'mlp_hidden_dims': [128, 128],
            'activation': 'relu',
            'dropout': 0.1,
            'pooling_strategy': 'dual',
            'residual_connections': True
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
    
    # Verify that the drug encoder is the enhanced GIN
    assert isinstance(model.drug_encoder, EnhancedGINDrugEncoder), \
        f"Expected EnhancedGINDrugEncoder, got {type(model.drug_encoder)}"
    
    # Create test data
    drug_data, protein_tokens = create_mock_data(batch_size=4)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(drug_data, protein_tokens)
    
    # Check output shape
    expected_shape = (4, 1)
    assert predictions.shape == expected_shape, \
        f"Model output shape mismatch: {predictions.shape} vs {expected_shape}"
    
    # Check that predictions are reasonable
    assert torch.isfinite(predictions).all(), "Model predictions contain NaN or infinite values"
    
    print("✓ Enhanced GIN Integration test passed")


def test_multiscale_gin_integration():
    """Test Multi-scale GIN encoder integration"""
    print("Testing Multi-scale GIN Integration...")
    
    config = {
        'protein_encoder_type': 'cnn',
        'drug_encoder_type': 'multiscale_gin',
        'use_fusion': True,
        'protein_config': {
            'output_dim': 128
        },
        'drug_config': {
            'node_features': 78,
            'hidden_dims': [64, 128, 256],
            'num_layers_per_scale': 2,
            'output_dim': 128,
            'dropout': 0.1
        },
        'fusion_config': {
            'hidden_dim': 256
        }
    }
    
    model = create_dta_model(config)
    
    # Verify encoder type
    assert isinstance(model.drug_encoder, MultiScaleGINEncoder), \
        f"Expected MultiScaleGINEncoder, got {type(model.drug_encoder)}"
    
    # Test forward pass
    drug_data, protein_tokens = create_mock_data(batch_size=2)
    
    model.eval()
    with torch.no_grad():
        predictions = model(drug_data, protein_tokens)
    
    assert predictions.shape == (2, 1), f"Multi-scale model output shape mismatch: {predictions.shape}"
    assert torch.isfinite(predictions).all(), "Multi-scale model predictions contain NaN or infinite values"
    
    print("✓ Multi-scale GIN Integration test passed")


def test_different_pooling_strategies():
    """Test different pooling strategies in the unified model"""
    print("Testing Different Pooling Strategies...")
    
    pooling_strategies = ['mean', 'max', 'dual', 'attention']
    
    for pooling in pooling_strategies:
        print(f"  Testing {pooling} pooling...")
        
        config = {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': False,  # Simpler for testing
            'protein_config': {'output_dim': 64},
            'drug_config': {
                'hidden_dim': 64,
                'num_layers': 2,
                'output_dim': 64,
                'pooling_strategy': pooling,
                'attention_heads': 4 if pooling == 'attention' else 1
            },
            'predictor_config': {'hidden_dims': [128]}
        }
        
        model = create_dta_model(config)
        
        # Test forward pass
        drug_data, protein_tokens = create_mock_data(batch_size=2)
        
        model.eval()
        with torch.no_grad():
            predictions = model(drug_data, protein_tokens)
        
        assert predictions.shape == (2, 1), \
            f"{pooling} pooling model output shape mismatch: {predictions.shape}"
        
        assert torch.isfinite(predictions).all(), \
            f"{pooling} pooling model predictions contain NaN or infinite values"
    
    print("✓ Different Pooling Strategies test passed")


def test_gin_encoder_features():
    """Test specific GIN encoder features"""
    print("Testing GIN Encoder Features...")
    
    config = {
        'protein_encoder_type': 'cnn',
        'drug_encoder_type': 'gin',
        'use_fusion': True,
        'drug_config': {
            'hidden_dim': 128,
            'num_layers': 4,
            'output_dim': 128,
            'mlp_hidden_dims': [256, 128],
            'activation': 'gelu',
            'dropout': 0.2,
            'residual_connections': True,
            'pooling_strategy': 'dual'
        }
    }
    
    model = create_dta_model(config)
    drug_encoder = model.drug_encoder
    
    # Test node embeddings extraction
    drug_data, _ = create_mock_data(batch_size=1)
    
    with torch.no_grad():
        node_embeddings = drug_encoder.get_node_embeddings(drug_data)
        layer_outputs = drug_encoder.get_layer_outputs(drug_data)
    
    # Check node embeddings
    assert node_embeddings.shape[1] == 128, \
        f"Node embeddings dimension mismatch: {node_embeddings.shape[1]} vs 128"
    
    # Check layer outputs
    assert len(layer_outputs) == drug_encoder.num_layers + 1, \
        f"Wrong number of layer outputs: {len(layer_outputs)} vs {drug_encoder.num_layers + 1}"
    
    # Check that layer outputs have correct dimensions
    for i, layer_output in enumerate(layer_outputs):
        assert layer_output.shape[1] == 128, \
            f"Layer {i} output dimension mismatch: {layer_output.shape[1]} vs 128"
    
    print("✓ GIN Encoder Features test passed")


def test_training_with_gin_encoder():
    """Test training mode with GIN encoder"""
    print("Testing Training with GIN Encoder...")
    
    config = {
        'protein_encoder_type': 'cnn',
        'drug_encoder_type': 'gin',
        'use_fusion': True,
        'drug_config': {
            'hidden_dim': 64,
            'num_layers': 2,
            'output_dim': 64,
            'dropout': 0.1
        },
        'protein_config': {'output_dim': 64},
        'fusion_config': {'hidden_dim': 128},
        'predictor_config': {'hidden_dims': [128]}
    }
    
    model = create_dta_model(config)
    model.train()
    
    # Test data
    drug_data, protein_tokens = create_mock_data(batch_size=2)
    
    # Forward pass with gradients
    predictions = model(drug_data, protein_tokens)
    
    # Test backward pass
    loss = predictions.sum()
    loss.backward()
    
    # Check that gradients were computed for drug encoder
    drug_encoder_has_gradients = False
    for param in model.drug_encoder.parameters():
        if param.grad is not None:
            drug_encoder_has_gradients = True
            break
    
    assert drug_encoder_has_gradients, "No gradients computed for drug encoder during backward pass"
    
    print("✓ Training with GIN Encoder test passed")


def run_integration_tests():
    """Run all integration tests"""
    print("Running Enhanced GIN Integration Tests...")
    print("=" * 50)
    
    try:
        test_enhanced_gin_in_unified_model()
        test_multiscale_gin_integration()
        test_different_pooling_strategies()
        test_gin_encoder_features()
        test_training_with_gin_encoder()
        
        print("=" * 50)
        print("✅ All integration tests passed successfully!")
        print("\nEnhanced GIN Integration Verified:")
        print("- ✓ Works with Unified DTA Model")
        print("- ✓ Multi-scale GIN encoder integration")
        print("- ✓ Multiple pooling strategies supported")
        print("- ✓ Node embedding and layer output extraction")
        print("- ✓ Proper gradient computation in training mode")
        print("- ✓ Compatible with fusion mechanisms")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
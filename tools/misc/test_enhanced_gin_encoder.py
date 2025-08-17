#!/usr/bin/env python3
"""
Test script for the Enhanced GIN Drug Encoder
Tests all the key features: residual connections, configurable MLPs, pooling strategies, etc.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from torch_geometric.data import Data, Batch

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.drug_encoders import (
    EnhancedGINDrugEncoder, 
    MultiScaleGINEncoder,
    ConfigurableMLPBlock,
    ResidualGINLayer
)


def create_mock_molecular_graphs(batch_size=4, min_nodes=10, max_nodes=30):
    """Create mock molecular graphs for testing"""
    graphs = []
    
    for i in range(batch_size):
        num_nodes = np.random.randint(min_nodes, max_nodes + 1)
        num_edges = np.random.randint(num_nodes, num_nodes * 3)
        
        # Node features (78-dimensional as per standard)
        x = torch.randn(num_nodes, 78)
        
        # Random edge indices (ensuring valid connections)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Create undirected edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        graph = Data(x=x, edge_index=edge_index)
        graphs.append(graph)
    
    return Batch.from_data_list(graphs)


def test_configurable_mlp_block():
    """Test the configurable MLP block"""
    print("Testing Configurable MLP Block...")
    
    # Test different configurations
    configs = [
        {
            'input_dim': 78,
            'hidden_dims': [128, 64],
            'output_dim': 32,
            'activation': 'relu',
            'dropout': 0.1,
            'use_batch_norm': True
        },
        {
            'input_dim': 64,
            'hidden_dims': [128],
            'output_dim': 64,
            'activation': 'gelu',
            'dropout': 0.0,
            'use_batch_norm': False
        }
    ]
    
    for i, config in enumerate(configs):
        mlp = ConfigurableMLPBlock(**config)
        
        # Test forward pass
        x = torch.randn(10, config['input_dim'])
        output = mlp(x)
        
        expected_shape = (10, config['output_dim'])
        assert output.shape == expected_shape, \
            f"MLP config {i} output shape mismatch: {output.shape} vs {expected_shape}"
    
    print("✓ Configurable MLP Block test passed")


def test_residual_gin_layer():
    """Test the residual GIN layer"""
    print("Testing Residual GIN Layer...")
    
    # Test with same dimensions (residual connection)
    layer = ResidualGINLayer(
        input_dim=64,
        hidden_dim=64,
        mlp_hidden_dims=[128, 64],
        residual_connection=True
    )
    
    # Create test data
    x = torch.randn(20, 64)
    edge_index = torch.randint(0, 20, (2, 40))
    
    output = layer(x, edge_index)
    assert output.shape == x.shape, f"Residual GIN layer output shape mismatch: {output.shape} vs {x.shape}"
    
    # Test with different dimensions (projection needed)
    layer_proj = ResidualGINLayer(
        input_dim=64,
        hidden_dim=128,
        mlp_hidden_dims=[256, 128],
        residual_connection=True
    )
    
    output_proj = layer_proj(x, edge_index)
    expected_shape = (20, 128)
    assert output_proj.shape == expected_shape, \
        f"Residual GIN layer projection shape mismatch: {output_proj.shape} vs {expected_shape}"
    
    print("✓ Residual GIN Layer test passed")


def test_enhanced_gin_encoder():
    """Test the complete Enhanced GIN Drug Encoder"""
    print("Testing Enhanced GIN Drug Encoder...")
    
    # Test with different pooling strategies
    pooling_strategies = ['mean', 'max', 'dual', 'attention']
    
    for pooling in pooling_strategies:
        print(f"  Testing {pooling} pooling...")
        
        encoder = EnhancedGINDrugEncoder(
            node_features=78,
            hidden_dim=128,
            num_layers=3,
            output_dim=128,
            mlp_hidden_dims=[128, 128],
            activation='relu',
            dropout=0.1,
            pooling_strategy=pooling,
            residual_connections=True
        )
        
        # Create test data
        batch_data = create_mock_molecular_graphs(batch_size=4)
        
        # Forward pass
        encoder.eval()
        with torch.no_grad():
            output = encoder(batch_data)
        
        expected_shape = (4, 128)
        assert output.shape == expected_shape, \
            f"Enhanced GIN encoder ({pooling}) output shape mismatch: {output.shape} vs {expected_shape}"
        
        # Test node embeddings
        node_embeddings = encoder.get_node_embeddings(batch_data)
        assert node_embeddings.shape[1] == 128, \
            f"Node embeddings dimension mismatch: {node_embeddings.shape[1]} vs 128"
        
        # Test layer outputs
        layer_outputs = encoder.get_layer_outputs(batch_data)
        assert len(layer_outputs) == encoder.num_layers + 1, \
            f"Wrong number of layer outputs: {len(layer_outputs)} vs {encoder.num_layers + 1}"
    
    print("✓ Enhanced GIN Drug Encoder test passed")


def test_multiscale_gin_encoder():
    """Test the Multi-scale GIN encoder"""
    print("Testing Multi-scale GIN Encoder...")
    
    encoder = MultiScaleGINEncoder(
        node_features=78,
        hidden_dims=[64, 128, 256],
        num_layers_per_scale=2,
        output_dim=128,
        dropout=0.1
    )
    
    # Test data
    batch_data = create_mock_molecular_graphs(batch_size=3)
    
    # Forward pass
    encoder.eval()
    with torch.no_grad():
        output = encoder(batch_data)
    
    expected_shape = (3, 128)
    assert output.shape == expected_shape, \
        f"Multi-scale GIN encoder output shape mismatch: {output.shape} vs {expected_shape}"
    
    # Check that we have the correct number of scale encoders
    assert len(encoder.encoders) == 3, f"Expected 3 scale encoders, got {len(encoder.encoders)}"
    
    print("✓ Multi-scale GIN Encoder test passed")


def test_configurable_parameters():
    """Test that the encoder works with different configurations"""
    print("Testing Configurable Parameters...")
    
    configs = [
        {
            'name': 'Lightweight',
            'config': {
                'node_features': 78,
                'hidden_dim': 64,
                'num_layers': 2,
                'output_dim': 64,
                'mlp_hidden_dims': [64],
                'activation': 'relu',
                'dropout': 0.1,
                'pooling_strategy': 'mean'
            }
        },
        {
            'name': 'High-capacity',
            'config': {
                'node_features': 78,
                'hidden_dim': 256,
                'num_layers': 6,
                'output_dim': 256,
                'mlp_hidden_dims': [512, 256],
                'activation': 'gelu',
                'dropout': 0.2,
                'pooling_strategy': 'attention',
                'attention_heads': 8
            }
        }
    ]
    
    for config_info in configs:
        name = config_info['name']
        config = config_info['config']
        
        encoder = EnhancedGINDrugEncoder(**config)
        
        # Test forward pass
        batch_data = create_mock_molecular_graphs(batch_size=2)
        
        encoder.eval()
        with torch.no_grad():
            output = encoder(batch_data)
        
        expected_shape = (2, config['output_dim'])
        assert output.shape == expected_shape, \
            f"{name} config output shape mismatch: {output.shape} vs {expected_shape}"
        
        # Count parameters
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        
        print(f"  {name}: {total_params:,} total params, {trainable_params:,} trainable")
    
    print("✓ Configurable Parameters test passed")


def test_residual_connections():
    """Test that residual connections work properly"""
    print("Testing Residual Connections...")
    
    # Create encoder with residual connections
    encoder_with_residual = EnhancedGINDrugEncoder(
        node_features=78,
        hidden_dim=128,
        num_layers=3,
        output_dim=128,
        residual_connections=True
    )
    
    # Create encoder without residual connections
    encoder_without_residual = EnhancedGINDrugEncoder(
        node_features=78,
        hidden_dim=128,
        num_layers=3,
        output_dim=128,
        residual_connections=False
    )
    
    # Test data
    batch_data = create_mock_molecular_graphs(batch_size=2)
    
    # Forward passes
    with torch.no_grad():
        output_with = encoder_with_residual(batch_data)
        output_without = encoder_without_residual(batch_data)
    
    # Both should have same shape
    assert output_with.shape == output_without.shape, \
        "Residual and non-residual encoders should have same output shape"
    
    # Outputs should be different (residual connections change the computation)
    assert not torch.allclose(output_with, output_without, atol=1e-6), \
        "Residual connections should change the output"
    
    print("✓ Residual Connections test passed")


def test_training_mode():
    """Test that the encoder works in training mode"""
    print("Testing Training Mode...")
    
    encoder = EnhancedGINDrugEncoder(
        node_features=78,
        hidden_dim=128,
        num_layers=3,
        output_dim=128,
        dropout=0.2
    )
    
    encoder.train()
    
    # Test data
    batch_data = create_mock_molecular_graphs(batch_size=2)
    
    # Forward pass with gradients
    output = encoder(batch_data)
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    # Check that gradients were computed
    has_gradients = False
    for param in encoder.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients were computed during backward pass"
    
    print("✓ Training mode test passed")


def run_all_tests():
    """Run all tests"""
    print("Running Enhanced GIN Drug Encoder Tests...")
    print("=" * 50)
    
    try:
        test_configurable_mlp_block()
        test_residual_gin_layer()
        test_enhanced_gin_encoder()
        test_multiscale_gin_encoder()
        test_configurable_parameters()
        test_residual_connections()
        test_training_mode()
        
        print("=" * 50)
        print("✅ All tests passed successfully!")
        print("\nEnhanced GIN Drug Encoder Features Verified:")
        print("- ✓ Configurable MLP layers with various activation functions")
        print("- ✓ Residual connections for training stability")
        print("- ✓ Advanced batch normalization and dropout regularization")
        print("- ✓ Multiple pooling strategies (mean, max, dual, attention)")
        print("- ✓ Multi-scale feature extraction")
        print("- ✓ Node-level embedding extraction")
        print("- ✓ Layer-wise output analysis")
        print("- ✓ Proper gradient computation in training mode")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Demo script for the Enhanced GIN Drug Encoder
Shows the key features and capabilities of the new encoder
"""

import torch
import numpy as np
from torch_geometric.data import Data, Batch
from core import EnhancedGINDrugEncoder, MultiScaleGINEncoder, create_dta_model


def create_demo_molecular_graphs(batch_size=4):
    """Create demo molecular graphs with realistic properties"""
    graphs = []
    
    # Simulate different molecule sizes (small to large)
    node_counts = [12, 18, 25, 35]  # Different molecule sizes
    
    for i in range(batch_size):
        num_nodes = node_counts[i] if i < len(node_counts) else np.random.randint(10, 40)
        
        # More realistic edge count (molecules are typically sparse)
        num_edges = int(num_nodes * 1.5)  # Realistic edge-to-node ratio
        
        # Node features (78-dimensional as per RDKit standard)
        x = torch.randn(num_nodes, 78)
        
        # Create more realistic edge connectivity
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Make edges undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        graph = Data(x=x, edge_index=edge_index)
        graphs.append(graph)
    
    return Batch.from_data_list(graphs)


def demo_enhanced_gin_features():
    """Demonstrate the key features of the Enhanced GIN Drug Encoder"""
    print("Enhanced GIN Drug Encoder Demo")
    print("=" * 50)
    
    # Create encoder with advanced configuration
    print("\n1. Creating Enhanced GIN Encoder with advanced features...")
    encoder = EnhancedGINDrugEncoder(
        node_features=78,
        hidden_dim=128,
        num_layers=5,
        output_dim=128,
        mlp_hidden_dims=[256, 128],  # Configurable MLP layers
        activation='gelu',           # Advanced activation
        dropout=0.1,
        use_batch_norm=True,
        residual_connections=True,   # Residual connections for stability
        pooling_strategy='dual',     # Mean + Max pooling
        eps=0.0,
        train_eps=False
    )
    
    print(f"   - Hidden dimension: {encoder.hidden_dim}")
    print(f"   - Number of layers: {encoder.num_layers}")
    print(f"   - MLP configuration: {[256, 128]}")
    print(f"   - Activation function: GELU")
    print(f"   - Pooling strategy: Dual (mean + max)")
    print(f"   - Residual connections: Enabled")
    print(f"   - Output dimension: {encoder.output_dim}")
    
    # Create sample molecular graphs
    batch_data = create_demo_molecular_graphs(batch_size=4)
    
    print(f"\n2. Processing batch of molecular graphs...")
    print(f"   - Batch size: {batch_data.batch.max().item() + 1}")
    print(f"   - Total nodes: {batch_data.x.shape[0]}")
    print(f"   - Total edges: {batch_data.edge_index.shape[1]}")
    print(f"   - Node feature dimension: {batch_data.x.shape[1]}")
    
    # Forward pass
    encoder.eval()
    with torch.no_grad():
        molecular_features = encoder(batch_data)
        node_embeddings = encoder.get_node_embeddings(batch_data)
        layer_outputs = encoder.get_layer_outputs(batch_data)
    
    print(f"   - Output molecular features shape: {molecular_features.shape}")
    print(f"   - Node embeddings shape: {node_embeddings.shape}")
    print(f"   - Number of layer outputs: {len(layer_outputs)}")
    
    # Analyze layer-wise feature evolution
    print(f"\n3. Layer-wise Feature Analysis:")
    for i, layer_output in enumerate(layer_outputs):
        mean_activation = layer_output.mean().item()
        std_activation = layer_output.std().item()
        print(f"   - Layer {i}: Mean={mean_activation:.4f}, Std={std_activation:.4f}, Shape={layer_output.shape}")
    
    # Compare different pooling strategies
    print(f"\n4. Comparing Pooling Strategies...")
    
    pooling_strategies = ['mean', 'max', 'dual', 'attention']
    pooling_results = {}
    
    for pooling in pooling_strategies:
        encoder_variant = EnhancedGINDrugEncoder(
            node_features=78,
            hidden_dim=128,
            num_layers=3,
            output_dim=128,
            pooling_strategy=pooling,
            attention_heads=4 if pooling == 'attention' else 1
        )
        
        encoder_variant.eval()
        with torch.no_grad():
            features = encoder_variant(batch_data)
        
        pooling_results[pooling] = features
        
        # Analyze feature statistics
        mean_val = features.mean().item()
        std_val = features.std().item()
        print(f"   - {pooling.capitalize()} pooling: Mean={mean_val:.4f}, Std={std_val:.4f}")
    
    # Multi-scale encoder demonstration
    print(f"\n5. Multi-scale GIN Encoder...")
    
    multiscale_encoder = MultiScaleGINEncoder(
        node_features=78,
        hidden_dims=[64, 128, 256],  # Different scales
        num_layers_per_scale=2,
        output_dim=128,
        dropout=0.1
    )
    
    multiscale_encoder.eval()
    with torch.no_grad():
        multiscale_features = multiscale_encoder(batch_data)
    
    print(f"   - Multi-scale encoder scales: {[64, 128, 256]}")
    print(f"   - Layers per scale: 2")
    print(f"   - Output features shape: {multiscale_features.shape}")
    
    # Parameter analysis
    print(f"\n6. Model Complexity Analysis...")
    
    configurations = [
        {
            'name': 'Lightweight',
            'config': {
                'hidden_dim': 64,
                'num_layers': 2,
                'output_dim': 64,
                'mlp_hidden_dims': [64]
            }
        },
        {
            'name': 'Standard',
            'config': {
                'hidden_dim': 128,
                'num_layers': 4,
                'output_dim': 128,
                'mlp_hidden_dims': [128, 128]
            }
        },
        {
            'name': 'High-capacity',
            'config': {
                'hidden_dim': 256,
                'num_layers': 6,
                'output_dim': 256,
                'mlp_hidden_dims': [512, 256]
            }
        }
    ]
    
    for config_info in configurations:
        name = config_info['name']
        config = config_info['config']
        
        encoder_variant = EnhancedGINDrugEncoder(node_features=78, **config)
        
        total_params = sum(p.numel() for p in encoder_variant.parameters())
        trainable_params = sum(p.numel() for p in encoder_variant.parameters() if p.requires_grad)
        
        print(f"   - {name}: {total_params:,} total params, {trainable_params:,} trainable")
    
    # Integration with complete DTA model
    print(f"\n7. Integration with Complete DTA Model...")
    
    model_config = {
        'protein_encoder_type': 'cnn',
        'drug_encoder_type': 'gin',
        'use_fusion': True,
        'protein_config': {
            'embed_dim': 128,
            'num_filters': [64, 128, 256],
            'kernel_sizes': [3, 5, 7],
            'output_dim': 128
        },
        'drug_config': {
            'node_features': 78,
            'hidden_dim': 128,
            'num_layers': 4,
            'output_dim': 128,
            'mlp_hidden_dims': [256, 128],
            'activation': 'gelu',
            'pooling_strategy': 'dual',
            'residual_connections': True
        },
        'fusion_config': {
            'hidden_dim': 256,
            'num_heads': 8
        }
    }
    
    dta_model = create_dta_model(model_config)
    
    # Count parameters
    total_model_params = sum(p.numel() for p in dta_model.parameters())
    drug_encoder_params = sum(p.numel() for p in dta_model.drug_encoder.parameters())
    protein_encoder_params = sum(p.numel() for p in dta_model.protein_encoder.parameters())
    
    print(f"   - Complete DTA model: {total_model_params:,} parameters")
    print(f"   - Enhanced GIN encoder: {drug_encoder_params:,} parameters")
    print(f"   - Enhanced CNN encoder: {protein_encoder_params:,} parameters")
    print(f"   - GIN encoder ratio: {drug_encoder_params/total_model_params*100:.1f}%")
    print(f"   - CNN encoder ratio: {protein_encoder_params/total_model_params*100:.1f}%")
    
    # Test complete model prediction
    protein_tokens = torch.randint(1, 26, (4, 150))  # Mock protein sequences
    
    dta_model.eval()
    with torch.no_grad():
        affinity_predictions = dta_model(batch_data, protein_tokens)
    
    print(f"   - Affinity predictions shape: {affinity_predictions.shape}")
    print(f"   - Sample predictions: {affinity_predictions.flatten()[:3].tolist()}")
    
    print(f"\n✅ Enhanced GIN Drug Encoder Demo Complete!")
    print(f"\nKey Features Demonstrated:")
    print(f"- ✓ Configurable MLP layers with advanced activations")
    print(f"- ✓ Residual connections for training stability")
    print(f"- ✓ Multiple pooling strategies (mean, max, dual, attention)")
    print(f"- ✓ Multi-scale feature extraction")
    print(f"- ✓ Layer-wise feature analysis capabilities")
    print(f"- ✓ Node-level embedding extraction")
    print(f"- ✓ Scalable architecture (lightweight to high-capacity)")
    print(f"- ✓ Integration with complete DTA prediction system")


if __name__ == "__main__":
    demo_enhanced_gin_features()
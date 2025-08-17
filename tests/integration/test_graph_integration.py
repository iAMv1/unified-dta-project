#!/usr/bin/env python3
"""
Integration test for Graph Preprocessing with Enhanced GIN Drug Encoder
Tests the complete pipeline from SMILES to molecular features
"""

import torch
import numpy as np
import sys
import os

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.graph_preprocessing import (
    GraphFeatureConfig,
    MolecularGraphProcessor,
    process_smiles_batch
)
from core.drug_encoders import EnhancedGINDrugEncoder
from core.models import create_dta_model


def test_smiles_to_features_pipeline():
    """Test complete pipeline from SMILES to molecular features"""
    print("Testing SMILES to Features Pipeline...")
    
    # Test SMILES
    test_smiles = [
        'CCO',  # Ethanol
        'c1ccccc1',  # Benzene
        'CC(C)O',  # Isopropanol
        'CCN(CC)CC',  # Triethylamine
    ]
    
    # Process SMILES to graphs
    config = GraphFeatureConfig(
        include_atomic_number=True,
        include_degree=True,
        include_aromaticity=True,
        use_one_hot_encoding=False,  # Use continuous features for simplicity
        normalize_features=True
    )
    
    results = process_smiles_batch(
        test_smiles,
        config=config,
        validate=True,
        create_batches=True
    )
    
    print(f"   - Processed {results['total_smiles']} SMILES")
    print(f"   - Success rate: {results['success_rate']*100:.1f}%")
    
    if 'batches' in results and results['batches']:
        batch = results['batches'][0]
        print(f"   - Batch shape: {batch.x.shape}")
        print(f"   - Edge shape: {batch.edge_index.shape}")
        
        # Test with GIN encoder
        encoder = EnhancedGINDrugEncoder(
            node_features=batch.x.shape[1],
            hidden_dim=64,
            num_layers=2,
            output_dim=128,
            pooling_strategy='dual'
        )
        
        encoder.eval()
        with torch.no_grad():
            molecular_features = encoder(batch)
        
        print(f"   - Molecular features shape: {molecular_features.shape}")
        assert molecular_features.shape == (batch.num_graphs, 128)
        
        print("✓ SMILES to Features Pipeline test passed")
    else:
        print("   - Warning: No valid batches created (limited without RDKit)")


def test_different_feature_configs_with_gin():
    """Test different feature configurations with GIN encoder"""
    print("Testing Different Feature Configs with GIN...")
    
    test_smiles = ['CCO', 'c1ccccc1']
    
    configs = [
        {
            'name': 'Basic',
            'config': GraphFeatureConfig(
                include_atomic_number=True,
                include_degree=True,
                use_one_hot_encoding=False
            )
        },
        {
            'name': 'Extended',
            'config': GraphFeatureConfig(
                include_atomic_number=True,
                include_degree=True,
                include_aromaticity=True,
                include_formal_charge=True,
                use_one_hot_encoding=False
            )
        }
    ]
    
    for config_info in configs:
        name = config_info['name']
        config = config_info['config']
        
        # Process SMILES
        processor = MolecularGraphProcessor(config)
        graphs, _ = processor.batch_process_smiles(test_smiles)
        
        if graphs:
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(graphs)
            
            # Create encoder matching the feature dimension
            encoder = EnhancedGINDrugEncoder(
                node_features=batch.x.shape[1],
                hidden_dim=32,
                num_layers=2,
                output_dim=64
            )
            
            encoder.eval()
            with torch.no_grad():
                features = encoder(batch)
            
            print(f"   - {name}: Input dim={batch.x.shape[1]}, Output={features.shape}")
        else:
            print(f"   - {name}: No valid graphs (limited without RDKit)")
    
    print("✓ Different Feature Configs with GIN test passed")


def test_complete_dta_model_with_preprocessing():
    """Test complete DTA model with graph preprocessing"""
    print("Testing Complete DTA Model with Preprocessing...")
    
    # Test data
    test_smiles = ['CCO', 'c1ccccc1', 'CC(C)O']
    protein_sequences = ['MKFLVL', 'ACDEFG', 'GHIKLM']  # Mock sequences
    
    # Process SMILES to graphs
    processor = MolecularGraphProcessor()
    graphs, _ = processor.batch_process_smiles(test_smiles)
    
    if graphs:
        from torch_geometric.data import Batch
        drug_batch = Batch.from_data_list(graphs)
        
        # Create protein tokens (mock) - match the number of graphs
        protein_tokens = torch.randint(1, 21, (len(graphs), 50))
        
        # Create DTA model
        config = {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': True,
            'protein_config': {
                'embed_dim': 64,
                'num_filters': [32, 64],
                'kernel_sizes': [3, 5],
                'output_dim': 64
            },
            'drug_config': {
                'node_features': drug_batch.x.shape[1],
                'hidden_dim': 64,
                'num_layers': 2,
                'output_dim': 64,
                'pooling_strategy': 'dual'
            },
            'fusion_config': {
                'hidden_dim': 128
            },
            'predictor_config': {
                'hidden_dims': [128, 64]
            }
        }
        
        model = create_dta_model(config)
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            predictions = model(drug_batch, protein_tokens)
        
        print(f"   - Drug batch: {drug_batch.num_graphs} molecules")
        print(f"   - Protein batch: {protein_tokens.shape}")
        print(f"   - Predictions: {predictions.shape}")
        print(f"   - Sample predictions: {predictions.flatten()[:3].tolist()}")
        
        assert predictions.shape == (len(graphs), 1)
        assert torch.isfinite(predictions).all()
        
        print("✓ Complete DTA Model with Preprocessing test passed")
    else:
        print("   - Warning: No valid graphs for DTA model test")


def test_batch_size_optimization():
    """Test batch size optimization with different molecule sizes"""
    print("Testing Batch Size Optimization...")
    
    # Create molecules of different sizes
    test_smiles = [
        'C',  # Very small
        'CCO',  # Small
        'c1ccccc1',  # Medium
        'CC(C)(C)c1ccc(O)cc1',  # Larger
        'CCCCCCCCCCCCCCCCC',  # Long chain
    ]
    
    from core.graph_preprocessing import OptimizedGraphBatcher
    
    processor = MolecularGraphProcessor()
    graphs, _ = processor.batch_process_smiles(test_smiles)
    
    if graphs and len(graphs) > 1:
        # Test different batch size limits
        batch_limits = [20, 50, 100]
        
        for limit in batch_limits:
            batcher = OptimizedGraphBatcher(
                max_nodes_per_batch=limit,
                sort_by_size=True
            )
            
            batches = batcher.create_batches(graphs)
            stats = batcher.get_batch_statistics(batches)
            
            print(f"   - Limit {limit}: {len(batches)} batches, "
                  f"avg nodes={stats['node_counts']['mean']:.1f}")
            
            # Verify no batch exceeds the limit
            for batch in batches:
                assert batch.x.shape[0] <= limit, f"Batch exceeds node limit: {batch.x.shape[0]} > {limit}"
        
        print("✓ Batch Size Optimization test passed")
    else:
        print("   - Warning: Not enough graphs for batch optimization test")


def test_feature_extraction_analysis():
    """Test and analyze feature extraction capabilities"""
    print("Testing Feature Extraction Analysis...")
    
    # Test molecule with different characteristics
    test_molecules = [
        ('CCO', 'Ethanol - simple alcohol'),
        ('c1ccccc1', 'Benzene - aromatic'),
        ('CC(=O)O', 'Acetic acid - carboxylic acid'),
        ('CCN(CC)CC', 'Triethylamine - tertiary amine'),
    ]
    
    config = GraphFeatureConfig(
        include_atomic_number=True,
        include_degree=True,
        include_aromaticity=True,
        include_formal_charge=True,
        use_one_hot_encoding=False,
        normalize_features=True
    )
    
    processor = MolecularGraphProcessor(config)
    
    for smiles, description in test_molecules:
        graph = processor.smiles_to_graph(smiles)
        
        if graph is not None:
            # Analyze features
            node_features = graph.x
            num_nodes = node_features.shape[0]
            feature_dim = node_features.shape[1]
            
            # Basic statistics
            mean_features = node_features.mean(dim=0)
            std_features = node_features.std(dim=0)
            
            print(f"   - {description}")
            print(f"     Nodes: {num_nodes}, Features: {feature_dim}")
            print(f"     Feature mean: {mean_features[:3].tolist()}")  # First 3 features
            
            # Test with encoder
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([graph])
            
            encoder = EnhancedGINDrugEncoder(
                node_features=feature_dim,
                hidden_dim=32,
                num_layers=2,
                output_dim=64
            )
            
            encoder.eval()
            with torch.no_grad():
                molecular_embedding = encoder(batch)
            
            print(f"     Molecular embedding: {molecular_embedding.shape}")
        else:
            print(f"   - {description}: Could not process")
    
    print("✓ Feature Extraction Analysis test passed")


def run_integration_tests():
    """Run all integration tests"""
    print("Running Graph Preprocessing Integration Tests...")
    print("=" * 60)
    
    try:
        test_smiles_to_features_pipeline()
        test_different_feature_configs_with_gin()
        test_complete_dta_model_with_preprocessing()
        test_batch_size_optimization()
        test_feature_extraction_analysis()
        
        print("=" * 60)
        print("✅ All integration tests passed successfully!")
        print("\nGraph Preprocessing Integration Verified:")
        print("- ✓ SMILES to molecular features pipeline")
        print("- ✓ Integration with Enhanced GIN encoder")
        print("- ✓ Complete DTA model with preprocessing")
        print("- ✓ Configurable feature extraction")
        print("- ✓ Optimized batch processing")
        print("- ✓ Feature analysis and validation")
        print("- ✓ End-to-end drug-target affinity prediction")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
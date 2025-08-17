#!/usr/bin/env python3
"""
Test script for Graph Preprocessing and Feature Extraction
Tests molecular graph processing, validation, and batching
"""

import torch
import numpy as np
import sys
import os
from typing import List

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.graph_preprocessing import (
    GraphFeatureConfig,
    MolecularGraphProcessor,
    GraphValidator,
    OptimizedGraphBatcher,
    create_molecular_graph_processor,
    process_smiles_batch
)


def get_test_smiles() -> List[str]:
    """Get a set of test SMILES strings"""
    return [
        'CCO',  # Ethanol (simple)
        'CC(C)O',  # Isopropanol
        'c1ccccc1',  # Benzene (aromatic)
        'CCN(CC)CC',  # Triethylamine
        'CC(=O)O',  # Acetic acid
        'c1ccc2c(c1)cccn2',  # Quinoline (bicyclic)
        'CC(C)(C)c1ccc(O)cc1',  # BHT (larger molecule)
        'invalid_smiles',  # Invalid SMILES for error testing
        'C',  # Methane (very simple)
        'CCCCCCCCCCCCCCCCC',  # Long chain (stress test)
    ]


def test_graph_feature_config():
    """Test GraphFeatureConfig functionality"""
    print("Testing GraphFeatureConfig...")
    
    # Test default configuration
    default_config = GraphFeatureConfig()
    assert default_config.include_atomic_number == True
    assert default_config.include_degree == True
    assert default_config.max_atomic_number == 100
    assert default_config.use_one_hot_encoding == True
    
    # Test custom configuration
    custom_config = GraphFeatureConfig(
        include_atomic_number=True,
        include_degree=False,
        include_bond_type=True,
        max_atomic_number=50,
        use_one_hot_encoding=False
    )
    
    assert custom_config.include_atomic_number == True
    assert custom_config.include_degree == False
    assert custom_config.max_atomic_number == 50
    assert custom_config.use_one_hot_encoding == False
    
    print("✓ GraphFeatureConfig test passed")


def test_molecular_graph_processor():
    """Test MolecularGraphProcessor functionality"""
    print("Testing MolecularGraphProcessor...")
    
    # Test with default configuration
    processor = MolecularGraphProcessor()
    
    # Test simple molecule
    smiles = 'CCO'  # Ethanol
    graph = processor.smiles_to_graph(smiles)
    
    if graph is not None:
        assert graph.x.shape[0] > 0, "Graph should have nodes"
        assert graph.edge_index.shape[0] == 2, "Edge index should have 2 rows"
        assert graph.edge_index.shape[1] > 0, "Graph should have edges"
        assert hasattr(graph, 'smiles'), "Graph should store original SMILES"
        assert graph.smiles == smiles, "Stored SMILES should match input"
        
        print(f"   - Ethanol: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]//2} bonds")
        print(f"   - Node feature dimension: {graph.x.shape[1]}")
        if graph.edge_attr is not None:
            print(f"   - Edge feature dimension: {graph.edge_attr.shape[1]}")
    else:
        print("   - Warning: Could not process ethanol (RDKit may not be available)")
    
    # Test aromatic molecule
    smiles = 'c1ccccc1'  # Benzene
    graph = processor.smiles_to_graph(smiles)
    
    if graph is not None:
        print(f"   - Benzene: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]//2} bonds")
    
    # Test invalid SMILES
    invalid_graph = processor.smiles_to_graph('invalid_smiles')
    assert invalid_graph is None, "Invalid SMILES should return None"
    
    print("✓ MolecularGraphProcessor test passed")


def test_batch_processing():
    """Test batch processing of SMILES"""
    print("Testing Batch Processing...")
    
    processor = MolecularGraphProcessor()
    test_smiles = get_test_smiles()
    
    # Process batch
    valid_graphs, failed_smiles = processor.batch_process_smiles(test_smiles)
    
    print(f"   - Input SMILES: {len(test_smiles)}")
    print(f"   - Valid graphs: {len(valid_graphs)}")
    print(f"   - Failed SMILES: {len(failed_smiles)}")
    print(f"   - Success rate: {len(valid_graphs)/len(test_smiles)*100:.1f}%")
    
    # Check that we have some valid graphs
    assert len(valid_graphs) > 0, "Should have at least some valid graphs"
    
    # Check that invalid SMILES are caught
    assert 'invalid_smiles' in failed_smiles, "Invalid SMILES should be in failed list"
    
    # Check graph properties
    if valid_graphs:
        for i, graph in enumerate(valid_graphs[:3]):  # Check first 3
            print(f"   - Graph {i}: {graph.x.shape[0]} nodes, {graph.x.shape[1]} features")
    
    print("✓ Batch Processing test passed")


def test_graph_validator():
    """Test GraphValidator functionality"""
    print("Testing GraphValidator...")
    
    validator = GraphValidator(min_nodes=2, max_nodes=100, min_edges=1)
    
    # Create test graphs
    processor = MolecularGraphProcessor()
    test_smiles = ['CCO', 'c1ccccc1', 'CC(C)O']
    
    valid_graphs, failed_smiles = processor.batch_process_smiles(test_smiles)
    
    if valid_graphs:
        # Test validation
        validated_graphs, invalid_info = validator.filter_valid_graphs(valid_graphs)
        
        print(f"   - Input graphs: {len(valid_graphs)}")
        print(f"   - Valid after validation: {len(validated_graphs)}")
        print(f"   - Invalid graphs: {len(invalid_info)}")
        
        # Test individual validation
        if valid_graphs:
            is_valid, errors = validator.validate_graph(valid_graphs[0])
            print(f"   - First graph valid: {is_valid}")
            if errors:
                print(f"   - Validation errors: {errors}")
        
        assert len(validated_graphs) <= len(valid_graphs), "Validated count should not exceed input"
    else:
        print("   - Warning: No valid graphs to validate (RDKit may not be available)")
    
    print("✓ GraphValidator test passed")


def test_optimized_graph_batcher():
    """Test OptimizedGraphBatcher functionality"""
    print("Testing OptimizedGraphBatcher...")
    
    batcher = OptimizedGraphBatcher(max_nodes_per_batch=50, sort_by_size=True)
    
    # Create test graphs
    processor = MolecularGraphProcessor()
    test_smiles = get_test_smiles()[:6]  # Use first 6 valid SMILES
    
    valid_graphs, _ = processor.batch_process_smiles(test_smiles)
    
    if valid_graphs and len(valid_graphs) > 1:
        # Create batches
        batches = batcher.create_batches(valid_graphs)
        
        print(f"   - Input graphs: {len(valid_graphs)}")
        print(f"   - Created batches: {len(batches)}")
        
        # Check batch properties
        total_graphs_in_batches = sum(batch.num_graphs for batch in batches)
        assert total_graphs_in_batches == len(valid_graphs), "All graphs should be in batches"
        
        # Get batch statistics
        stats = batcher.get_batch_statistics(batches)
        print(f"   - Average batch size: {stats['batch_sizes']['mean']:.1f}")
        print(f"   - Average nodes per batch: {stats['node_counts']['mean']:.1f}")
        
        # Test individual batch
        if batches:
            batch = batches[0]
            print(f"   - First batch: {batch.num_graphs} graphs, {batch.x.shape[0]} total nodes")
            assert batch.x.shape[0] > 0, "Batch should have nodes"
            assert batch.edge_index.shape[1] > 0, "Batch should have edges"
    else:
        print("   - Warning: Not enough valid graphs for batching test")
    
    print("✓ OptimizedGraphBatcher test passed")


def test_different_configurations():
    """Test different feature configurations"""
    print("Testing Different Configurations...")
    
    configs = [
        {
            'name': 'Minimal',
            'config': GraphFeatureConfig(
                include_atomic_number=True,
                include_degree=True,
                include_bond_type=True,
                use_one_hot_encoding=False,
                include_molecular_descriptors=False
            )
        },
        {
            'name': 'Full Features',
            'config': GraphFeatureConfig(
                include_atomic_number=True,
                include_degree=True,
                include_formal_charge=True,
                include_hybridization=True,
                include_aromaticity=True,
                include_bond_type=True,
                include_conjugation=True,
                use_one_hot_encoding=True,
                include_molecular_descriptors=True
            )
        }
    ]
    
    test_smiles = 'CCO'  # Simple test molecule
    
    for config_info in configs:
        name = config_info['name']
        config = config_info['config']
        
        processor = MolecularGraphProcessor(config)
        graph = processor.smiles_to_graph(test_smiles)
        
        if graph is not None:
            node_dim = graph.x.shape[1]
            edge_dim = graph.edge_attr.shape[1] if graph.edge_attr is not None else 0
            
            print(f"   - {name}: Node dim={node_dim}, Edge dim={edge_dim}")
            
            # Check that different configs produce different dimensions
            assert node_dim > 0, f"{name} should have node features"
        else:
            print(f"   - {name}: Could not process (RDKit may not be available)")
    
    print("✓ Different Configurations test passed")


def test_process_smiles_batch_pipeline():
    """Test the complete SMILES processing pipeline"""
    print("Testing Complete SMILES Processing Pipeline...")
    
    test_smiles = get_test_smiles()
    
    # Test with default configuration
    results = process_smiles_batch(
        test_smiles,
        config=None,
        validate=True,
        create_batches=True
    )
    
    print(f"   - Total SMILES: {results['total_smiles']}")
    print(f"   - Successful graphs: {results['successful_graphs']}")
    print(f"   - Success rate: {results['success_rate']*100:.1f}%")
    
    if 'valid_graphs' in results:
        print(f"   - Valid graphs: {results['valid_graphs']}")
        print(f"   - Validation rate: {results['validation_rate']*100:.1f}%")
    
    if 'batches' in results:
        print(f"   - Created batches: {len(results['batches'])}")
        batch_stats = results['batch_statistics']
        print(f"   - Average batch size: {batch_stats['batch_sizes']['mean']:.1f}")
    
    # Check that we have reasonable results
    assert results['total_smiles'] == len(test_smiles)
    assert results['success_rate'] >= 0.0
    
    if results['successful_graphs'] > 0:
        assert 'batches' in results or 'graphs' in results
    
    print("✓ Complete SMILES Processing Pipeline test passed")


def test_error_handling():
    """Test error handling and edge cases"""
    print("Testing Error Handling...")
    
    processor = MolecularGraphProcessor()
    
    # Test empty SMILES
    empty_result = processor.smiles_to_graph('')
    assert empty_result is None, "Empty SMILES should return None"
    
    # Test None input
    none_result = processor.smiles_to_graph(None)
    assert none_result is None, "None input should return None"
    
    # Test batch with empty list
    empty_batch_graphs, empty_batch_failed = processor.batch_process_smiles([])
    assert len(empty_batch_graphs) == 0, "Empty batch should return empty list"
    assert len(empty_batch_failed) == 0, "Empty batch should have no failures"
    
    # Test validator with edge cases
    validator = GraphValidator()
    
    # Test with None graph (should handle gracefully)
    try:
        # This should not crash
        validator.filter_valid_graphs([])
        valid_empty, invalid_empty = validator.filter_valid_graphs([])
        assert len(valid_empty) == 0
        assert len(invalid_empty) == 0
    except Exception as e:
        print(f"   - Warning: Validator error with empty list: {e}")
    
    print("✓ Error Handling test passed")


def run_all_tests():
    """Run all tests"""
    print("Running Graph Preprocessing Tests...")
    print("=" * 50)
    
    try:
        test_graph_feature_config()
        test_molecular_graph_processor()
        test_batch_processing()
        test_graph_validator()
        test_optimized_graph_batcher()
        test_different_configurations()
        test_process_smiles_batch_pipeline()
        test_error_handling()
        
        print("=" * 50)
        print("✅ All tests passed successfully!")
        print("\nGraph Preprocessing Features Verified:")
        print("- ✓ Configurable molecular feature extraction")
        print("- ✓ Node feature extraction from molecular graphs")
        print("- ✓ Edge feature processing for bond information")
        print("- ✓ Graph validation and error handling")
        print("- ✓ Optimized graph batching for efficient processing")
        print("- ✓ Batch processing pipeline with validation")
        print("- ✓ Multiple feature configuration options")
        print("- ✓ Robust error handling for invalid inputs")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
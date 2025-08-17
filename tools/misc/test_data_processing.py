#!/usr/bin/env python3
"""
Test script for data processing pipeline
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import logging

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

def create_sample_data():
    """Create sample DTA data for testing"""
    sample_data = {
        'compound_iso_smiles': [
            'CCO',  # Ethanol
            'C1=CC=CC=C1',  # Benzene
            'CC(=O)O',  # Acetic acid
            'invalid_smiles',  # Invalid SMILES
            'C1=CC=C(C=C1)O'  # Phenol
        ],
        'target_sequence': [
            'ACDEFGHIKLMNPQRSTVWY',  # Valid protein sequence
            'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY',  # Longer sequence
            'ACDEFG',  # Short sequence
            'INVALID123',  # Invalid sequence
            'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY'  # Very long
        ],
        'affinity': [5.2, 6.8, 4.1, 7.5, 3.9]
    }
    
    return pd.DataFrame(sample_data)

def test_smiles_validation():
    """Test SMILES validation functionality"""
    print("Testing SMILES validation...")
    
    try:
        from core import SMILESValidator
        
        validator = SMILESValidator()
        test_smiles = ['CCO', 'C1=CC=CC=C1', 'invalid_smiles', '']
        
        results = validator.validate_batch(test_smiles)
        stats = validator.get_statistics()
        
        print(f"✓ SMILES validation results: {results}")
        print(f"✓ Validation statistics: {stats}")
        
        # Check that we have some valid and some invalid
        assert any(results), "Should have at least one valid SMILES"
        assert not all(results), "Should have at least one invalid SMILES"
        
        return True
    except Exception as e:
        print(f"✗ SMILES validation test failed: {e}")
        return False

def test_protein_processing():
    """Test protein sequence processing"""
    print("\nTesting protein processing...")
    
    try:
        from core import ProteinProcessor
        
        processor = ProteinProcessor(max_length=10)
        test_sequences = [
            'ACDEFGHIKLMNPQRSTVWY',  # Long sequence (should be truncated)
            'INVALID123',  # Invalid characters (should be cleaned)
            'SHORT'  # Short sequence (should be padded)
        ]
        
        for seq in test_sequences:
            tokens = processor.process_sequence(seq)
            print(f"  {seq[:20]}... → {len(tokens)} tokens")
            assert len(tokens) == 10, f"Expected 10 tokens, got {len(tokens)}"
        
        stats = processor.get_statistics()
        print(f"✓ Processing statistics: {stats}")
        
        return True
    except Exception as e:
        print(f"✗ Protein processing test failed: {e}")
        return False

def test_molecular_graph_conversion():
    """Test molecular graph conversion"""
    print("\nTesting molecular graph conversion...")
    
    try:
        from core import MolecularGraphConverter
        
        converter = MolecularGraphConverter()
        test_smiles = ['CCO', 'C1=CC=CC=C1', 'invalid_smiles']
        
        for smiles in test_smiles:
            graph = converter.smiles_to_graph(smiles)
            if graph is not None:
                print(f"  {smiles} → Graph with {graph.num_nodes} nodes")
                assert hasattr(graph, 'x'), "Graph should have node features"
                assert hasattr(graph, 'edge_index'), "Graph should have edge indices"
            else:
                print(f"  {smiles} → Conversion failed (expected for invalid SMILES)")
        
        stats = converter.get_statistics()
        print(f"✓ Conversion statistics: {stats}")
        
        return True
    except Exception as e:
        print(f"✗ Molecular graph conversion test failed: {e}")
        return False

def test_data_validation():
    """Test comprehensive data validation"""
    print("\nTesting data validation...")
    
    try:
        from core import DataValidator
        
        validator = DataValidator()
        sample_df = create_sample_data()
        
        # Test individual sample validation
        sample_result = validator.validate_dta_sample('CCO', 'ACDEFG', 5.2)
        print(f"✓ Sample validation result: {sample_result}")
        assert sample_result['valid'], "Valid sample should pass validation"
        
        # Test dataset validation
        dataset_stats = validator.validate_dataset(sample_df)
        print(f"✓ Dataset validation statistics:")
        for key, value in dataset_stats.items():
            if key != 'validation_errors':
                print(f"    {key}: {value}")
        
        assert dataset_stats['total_samples'] == 5, "Should process 5 samples"
        
        return True
    except Exception as e:
        print(f"✗ Data validation test failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading and preprocessing"""
    print("\nTesting dataset loading...")
    
    try:
        from core import load_dta_dataset, preprocess_dta_dataset
        
        # Create temporary CSV file
        sample_df = create_sample_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test loading
            loaded_df, validation_stats = load_dta_dataset(temp_file, validate_data=True)
            print(f"✓ Loaded {len(loaded_df)} samples")
            print(f"✓ Validation stats: {validation_stats['valid_samples']}/{validation_stats['total_samples']} valid")
            
            # Test preprocessing
            processed_df = preprocess_dta_dataset(loaded_df, remove_invalid=True)
            print(f"✓ Preprocessed to {len(processed_df)} samples")
            
            assert len(processed_df) <= len(loaded_df), "Preprocessing should not increase sample count"
            
        finally:
            # Clean up temporary file
            Path(temp_file).unlink()
        
        return True
    except Exception as e:
        print(f"✗ Dataset loading test failed: {e}")
        return False

def test_dta_dataset():
    """Test DTADataset class"""
    print("\nTesting DTADataset class...")
    
    try:
        from core import DTADataset
        
        # Create sample data
        sample_df = create_sample_data()
        
        # Filter to only valid samples for this test
        valid_df = sample_df[sample_df['compound_iso_smiles'].isin(['CCO', 'C1=CC=CC=C1', 'CC(=O)O'])].copy()
        
        # Create dataset
        dataset = DTADataset(
            data_path=valid_df,
            max_protein_length=50,
            preprocess=True,
            dataset_name="test_dataset"
        )
        
        print(f"✓ Created dataset with {len(dataset)} samples")
        
        # Test getting a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample structure: {type(sample)}")
            print(f"    SMILES: {sample.smiles}")
            print(f"    Protein length: {len(sample.protein_sequence)}")
            print(f"    Affinity: {sample.affinity}")
            
            if sample.graph is not None:
                print(f"    Graph nodes: {sample.graph.num_nodes}")
            if sample.protein_tokens is not None:
                print(f"    Protein tokens shape: {sample.protein_tokens.shape}")
        
        # Test statistics
        stats = dataset.get_statistics()
        print(f"✓ Dataset statistics: {stats['num_samples']} samples")
        
        return True
    except Exception as e:
        print(f"✗ DTADataset test failed: {e}")
        return False

def test_data_loader():
    """Test DTADataLoader"""
    print("\nTesting DTADataLoader...")
    
    try:
        from core import DTADataset, DTADataLoader
        
        # Create sample data with only valid SMILES
        valid_data = {
            'compound_iso_smiles': ['CCO', 'C1=CC=CC=C1', 'CC(=O)O'],
            'target_sequence': ['ACDEFG', 'ACDEFGH', 'ACDEFGHI'],
            'affinity': [5.2, 6.8, 4.1]
        }
        valid_df = pd.DataFrame(valid_data)
        
        # Create dataset
        dataset = DTADataset(
            data_path=valid_df,
            max_protein_length=20,
            dataset_name="test_loader"
        )
        
        if len(dataset) > 0:
            # Create data loader
            dataloader = DTADataLoader(dataset, batch_size=2, shuffle=False)
            
            print(f"✓ Created data loader with {len(dataloader)} batches")
            
            # Test one batch
            for batch in dataloader:
                print(f"✓ Batch structure:")
                print(f"    Batch size: {batch['batch_size']}")
                print(f"    Drug data type: {type(batch['drug_data'])}")
                print(f"    Protein tokens shape: {batch['protein_tokens'].shape}")
                print(f"    Affinities shape: {batch['affinities'].shape}")
                break  # Only test first batch
        else:
            print("⚠ No valid samples in dataset, skipping data loader test")
        
        return True
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def main():
    """Run all data processing tests"""
    print("=" * 60)
    print("UNIFIED DTA SYSTEM - DATA PROCESSING TESTS")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    tests = [
        test_smiles_validation,
        test_protein_processing,
        test_molecular_graph_conversion,
        test_data_validation,
        test_dataset_loading,
        test_dta_dataset,
        test_data_loader
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All data processing tests passed!")
        print("✓ SMILES validation working")
        print("✓ Protein processing working")
        print("✓ Molecular graph conversion working")
        print("✓ Data validation working")
        print("✓ Dataset loading working")
        print("✓ DTADataset class working")
        print("✓ Data loader working")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
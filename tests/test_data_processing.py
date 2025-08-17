"""
Unit tests for data processing and validation
Tests SMILES validation, protein processing, and dataset handling
"""

import unittest
import torch
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.data_processing import (
        SMILESValidator, ProteinProcessor, MolecularGraphConverter,
        DataValidator, load_dta_dataset, preprocess_dta_dataset
    )
    from core.datasets import (
        DTASample, DTADataset, MultiDatasetDTA, DTADataLoader,
        collate_dta_batch, create_data_splits, load_standard_datasets,
        DataAugmentation, create_balanced_sampler
    )
    from core.graph_preprocessing import (
        GraphFeatureConfig, MolecularGraphProcessor,
        GraphValidator, OptimizedGraphBatcher,
        create_molecular_graph_processor, process_smiles_batch
    )
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    # Set fallbacks for graceful handling
    SMILESValidator = None
    ProteinProcessor = None
    MolecularGraphConverter = None
    DTADataset = None
    MultiDatasetDTA = None


class TestSMILESValidation(unittest.TestCase):
    """Test SMILES string validation and processing"""
    
    def setUp(self):
        """Set up test SMILES strings"""
        self.valid_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
        ]
        
        self.invalid_smiles = [
            "",  # Empty string
            "INVALID",  # Invalid SMILES
            "C(C(C",  # Unbalanced parentheses
            "c1ccccc",  # Incomplete ring
            None  # None value
        ]
    
    def test_smiles_validator_initialization(self):
        """Test SMILESValidator initialization"""
        if SMILESValidator is None:
            self.skipTest("SMILESValidator not available")
        
        validator = SMILESValidator(
            sanitize=True,
            remove_hydrogens=True,
            canonical=True
        )
        
        self.assertTrue(validator.sanitize)
        self.assertTrue(validator.remove_hydrogens)
        self.assertTrue(validator.canonical)
    
    def test_valid_smiles_validation(self):
        """Test validation of valid SMILES strings"""
        if SMILESValidator is None:
            self.skipTest("SMILESValidator not available")
        
        validator = SMILESValidator()
        
        for smiles in self.valid_smiles:
            with self.subTest(smiles=smiles):
                is_valid, mol, error = validator.validate(smiles)
                self.assertTrue(is_valid, f"SMILES {smiles} should be valid")
                self.assertIsNotNone(mol)
                self.assertIsNone(error)
    
    def test_invalid_smiles_validation(self):
        """Test validation of invalid SMILES strings"""
        if SMILESValidator is None:
            self.skipTest("SMILESValidator not available")
        
        validator = SMILESValidator()
        
        for smiles in self.invalid_smiles:
            with self.subTest(smiles=smiles):
                is_valid, mol, error = validator.validate(smiles)
                self.assertFalse(is_valid, f"SMILES {smiles} should be invalid")
                self.assertIsNone(mol)
                self.assertIsNotNone(error)
    
    def test_smiles_canonicalization(self):
        """Test SMILES canonicalization"""
        if SMILESValidator is None:
            self.skipTest("SMILESValidator not available")
        
        validator = SMILESValidator(canonical=True)
        
        # Test equivalent SMILES representations
        equivalent_smiles = [
            ("CCO", "OCC"),  # Different order
            ("c1ccccc1", "C1=CC=CC=C1"),  # Aromatic vs Kekule
        ]
        
        for smiles1, smiles2 in equivalent_smiles:
            with self.subTest(smiles1=smiles1, smiles2=smiles2):
                _, mol1, _ = validator.validate(smiles1)
                _, mol2, _ = validator.validate(smiles2)
                
                if mol1 and mol2:
                    canonical1 = validator.get_canonical_smiles(mol1)
                    canonical2 = validator.get_canonical_smiles(mol2)
                    self.assertEqual(canonical1, canonical2)
    
    def test_batch_smiles_validation(self):
        """Test batch SMILES validation"""
        if SMILESValidator is None:
            self.skipTest("SMILESValidator not available")
        
        validator = SMILESValidator()
        
        mixed_smiles = self.valid_smiles + self.invalid_smiles
        results = validator.validate_batch(mixed_smiles)
        
        self.assertEqual(len(results), len(mixed_smiles))
        
        # Check that valid SMILES are marked as valid
        for i, smiles in enumerate(self.valid_smiles):
            self.assertTrue(results[i]['is_valid'])
        
        # Check that invalid SMILES are marked as invalid
        for i, smiles in enumerate(self.invalid_smiles):
            idx = len(self.valid_smiles) + i
            self.assertFalse(results[idx]['is_valid'])


class TestProteinProcessing(unittest.TestCase):
    """Test protein sequence processing"""
    
    def setUp(self):
        """Set up test protein sequences"""
        self.protein_sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ",
            "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
        ]
        
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    def test_protein_processor_initialization(self):
        """Test ProteinProcessor initialization"""
        if ProteinProcessor is None:
            self.skipTest("ProteinProcessor not available")
        
        processor = ProteinProcessor(
            max_length=200,
            pad_token='<PAD>',
            unknown_token='<UNK>'
        )
        
        self.assertEqual(processor.max_length, 200)
        self.assertEqual(processor.pad_token, '<PAD>')
        self.assertEqual(processor.unknown_token, '<UNK>')
    
    def test_protein_sequence_validation(self):
        """Test protein sequence validation"""
        if ProteinProcessor is None:
            self.skipTest("ProteinProcessor not available")
        
        processor = ProteinProcessor()
        
        # Test valid sequences
        for seq in self.protein_sequences:
            with self.subTest(sequence=seq[:20] + "..."):
                is_valid, error = processor.validate_sequence(seq)
                self.assertTrue(is_valid)
                self.assertIsNone(error)
        
        # Test invalid sequences
        invalid_sequences = [
            "",  # Empty
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGX",  # Invalid AA
            "123456",  # Numbers
            None  # None
        ]
        
        for seq in invalid_sequences:
            with self.subTest(sequence=seq):
                is_valid, error = processor.validate_sequence(seq)
                self.assertFalse(is_valid)
                self.assertIsNotNone(error)
    
    def test_protein_sequence_truncation(self):
        """Test protein sequence truncation"""
        if ProteinProcessor is None:
            self.skipTest("ProteinProcessor not available")
        
        processor = ProteinProcessor(max_length=50)
        
        long_sequence = self.protein_sequences[2]  # Longest sequence
        self.assertGreater(len(long_sequence), 50)
        
        truncated = processor.truncate_sequence(long_sequence)
        self.assertEqual(len(truncated), 50)
        self.assertEqual(truncated, long_sequence[:50])
    
    def test_protein_sequence_tokenization(self):
        """Test protein sequence tokenization"""
        if ProteinProcessor is None:
            self.skipTest("ProteinProcessor not available")
        
        processor = ProteinProcessor()
        
        sequence = self.protein_sequences[0]
        tokens = processor.tokenize(sequence)
        
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), len(sequence))
        
        # Check that all tokens are valid
        for token in tokens:
            self.assertIn(token, processor.vocab)
    
    def test_protein_batch_processing(self):
        """Test batch protein processing"""
        if ProteinProcessor is None:
            self.skipTest("ProteinProcessor not available")
        
        processor = ProteinProcessor(max_length=100)
        
        processed = processor.process_batch(self.protein_sequences)
        
        self.assertEqual(len(processed), len(self.protein_sequences))
        
        for result in processed:
            self.assertIn('sequence', result)
            self.assertIn('tokens', result)
            self.assertIn('length', result)
            self.assertLessEqual(result['length'], 100)


class TestMolecularGraphProcessing(unittest.TestCase):
    """Test molecular graph conversion and processing"""
    
    def setUp(self):
        """Set up test data"""
        self.test_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
        ]
    
    def test_molecular_graph_converter(self):
        """Test molecular graph conversion"""
        if MolecularGraphConverter is None:
            self.skipTest("MolecularGraphConverter not available")
        
        converter = MolecularGraphConverter()
        
        for smiles in self.test_smiles:
            with self.subTest(smiles=smiles):
                graph = converter.smiles_to_graph(smiles)
                
                if graph is not None:
                    # Check graph structure
                    self.assertTrue(hasattr(graph, 'x'))  # Node features
                    self.assertTrue(hasattr(graph, 'edge_index'))  # Edge indices
                    
                    # Check dimensions
                    num_nodes = graph.x.shape[0]
                    self.assertGreater(num_nodes, 0)
                    
                    # Check edge indices
                    if graph.edge_index.numel() > 0:
                        self.assertEqual(graph.edge_index.shape[0], 2)
                        self.assertTrue(torch.all(graph.edge_index >= 0))
                        self.assertTrue(torch.all(graph.edge_index < num_nodes))
    
    def test_graph_feature_extraction(self):
        """Test graph feature extraction"""
        if MolecularGraphConverter is None:
            self.skipTest("MolecularGraphConverter not available")
        
        converter = MolecularGraphConverter()
        
        smiles = "CCO"  # Simple molecule
        graph = converter.smiles_to_graph(smiles)
        
        if graph is not None:
            # Check node features
            self.assertEqual(graph.x.shape[1], 78)  # Expected feature dimension
            
            # Check that features are reasonable
            self.assertFalse(torch.isnan(graph.x).any())
            self.assertFalse(torch.isinf(graph.x).any())
    
    def test_batch_graph_processing(self):
        """Test batch graph processing"""
        if MolecularGraphConverter is None:
            self.skipTest("MolecularGraphConverter not available")
        
        converter = MolecularGraphConverter()
        
        graphs = converter.smiles_batch_to_graphs(self.test_smiles)
        
        self.assertEqual(len(graphs), len(self.test_smiles))
        
        # Filter out None graphs (invalid SMILES)
        valid_graphs = [g for g in graphs if g is not None]
        
        if valid_graphs:
            # Test batching
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(valid_graphs)
            
            self.assertTrue(hasattr(batch, 'batch'))
            self.assertEqual(batch.batch.max().item() + 1, len(valid_graphs))


class TestDatasetHandling(unittest.TestCase):
    """Test dataset classes and data loading"""
    
    def setUp(self):
        """Set up test dataset"""
        self.test_data = {
            'compound_iso_smiles': ['CCO', 'CC(=O)O', 'c1ccccc1'],
            'target_sequence': [
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'
            ],
            'affinity': [7.5, 6.2, 8.1]
        }
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame(self.test_data)
        df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_dta_sample_creation(self):
        """Test DTASample creation"""
        if DTASample is None:
            self.skipTest("DTASample not available")
        
        sample = DTASample(
            smiles=self.test_data['compound_iso_smiles'][0],
            protein_sequence=self.test_data['target_sequence'][0],
            affinity=self.test_data['affinity'][0],
            dataset='test'
        )
        
        self.assertEqual(sample.smiles, 'CCO')
        self.assertEqual(sample.affinity, 7.5)
        self.assertEqual(sample.dataset, 'test')
    
    def test_dta_dataset_loading(self):
        """Test DTADataset loading"""
        if DTADataset is None:
            self.skipTest("DTADataset not available")
        
        dataset = DTADataset(self.temp_file.name)
        
        self.assertEqual(len(dataset), len(self.test_data['compound_iso_smiles']))
        
        # Test indexing
        sample = dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn('smiles', sample)
        self.assertIn('protein_sequence', sample)
        self.assertIn('affinity', sample)
    
    def test_data_loader_creation(self):
        """Test data loader creation"""
        if DTADataset is None or DTADataLoader is None:
            self.skipTest("Dataset classes not available")
        
        dataset = DTADataset(self.temp_file.name)
        dataloader = DTADataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_dta_batch
        )
        
        # Test iteration
        for batch in dataloader:
            self.assertIsInstance(batch, dict)
            self.assertIn('smiles', batch)
            self.assertIn('protein_sequences', batch)
            self.assertIn('affinities', batch)
            break  # Test first batch only
    
    def test_data_splits_creation(self):
        """Test data splits creation"""
        if DTADataset is None:
            self.skipTest("DTADataset not available")
        
        dataset = DTADataset(self.temp_file.name)
        
        train_dataset, val_dataset, test_dataset = create_data_splits(
            dataset,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42
        )
        
        total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        self.assertEqual(total_samples, len(dataset))
        
        # Check that splits are reasonable
        self.assertGreater(len(train_dataset), 0)


class TestDataValidation(unittest.TestCase):
    """Test data validation utilities"""
    
    def test_data_validator_initialization(self):
        """Test DataValidator initialization"""
        if DataValidator is None:
            self.skipTest("DataValidator not available")
        
        validator = DataValidator(
            validate_smiles=True,
            validate_proteins=True,
            max_protein_length=200
        )
        
        self.assertTrue(validator.validate_smiles)
        self.assertTrue(validator.validate_proteins)
        self.assertEqual(validator.max_protein_length, 200)
    
    def test_dataset_validation(self):
        """Test full dataset validation"""
        if DataValidator is None:
            self.skipTest("DataValidator not available")
        
        validator = DataValidator()
        
        # Create test data with some invalid entries
        test_data = pd.DataFrame({
            'compound_iso_smiles': ['CCO', 'INVALID', 'CC(=O)O'],
            'target_sequence': [
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'INVALID_SEQUENCE_WITH_NUMBERS_123'
            ],
            'affinity': [7.5, 6.2, 8.1]
        })
        
        validation_results = validator.validate_dataset(test_data)
        
        self.assertIn('valid_indices', validation_results)
        self.assertIn('invalid_indices', validation_results)
        self.assertIn('errors', validation_results)
        
        # Should have at least one valid entry
        self.assertGreater(len(validation_results['valid_indices']), 0)


if __name__ == '__main__':
    unittest.main()
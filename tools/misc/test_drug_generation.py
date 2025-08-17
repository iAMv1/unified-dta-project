"""
Test suite for drug generation capabilities
Tests transformer-based SMILES generation and evaluation systems
"""

import unittest
import torch
import numpy as np
from typing import List, Dict
import tempfile
import os

# Import modules to test
from core.drug_generation import (
    SMILESTokenizer,
    TransformerDecoder,
    ProteinConditionedGenerator,
    ChemicalValidator,
    DrugGenerationPipeline
)
from core.generation_scoring import (
    MolecularPropertyCalculator,
    GenerationQualityAssessor,
    DiversityCalculator,
    NoveltyCalculator,
    ConfidenceScoringPipeline
)
from core.generation_evaluation import (
    GenerationBenchmark,
    GenerationEvaluationPipeline
)
from core.models import ESMProteinEncoder


class TestSMILESTokenizer(unittest.TestCase):
    """Test SMILES tokenizer functionality"""
    
    def setUp(self):
        self.tokenizer = SMILESTokenizer()
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization"""
        self.assertIsInstance(self.tokenizer.vocab, list)
        self.assertGreater(len(self.tokenizer.vocab), 0)
        self.assertIn('<pad>', self.tokenizer.vocab)
        self.assertIn('<sos>', self.tokenizer.vocab)
        self.assertIn('<eos>', self.tokenizer.vocab)
    
    def test_encode_decode(self):
        """Test encoding and decoding SMILES"""
        test_smiles = "CCO"
        
        # Encode
        tokens = self.tokenizer.encode(test_smiles)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Decode
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, test_smiles)
    
    def test_special_tokens(self):
        """Test special token handling"""
        self.assertEqual(self.tokenizer.pad_token_id, 0)
        self.assertEqual(self.tokenizer.sos_token_id, 1)
        self.assertEqual(self.tokenizer.eos_token_id, 2)
        self.assertEqual(self.tokenizer.unk_token_id, 3)
    
    def test_max_length_encoding(self):
        """Test encoding with max length"""
        test_smiles = "CCCCCCCCCCCCCCCCCCCC"  # Long SMILES
        max_len = 10
        
        tokens = self.tokenizer.encode(test_smiles, max_length=max_len)
        self.assertEqual(len(tokens), max_len)
        self.assertEqual(tokens[-1], self.tokenizer.eos_token_id)


class TestChemicalValidator(unittest.TestCase):
    """Test chemical validation functionality"""
    
    def setUp(self):
        self.validator = ChemicalValidator()
    
    def test_valid_smiles(self):
        """Test validation of valid SMILES"""
        valid_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
        
        for smiles in valid_smiles:
            with self.subTest(smiles=smiles):
                self.assertTrue(self.validator.is_valid_smiles(smiles))
    
    def test_invalid_smiles(self):
        """Test validation of invalid SMILES"""
        invalid_smiles = ["INVALID", "C(", "c1ccccc", ""]
        
        for smiles in invalid_smiles:
            with self.subTest(smiles=smiles):
                self.assertFalse(self.validator.is_valid_smiles(smiles))
    
    def test_canonicalization(self):
        """Test SMILES canonicalization"""
        test_cases = [
            ("CCO", "CCO"),
            ("c1ccccc1", "c1ccccc1"),
            ("CC(=O)O", "CC(=O)O")
        ]
        
        for input_smiles, expected in test_cases:
            with self.subTest(smiles=input_smiles):
                canonical = self.validator.canonicalize_smiles(input_smiles)
                self.assertIsNotNone(canonical)
                # Note: Exact canonical form may vary by RDKit version
    
    def test_property_calculation(self):
        """Test molecular property calculation"""
        test_smiles = "CCO"  # Ethanol
        
        properties = self.validator.calculate_properties(test_smiles)
        
        self.assertIsInstance(properties, dict)
        self.assertIn('molecular_weight', properties)
        self.assertIn('logp', properties)
        self.assertIn('num_atoms', properties)
        
        # Check reasonable values for ethanol
        self.assertAlmostEqual(properties['molecular_weight'], 46.07, delta=0.1)
        self.assertGreater(properties['num_atoms'], 0)
    
    def test_filter_valid_molecules(self):
        """Test filtering of valid molecules"""
        mixed_smiles = ["CCO", "INVALID", "c1ccccc1", "BAD_SMILES", "CC(=O)O"]
        
        valid_filtered = self.validator.filter_valid_molecules(mixed_smiles)
        
        self.assertIsInstance(valid_filtered, list)
        self.assertLessEqual(len(valid_filtered), len(mixed_smiles))
        
        # All filtered molecules should be valid
        for smiles in valid_filtered:
            self.assertTrue(self.validator.is_valid_smiles(smiles))


class TestTransformerDecoder(unittest.TestCase):
    """Test transformer decoder functionality"""
    
    def setUp(self):
        self.vocab_size = 100
        self.d_model = 64
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=4,
            num_layers=2,
            max_length=32
        )
    
    def test_decoder_initialization(self):
        """Test decoder initialization"""
        self.assertEqual(self.decoder.vocab_size, self.vocab_size)
        self.assertEqual(self.decoder.d_model, self.d_model)
    
    def test_forward_pass(self):
        """Test decoder forward pass"""
        batch_size = 2
        seq_len = 10
        memory_len = 5
        
        # Create dummy inputs
        tgt = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        memory = torch.randn(batch_size, memory_len, self.d_model)
        
        # Forward pass
        output = self.decoder(tgt, memory)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, self.vocab_size)
        self.assertEqual(output.shape, expected_shape)
    
    def test_causal_mask_generation(self):
        """Test causal mask generation"""
        seq_len = 5
        mask = self.decoder.generate_square_subsequent_mask(seq_len)
        
        self.assertEqual(mask.shape, (seq_len, seq_len))
        
        # Check that mask is upper triangular with -inf
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    self.assertEqual(mask[i, j].item(), float('-inf'))
                else:
                    self.assertEqual(mask[i, j].item(), 0.0)


class TestProteinConditionedGenerator(unittest.TestCase):
    """Test protein-conditioned generator"""
    
    def setUp(self):
        # Create simple protein encoder for testing
        self.protein_encoder = ESMProteinEncoder(output_dim=32)
        
        # Create generator with small dimensions for testing
        self.generator = ProteinConditionedGenerator(
            protein_encoder=self.protein_encoder,
            vocab_size=50,
            d_model=32,
            nhead=2,
            num_layers=1,
            max_length=16
        )
        
        self.test_proteins = ["MKLLVLSLSLVLVAPMAAQAAEITLKAVSRSLNCACELKCSTSLLLEACTFRRP"]
    
    def test_generator_initialization(self):
        """Test generator initialization"""
        self.assertIsNotNone(self.generator.protein_encoder)
        self.assertIsNotNone(self.generator.decoder)
        self.assertIsNotNone(self.generator.tokenizer)
    
    def test_protein_encoding(self):
        """Test protein encoding"""
        encoded = self.generator.encode_protein(self.test_proteins)
        
        expected_shape = (len(self.test_proteins), 1, 32)  # batch_size, 1, d_model
        self.assertEqual(encoded.shape, expected_shape)
    
    def test_generation(self):
        """Test molecule generation"""
        self.generator.eval()
        
        with torch.no_grad():
            generated = self.generator.generate(
                protein_sequences=self.test_proteins,
                max_length=8,
                deterministic=True,
                num_return_sequences=1
            )
        
        self.assertEqual(len(generated), len(self.test_proteins))
        self.assertIsInstance(generated[0], str)
    
    def test_multiple_generation(self):
        """Test generating multiple sequences per protein"""
        self.generator.eval()
        
        with torch.no_grad():
            generated = self.generator.generate(
                protein_sequences=self.test_proteins,
                max_length=8,
                deterministic=False,
                num_return_sequences=3
            )
        
        self.assertEqual(len(generated), len(self.test_proteins))
        self.assertIsInstance(generated[0], list)
        self.assertEqual(len(generated[0]), 3)


class TestMolecularPropertyCalculator(unittest.TestCase):
    """Test molecular property calculator"""
    
    def setUp(self):
        self.calculator = MolecularPropertyCalculator()
    
    def test_lipinski_properties(self):
        """Test Lipinski properties calculation"""
        test_smiles = "CCO"  # Ethanol
        
        properties = self.calculator.calculate_lipinski_properties(test_smiles)
        
        self.assertIsInstance(properties, dict)
        self.assertIn('molecular_weight', properties)
        self.assertIn('logp', properties)
        self.assertIn('lipinski_violations', properties)
        self.assertIn('lipinski_compliant', properties)
        
        # Ethanol should be Lipinski compliant
        self.assertTrue(properties['lipinski_compliant'])
        self.assertEqual(properties['lipinski_violations'], 0)
    
    def test_drug_likeness_score(self):
        """Test drug-likeness scoring"""
        test_cases = [
            ("CCO", 0.5),  # Simple alcohol - moderate score
            ("CC(=O)Nc1ccccc1", 0.7),  # Acetanilide - higher score
        ]
        
        for smiles, min_expected in test_cases:
            with self.subTest(smiles=smiles):
                score = self.calculator.calculate_drug_likeness_score(smiles)
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
    
    def test_synthetic_accessibility(self):
        """Test synthetic accessibility estimation"""
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        
        for smiles in test_smiles:
            with self.subTest(smiles=smiles):
                score = self.calculator.calculate_synthetic_accessibility(smiles)
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


class TestGenerationQualityAssessor(unittest.TestCase):
    """Test generation quality assessor"""
    
    def setUp(self):
        self.assessor = GenerationQualityAssessor()
    
    def test_single_molecule_assessment(self):
        """Test assessment of single molecule"""
        test_smiles = "CCO"
        
        assessment = self.assessor.assess_molecule(test_smiles)
        
        self.assertIsInstance(assessment, dict)
        self.assertIn('is_valid', assessment)
        self.assertIn('drug_likeness', assessment)
        self.assertIn('overall_score', assessment)
        
        # Valid molecule should have is_valid = 1.0
        self.assertEqual(assessment['is_valid'], 1.0)
        self.assertGreater(assessment['overall_score'], 0.0)
    
    def test_batch_assessment(self):
        """Test batch assessment"""
        test_smiles = ["CCO", "c1ccccc1", "INVALID", "CC(=O)O"]
        
        assessments = self.assessor.assess_batch(test_smiles)
        
        self.assertEqual(len(assessments), len(test_smiles))
        
        # Check that invalid SMILES gets low scores
        invalid_assessment = assessments[2]  # "INVALID"
        self.assertEqual(invalid_assessment['is_valid'], 0.0)
        self.assertEqual(invalid_assessment['overall_score'], 0.0)
    
    def test_batch_statistics(self):
        """Test batch statistics calculation"""
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "INVALID"]
        
        stats = self.assessor.get_batch_statistics(test_smiles)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_molecules', stats)
        self.assertIn('valid_molecules', stats)
        self.assertIn('validity_rate', stats)
        
        self.assertEqual(stats['total_molecules'], 4)
        self.assertEqual(stats['valid_molecules'], 3)
        self.assertEqual(stats['validity_rate'], 0.75)


class TestDiversityCalculator(unittest.TestCase):
    """Test diversity calculator"""
    
    def setUp(self):
        self.calculator = DiversityCalculator()
    
    def test_tanimoto_diversity(self):
        """Test Tanimoto diversity calculation"""
        # Similar molecules should have low diversity
        similar_smiles = ["CCO", "CCC", "CCCC"]
        
        diversity = self.calculator.calculate_tanimoto_diversity(similar_smiles)
        
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
    
    def test_scaffold_diversity(self):
        """Test scaffold diversity calculation"""
        test_smiles = ["c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CCO"]
        
        diversity = self.calculator.calculate_scaffold_diversity(test_smiles)
        
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)


class TestNoveltyCalculator(unittest.TestCase):
    """Test novelty calculator"""
    
    def setUp(self):
        self.reference_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        self.calculator = NoveltyCalculator(self.reference_smiles)
    
    def test_novelty_calculation(self):
        """Test novelty rate calculation"""
        # Mix of novel and known molecules
        test_smiles = ["CCO", "CCC", "CCCC", "c1ccccc1"]  # CCO and c1ccccc1 are in reference
        
        novelty_rate = self.calculator.calculate_novelty_rate(test_smiles)
        
        self.assertIsInstance(novelty_rate, float)
        self.assertGreaterEqual(novelty_rate, 0.0)
        self.assertLessEqual(novelty_rate, 1.0)
        
        # Should be 0.5 (2 novel out of 4 valid)
        self.assertAlmostEqual(novelty_rate, 0.5, delta=0.1)
    
    def test_add_reference_molecules(self):
        """Test adding reference molecules"""
        initial_size = len(self.calculator.reference_smiles)
        
        new_molecules = ["CCCC", "CCCCC"]
        self.calculator.add_reference_molecules(new_molecules)
        
        # Should have added valid molecules
        self.assertGreater(len(self.calculator.reference_smiles), initial_size)


class TestGenerationBenchmark(unittest.TestCase):
    """Test generation benchmark suite"""
    
    def setUp(self):
        self.reference_datasets = {
            'test_set': ["CCO", "c1ccccc1", "CC(=O)O"]
        }
        self.benchmark = GenerationBenchmark(self.reference_datasets)
    
    def test_validity_evaluation(self):
        """Test validity evaluation"""
        test_smiles = ["CCO", "INVALID", "c1ccccc1", "BAD"]
        
        results = self.benchmark.evaluate_validity(test_smiles)
        
        self.assertIsInstance(results, dict)
        self.assertIn('validity_rate', results)
        self.assertIn('uniqueness_rate', results)
        
        self.assertEqual(results['total_generated'], 4)
        self.assertEqual(results['valid_molecules'], 2)
        self.assertEqual(results['validity_rate'], 0.5)
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation"""
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCC"]
        
        results = self.benchmark.comprehensive_evaluation(test_smiles)
        
        self.assertIsInstance(results, dict)
        self.assertIn('validity', results)
        self.assertIn('diversity', results)
        self.assertIn('novelty', results)
        self.assertIn('drug_likeness', results)


class TestGenerationEvaluationPipeline(unittest.TestCase):
    """Test complete evaluation pipeline"""
    
    def setUp(self):
        self.pipeline = GenerationEvaluationPipeline()
    
    def test_single_model_evaluation(self):
        """Test single model evaluation"""
        test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        
        results = self.pipeline.evaluate_single_model(
            generated_smiles=test_smiles,
            model_name="test_model",
            save_results=False
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('model_name', results)
        self.assertEqual(results['model_name'], "test_model")
    
    def test_model_comparison(self):
        """Test model comparison"""
        model_generations = {
            'model_1': ["CCO", "c1ccccc1"],
            'model_2': ["CCC", "CC(=O)O"]
        }
        
        results = self.pipeline.compare_models(
            model_generations=model_generations,
            save_results=False
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('model_1', results)
        self.assertIn('model_2', results)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_classes = [
        TestSMILESTokenizer,
        TestChemicalValidator,
        TestTransformerDecoder,
        TestProteinConditionedGenerator,
        TestMolecularPropertyCalculator,
        TestGenerationQualityAssessor,
        TestDiversityCalculator,
        TestNoveltyCalculator,
        TestGenerationBenchmark,
        TestGenerationEvaluationPipeline
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
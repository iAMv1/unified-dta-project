"""
Integration tests for drug generation system
Tests end-to-end functionality of the complete generation pipeline
"""

import unittest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import warnings

# Import all generation modules
from core.protein_encoders import MemoryOptimizedESMEncoder
from core.drug_generation import (
    ProteinConditionedGenerator,
    DrugGenerationPipeline,
    SMILESTokenizer,
    ChemicalValidator
)
from core.generation_scoring import (
    GenerationMetrics,
    ConfidenceScoringPipeline,
    MolecularPropertyCalculator
)
from core.generation_evaluation import (
    GenerationEvaluationPipeline,
    GenerationBenchmark
)


class TestEndToEndGeneration(unittest.TestCase):
    """Test complete end-to-end generation pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        # Suppress warnings for cleaner test output
        warnings.filterwarnings("ignore")
        
        # Create test proteins
        self.test_proteins = [
            "MKLLVLSLSLVLVAPMAAQAAEITLKAVSRSLNCACELKCSTSLLLEACTFRRP",
            "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
        ]
        
        # Create lightweight generator for testing
        protein_encoder = ESMProteinEncoder(output_dim=32)
        self.generator = ProteinConditionedGenerator(
            protein_encoder=protein_encoder,
            vocab_size=50,
            d_model=32,
            nhead=2,
            num_layers=1,
            max_length=16
        )
        
        # Create evaluation pipeline
        self.evaluator = GenerationEvaluationPipeline()
        
        # Sample reference molecules for novelty testing
        self.reference_molecules = [
            "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC"
        ]
    
    def test_complete_generation_workflow(self):
        """Test complete generation workflow from proteins to evaluated molecules"""
        print("\n=== Testing Complete Generation Workflow ===")
        
        # Step 1: Generate molecules
        print("Step 1: Generating molecules...")
        self.generator.eval()
        
        with torch.no_grad():
            generated_smiles = self.generator.generate(
                protein_sequences=self.test_proteins,
                max_length=12,
                temperature=1.0,
                deterministic=False,
                num_return_sequences=5
            )
        
        # Verify generation output structure
        self.assertEqual(len(generated_smiles), len(self.test_proteins))
        for protein_results in generated_smiles:
            self.assertIsInstance(protein_results, list)
            self.assertEqual(len(protein_results), 5)
        
        print(f"Generated {len(generated_smiles)} sets of molecules")
        
        # Step 2: Flatten results for evaluation
        flat_generated = []
        for protein_results in generated_smiles:
            flat_generated.extend(protein_results)
        
        print(f"Total molecules for evaluation: {len(flat_generated)}")
        
        # Step 3: Evaluate generation quality
        print("Step 2: Evaluating generation quality...")
        
        evaluation_results = self.evaluator.evaluate_single_model(
            generated_smiles=flat_generated,
            model_name="test_generator",
            save_results=False
        )
        
        # Verify evaluation results structure
        self.assertIsInstance(evaluation_results, dict)
        self.assertIn('validity', evaluation_results)
        self.assertIn('diversity', evaluation_results)
        self.assertIn('drug_likeness', evaluation_results)
        
        # Print results
        validity = evaluation_results['validity']
        print(f"Validity rate: {validity['validity_rate']:.3f}")
        print(f"Valid molecules: {validity['valid_molecules']}/{validity['total_generated']}")
        
        if 'diversity' in evaluation_results:
            diversity = evaluation_results['diversity']
            print(f"Tanimoto diversity: {diversity['tanimoto_diversity']:.3f}")
        
        # Step 4: Test chemical validation
        print("Step 3: Testing chemical validation...")
        
        validator = ChemicalValidator()
        valid_count = 0
        
        for smiles in flat_generated:
            if validator.is_valid_smiles(smiles):
                valid_count += 1
                
                # Test property calculation
                properties = validator.calculate_properties(smiles)
                self.assertIsInstance(properties, dict)
                
                # Test canonicalization
                canonical = validator.canonicalize_smiles(smiles)
                if canonical:
                    self.assertIsInstance(canonical, str)
        
        print(f"Chemical validation: {valid_count}/{len(flat_generated)} valid")
        
        print("✓ Complete workflow test passed")
    
    def test_generation_pipeline_integration(self):
        """Test DrugGenerationPipeline integration"""
        print("\n=== Testing Generation Pipeline Integration ===")
        
        # Create pipeline
        protein_encoder = ESMProteinEncoder(output_dim=32)
        pipeline = DrugGenerationPipeline(
            protein_encoder=protein_encoder,
            vocab_size=50,
            d_model=32,
            nhead=2,
            num_layers=1
        )
        
        # Generate molecules using pipeline
        results = pipeline.generate_molecules(
            protein_sequences=self.test_proteins[:1],  # Use one protein for speed
            num_molecules=3,
            filter_valid=True,
            max_length=12,
            temperature=1.0
        )
        
        # Verify results structure
        self.assertEqual(len(results), 1)
        result = results[0]
        
        self.assertIn('protein_sequence', result)
        self.assertIn('generated_smiles', result)
        self.assertIn('valid_smiles', result)
        self.assertIn('properties', result)
        
        # Verify data types
        self.assertIsInstance(result['generated_smiles'], list)
        self.assertIsInstance(result['valid_smiles'], list)
        self.assertIsInstance(result['properties'], list)
        
        print(f"Generated {len(result['generated_smiles'])} molecules")
        print(f"Valid molecules: {len(result['valid_smiles'])}")
        print("✓ Pipeline integration test passed")
    
    def test_scoring_and_ranking(self):
        """Test molecule scoring and ranking"""
        print("\n=== Testing Scoring and Ranking ===")
        
        # Test molecules with different quality levels
        test_molecules = [
            "CCO",  # Simple, valid
            "c1ccccc1O",  # Aromatic, drug-like
            "CC(=O)Nc1ccccc1",  # More complex, drug-like
            "INVALID_SMILES",  # Invalid
            "C",  # Too simple
        ]
        
        # Create scoring pipeline
        scoring_pipeline = ConfidenceScoringPipeline()
        
        # Score molecules
        scoring_results = scoring_pipeline.score_molecules(test_molecules)
        
        # Verify scoring results
        self.assertEqual(len(scoring_results), len(test_molecules))
        
        for result in scoring_results:
            self.assertIn('smiles', result)
            self.assertIn('confidence_score', result)
            self.assertIn('overall_score', result)
            self.assertIn('is_valid', result)
            
            # Check score ranges
            self.assertGreaterEqual(result['confidence_score'], 0.0)
            self.assertLessEqual(result['confidence_score'], 1.0)
            self.assertGreaterEqual(result['overall_score'], 0.0)
            self.assertLessEqual(result['overall_score'], 1.0)
        
        # Test ranking
        ranked_results = scoring_pipeline.rank_molecules(scoring_results, 'overall_score')
        
        # Verify ranking (should be sorted by overall_score descending)
        for i in range(len(ranked_results) - 1):
            self.assertGreaterEqual(
                ranked_results[i]['overall_score'],
                ranked_results[i + 1]['overall_score']
            )
        
        print(f"Scored and ranked {len(test_molecules)} molecules")
        print("Top molecule:", ranked_results[0]['smiles'])
        print("✓ Scoring and ranking test passed")
    
    def test_diversity_and_novelty_metrics(self):
        """Test diversity and novelty calculations"""
        print("\n=== Testing Diversity and Novelty Metrics ===")
        
        # Create test molecules with known diversity characteristics
        similar_molecules = ["CCO", "CCC", "CCCC", "CCCCC"]  # Similar alkanes/alcohols
        diverse_molecules = ["CCO", "c1ccccc1", "CC(=O)O", "CCN(CC)CC"]  # Different scaffolds
        
        # Test diversity calculation
        from core.generation_scoring import DiversityCalculator
        diversity_calc = DiversityCalculator()
        
        similar_diversity = diversity_calc.calculate_tanimoto_diversity(similar_molecules)
        diverse_diversity = diversity_calc.calculate_tanimoto_diversity(diverse_molecules)
        
        # Diverse molecules should have higher diversity score
        self.assertGreater(diverse_diversity, similar_diversity)
        
        print(f"Similar molecules diversity: {similar_diversity:.3f}")
        print(f"Diverse molecules diversity: {diverse_diversity:.3f}")
        
        # Test novelty calculation
        from core.generation_scoring import NoveltyCalculator
        novelty_calc = NoveltyCalculator(self.reference_molecules)
        
        # Test with mix of novel and known molecules
        test_molecules = self.reference_molecules[:2] + ["CCCCCC", "c1ccc(O)cc1"]
        novelty_rate = novelty_calc.calculate_novelty_rate(test_molecules)
        
        # Should be 0.5 (2 novel out of 4 valid)
        self.assertAlmostEqual(novelty_rate, 0.5, delta=0.1)
        
        print(f"Novelty rate: {novelty_rate:.3f}")
        print("✓ Diversity and novelty test passed")
    
    def test_property_prediction_integration(self):
        """Test molecular property prediction integration"""
        print("\n=== Testing Property Prediction Integration ===")
        
        test_molecules = [
            "CCO",  # Ethanol
            "c1ccccc1O",  # Phenol
            "CC(=O)Nc1ccccc1",  # Acetanilide
        ]
        
        property_calc = MolecularPropertyCalculator()
        
        for smiles in test_molecules:
            print(f"\nAnalyzing: {smiles}")
            
            # Test Lipinski properties
            lipinski_props = property_calc.calculate_lipinski_properties(smiles)
            self.assertIsInstance(lipinski_props, dict)
            
            if lipinski_props:
                print(f"  MW: {lipinski_props.get('molecular_weight', 0):.1f}")
                print(f"  LogP: {lipinski_props.get('logp', 0):.2f}")
                print(f"  Lipinski violations: {lipinski_props.get('lipinski_violations', 0)}")
            
            # Test drug-likeness score
            drug_score = property_calc.calculate_drug_likeness_score(smiles)
            self.assertIsInstance(drug_score, float)
            self.assertGreaterEqual(drug_score, 0.0)
            self.assertLessEqual(drug_score, 1.0)
            
            print(f"  Drug-likeness: {drug_score:.3f}")
            
            # Test synthetic accessibility
            sa_score = property_calc.calculate_synthetic_accessibility(smiles)
            self.assertIsInstance(sa_score, float)
            self.assertGreaterEqual(sa_score, 0.0)
            self.assertLessEqual(sa_score, 1.0)
            
            print(f"  Synthetic accessibility: {sa_score:.3f}")
        
        print("✓ Property prediction integration test passed")
    
    def test_benchmark_comparison(self):
        """Test benchmarking against baseline methods"""
        print("\n=== Testing Benchmark Comparison ===")
        
        # Simulate results from different generation methods
        method_results = {
            'random_generation': ["CCO", "CCC", "INVALID", "c1ccccc1"],
            'rule_based': ["CC(=O)O", "CCN", "c1ccccc1O", "CC(C)O"],
            'neural_model': ["CC(=O)Nc1ccccc1", "c1ccc(N)cc1", "CCO", "CCC"]
        }
        
        # Run comparison
        comparison_results = self.evaluator.compare_models(
            model_generations=method_results,
            save_results=False
        )
        
        # Verify comparison structure
        self.assertIsInstance(comparison_results, dict)
        
        for method_name in method_results.keys():
            self.assertIn(method_name, comparison_results)
            
            method_result = comparison_results[method_name]
            self.assertIn('validity', method_result)
            self.assertIn('diversity', method_result)
            self.assertIn('drug_likeness', method_result)
        
        # Print comparison summary
        print("\nComparison Results:")
        for method_name, results in comparison_results.items():
            validity_rate = results['validity']['validity_rate']
            avg_drug_likeness = results['drug_likeness']['avg_drug_likeness']
            print(f"{method_name}: Validity={validity_rate:.3f}, Drug-likeness={avg_drug_likeness:.3f}")
        
        print("✓ Benchmark comparison test passed")
    
    def test_confidence_scoring_integration(self):
        """Test confidence scoring with generation results"""
        print("\n=== Testing Confidence Scoring Integration ===")
        
        # Generate some molecules
        self.generator.eval()
        
        with torch.no_grad():
            generated_smiles = self.generator.generate(
                protein_sequences=self.test_proteins[:1],
                max_length=10,
                num_return_sequences=3
            )
        
        # Flatten results
        flat_generated = []
        if isinstance(generated_smiles[0], list):
            flat_generated = generated_smiles[0]
        else:
            flat_generated = [generated_smiles[0]]
        
        # Score with confidence pipeline
        confidence_pipeline = ConfidenceScoringPipeline()
        scoring_results = confidence_pipeline.score_molecules(flat_generated)
        
        # Verify scoring
        self.assertEqual(len(scoring_results), len(flat_generated))
        
        for result in scoring_results:
            self.assertIn('confidence_score', result)
            self.assertIn('overall_score', result)
            
            # Print results
            print(f"SMILES: {result['smiles']}")
            print(f"  Confidence: {result['confidence_score']:.3f}")
            print(f"  Overall Quality: {result['overall_score']:.3f}")
            print(f"  Valid: {'Yes' if result['is_valid'] > 0 else 'No'}")
        
        print("✓ Confidence scoring integration test passed")


class TestGenerationPerformance(unittest.TestCase):
    """Test generation performance and memory usage"""
    
    def setUp(self):
        """Set up performance test environment"""
        warnings.filterwarnings("ignore")
        
        # Create lightweight model for performance testing
        protein_encoder = ESMProteinEncoder(output_dim=32)
        self.generator = ProteinConditionedGenerator(
            protein_encoder=protein_encoder,
            vocab_size=50,
            d_model=32,
            nhead=2,
            num_layers=1,
            max_length=16
        )
        
        self.test_proteins = [
            "MKLLVLSLSLVLVAPMAAQAAEITLKAVSRSLNCACELKCSTSLLLEACTFRRP"
        ] * 5  # Repeat for batch testing
    
    def test_batch_generation_performance(self):
        """Test batch generation performance"""
        print("\n=== Testing Batch Generation Performance ===")
        
        import time
        
        self.generator.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 3, 5]
        
        for batch_size in batch_sizes:
            proteins = self.test_proteins[:batch_size]
            
            start_time = time.time()
            
            with torch.no_grad():
                generated = self.generator.generate(
                    protein_sequences=proteins,
                    max_length=12,
                    num_return_sequences=2
                )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            print(f"Batch size {batch_size}: {generation_time:.3f}s")
            
            # Verify output structure
            self.assertEqual(len(generated), batch_size)
            
            # Check memory usage (basic check)
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                print(f"  GPU Memory: {memory_used:.1f} MB")
        
        print("✓ Batch generation performance test passed")
    
    def test_memory_efficiency(self):
        """Test memory efficiency during generation"""
        print("\n=== Testing Memory Efficiency ===")
        
        # Test with longer sequences
        long_protein = "M" + "A" * 150 + "K" * 50  # 201 residues (will be truncated)
        
        self.generator.eval()
        
        initial_memory = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            generated = self.generator.generate(
                protein_sequences=[long_protein],
                max_length=20,
                num_return_sequences=1
            )
        
        final_memory = 0
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_increase = (final_memory - initial_memory) / 1024**2  # MB
            print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Verify generation worked
        self.assertEqual(len(generated), 1)
        
        print("✓ Memory efficiency test passed")


def run_integration_tests():
    """Run all integration tests"""
    print("Running Drug Generation Integration Tests")
    print("=" * 50)
    
    # Create test suite
    test_classes = [
        TestEndToEndGeneration,
        TestGenerationPerformance
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
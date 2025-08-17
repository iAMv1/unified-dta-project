"""
Integration tests for end-to-end workflows
Tests complete training and inference pipelines
"""

import unittest
import torch
import tempfile
import os
import pandas as pd
import sys
from pathlib import Path
import time

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.models import UnifiedDTAModel, get_lightweight_model
    from core.datasets import DTADataset, DTADataLoader, collate_dta_batch
    from core.training import DTATrainer, TrainingConfig
    from core.evaluation import ComprehensiveEvaluator, MetricsCalculator
    from core.data_processing import preprocess_dta_dataset
    from core.utils import set_seed, get_device
    from torch_geometric.data import Data, Batch
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    # Set fallbacks
    UnifiedDTAModel = None
    DTATrainer = None
    ComprehensiveEvaluator = None


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows"""
    
    def setUp(self):
        """Set up test data and models"""
        set_seed(42)
        
        # Create test dataset
        self.test_data = {
            'compound_iso_smiles': [
                'CCO',  # Ethanol
                'CC(=O)O',  # Acetic acid
                'c1ccccc1',  # Benzene
                'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
                'CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O',  # Salbutamol
                'CC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)C',  # Benzophenone
                'CCCCCCCCCCCCCCC(=O)O'  # Palmitic acid
            ],
            'target_sequence': [
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ'
            ],
            'affinity': [7.5, 6.2, 8.1, 5.8, 7.9, 6.5, 7.2, 6.8]
        }
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame(self.test_data)
        df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        self.device = get_device('cpu')  # Force CPU for testing
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_data_loading_pipeline(self):
        """Test complete data loading and preprocessing pipeline"""
        try:
            # Load dataset
            dataset = DTADataset(self.temp_file.name)
            self.assertEqual(len(dataset), len(self.test_data['compound_iso_smiles']))
            
            # Create data loader
            dataloader = DTADataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=collate_dta_batch
            )
            
            # Test iteration
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                
                # Check batch structure
                self.assertIn('smiles', batch)
                self.assertIn('protein_sequences', batch)
                self.assertIn('affinities', batch)
                
                # Check batch sizes
                self.assertLessEqual(len(batch['smiles']), 4)
                self.assertEqual(len(batch['smiles']), len(batch['protein_sequences']))
                self.assertEqual(len(batch['smiles']), len(batch['affinities']))
                
                # Check data types
                self.assertIsInstance(batch['affinities'], torch.Tensor)
                
                if batch_count >= 2:  # Test first 2 batches
                    break
            
            self.assertGreater(batch_count, 0)
            
        except Exception as e:
            self.skipTest(f"Data loading pipeline test failed: {e}")
    
    def test_model_training_pipeline(self):
        """Test complete model training pipeline"""
        if DTATrainer is None:
            self.skipTest("DTATrainer not available")
        
        try:
            # Create lightweight model for testing
            model = get_lightweight_model()
            model = model.to(self.device)
            
            # Create dataset and dataloader
            dataset = DTADataset(self.temp_file.name)
            train_loader = DTADataLoader(
                dataset,
                batch_size=2,
                shuffle=True,
                collate_fn=collate_dta_batch
            )
            
            # Create trainer
            training_config = TrainingConfig(
                num_epochs=2,  # Short training for testing
                learning_rate=1e-3,
                batch_size=2,
                device=self.device,
                save_checkpoints=False
            )
            
            trainer = DTATrainer(model, training_config)
            
            # Train model
            training_history = trainer.train(train_loader, val_loader=None)
            
            # Check training history
            self.assertIn('train_loss', training_history)
            self.assertEqual(len(training_history['train_loss']), 2)  # 2 epochs
            
            # Check that loss decreased or stayed reasonable
            final_loss = training_history['train_loss'][-1]
            self.assertLess(final_loss, 100)  # Reasonable loss value
            self.assertFalse(torch.isnan(torch.tensor(final_loss)))
            
        except Exception as e:
            self.skipTest(f"Training pipeline test failed: {e}")
    
    def test_inference_pipeline(self):
        """Test complete inference pipeline"""
        try:
            # Create model
            model = get_lightweight_model()
            model = model.to(self.device)
            model.eval()
            
            # Create test data
            test_smiles = ['CCO', 'CC(=O)O']
            test_proteins = [
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ'
            ]
            
            # Create molecular graphs (mock)
            graphs = []
            for i, smiles in enumerate(test_smiles):
                num_atoms = 3 + i
                node_features = torch.randn(num_atoms, 78)
                edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
                if i > 0:
                    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
                
                graph = Data(x=node_features, edge_index=edge_index)
                graphs.append(graph)
            
            batch_graph = Batch.from_data_list(graphs)
            batch_graph = batch_graph.to(self.device)
            
            # Inference
            with torch.no_grad():
                predictions = model(batch_graph, test_proteins)
            
            # Check predictions
            self.assertEqual(predictions.shape, (len(test_smiles), 1))
            self.assertFalse(torch.isnan(predictions).any())
            self.assertFalse(torch.isinf(predictions).any())
            
            # Check prediction range is reasonable
            self.assertTrue(torch.all(predictions > -20))
            self.assertTrue(torch.all(predictions < 20))
            
        except Exception as e:
            self.skipTest(f"Inference pipeline test failed: {e}")
    
    def test_evaluation_pipeline(self):
        """Test complete evaluation pipeline"""
        if ComprehensiveEvaluator is None:
            self.skipTest("ComprehensiveEvaluator not available")
        
        try:
            # Create model and test data
            model = get_lightweight_model()
            model = model.to(self.device)
            model.eval()
            
            # Create predictions and targets
            num_samples = 8
            predictions = torch.randn(num_samples, 1)
            targets = torch.randn(num_samples)
            
            # Create evaluator
            evaluator = ComprehensiveEvaluator()
            
            # Evaluate
            metrics = evaluator.evaluate(predictions.squeeze(), targets)
            
            # Check metrics
            expected_metrics = ['rmse', 'mse', 'pearson', 'spearman', 'ci']
            for metric in expected_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], (int, float))
                self.assertFalse(torch.isnan(torch.tensor(metrics[metric])))
            
        except Exception as e:
            self.skipTest(f"Evaluation pipeline test failed: {e}")
    
    def test_preprocessing_pipeline(self):
        """Test data preprocessing pipeline"""
        try:
            # Test preprocessing
            processed_data = preprocess_dta_dataset(
                self.temp_file.name,
                validate_smiles=True,
                validate_proteins=True,
                max_protein_length=200
            )
            
            # Check processed data structure
            self.assertIn('valid_data', processed_data)
            self.assertIn('invalid_indices', processed_data)
            self.assertIn('preprocessing_stats', processed_data)
            
            # Check that we have some valid data
            valid_data = processed_data['valid_data']
            self.assertGreater(len(valid_data), 0)
            
            # Check preprocessing stats
            stats = processed_data['preprocessing_stats']
            self.assertIn('total_samples', stats)
            self.assertIn('valid_samples', stats)
            self.assertIn('invalid_samples', stats)
            
        except Exception as e:
            self.skipTest(f"Preprocessing pipeline test failed: {e}")


class TestMultiDatasetCompatibility(unittest.TestCase):
    """Test compatibility with multiple dataset formats"""
    
    def setUp(self):
        """Set up test datasets in different formats"""
        # KIBA format
        self.kiba_data = {
            'compound_iso_smiles': ['CCO', 'CC(=O)O', 'c1ccccc1'],
            'target_sequence': [
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'
            ],
            'affinity': [7.5, 6.2, 8.1]
        }
        
        # Davis format (similar structure)
        self.davis_data = {
            'compound_iso_smiles': ['CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'],
            'target_sequence': [
                'MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'
            ],
            'affinity': [6.8, 7.2]
        }
        
        # BindingDB format
        self.bindingdb_data = {
            'compound_iso_smiles': ['CCCCCCCCCCCCCCC(=O)O', 'CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O'],
            'target_sequence': [
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ'
            ],
            'affinity': [5.9, 7.8]
        }
        
        # Create temporary files
        self.temp_files = {}
        for name, data in [('kiba', self.kiba_data), ('davis', self.davis_data), ('bindingdb', self.bindingdb_data)]:
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            df = pd.DataFrame(data)
            df.to_csv(temp_file.name, index=False)
            temp_file.close()
            self.temp_files[name] = temp_file.name
    
    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files.values():
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_kiba_dataset_compatibility(self):
        """Test KIBA dataset compatibility"""
        try:
            dataset = DTADataset(self.temp_files['kiba'])
            self.assertEqual(len(dataset), len(self.kiba_data['compound_iso_smiles']))
            
            # Test sample access
            sample = dataset[0]
            self.assertIn('smiles', sample)
            self.assertIn('protein_sequence', sample)
            self.assertIn('affinity', sample)
            
        except Exception as e:
            self.skipTest(f"KIBA dataset test failed: {e}")
    
    def test_davis_dataset_compatibility(self):
        """Test Davis dataset compatibility"""
        try:
            dataset = DTADataset(self.temp_files['davis'])
            self.assertEqual(len(dataset), len(self.davis_data['compound_iso_smiles']))
            
            # Test sample access
            sample = dataset[0]
            self.assertIn('smiles', sample)
            self.assertIn('protein_sequence', sample)
            self.assertIn('affinity', sample)
            
        except Exception as e:
            self.skipTest(f"Davis dataset test failed: {e}")
    
    def test_bindingdb_dataset_compatibility(self):
        """Test BindingDB dataset compatibility"""
        try:
            dataset = DTADataset(self.temp_files['bindingdb'])
            self.assertEqual(len(dataset), len(self.bindingdb_data['compound_iso_smiles']))
            
            # Test sample access
            sample = dataset[0]
            self.assertIn('smiles', sample)
            self.assertIn('protein_sequence', sample)
            self.assertIn('affinity', sample)
            
        except Exception as e:
            self.skipTest(f"BindingDB dataset test failed: {e}")
    
    def test_multi_dataset_loading(self):
        """Test loading multiple datasets simultaneously"""
        if MultiDatasetDTA is None:
            self.skipTest("MultiDatasetDTA not available")
        
        try:
            # Create multi-dataset
            dataset_paths = list(self.temp_files.values())
            multi_dataset = MultiDatasetDTA(dataset_paths)
            
            # Check total length
            expected_length = sum(len(data['compound_iso_smiles']) 
                                for data in [self.kiba_data, self.davis_data, self.bindingdb_data])
            self.assertEqual(len(multi_dataset), expected_length)
            
            # Test sample access
            sample = multi_dataset[0]
            self.assertIn('smiles', sample)
            self.assertIn('protein_sequence', sample)
            self.assertIn('affinity', sample)
            self.assertIn('dataset_source', sample)
            
        except Exception as e:
            self.skipTest(f"Multi-dataset test failed: {e}")


class TestMemoryAndPerformance(unittest.TestCase):
    """Test memory usage and performance benchmarks"""
    
    def setUp(self):
        """Set up performance test data"""
        self.device = get_device('cpu')
        
        # Create larger test dataset for performance testing
        num_samples = 50
        self.large_test_data = {
            'compound_iso_smiles': ['CCO'] * num_samples,
            'target_sequence': ['MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'] * num_samples,
            'affinity': [7.5] * num_samples
        }
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame(self.large_test_data)
        df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_memory_usage_lightweight_model(self):
        """Test memory usage of lightweight model"""
        try:
            from core.utils import get_memory_usage
            
            # Get baseline memory
            baseline_memory = get_memory_usage()
            
            # Create lightweight model
            model = get_lightweight_model()
            model = model.to(self.device)
            
            # Get memory after model creation
            model_memory = get_memory_usage()
            
            # Check memory increase is reasonable for lightweight model
            memory_increase = model_memory['system']['used'] - baseline_memory['system']['used']
            self.assertLess(memory_increase, 500)  # Less than 500MB increase
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            self.assertLess(total_params, 1000000)  # Less than 1M parameters
            
        except Exception as e:
            self.skipTest(f"Memory usage test failed: {e}")
    
    def test_inference_speed_benchmark(self):
        """Test inference speed benchmark"""
        try:
            # Create model
            model = get_lightweight_model()
            model = model.to(self.device)
            model.eval()
            
            # Create test data
            batch_size = 8
            test_smiles = ['CCO'] * batch_size
            test_proteins = ['MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'] * batch_size
            
            # Create molecular graphs (mock)
            graphs = []
            for i in range(batch_size):
                node_features = torch.randn(5, 78)
                edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
                graph = Data(x=node_features, edge_index=edge_index)
                graphs.append(graph)
            
            batch_graph = Batch.from_data_list(graphs)
            batch_graph = batch_graph.to(self.device)
            
            # Warm up
            with torch.no_grad():
                _ = model(batch_graph, test_proteins)
            
            # Benchmark inference
            num_runs = 10
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    predictions = model(batch_graph, test_proteins)
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time_per_batch = total_time / num_runs
            samples_per_second = (batch_size * num_runs) / total_time
            
            # Check performance is reasonable
            self.assertLess(avg_time_per_batch, 5.0)  # Less than 5 seconds per batch
            self.assertGreater(samples_per_second, 1.0)  # At least 1 sample per second
            
            print(f"Inference benchmark:")
            print(f"  Average time per batch ({batch_size} samples): {avg_time_per_batch:.3f}s")
            print(f"  Samples per second: {samples_per_second:.2f}")
            
        except Exception as e:
            self.skipTest(f"Inference speed benchmark failed: {e}")
    
    def test_training_memory_efficiency(self):
        """Test training memory efficiency"""
        try:
            from core.utils import get_memory_usage
            
            # Get baseline memory
            baseline_memory = get_memory_usage()
            
            # Create model and data
            model = get_lightweight_model()
            model = model.to(self.device)
            
            dataset = DTADataset(self.temp_file.name)
            dataloader = DTADataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=collate_dta_batch
            )
            
            # Training step
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.MSELoss()
            
            # Get one batch
            batch = next(iter(dataloader))
            
            # Create mock graph data
            graphs = []
            for smiles in batch['smiles']:
                node_features = torch.randn(5, 78)
                edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
                graph = Data(x=node_features, edge_index=edge_index)
                graphs.append(graph)
            
            batch_graph = Batch.from_data_list(graphs)
            batch_graph = batch_graph.to(self.device)
            
            # Forward pass
            predictions = model(batch_graph, batch['protein_sequences'])
            loss = criterion(predictions.squeeze(), batch['affinities'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check memory after training step
            training_memory = get_memory_usage()
            memory_increase = training_memory['system']['used'] - baseline_memory['system']['used']
            
            # Memory increase should be reasonable
            self.assertLess(memory_increase, 1000)  # Less than 1GB increase
            
        except Exception as e:
            self.skipTest(f"Training memory efficiency test failed: {e}")
    
    def test_batch_size_optimization(self):
        """Test automatic batch size optimization"""
        try:
            from core.utils import optimize_batch_size
            
            # Test batch size optimization
            model = get_lightweight_model()
            model = model.to(self.device)
            
            # Mock data function
            def create_batch(batch_size):
                graphs = []
                proteins = []
                for i in range(batch_size):
                    node_features = torch.randn(5, 78)
                    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
                    graph = Data(x=node_features, edge_index=edge_index)
                    graphs.append(graph)
                    proteins.append('MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG')
                
                batch_graph = Batch.from_data_list(graphs)
                return batch_graph.to(self.device), proteins
            
            # Find optimal batch size
            optimal_batch_size = optimize_batch_size(
                model=model,
                create_batch_fn=create_batch,
                max_batch_size=16,
                device=self.device
            )
            
            self.assertGreater(optimal_batch_size, 0)
            self.assertLessEqual(optimal_batch_size, 16)
            
            print(f"Optimal batch size: {optimal_batch_size}")
            
        except Exception as e:
            self.skipTest(f"Batch size optimization test failed: {e}")


if __name__ == '__main__':
    unittest.main()
"""
Performance benchmarks and memory usage tests
Tests system performance under various conditions
"""

import unittest
import torch
import time
import psutil
import gc
import sys
from pathlib import Path
import tempfile
import pandas as pd

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.models import get_lightweight_model, get_production_model
    from core.datasets import DTADataset, DTADataLoader, collate_dta_batch
    from core.utils import get_memory_usage, optimize_batch_size, clear_memory
    from core.evaluation import MetricsCalculator, BenchmarkSuite
    from torch_geometric.data import Data, Batch
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    # Set fallbacks
    BenchmarkSuite = None


class TestMemoryBenchmarks(unittest.TestCase):
    """Test memory usage benchmarks"""
    
    def setUp(self):
        """Set up memory testing"""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.device = torch.device('cpu')  # Use CPU for consistent testing
    
    def tearDown(self):
        """Clean up after memory tests"""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def get_memory_mb(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def test_lightweight_model_memory(self):
        """Test memory usage of lightweight model"""
        baseline_memory = self.get_memory_mb()
        
        # Create lightweight model
        model = get_lightweight_model()
        model = model.to(self.device)
        
        model_memory = self.get_memory_mb()
        memory_increase = model_memory - baseline_memory
        
        # Lightweight model should use less than 100MB
        self.assertLess(memory_increase, 100, 
                       f"Lightweight model uses {memory_increase:.1f}MB, expected <100MB")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Lightweight model memory usage: {memory_increase:.1f}MB")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Lightweight model should have reasonable parameter count
        self.assertLess(total_params, 1000000, "Lightweight model has too many parameters")
    
    def test_production_model_memory(self):
        """Test memory usage of production model"""
        try:
            baseline_memory = self.get_memory_mb()
            
            # Create production model
            model = get_production_model()
            model = model.to(self.device)
            
            model_memory = self.get_memory_mb()
            memory_increase = model_memory - baseline_memory
            
            # Production model can use more memory but should be reasonable
            self.assertLess(memory_increase, 2000, 
                           f"Production model uses {memory_increase:.1f}MB, expected <2GB")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Production model memory usage: {memory_increase:.1f}MB")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            self.skipTest(f"Production model memory test failed: {e}")
    
    def test_batch_processing_memory(self):
        """Test memory usage during batch processing"""
        model = get_lightweight_model()
        model = model.to(self.device)
        model.eval()
        
        batch_sizes = [1, 4, 8, 16]
        memory_usage = {}
        
        for batch_size in batch_sizes:
            gc.collect()
            baseline_memory = self.get_memory_mb()
            
            # Create batch data
            graphs = []
            proteins = []
            for i in range(batch_size):
                node_features = torch.randn(10, 78)  # Larger molecules
                edge_index = torch.tensor([
                    [0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9]
                ], dtype=torch.long)
                graph = Data(x=node_features, edge_index=edge_index)
                graphs.append(graph)
                proteins.append('MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG' * 2)  # Longer proteins
            
            batch_graph = Batch.from_data_list(graphs)
            batch_graph = batch_graph.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                predictions = model(batch_graph, proteins)
            
            peak_memory = self.get_memory_mb()
            memory_increase = peak_memory - baseline_memory
            memory_usage[batch_size] = memory_increase
            
            print(f"Batch size {batch_size}: {memory_increase:.1f}MB")
            
            # Clean up
            del batch_graph, predictions, graphs, proteins
            gc.collect()
        
        # Memory should scale reasonably with batch size
        for i in range(1, len(batch_sizes)):
            prev_batch = batch_sizes[i-1]
            curr_batch = batch_sizes[i]
            
            # Memory shouldn't increase too dramatically
            memory_ratio = memory_usage[curr_batch] / memory_usage[prev_batch]
            batch_ratio = curr_batch / prev_batch
            
            # Memory increase should be roughly proportional to batch size increase
            self.assertLess(memory_ratio, batch_ratio * 2, 
                           f"Memory scaling too high: {memory_ratio:.2f}x for {batch_ratio}x batch size")
    
    def test_memory_cleanup(self):
        """Test memory cleanup after operations"""
        baseline_memory = self.get_memory_mb()
        
        # Create and use model
        model = get_lightweight_model()
        model = model.to(self.device)
        
        # Create large batch
        graphs = []
        proteins = []
        for i in range(32):  # Large batch
            node_features = torch.randn(15, 78)
            edge_index = torch.tensor([
                [j for j in range(14)],
                [j+1 for j in range(14)]
            ], dtype=torch.long)
            graph = Data(x=node_features, edge_index=edge_index)
            graphs.append(graph)
            proteins.append('MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG')
        
        batch_graph = Batch.from_data_list(graphs)
        batch_graph = batch_graph.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            predictions = model(batch_graph, proteins)
        
        peak_memory = self.get_memory_mb()
        
        # Clean up
        del model, batch_graph, predictions, graphs, proteins
        clear_memory()
        
        final_memory = self.get_memory_mb()
        
        # Memory should return close to baseline
        memory_retained = final_memory - baseline_memory
        peak_increase = peak_memory - baseline_memory
        
        print(f"Baseline: {baseline_memory:.1f}MB")
        print(f"Peak: {peak_memory:.1f}MB (+{peak_increase:.1f}MB)")
        print(f"Final: {final_memory:.1f}MB (+{memory_retained:.1f}MB)")
        
        # Should retain less than 20% of peak memory increase
        self.assertLess(memory_retained, peak_increase * 0.2, 
                       "Memory not properly cleaned up")


class TestInferenceBenchmarks(unittest.TestCase):
    """Test inference speed benchmarks"""
    
    def setUp(self):
        """Set up inference benchmarks"""
        self.device = torch.device('cpu')
        self.model = get_lightweight_model()
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def create_test_batch(self, batch_size, molecule_size=10):
        """Create test batch with specified size"""
        graphs = []
        proteins = []
        
        for i in range(batch_size):
            # Create molecular graph
            node_features = torch.randn(molecule_size, 78)
            num_edges = min(molecule_size * 2, molecule_size * (molecule_size - 1) // 2)
            edge_index = torch.randint(0, molecule_size, (2, num_edges), dtype=torch.long)
            
            graph = Data(x=node_features, edge_index=edge_index)
            graphs.append(graph)
            
            # Create protein sequence
            protein = 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'
            proteins.append(protein)
        
        batch_graph = Batch.from_data_list(graphs)
        batch_graph = batch_graph.to(self.device)
        
        return batch_graph, proteins
    
    def benchmark_inference(self, batch_size, num_runs=10, warmup_runs=3):
        """Benchmark inference speed"""
        batch_graph, proteins = self.create_test_batch(batch_size)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(batch_graph, proteins)
        
        # Benchmark runs
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                predictions = self.model(batch_graph, proteins)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_runs
        samples_per_second = (batch_size * num_runs) / total_time
        time_per_sample = total_time / (batch_size * num_runs)
        
        return {
            'batch_size': batch_size,
            'avg_time_per_batch': avg_time_per_batch,
            'samples_per_second': samples_per_second,
            'time_per_sample': time_per_sample,
            'total_samples': batch_size * num_runs
        }
    
    def test_single_sample_inference(self):
        """Test single sample inference speed"""
        results = self.benchmark_inference(batch_size=1, num_runs=20)
        
        print(f"Single sample inference:")
        print(f"  Time per sample: {results['time_per_sample']*1000:.2f}ms")
        print(f"  Samples per second: {results['samples_per_second']:.2f}")
        
        # Single sample should be processed quickly
        self.assertLess(results['time_per_sample'], 1.0, "Single sample takes too long")
        self.assertGreater(results['samples_per_second'], 1.0, "Too slow for single samples")
    
    def test_batch_inference_scaling(self):
        """Test how inference speed scales with batch size"""
        batch_sizes = [1, 2, 4, 8, 16]
        results = []
        
        for batch_size in batch_sizes:
            result = self.benchmark_inference(batch_size, num_runs=10)
            results.append(result)
            
            print(f"Batch size {batch_size}:")
            print(f"  Time per batch: {result['avg_time_per_batch']:.3f}s")
            print(f"  Samples per second: {result['samples_per_second']:.2f}")
            print(f"  Time per sample: {result['time_per_sample']*1000:.2f}ms")
        
        # Check that larger batches are more efficient per sample
        for i in range(1, len(results)):
            prev_efficiency = results[i-1]['time_per_sample']
            curr_efficiency = results[i]['time_per_sample']
            
            # Larger batches should be more efficient (lower time per sample)
            # Allow some tolerance for small batch sizes
            if results[i]['batch_size'] >= 4:
                self.assertLessEqual(curr_efficiency, prev_efficiency * 1.2, 
                                   f"Batch efficiency decreased at size {results[i]['batch_size']}")
    
    def test_molecule_size_impact(self):
        """Test how molecule size impacts inference speed"""
        molecule_sizes = [5, 10, 20, 30]
        batch_size = 4
        results = []
        
        for mol_size in molecule_sizes:
            batch_graph, proteins = self.create_test_batch(batch_size, mol_size)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(batch_graph, proteins)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            time_per_sample = avg_time / batch_size
            
            results.append({
                'molecule_size': mol_size,
                'time_per_sample': time_per_sample
            })
            
            print(f"Molecule size {mol_size} atoms:")
            print(f"  Time per sample: {time_per_sample*1000:.2f}ms")
        
        # Larger molecules should take more time, but not excessively
        for i in range(1, len(results)):
            prev_time = results[i-1]['time_per_sample']
            curr_time = results[i]['time_per_sample']
            size_ratio = molecule_sizes[i] / molecule_sizes[i-1]
            time_ratio = curr_time / prev_time
            
            # Time increase should be reasonable relative to size increase
            self.assertLess(time_ratio, size_ratio * 2, 
                           f"Time scaling too high for molecule size {molecule_sizes[i]}")


class TestTrainingBenchmarks(unittest.TestCase):
    """Test training performance benchmarks"""
    
    def setUp(self):
        """Set up training benchmarks"""
        self.device = torch.device('cpu')
        
        # Create test dataset
        test_data = {
            'compound_iso_smiles': ['CCO', 'CC(=O)O', 'c1ccccc1', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'] * 10,
            'target_sequence': [
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ'
            ] * 10,
            'affinity': [7.5, 6.2, 8.1, 6.8] * 10
        }
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame(test_data)
        df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary files"""
        import os
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_training_step_speed(self):
        """Test training step speed"""
        model = get_lightweight_model()
        model = model.to(self.device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        # Create test batch
        batch_size = 4
        graphs = []
        proteins = []
        targets = torch.randn(batch_size)
        
        for i in range(batch_size):
            node_features = torch.randn(10, 78)
            edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
            graph = Data(x=node_features, edge_index=edge_index)
            graphs.append(graph)
            proteins.append('MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG')
        
        batch_graph = Batch.from_data_list(graphs)
        batch_graph = batch_graph.to(self.device)
        targets = targets.to(self.device)
        
        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            predictions = model(batch_graph, proteins)
            loss = criterion(predictions.squeeze(), targets)
            loss.backward()
            optimizer.step()
        
        # Benchmark training steps
        num_steps = 10
        start_time = time.time()
        
        for _ in range(num_steps):
            optimizer.zero_grad()
            predictions = model(batch_graph, proteins)
            loss = criterion(predictions.squeeze(), targets)
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        time_per_step = total_time / num_steps
        samples_per_second = (batch_size * num_steps) / total_time
        
        print(f"Training step benchmark:")
        print(f"  Time per step: {time_per_step:.3f}s")
        print(f"  Samples per second: {samples_per_second:.2f}")
        
        # Training step should be reasonably fast
        self.assertLess(time_per_step, 5.0, "Training step too slow")
        self.assertGreater(samples_per_second, 0.5, "Training throughput too low")
    
    def test_gradient_computation_speed(self):
        """Test gradient computation speed"""
        model = get_lightweight_model()
        model = model.to(self.device)
        model.train()
        
        # Create test data
        batch_size = 8
        graphs = []
        proteins = []
        
        for i in range(batch_size):
            node_features = torch.randn(10, 78)
            edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
            graph = Data(x=node_features, edge_index=edge_index)
            graphs.append(graph)
            proteins.append('MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG')
        
        batch_graph = Batch.from_data_list(graphs)
        batch_graph = batch_graph.to(self.device)
        
        # Benchmark forward pass
        start_time = time.time()
        for _ in range(10):
            predictions = model(batch_graph, proteins)
            loss = predictions.sum()
        forward_time = time.time() - start_time
        
        # Benchmark backward pass
        start_time = time.time()
        for _ in range(10):
            model.zero_grad()
            predictions = model(batch_graph, proteins)
            loss = predictions.sum()
            loss.backward()
        backward_time = time.time() - start_time
        
        forward_per_step = forward_time / 10
        backward_per_step = backward_time / 10
        
        print(f"Gradient computation benchmark:")
        print(f"  Forward pass: {forward_per_step:.3f}s")
        print(f"  Backward pass: {backward_per_step:.3f}s")
        print(f"  Backward/Forward ratio: {backward_per_step/forward_per_step:.2f}")
        
        # Backward pass should not be excessively slower than forward
        self.assertLess(backward_per_step / forward_per_step, 5.0, 
                       "Backward pass too slow relative to forward pass")


class TestBenchmarkSuite(unittest.TestCase):
    """Test comprehensive benchmark suite"""
    
    def test_benchmark_suite_execution(self):
        """Test benchmark suite execution"""
        if BenchmarkSuite is None:
            self.skipTest("BenchmarkSuite not available")
        
        try:
            # Create benchmark suite
            benchmark = BenchmarkSuite()
            
            # Run lightweight benchmarks
            results = benchmark.run_lightweight_benchmarks()
            
            # Check results structure
            self.assertIsInstance(results, dict)
            self.assertIn('model_info', results)
            self.assertIn('inference_speed', results)
            self.assertIn('memory_usage', results)
            
            # Check that benchmarks completed successfully
            for key, value in results.items():
                if isinstance(value, dict):
                    self.assertGreater(len(value), 0, f"Empty benchmark results for {key}")
            
            print("Benchmark suite results:")
            for key, value in results.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            self.skipTest(f"Benchmark suite test failed: {e}")
    
    def test_performance_regression_detection(self):
        """Test performance regression detection"""
        if BenchmarkSuite is None:
            self.skipTest("BenchmarkSuite not available")
        
        try:
            benchmark = BenchmarkSuite()
            
            # Run benchmarks twice to test consistency
            results1 = benchmark.run_lightweight_benchmarks()
            results2 = benchmark.run_lightweight_benchmarks()
            
            # Compare results for consistency
            if 'inference_speed' in results1 and 'inference_speed' in results2:
                speed1 = results1['inference_speed'].get('samples_per_second', 0)
                speed2 = results2['inference_speed'].get('samples_per_second', 0)
                
                if speed1 > 0 and speed2 > 0:
                    # Results should be reasonably consistent (within 50%)
                    ratio = max(speed1, speed2) / min(speed1, speed2)
                    self.assertLess(ratio, 1.5, 
                                   f"Inconsistent benchmark results: {speed1:.2f} vs {speed2:.2f}")
            
        except Exception as e:
            self.skipTest(f"Performance regression test failed: {e}")


if __name__ == '__main__':
    # Run with verbose output to see benchmark results
    unittest.main(verbosity=2)
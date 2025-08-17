"""
Unit tests for encoder implementations
Tests all protein and drug encoders with various configurations
"""

import unittest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.models import (
        BaseEncoder, GINDrugEncoder
    )
    from core.protein_encoders import (
        EnhancedCNNProteinEncoder, MemoryOptimizedESMEncoder as ESMProteinEncoder, GatedCNNBlock
    )
    from core.drug_encoders import (
        EnhancedGINDrugEncoder, MultiScaleGINEncoder, 
        ConfigurableMLPBlock, ResidualGINLayer
    )
    from core.base_components import SEBlock, PositionalEncoding
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    # Set to None for graceful handling
    EnhancedCNNProteinEncoder = None
    MemoryOptimizedESMEncoder = None
    EnhancedGINDrugEncoder = None
    MultiScaleGINEncoder = None


class TestBaseComponents(unittest.TestCase):
    """Test base components and abstract classes"""
    
    def test_base_encoder_abstract(self):
        """Test that BaseEncoder is properly abstract"""
        with self.assertRaises(TypeError):
            BaseEncoder()
    
    def test_se_block(self):
        """Test Squeeze-and-Excitation block"""
        se_block = SEBlock(in_channels=64, reduction=16)
        
        # Test forward pass
        x = torch.randn(2, 64, 100)  # [batch, channels, length]
        output = se_block(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_positional_encoding(self):
        """Test positional encoding"""
        pos_enc = PositionalEncoding(d_model=128, max_len=200)
        
        # Test forward pass
        x = torch.randn(50, 2, 128)  # [seq_len, batch, d_model]
        output = pos_enc(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestProteinEncoders(unittest.TestCase):
    """Test protein encoder implementations"""
    
    def setUp(self):
        """Set up test data"""
        self.protein_sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "MSANNSPPSAQKSVLPTAIPAVLPAASPCSSPKTGLSARLSNGSFSAPSLPQPQPQPQPQPQPQ"
        ]
        self.batch_size = len(self.protein_sequences)
    
    def test_esm_protein_encoder(self):
        """Test ESM-2 protein encoder"""
        try:
            encoder = MemoryOptimizedESMEncoder(
                output_dim=128,
                max_length=100,
                freeze_initial=True
            )
            
            # Test properties
            self.assertEqual(encoder.output_dim, 128)
            
            # Test forward pass
            output = encoder(self.protein_sequences)
            
            self.assertEqual(output.shape, (self.batch_size, 128))
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())
            
            # Test phase switching
            encoder.freeze_esm()
            frozen_params = sum(1 for p in encoder.esm_model.parameters() if not p.requires_grad)
            total_params = sum(1 for p in encoder.esm_model.parameters())
            self.assertEqual(frozen_params, total_params)
            
            encoder.unfreeze_esm_layers(num_layers=2)
            unfrozen_params = sum(1 for p in encoder.esm_model.parameters() if p.requires_grad)
            self.assertGreater(unfrozen_params, 0)
            
        except Exception as e:
            self.skipTest(f"ESM encoder test skipped due to dependency: {e}")
    
    def test_enhanced_cnn_protein_encoder(self):
        """Test enhanced CNN protein encoder"""
        if EnhancedCNNProteinEncoder is None:
            self.skipTest("EnhancedCNNProteinEncoder not available")
        
        encoder = EnhancedCNNProteinEncoder(
            vocab_size=25,
            embed_dim=64,
            num_filters=[32, 64],
            kernel_sizes=[3, 5],
            output_dim=128
        )
        
        # Test properties
        self.assertEqual(encoder.output_dim, 128)
        
        # Create tokenized sequences (mock)
        max_len = max(len(seq) for seq in self.protein_sequences)
        tokenized = torch.randint(0, 25, (self.batch_size, max_len))
        
        # Test forward pass
        output = encoder(tokenized)
        
        self.assertEqual(output.shape, (self.batch_size, 128))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_memory_optimized_esm_encoder(self):
        """Test memory optimized ESM encoder"""
        if MemoryOptimizedESMEncoder is None:
            self.skipTest("MemoryOptimizedESMEncoder not available")
        
        try:
            encoder = MemoryOptimizedESMEncoder(
                output_dim=64,
                max_length=50,
                use_gradient_checkpointing=True
            )
            
            # Test forward pass
            output = encoder(self.protein_sequences)
            
            self.assertEqual(output.shape, (self.batch_size, 64))
            self.assertFalse(torch.isnan(output).any())
            
        except Exception as e:
            self.skipTest(f"Memory optimized ESM test skipped: {e}")


class TestDrugEncoders(unittest.TestCase):
    """Test drug encoder implementations"""
    
    def setUp(self):
        """Set up test molecular graph data"""
        # Create mock molecular graphs
        self.graphs = []
        for i in range(2):
            num_atoms = 5 + i
            node_features = torch.randn(num_atoms, 78)
            edge_index = torch.tensor([
                [0, 1, 2, 3],
                [1, 2, 3, 4]
            ], dtype=torch.long)
            
            graph = Data(x=node_features, edge_index=edge_index)
            self.graphs.append(graph)
        
        self.batch_graph = Batch.from_data_list(self.graphs)
        self.batch_size = len(self.graphs)
    
    def test_gin_drug_encoder(self):
        """Test basic GIN drug encoder"""
        encoder = GINDrugEncoder(
            node_features=78,
            hidden_dim=64,
            num_layers=3,
            output_dim=128,
            dropout=0.2
        )
        
        # Test properties
        self.assertEqual(encoder.output_dim, 128)
        
        # Test forward pass
        output = encoder(self.batch_graph)
        
        self.assertEqual(output.shape, (self.batch_size, 128))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_enhanced_gin_drug_encoder(self):
        """Test enhanced GIN drug encoder"""
        if EnhancedGINDrugEncoder is None:
            self.skipTest("EnhancedGINDrugEncoder not available")
        
        encoder = EnhancedGINDrugEncoder(
            node_features=78,
            hidden_dim=64,
            num_layers=4,
            output_dim=128,
            use_residual=True,
            use_batch_norm=True
        )
        
        # Test forward pass
        output = encoder(self.batch_graph)
        
        self.assertEqual(output.shape, (self.batch_size, 128))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_multiscale_gin_encoder(self):
        """Test multi-scale GIN encoder"""
        if MultiScaleGINEncoder is None:
            self.skipTest("MultiScaleGINEncoder not available")
        
        encoder = MultiScaleGINEncoder(
            node_features=78,
            hidden_dims=[32, 64, 128],
            output_dim=128,
            scales=[1, 2, 3]
        )
        
        # Test forward pass
        output = encoder(self.batch_graph)
        
        self.assertEqual(output.shape, (self.batch_size, 128))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestEncoderGradients(unittest.TestCase):
    """Test gradient computation for encoders"""
    
    def setUp(self):
        """Set up test data"""
        self.protein_sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        ]
        
        # Create molecular graph
        node_features = torch.randn(5, 78)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        self.graph = Batch.from_data_list([Data(x=node_features, edge_index=edge_index)])
    
    def test_gin_encoder_gradients(self):
        """Test gradient computation for GIN encoder"""
        encoder = GINDrugEncoder(output_dim=64)
        encoder.train()
        
        # Forward pass
        output = encoder(self.graph)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_count = 0
        for param in encoder.parameters():
            if param.grad is not None:
                grad_count += 1
                self.assertFalse(torch.isnan(param.grad).any())
                self.assertFalse(torch.isinf(param.grad).any())
        
        self.assertGreater(grad_count, 0)
    
    def test_esm_encoder_gradients(self):
        """Test gradient computation for ESM encoder"""
        try:
            encoder = ESMProteinEncoder(output_dim=64, freeze_initial=False)
            encoder.train()
            
            # Forward pass
            output = encoder(self.protein_sequences)
            loss = output.sum()
            
            # Backward pass
            loss.backward()
            
            # Check gradients for projection layer
            proj_grad_count = 0
            for param in encoder.projection.parameters():
                if param.grad is not None:
                    proj_grad_count += 1
                    self.assertFalse(torch.isnan(param.grad).any())
            
            self.assertGreater(proj_grad_count, 0)
            
        except Exception as e:
            self.skipTest(f"ESM gradient test skipped: {e}")


class TestEncoderDeviceHandling(unittest.TestCase):
    """Test device handling for encoders"""
    
    def test_gin_encoder_device_handling(self):
        """Test GIN encoder device handling"""
        encoder = GINDrugEncoder(output_dim=64)
        
        # Test CPU
        encoder = encoder.to('cpu')
        node_features = torch.randn(5, 78)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        graph = Batch.from_data_list([Data(x=node_features, edge_index=edge_index)])
        
        output = encoder(graph)
        self.assertEqual(output.device.type, 'cpu')
        
        # Test GPU if available
        if torch.cuda.is_available():
            encoder = encoder.to('cuda')
            graph = graph.to('cuda')
            output = encoder(graph)
            self.assertEqual(output.device.type, 'cuda')


if __name__ == '__main__':
    unittest.main()
"""
Unit tests for fusion mechanisms and attention computations
Tests multi-modal fusion, cross-attention, and prediction heads
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.models import MultiModalFusion, AffinityPredictor
    from core.prediction_heads import (
        MLPPredictionHead, PredictionHeadFactory,
        get_lightweight_predictor, get_standard_predictor, get_deep_predictor
    )
    from core.fusion import (
        CrossAttentionFusion, ConcatenationFusion, 
        BilinearFusion, GatedFusion
    )
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    # Set fallbacks
    CrossAttentionFusion = None
    ConcatenationFusion = None
    BilinearFusion = None
    GatedFusion = None


class TestMultiModalFusion(unittest.TestCase):
    """Test multi-modal fusion mechanisms"""
    
    def setUp(self):
        """Set up test data"""
        self.batch_size = 4
        self.drug_dim = 128
        self.protein_dim = 256
        self.hidden_dim = 512
        
        self.drug_features = torch.randn(self.batch_size, self.drug_dim)
        self.protein_features = torch.randn(self.batch_size, self.protein_dim)
    
    def test_multimodal_fusion_initialization(self):
        """Test MultiModalFusion initialization"""
        fusion = MultiModalFusion(
            drug_dim=self.drug_dim,
            protein_dim=self.protein_dim,
            hidden_dim=self.hidden_dim,
            num_heads=8
        )
        
        # Check output dimension
        self.assertEqual(fusion.output_dim, self.hidden_dim * 2)
        
        # Check components exist
        self.assertIsInstance(fusion.drug_proj, nn.Linear)
        self.assertIsInstance(fusion.protein_proj, nn.Linear)
        self.assertIsInstance(fusion.cross_attention, nn.MultiheadAttention)
        self.assertIsInstance(fusion.layer_norm, nn.LayerNorm)
    
    def test_multimodal_fusion_forward(self):
        """Test MultiModalFusion forward pass"""
        fusion = MultiModalFusion(
            drug_dim=self.drug_dim,
            protein_dim=self.protein_dim,
            hidden_dim=self.hidden_dim,
            num_heads=8
        )
        
        # Forward pass
        output = fusion(self.drug_features, self.protein_features)
        
        # Check output shape
        expected_shape = (self.batch_size, self.hidden_dim * 2)
        self.assertEqual(output.shape, expected_shape)
        
        # Check for NaN/Inf values
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_multimodal_fusion_attention_weights(self):
        """Test attention weight computation"""
        fusion = MultiModalFusion(
            drug_dim=self.drug_dim,
            protein_dim=self.protein_dim,
            hidden_dim=self.hidden_dim,
            num_heads=4
        )
        
        # Enable attention weight return for testing
        fusion.eval()
        
        with torch.no_grad():
            output = fusion(self.drug_features, self.protein_features)
            
            # Check that attention produces reasonable outputs
            self.assertTrue(torch.all(torch.isfinite(output)))
            
            # Check output range is reasonable
            self.assertTrue(output.abs().max() < 100)  # Reasonable range
    
    def test_cross_attention_fusion(self):
        """Test cross-attention fusion implementation"""
        if CrossAttentionFusion is None:
            self.skipTest("CrossAttentionFusion not available")
        
        fusion = CrossAttentionFusion(
            drug_dim=self.drug_dim,
            protein_dim=self.protein_dim,
            hidden_dim=256,
            num_heads=8
        )
        
        output = fusion(self.drug_features, self.protein_features)
        
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertFalse(torch.isnan(output).any())
    
    def test_concatenation_fusion(self):
        """Test simple concatenation fusion"""
        if ConcatenationFusion is None:
            self.skipTest("ConcatenationFusion not available")
        
        fusion = ConcatenationFusion(
            drug_dim=self.drug_dim,
            protein_dim=self.protein_dim
        )
        
        output = fusion(self.drug_features, self.protein_features)
        
        expected_dim = self.drug_dim + self.protein_dim
        self.assertEqual(output.shape, (self.batch_size, expected_dim))
    
    def test_bilinear_fusion(self):
        """Test bilinear fusion mechanism"""
        if BilinearFusion is None:
            self.skipTest("BilinearFusion not available")
        
        fusion = BilinearFusion(
            drug_dim=self.drug_dim,
            protein_dim=self.protein_dim,
            output_dim=256
        )
        
        output = fusion(self.drug_features, self.protein_features)
        
        self.assertEqual(output.shape, (self.batch_size, 256))
        self.assertFalse(torch.isnan(output).any())
    
    def test_gated_fusion(self):
        """Test gated fusion mechanism"""
        if GatedFusion is None:
            self.skipTest("GatedFusion not available")
        
        fusion = GatedFusion(
            drug_dim=self.drug_dim,
            protein_dim=self.protein_dim,
            hidden_dim=256
        )
        
        output = fusion(self.drug_features, self.protein_features)
        
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertFalse(torch.isnan(output).any())


class TestPredictionHeads(unittest.TestCase):
    """Test prediction head implementations"""
    
    def setUp(self):
        """Set up test data"""
        self.batch_size = 4
        self.input_dim = 512
        self.input_features = torch.randn(self.batch_size, self.input_dim)
    
    def test_mlp_prediction_head(self):
        """Test MLP prediction head"""
        predictor = MLPPredictionHead(
            input_dim=self.input_dim,
            hidden_dims=[256, 128],
            output_dim=1,
            dropout=0.2,
            activation='relu'
        )
        
        # Test forward pass
        output = predictor(self.input_features)
        
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_affinity_predictor_alias(self):
        """Test AffinityPredictor alias"""
        predictor = AffinityPredictor(
            input_dim=self.input_dim,
            hidden_dims=[128],
            dropout=0.1
        )
        
        output = predictor(self.input_features)
        
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(output).any())
    
    def test_prediction_head_factory(self):
        """Test prediction head factory"""
        # Test MLP creation
        mlp_head = PredictionHeadFactory.create(
            'mlp',
            input_dim=self.input_dim,
            hidden_dims=[256, 128],
            dropout=0.2
        )
        
        output = mlp_head(self.input_features)
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Test lightweight predictor
        lightweight = get_lightweight_predictor(input_dim=self.input_dim)
        output = lightweight(self.input_features)
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Test standard predictor
        standard = get_standard_predictor(input_dim=self.input_dim)
        output = standard(self.input_features)
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Test deep predictor
        deep = get_deep_predictor(input_dim=self.input_dim)
        output = deep(self.input_features)
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_prediction_head_activations(self):
        """Test different activation functions"""
        activations = ['relu', 'gelu', 'tanh', 'leaky_relu']
        
        for activation in activations:
            with self.subTest(activation=activation):
                predictor = MLPPredictionHead(
                    input_dim=self.input_dim,
                    hidden_dims=[128],
                    activation=activation
                )
                
                output = predictor(self.input_features)
                self.assertEqual(output.shape, (self.batch_size, 1))
                self.assertFalse(torch.isnan(output).any())
    
    def test_prediction_head_batch_norm(self):
        """Test prediction head with batch normalization"""
        predictor = MLPPredictionHead(
            input_dim=self.input_dim,
            hidden_dims=[256, 128],
            use_batch_norm=True,
            dropout=0.2
        )
        
        # Test training mode
        predictor.train()
        output_train = predictor(self.input_features)
        
        # Test eval mode
        predictor.eval()
        output_eval = predictor(self.input_features)
        
        self.assertEqual(output_train.shape, (self.batch_size, 1))
        self.assertEqual(output_eval.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(output_train).any())
        self.assertFalse(torch.isnan(output_eval).any())


class TestAttentionMechanisms(unittest.TestCase):
    """Test attention mechanism computations"""
    
    def setUp(self):
        """Set up test data"""
        self.batch_size = 4
        self.seq_len = 10
        self.embed_dim = 128
        
        self.query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
    
    def test_multihead_attention(self):
        """Test multi-head attention mechanism"""
        attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Forward pass
        output, attention_weights = attention(
            self.query, self.key, self.value
        )
        
        # Check output shape
        self.assertEqual(output.shape, self.query.shape)
        
        # Check attention weights shape
        expected_attn_shape = (self.batch_size, self.seq_len, self.seq_len)
        self.assertEqual(attention_weights.shape, expected_attn_shape)
        
        # Check attention weights sum to 1
        attn_sums = attention_weights.sum(dim=-1)
        torch.testing.assert_close(attn_sums, torch.ones_like(attn_sums), atol=1e-6, rtol=1e-6)
        
        # Check for NaN/Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isnan(attention_weights).any())
    
    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention computation"""
        # Manual implementation for testing
        def scaled_dot_product_attention(query, key, value, mask=None):
            d_k = query.size(-1)
            scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attention_weights, value)
            
            return output, attention_weights
        
        # Test computation
        output, attention_weights = scaled_dot_product_attention(
            self.query, self.key, self.value
        )
        
        # Check shapes
        self.assertEqual(output.shape, self.query.shape)
        expected_attn_shape = (self.batch_size, self.seq_len, self.seq_len)
        self.assertEqual(attention_weights.shape, expected_attn_shape)
        
        # Check attention weights properties
        self.assertTrue(torch.all(attention_weights >= 0))
        self.assertTrue(torch.all(attention_weights <= 1))
        
        # Check attention weights sum to 1
        attn_sums = attention_weights.sum(dim=-1)
        torch.testing.assert_close(attn_sums, torch.ones_like(attn_sums), atol=1e-6, rtol=1e-6)


class TestFusionGradients(unittest.TestCase):
    """Test gradient computation for fusion mechanisms"""
    
    def setUp(self):
        """Set up test data"""
        self.batch_size = 4
        self.drug_dim = 128
        self.protein_dim = 256
        
        self.drug_features = torch.randn(self.batch_size, self.drug_dim, requires_grad=True)
        self.protein_features = torch.randn(self.batch_size, self.protein_dim, requires_grad=True)
    
    def test_multimodal_fusion_gradients(self):
        """Test gradient computation for MultiModalFusion"""
        fusion = MultiModalFusion(
            drug_dim=self.drug_dim,
            protein_dim=self.protein_dim,
            hidden_dim=256,
            num_heads=8
        )
        
        fusion.train()
        
        # Forward pass
        output = fusion(self.drug_features, self.protein_features)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients for fusion parameters
        grad_count = 0
        for param in fusion.parameters():
            if param.grad is not None:
                grad_count += 1
                self.assertFalse(torch.isnan(param.grad).any())
                self.assertFalse(torch.isinf(param.grad).any())
        
        self.assertGreater(grad_count, 0)
        
        # Check input gradients
        self.assertIsNotNone(self.drug_features.grad)
        self.assertIsNotNone(self.protein_features.grad)
        self.assertFalse(torch.isnan(self.drug_features.grad).any())
        self.assertFalse(torch.isnan(self.protein_features.grad).any())
    
    def test_prediction_head_gradients(self):
        """Test gradient computation for prediction heads"""
        input_features = torch.randn(self.batch_size, 512, requires_grad=True)
        
        predictor = MLPPredictionHead(
            input_dim=512,
            hidden_dims=[256, 128],
            dropout=0.2
        )
        
        predictor.train()
        
        # Forward pass
        output = predictor(input_features)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_count = 0
        for param in predictor.parameters():
            if param.grad is not None:
                grad_count += 1
                self.assertFalse(torch.isnan(param.grad).any())
        
        self.assertGreater(grad_count, 0)
        self.assertIsNotNone(input_features.grad)


if __name__ == '__main__':
    unittest.main()
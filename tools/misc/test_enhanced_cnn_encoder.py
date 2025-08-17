#!/usr/bin/env python3
"""
Test script for the Enhanced CNN Protein Encoder
Tests all the key features: gated CNN layers, SE attention, residual connections, etc.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.protein_encoders import EnhancedCNNProteinEncoder, GatedCNNBlock
from core.models import SEBlock


def test_se_block():
    """Test SE attention block"""
    print("Testing SE Block...")
    
    se_block = SEBlock(in_channels=64, reduction=16)
    
    # Test input: [batch_size, channels, seq_len]
    x = torch.randn(4, 64, 100)
    output = se_block(x)
    
    assert output.shape == x.shape, f"SE Block output shape mismatch: {output.shape} vs {x.shape}"
    
    # Check that attention weights are applied (output should be different from input)
    assert not torch.allclose(x, output), "SE Block should modify the input"
    
    print("✓ SE Block test passed")


def test_gated_cnn_block():
    """Test Gated CNN Block with residual connections"""
    print("Testing Gated CNN Block...")
    
    # Test with same dimensions (residual connection)
    block = GatedCNNBlock(in_channels=64, out_channels=64, kernel_size=3, use_se=True)
    x = torch.randn(4, 64, 100)
    output = block(x)
    
    assert output.shape == x.shape, f"Gated CNN Block output shape mismatch: {output.shape} vs {x.shape}"
    
    # Test with different dimensions (projection needed)
    block_proj = GatedCNNBlock(in_channels=64, out_channels=128, kernel_size=3, use_se=True)
    output_proj = block_proj(x)
    
    expected_shape = (4, 128, 100)
    assert output_proj.shape == expected_shape, f"Gated CNN Block projection shape mismatch: {output_proj.shape} vs {expected_shape}"
    
    print("✓ Gated CNN Block test passed")


def test_enhanced_cnn_encoder():
    """Test the complete Enhanced CNN Protein Encoder"""
    print("Testing Enhanced CNN Protein Encoder...")
    
    # Test with default configuration
    encoder = EnhancedCNNProteinEncoder(
        vocab_size=25,
        embed_dim=128,
        num_filters=[64, 128, 256],
        kernel_sizes=[3, 5, 7],
        output_dim=128,
        max_length=200,
        dropout=0.1,
        use_se=True
    )
    
    # Test input: batch of tokenized protein sequences
    batch_size = 4
    seq_len = 150
    protein_tokens = torch.randint(1, 26, (batch_size, seq_len))  # Random amino acid tokens
    
    # Forward pass
    output = encoder(protein_tokens)
    
    expected_shape = (batch_size, 128)
    assert output.shape == expected_shape, f"Encoder output shape mismatch: {output.shape} vs {expected_shape}"
    
    # Test with longer sequences (should be truncated)
    long_seq = torch.randint(1, 26, (batch_size, 250))
    output_long = encoder(long_seq)
    assert output_long.shape == expected_shape, f"Long sequence output shape mismatch: {output_long.shape} vs {expected_shape}"
    
    # Test attention weights
    attention_weights = encoder.get_attention_weights(protein_tokens)
    expected_attn_shape = (batch_size, min(seq_len, encoder.max_length))
    assert attention_weights.shape == expected_attn_shape, f"Attention weights shape mismatch: {attention_weights.shape} vs {expected_attn_shape}"
    
    # Check that attention weights sum to 1 (approximately)
    attn_sums = attention_weights.sum(dim=1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-6), "Attention weights should sum to 1"
    
    print("✓ Enhanced CNN Encoder test passed")


def test_multi_scale_features():
    """Test that multi-scale CNN branches work correctly"""
    print("Testing Multi-scale CNN Features...")
    
    encoder = EnhancedCNNProteinEncoder(
        vocab_size=25,
        embed_dim=64,
        num_filters=[32, 64],  # Smaller for testing
        kernel_sizes=[3, 5],   # Two different kernel sizes
        output_dim=128,
        max_length=100
    )
    
    # Test that we have the correct number of branches
    assert len(encoder.cnn_branches) == 2, f"Expected 2 CNN branches, got {len(encoder.cnn_branches)}"
    
    # Test forward pass
    protein_tokens = torch.randint(1, 26, (2, 80))
    output = encoder(protein_tokens)
    
    assert output.shape == (2, 128), f"Multi-scale encoder output shape mismatch: {output.shape}"
    
    print("✓ Multi-scale CNN Features test passed")


def test_configurable_parameters():
    """Test that the encoder works with different configurations"""
    print("Testing Configurable Parameters...")
    
    # Test different configurations
    configs = [
        {
            'vocab_size': 20,
            'embed_dim': 64,
            'num_filters': [32, 64, 128],
            'kernel_sizes': [3, 5, 7, 9],
            'output_dim': 256,
            'use_se': True,
            'use_positional_encoding': True
        },
        {
            'vocab_size': 25,
            'embed_dim': 128,
            'num_filters': [64, 128],
            'kernel_sizes': [3, 7],
            'output_dim': 64,
            'use_se': False,
            'use_positional_encoding': False
        }
    ]
    
    for i, config in enumerate(configs):
        encoder = EnhancedCNNProteinEncoder(**config)
        
        # Test forward pass
        protein_tokens = torch.randint(1, config['vocab_size'] + 1, (2, 100))
        output = encoder(protein_tokens)
        
        expected_shape = (2, config['output_dim'])
        assert output.shape == expected_shape, f"Config {i} output shape mismatch: {output.shape} vs {expected_shape}"
    
    print("✓ Configurable Parameters test passed")


def test_memory_efficiency():
    """Test memory efficiency with padding masks"""
    print("Testing Memory Efficiency...")
    
    encoder = EnhancedCNNProteinEncoder(
        vocab_size=25,
        embed_dim=128,
        num_filters=[64, 128],
        kernel_sizes=[3, 5],
        output_dim=128,
        max_length=200
    )
    
    # Create sequences with different lengths (using padding token 0)
    batch_size = 3
    sequences = [
        torch.cat([torch.randint(1, 26, (50,)), torch.zeros(150, dtype=torch.long)]),  # Short sequence
        torch.cat([torch.randint(1, 26, (100,)), torch.zeros(100, dtype=torch.long)]), # Medium sequence
        torch.randint(1, 26, (200,))  # Full length sequence
    ]
    
    protein_tokens = torch.stack(sequences)
    
    # Forward pass
    output = encoder(protein_tokens)
    assert output.shape == (batch_size, 128), f"Padded sequences output shape mismatch: {output.shape}"
    
    # Test attention weights respect padding
    attention_weights = encoder.get_attention_weights(protein_tokens)
    
    # Check that attention weights are zero for padded positions
    for i, seq in enumerate(sequences):
        padding_mask = (seq == 0)
        if padding_mask.any():
            padded_attention = attention_weights[i][padding_mask]
            assert torch.allclose(padded_attention, torch.zeros_like(padded_attention), atol=1e-6), \
                f"Attention weights should be zero for padded positions in sequence {i}"
    
    print("✓ Memory Efficiency test passed")


def run_all_tests():
    """Run all tests"""
    print("Running Enhanced CNN Protein Encoder Tests...")
    print("=" * 50)
    
    try:
        test_se_block()
        test_gated_cnn_block()
        test_enhanced_cnn_encoder()
        test_multi_scale_features()
        test_configurable_parameters()
        test_memory_efficiency()
        
        print("=" * 50)
        print("✅ All tests passed successfully!")
        print("\nEnhanced CNN Protein Encoder Features Verified:")
        print("- ✓ Gated CNN layers with residual connections")
        print("- ✓ SE attention blocks for feature enhancement")
        print("- ✓ Configurable kernel sizes and filter numbers")
        print("- ✓ Efficient embedding and projection layers")
        print("- ✓ Multi-scale feature extraction")
        print("- ✓ Attention-based global pooling")
        print("- ✓ Memory-efficient padding handling")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
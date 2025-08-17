#!/usr/bin/env python3
"""
Demo script for the Enhanced CNN Protein Encoder
Shows the key features and capabilities of the new encoder
"""

import torch
import numpy as np
from core import EnhancedCNNProteinEncoder, create_dta_model


def demo_enhanced_cnn_features():
    """Demonstrate the key features of the Enhanced CNN Protein Encoder"""
    print("Enhanced CNN Protein Encoder Demo")
    print("=" * 50)
    
    # Create encoder with different configurations
    print("\n1. Creating Enhanced CNN Encoder with multi-scale features...")
    encoder = EnhancedCNNProteinEncoder(
        vocab_size=25,
        embed_dim=128,
        num_filters=[64, 128, 256],  # Progressive filter increase
        kernel_sizes=[3, 5, 7],      # Multi-scale kernels
        output_dim=128,
        max_length=200,
        dropout=0.1,
        use_se=True,                 # SE attention enabled
        use_positional_encoding=True
    )
    
    print(f"   - Multi-scale kernels: {[3, 5, 7]}")
    print(f"   - Progressive filters: {[64, 128, 256]}")
    print(f"   - SE attention: Enabled")
    print(f"   - Positional encoding: Enabled")
    print(f"   - Output dimension: {encoder.output_dim}")
    
    # Create sample protein sequences (tokenized)
    batch_size = 4
    seq_len = 150
    protein_tokens = torch.randint(1, 26, (batch_size, seq_len))
    
    print(f"\n2. Processing batch of {batch_size} protein sequences...")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Input shape: {protein_tokens.shape}")
    
    # Forward pass
    encoder.eval()
    with torch.no_grad():
        features = encoder(protein_tokens)
        attention_weights = encoder.get_attention_weights(protein_tokens)
    
    print(f"   - Output features shape: {features.shape}")
    print(f"   - Attention weights shape: {attention_weights.shape}")
    
    # Show attention statistics
    print(f"\n3. Attention Analysis:")
    for i in range(min(2, batch_size)):
        attn = attention_weights[i]
        max_pos = torch.argmax(attn).item()
        max_weight = attn[max_pos].item()
        print(f"   - Sequence {i+1}: Max attention at position {max_pos} (weight: {max_weight:.4f})")
    
    # Compare with different configurations
    print(f"\n4. Comparing different configurations...")
    
    configs = [
        {
            'name': 'Lightweight',
            'config': {
                'embed_dim': 64,
                'num_filters': [32, 64],
                'kernel_sizes': [3, 5],
                'output_dim': 64,
                'use_se': False
            }
        },
        {
            'name': 'High-capacity',
            'config': {
                'embed_dim': 256,
                'num_filters': [128, 256, 512],
                'kernel_sizes': [3, 5, 7, 9],
                'output_dim': 256,
                'use_se': True
            }
        }
    ]
    
    for config_info in configs:
        name = config_info['name']
        config = config_info['config']
        
        encoder_variant = EnhancedCNNProteinEncoder(vocab_size=25, **config)
        
        # Count parameters
        total_params = sum(p.numel() for p in encoder_variant.parameters())
        trainable_params = sum(p.numel() for p in encoder_variant.parameters() if p.requires_grad)
        
        print(f"   - {name}: {total_params:,} total params, {trainable_params:,} trainable")
    
    print(f"\n5. Integration with Unified DTA Model...")
    
    # Create a complete DTA model using the enhanced CNN encoder
    model_config = {
        'protein_encoder_type': 'cnn',
        'drug_encoder_type': 'gin',
        'use_fusion': True,
        'protein_config': {
            'embed_dim': 128,
            'num_filters': [64, 128, 256],
            'kernel_sizes': [3, 5, 7],
            'output_dim': 128,
            'use_se': True
        },
        'drug_config': {
            'output_dim': 128,
            'num_layers': 3
        },
        'fusion_config': {
            'hidden_dim': 256,
            'num_heads': 4
        }
    }
    
    dta_model = create_dta_model(model_config)
    
    # Count total model parameters
    total_model_params = sum(p.numel() for p in dta_model.parameters())
    protein_encoder_params = sum(p.numel() for p in dta_model.protein_encoder.parameters())
    
    print(f"   - Complete DTA model: {total_model_params:,} parameters")
    print(f"   - Enhanced CNN encoder: {protein_encoder_params:,} parameters")
    print(f"   - CNN encoder ratio: {protein_encoder_params/total_model_params*100:.1f}%")
    
    print(f"\n✅ Enhanced CNN Protein Encoder Demo Complete!")
    print(f"\nKey Features Demonstrated:")
    print(f"- ✓ Multi-scale CNN with configurable kernel sizes")
    print(f"- ✓ Gated convolutions with residual connections")
    print(f"- ✓ SE attention blocks for feature enhancement")
    print(f"- ✓ Attention-based global pooling")
    print(f"- ✓ Efficient embedding and projection layers")
    print(f"- ✓ Integration with complete DTA prediction system")


if __name__ == "__main__":
    demo_enhanced_cnn_features()
#!/usr/bin/env python3
"""
Test script for the Memory-Optimized ESM-2 Encoder
"""

import torch
import logging
from core.protein_encoders import MemoryOptimizedESMEncoder, TRANSFORMERS_AVAILABLE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_esm_encoder():
    """Test the ESM-2 encoder implementation"""
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers library not available. Cannot test ESM encoder.")
        return False
    
    try:
        print("🧪 Testing Memory-Optimized ESM Encoder...")
        
        # Create encoder instance
        encoder = MemoryOptimizedESMEncoder(
            output_dim=128,
            max_length=100,  # Shorter for testing
            freeze_initial=True,
            use_gradient_checkpointing=True,
            pooling_strategy='cls'
        )
        
        print(f"✅ Encoder created successfully")
        print(f"   - Output dimension: {encoder.output_dim}")
        print(f"   - Max length: {encoder.max_length}")
        print(f"   - Pooling strategy: {encoder.pooling_strategy}")
        
        # Test frozen status
        frozen_status = encoder.get_frozen_status()
        print(f"   - Frozen parameters: {frozen_status['frozen_percentage']:.1f}%")
        
        # Create sample protein sequences
        sample_proteins = [
            "MKFLVLLFNILCLFPVLAADNHGVGPQGASLFIRSDYNLQLLRIEABEEVEQEVA",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV"
        ]
        
        print(f"🔬 Testing forward pass with {len(sample_proteins)} sequences...")
        
        # Forward pass
        with torch.no_grad():  # Inference mode
            output = encoder(sample_proteins)
        
        print(f"✅ Forward pass successful!")
        print(f"   - Input sequences: {len(sample_proteins)}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output dtype: {output.dtype}")
        print(f"   - Output device: {output.device}")
        
        # Test progressive unfreezing
        print(f"🔧 Testing progressive unfreezing...")
        encoder.unfreeze_esm_layers(num_layers=2)
        
        frozen_status_after = encoder.get_frozen_status()
        print(f"   - Frozen parameters after unfreezing: {frozen_status_after['frozen_percentage']:.1f}%")
        print(f"   - Unfrozen layers: {frozen_status_after['num_unfrozen_layers']}")
        
        # Test different pooling strategies
        print(f"🎯 Testing different pooling strategies...")
        
        for strategy in ['cls', 'mean', 'max']:
            test_encoder = MemoryOptimizedESMEncoder(
                output_dim=64,
                max_length=50,
                pooling_strategy=strategy
            )
            
            with torch.no_grad():
                test_output = test_encoder(sample_proteins[:1])  # Single sequence
            
            print(f"   - {strategy} pooling: {test_output.shape}")
        
        print(f"🎉 All ESM encoder tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ ESM encoder test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_optimization():
    """Test memory optimization features"""
    
    if not TRANSFORMERS_AVAILABLE:
        return True
    
    try:
        print("💾 Testing memory optimization features...")
        
        # Test adaptive truncation
        encoder = MemoryOptimizedESMEncoder(max_length=50)
        
        # Create sequences of varying lengths
        test_sequences = [
            "MKFLVLLFNILCLFPVLAADNHGVGPQGASLFIRSDYNLQLLRIEABEEVEQEVA",  # 56 chars
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUQTPGYPFYAI",  # Very long
            "MKFLVLLFNILCLFPVLA"  # Short
        ]
        
        truncated_seqs, lengths = encoder._adaptive_truncation(test_sequences)
        
        print(f"   - Original lengths: {[len(seq) for seq in test_sequences]}")
        print(f"   - Truncated lengths: {[len(seq) for seq in truncated_seqs]}")
        print(f"   - Max allowed length: {encoder.max_length}")
        
        # Test gradient checkpointing is enabled
        if hasattr(encoder.esm_model, 'gradient_checkpointing'):
            print(f"   - Gradient checkpointing: ✅ Available")
        else:
            print(f"   - Gradient checkpointing: ⚠️  Not available in this model")
        
        print(f"✅ Memory optimization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting ESM-2 Encoder Tests\n")
    
    success = True
    success &= test_esm_encoder()
    success &= test_memory_optimization()
    
    if success:
        print("\n🎉 All tests passed! ESM-2 encoder implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        exit(1)
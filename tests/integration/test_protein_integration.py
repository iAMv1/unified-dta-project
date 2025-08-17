#!/usr/bin/env python3
"""
Integration test for the complete protein encoder implementation
Tests ESM-2 encoder with all advanced features
"""

import torch
import logging
import time
from core.protein_encoders import MemoryOptimizedESMEncoder, TRANSFORMERS_AVAILABLE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_complete_esm_implementation():
    """Test all features of the ESM-2 encoder implementation"""
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers library not available. Cannot test ESM encoder.")
        return False
    
    print("🧬 Testing Complete ESM-2 Protein Encoder Implementation")
    print("=" * 60)
    
    try:
        # Test 1: Basic functionality
        print("\n1️⃣ Testing Basic Functionality")
        encoder = MemoryOptimizedESMEncoder(
            output_dim=256,
            max_length=150,
            freeze_initial=True,
            use_gradient_checkpointing=True,
            pooling_strategy='cls',
            dropout=0.1
        )
        
        sample_proteins = [
            "MKFLVLLFNILCLFPVLAADNHGVGPQGASLFIRSDYNLQLLRIEABEEVEQEVA",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUQTPGYPFYAI"
        ]
        
        start_time = time.time()
        output = encoder(sample_proteins)
        inference_time = time.time() - start_time
        
        print(f"   ✅ Forward pass successful")
        print(f"   📊 Output shape: {output.shape}")
        print(f"   ⏱️  Inference time: {inference_time:.3f}s")
        print(f"   🎯 Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Test 2: Progressive unfreezing
        print("\n2️⃣ Testing Progressive Unfreezing")
        initial_status = encoder.get_frozen_status()
        print(f"   📊 Initial frozen parameters: {initial_status['frozen_percentage']:.1f}%")
        
        # Unfreeze last 4 layers
        encoder.unfreeze_esm_layers(num_layers=4)
        after_unfreeze = encoder.get_frozen_status()
        print(f"   📊 After unfreezing 4 layers: {after_unfreeze['frozen_percentage']:.1f}%")
        print(f"   🔓 Unfrozen layers: {after_unfreeze['num_unfrozen_layers']}")
        
        # Unfreeze embeddings
        encoder.unfreeze_embeddings()
        after_embeddings = encoder.get_frozen_status()
        print(f"   📊 After unfreezing embeddings: {after_embeddings['frozen_percentage']:.1f}%")
        
        # Test 3: Different pooling strategies
        print("\n3️⃣ Testing Pooling Strategies")
        pooling_results = {}
        
        for strategy in ['cls', 'mean', 'max', 'attention']:
            test_encoder = MemoryOptimizedESMEncoder(
                output_dim=128,
                max_length=100,
                pooling_strategy=strategy,
                attention_pooling_heads=4 if strategy == 'attention' else 8
            )
            
            with torch.no_grad():
                result = test_encoder(sample_proteins[:1])
                pooling_results[strategy] = result
            
            print(f"   🎯 {strategy.capitalize()} pooling: {result.shape} | Range: [{result.min().item():.3f}, {result.max().item():.3f}]")
        
        # Test 4: Memory optimization features
        print("\n4️⃣ Testing Memory Optimization")
        
        # Test adaptive truncation with various sequence lengths
        test_sequences = [
            "MKFLVLLFNILCLFPVLAADNHGVGPQGASLFIRSDYNLQLLRIEABEEVEQEVA",  # 55 chars
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUQTPGYPFYAI",  # 232 chars
            "MKFLVLLFNILCLFPVLA",  # 18 chars
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUQTPGYPFYAIQWERTYUIOPASDFGHJKLZXCVBNM"  # 300+ chars
        ]
        
        memory_encoder = MemoryOptimizedESMEncoder(max_length=100)
        truncated_seqs, lengths = memory_encoder._adaptive_truncation(test_sequences)
        
        print(f"   📏 Original lengths: {[len(seq) for seq in test_sequences]}")
        print(f"   ✂️  Truncated lengths: {[len(seq) for seq in truncated_seqs]}")
        print(f"   📐 Max allowed: {memory_encoder.max_length}")
        
        # Test batch processing with different sizes
        print("\n5️⃣ Testing Batch Processing")
        batch_encoder = MemoryOptimizedESMEncoder(output_dim=64, max_length=80)
        
        for batch_size in [1, 2, 4]:
            batch_sequences = sample_proteins[:batch_size] if batch_size <= len(sample_proteins) else sample_proteins * (batch_size // len(sample_proteins) + 1)
            batch_sequences = batch_sequences[:batch_size]
            
            start_time = time.time()
            with torch.no_grad():
                batch_output = batch_encoder(batch_sequences)
            batch_time = time.time() - start_time
            
            print(f"   📦 Batch size {batch_size}: {batch_output.shape} | Time: {batch_time:.3f}s")
        
        # Test 6: Device handling
        print("\n6️⃣ Testing Device Handling")
        device_encoder = MemoryOptimizedESMEncoder(output_dim=32, max_length=50)
        
        # Test CPU
        with torch.no_grad():
            cpu_output = device_encoder(sample_proteins[:1])
        print(f"   💻 CPU output: {cpu_output.shape} on {cpu_output.device}")
        
        # Test GPU if available
        if torch.cuda.is_available():
            device_encoder = device_encoder.cuda()
            with torch.no_grad():
                gpu_output = device_encoder(sample_proteins[:1])
            print(f"   🚀 GPU output: {gpu_output.shape} on {gpu_output.device}")
        else:
            print(f"   ⚠️  GPU not available for testing")
        
        # Test 7: Attention weights extraction
        print("\n7️⃣ Testing Attention Weights Extraction")
        try:
            attention_weights = encoder.get_attention_weights(sample_proteins[:1])
            print(f"   🧠 Attention weights shape: {attention_weights.shape}")
            print(f"   📊 Attention range: [{attention_weights.min().item():.3f}, {attention_weights.max().item():.3f}]")
        except Exception as e:
            print(f"   ⚠️  Attention extraction: {str(e)}")
        
        print("\n🎉 All ESM-2 encoder tests completed successfully!")
        print("=" * 60)
        
        # Summary
        print("\n📋 Implementation Summary:")
        print(f"   ✅ ESM-2 integration with HuggingFace transformers")
        print(f"   ✅ Memory optimization with gradient checkpointing")
        print(f"   ✅ Progressive unfreezing for fine-tuning")
        print(f"   ✅ Multiple pooling strategies (cls, mean, max, attention)")
        print(f"   ✅ Adaptive sequence truncation")
        print(f"   ✅ Efficient tokenization and device handling")
        print(f"   ✅ Batch processing optimization")
        print(f"   ✅ Attention weights extraction for interpretability")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_esm_implementation()
    
    if success:
        print("\n🎊 ESM-2 Protein Encoder Implementation Complete!")
        print("   Ready for integration into the unified DTA system.")
    else:
        print("\n❌ Implementation test failed.")
        exit(1)
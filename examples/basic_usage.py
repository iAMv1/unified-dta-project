#!/usr/bin/env python3
"""
Basic Usage Example for Unified DTA System

This example demonstrates the simplest way to use the Unified DTA System
for drug-target affinity prediction.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unified_dta import UnifiedDTAModel
import torch

def main():
    print("=== Unified DTA System - Basic Usage Example ===\n")
    
    # Example drug and protein data
    smiles = "CCO"  # Ethanol
    protein_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    print(f"Drug (SMILES): {smiles}")
    print(f"Protein sequence: {protein_sequence[:50]}...")
    print()
    
    # 1. Load lightweight model (good for testing)
    print("1. Loading lightweight model...")
    try:
        model = UnifiedDTAModel.from_pretrained('lightweight')
        print("✓ Lightweight model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load lightweight model: {e}")
        print("Creating simple model for demonstration...")
        
        # Fallback: create a simple model configuration
        config = {
            'model': {
                'protein_encoder_type': 'cnn',
                'drug_encoder_type': 'gin',
                'use_fusion': False
            },
            'protein_config': {
                'vocab_size': 25,
                'embed_dim': 64,
                'output_dim': 64
            },
            'drug_config': {
                'input_dim': 78,
                'hidden_dim': 64,
                'num_layers': 3,
                'output_dim': 64
            },
            'predictor_config': {
                'hidden_dims': [128],
                'dropout': 0.2
            }
        }
        model = UnifiedDTAModel.from_config(config)
        print("✓ Simple model created")
    
    print()
    
    # 2. Make a single prediction
    print("2. Making single prediction...")
    try:
        prediction = model.predict(smiles, protein_sequence)
        print(f"✓ Predicted binding affinity: {prediction:.4f}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return
    
    print()
    
    # 3. Make batch predictions
    print("3. Making batch predictions...")
    smiles_list = [
        "CCO",           # Ethanol
        "CC(=O)O",       # Acetic acid
        "CC(C)O",        # Isopropanol
        "C1=CC=CC=C1"    # Benzene
    ]
    
    protein_list = [protein_sequence] * len(smiles_list)
    
    try:
        predictions = model.predict_batch(smiles_list, protein_list)
        print("✓ Batch predictions completed:")
        for i, (smi, pred) in enumerate(zip(smiles_list, predictions)):
            print(f"   {i+1}. {smi:12} -> {pred:.4f}")
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
    
    print()
    
    # 4. Model information
    print("4. Model information:")
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"   CUDA available: Yes ({torch.cuda.get_device_name(0)})")
        else:
            print("   CUDA available: No (using CPU)")
            
    except Exception as e:
        print(f"   Could not get model info: {e}")
    
    print()
    
    # 5. Save and load model (optional)
    print("5. Model persistence:")
    try:
        # Save model
        model_path = "example_model.pt"
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Load model
        loaded_model = UnifiedDTAModel.load(model_path)
        print("✓ Model loaded successfully")
        
        # Verify loaded model works
        test_prediction = loaded_model.predict(smiles, protein_sequence)
        print(f"✓ Loaded model prediction: {test_prediction:.4f}")
        
        # Clean up
        os.remove(model_path)
        print("✓ Temporary model file cleaned up")
        
    except Exception as e:
        print(f"✗ Model persistence failed: {e}")
    
    print("\n=== Example completed successfully! ===")

if __name__ == "__main__":
    main()
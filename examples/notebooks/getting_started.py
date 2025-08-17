#!/usr/bin/env python3
"""
Getting Started Tutorial - Python Script Version

This is a Python script version of the getting started tutorial.
You can run this directly or convert it to a Jupyter notebook.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def main():
    print("=== Unified DTA System - Getting Started Tutorial ===\n")
    
    # Import after path setup
    from unified_dta import UnifiedDTAModel
    from unified_dta.utils import MemoryMonitor
    
    # 1. Load model
    print("1. Loading model...")
    try:
        model = UnifiedDTAModel.from_pretrained('lightweight')
        print("✓ Lightweight model loaded")
    except Exception as e:
        print(f"Creating simple model: {e}")
        config = {
            'model': {'protein_encoder_type': 'cnn', 'drug_encoder_type': 'gin', 'use_fusion': False},
            'protein_config': {'output_dim': 64},
            'drug_config': {'output_dim': 64, 'num_layers': 3},
            'predictor_config': {'hidden_dims': [128]}
        }
        model = UnifiedDTAModel.from_config(config)
        print("✓ Simple model created")
    
    # 2. Single prediction
    print("\n2. Making single prediction...")
    drug_smiles = "CCO"
    protein_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    prediction = model.predict(drug_smiles, protein_sequence)
    print(f"✓ Predicted affinity: {prediction:.4f}")
    
    # 3. Batch predictions
    print("\n3. Making batch predictions...")
    drugs = ["CCO", "CC(=O)O", "CC(C)O", "C1=CC=CC=C1", "CC(C)(C)O"]
    proteins = [protein_sequence] * len(drugs)
    
    start_time = time.time()
    predictions = model.predict_batch(drugs, proteins)
    batch_time = time.time() - start_time
    
    print(f"✓ Batch prediction completed in {batch_time:.3f}s")
    for i, (drug, pred) in enumerate(zip(drugs, predictions)):
        print(f"   {i+1}. {drug:12} -> {pred:.4f}")
    
    # 4. Model statistics
    print("\n4. Model information:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Estimated size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # 5. Memory monitoring
    print("\n5. Memory usage monitoring...")
    with MemoryMonitor() as monitor:
        large_predictions = model.predict_batch(drugs * 10, proteins * 10) main()
   __":"__main __name__ ==")

if own modelsng your ry trainint("- T)
    pri"s/ation in docthe documentRead   print("- 
  )tory"recexamples/ di the mples inother exa"- Explore print(  ")
  ps:\nNext stent("    pri ===")
cessfully!mpleted succoal == Tutorin="\rint(    
    pns)}")
_predictioen(largens: {ldictioPre(f"   
    print")1f} MBmory_mb:..peak_mery: {monitor memo"   Peak print(f
    
   
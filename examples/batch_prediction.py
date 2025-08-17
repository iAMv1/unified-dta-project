#!/usr/bin/env python3
"""
Batch Prediction Example for Unified DTA System

This example demonstrates how to efficiently process large datasets
using batch prediction capabilities.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unified_dta import UnifiedDTAModel
from unified_dta.utils import MemoryMonitor

def create_sample_data(n_samples: int = 100) -> Tuple[List[str], List[str], List[float]]:
    """Create sample data for demonstration."""
    
    # Sample SMILES strings (common drug-like molecules)
    sample_smiles = [
        "CCO",                    # Ethanol
        "CC(=O)O",               # Acetic acid
        "CC(C)O",                # Isopropanol
        "C1=CC=CC=C1",           # Benzene
        "CC(C)(C)O",             # tert-Butanol
        "CCCCCCCCCCCCCCCCCC(=O)O", # Stearic acid
        "CC(C)CC(C)(C)C",        # Isooctane
        "C1=CC=C(C=C1)O",        # Phenol
        "CC(=O)N",               # Acetamide
        "CCCCO"                  # Butanol
    ]
    
    # Sample protein sequence (kinase domain)
    protein_sequence = (
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        "RHPKMWVGVLLFRIGGGSSVGAGTTMGKSTTSAAITAACWSRDVLKKNKHVDGCMYEEQNLSV"
        "IRGSIAHAIYLNTLTNMDGTERELLESYIDGRRLVRGDGSFAKLVRPCSLDYVAIHGFLTNYH"
    )
    
    # Generate random combinations
    np.random.seed(42)  # For reproducibility
    smiles_list = np.random.choice(sample_smiles, n_samples).tolist()
    protein_list = [protein_sequence] * n_samples
    
    # Generate synthetic affinities (for demonstration)
    affinities = np.random.normal(5.0, 1.5, n_samples).tolist()
    
    return smiles_list, protein_list, affinities

def benchmark_prediction_methods(model: UnifiedDTAModel, 
                               smiles_list: List[str], 
                               protein_list: List[str]) -> None:
    """Compare different prediction methods."""
    
    print("=== Prediction Method Comparison ===\n")
    
    n_samples = len(smiles_list)
    
    # Method 1: Individual predictions
    print(f"1. Individual predictions ({n_samples} samples)...")
    start_time = time.time()
    individual_predictions = []
    
    for smiles, protein in zip(smiles_list[:10], protein_list[:10]):  # Limit for demo
        pred = model.predict(smiles, protein)
        individual_predictions.append(pred)
    
    individual_time = time.time() - start_time
    print(f"   Time: {individual_time:.3f}s ({individual_time/10:.3f}s per sample)")
    print(f"   Sample predictions: {individual_predictions[:3]}")
    
    # Method 2: Batch prediction
    print(f"\n2. Batch prediction ({n_samples} samples)...")
    start_time = time.time()
    
    batch_predictions = model.predict_batch(smiles_list, protein_list)
    
    batch_time = time.time() - start_time
    print(f"   Time: {batch_time:.3f}s ({batch_time/n_samples:.3f}s per sample)")
    print(f"   Sample predictions: {batch_predictions[:3]}")
    
    # Speed comparison
    if individual_time > 0:
        speedup = (individual_time / 10) / (batch_time / n_samples)
        print(f"\n   Batch processing speedup: {speedup:.1f}x")

def process_with_memory_monitoring(model: UnifiedDTAModel,
                                 smiles_list: List[str],
                                 protein_list: List[str]) -> None:
    """Demonstrate memory monitoring during batch processing."""
    
    print("\n=== Memory Usage Monitoring ===\n")
    
    with MemoryMonitor() as monitor:
        print("Processing batch with memory monitoring...")
        predictions = model.predict_batch(smiles_list, protein_list)
        
    print(f"Peak memory usage: {monitor.peak_memory_mb:.1f} MB")
    print(f"Memory efficiency: {len(predictions) / monitor.peak_memory_mb:.1f} predictions/MB")

def save_results_to_csv(smiles_list: List[str],
                       protein_list: List[str],
                       predictions: List[float],
                       true_affinities: List[float],
                       filename: str = "batch_predictions.csv") -> None:
    """Save batch prediction results to CSV."""
    
    print(f"\n=== Saving Results to {filename} ===\n")
    
    # Create DataFrame
    df = pd.DataFrame({
        'smiles': smiles_list,
        'protein_sequence': [seq[:50] + "..." for seq in protein_list],  # Truncate for readability
        'predicted_affinity': predictions,
        'true_affinity': true_affinities,
        'absolute_error': [abs(pred - true) for pred, true in zip(predictions, true_affinities)]
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"✓ Results saved to {filename}")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"   Mean Absolute Error: {df['absolute_error'].mean():.3f}")
    print(f"   RMSE: {np.sqrt(np.mean(df['absolute_error']**2)):.3f}")
    print(f"   Correlation: {df['predicted_affinity'].corr(df['true_affinity']):.3f}")
    
    # Display first few rows
    print(f"\nFirst 5 predictions:")
    print(df[['smiles', 'predicted_affinity', 'true_affinity']].head())

def demonstrate_chunked_processing(model: UnifiedDTAModel,
                                 smiles_list: List[str],
                                 protein_list: List[str],
                                 chunk_size: int = 32) -> List[float]:
    """Demonstrate processing large datasets in chunks."""
    
    print(f"\n=== Chunked Processing (chunk_size={chunk_size}) ===\n")
    
    all_predictions = []
    n_samples = len(smiles_list)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    print(f"Processing {n_samples} samples in {n_chunks} chunks...")
    
    for i in range(0, n_samples, chunk_size):
        chunk_smiles = smiles_list[i:i+chunk_size]
        chunk_proteins = protein_list[i:i+chunk_size]
        
        print(f"   Chunk {i//chunk_size + 1}/{n_chunks}: {len(chunk_smiles)} samples")
        
        chunk_predictions = model.predict_batch(chunk_smiles, chunk_proteins)
        all_predictions.extend(chunk_predictions)
    
    print(f"✓ Completed chunked processing: {len(all_predictions)} predictions")
    return all_predictions

def main():
    print("=== Unified DTA System - Batch Prediction Example ===\n")
    
    # 1. Create sample data
    print("1. Creating sample dataset...")
    n_samples = 200
    smiles_list, protein_list, true_affinities = create_sample_data(n_samples)
    print(f"✓ Created dataset with {n_samples} samples")
    print(f"   Sample SMILES: {smiles_list[:3]}")
    print(f"   Protein length: {len(protein_list[0])} residues")
    
    # 2. Load model
    print("\n2. Loading model...")
    try:
        model = UnifiedDTAModel.from_pretrained('lightweight')
        print("✓ Lightweight model loaded")
    except Exception as e:
        print(f"Could not load pretrained model: {e}")
        print("Creating simple model for demonstration...")
        
        config = {
            'model': {
                'protein_encoder_type': 'cnn',
                'drug_encoder_type': 'gin',
                'use_fusion': False
            },
            'protein_config': {'output_dim': 64},
            'drug_config': {'output_dim': 64, 'num_layers': 3},
            'predictor_config': {'hidden_dims': [128]}
        }
        model = UnifiedDTAModel.from_config(config)
        print("✓ Simple model created")
    
    # 3. Benchmark prediction methods
    benchmark_prediction_methods(model, smiles_list, protein_list)
    
    # 4. Full batch prediction with memory monitoring
    process_with_memory_monitoring(model, smiles_list, protein_list)
    
    # 5. Chunked processing demonstration
    chunk_predictions = demonstrate_chunked_processing(
        model, smiles_list, protein_list, chunk_size=50
    )
    
    # 6. Save results
    save_results_to_csv(
        smiles_list, protein_list, chunk_predictions, true_affinities
    )
    
    # 7. Performance tips
    print("\n=== Performance Tips ===\n")
    print("1. Use batch prediction for multiple samples (much faster)")
    print("2. Process large datasets in chunks to manage memory")
    print("3. Monitor memory usage with MemoryMonitor")
    print("4. Use lightweight models for development/testing")
    print("5. Consider GPU acceleration for large batches")
    
    # Clean up
    if os.path.exists("batch_predictions.csv"):
        os.remove("batch_predictions.csv")
        print("\n✓ Cleaned up temporary files")
    
    print("\n=== Batch prediction example completed! ===")

if __name__ == "__main__":
    main()
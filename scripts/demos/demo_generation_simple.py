"""
Simple demo for drug generation capabilities
Demonstrates transformer-based SMILES generation without heavy dependencies
"""

import torch
import argparse
from test_generation_standalone import (
    SMILESTokenizer,
    SimpleProteinEncoder,
    ProteinConditionedGenerator
)


def demo_basic_generation():
    """Demonstrate basic drug generation"""
    print("=== Basic Drug Generation Demo ===\n")
    
    # Sample protein sequences (shortened for demo)
    proteins = [
        "MKLLVLSLSLVLVAPMAAQAAEITLKAVSRSLNCACELKCSTSLLLEACTFRRP",  # Insulin receptor fragment
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLL",  # KRAS fragment
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQW"   # p53 fragment
    ]
    
    print(f"Loaded {len(proteins)} sample protein sequences")
    for i, protein in enumerate(proteins):
        print(f"  Protein {i+1}: {protein[:30]}... (length: {len(protein)})")
    print()
    
    # Create lightweight generator
    print("Creating lightweight generator...")
    protein_encoder = SimpleProteinEncoder(output_dim=64)
    tokenizer = SMILESTokenizer()
    
    generator = ProteinConditionedGenerator(
        protein_encoder=protein_encoder,
        vocab_size=len(tokenizer),
        d_model=64,
        nhead=4,
        num_layers=2,
        max_length=32
    )
    
    print(f"Model parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print()
    
    # Generate molecules
    print("Generating molecules...")
    generator.eval()
    
    try:
        with torch.no_grad():
            # Generate with different strategies
            print("1. Deterministic generation:")
            deterministic_results = generator.generate(
                protein_sequences=proteins,
                max_length=20,
                deterministic=True,
                num_return_sequences=1
            )
            
            for i, (protein, smiles) in enumerate(zip(proteins, deterministic_results)):
                print(f"   Protein {i+1}: {smiles}")
            
            print("\n2. Stochastic generation (multiple per protein):")
            stochastic_results = generator.generate(
                protein_sequences=proteins[:2],  # Use first 2 for demo
                max_length=20,
                temperature=1.2,
                deterministic=False,
                num_return_sequences=3
            )
            
            for i, (protein, smiles_list) in enumerate(zip(proteins[:2], stochastic_results)):
                print(f"   Protein {i+1}:")
                for j, smiles in enumerate(smiles_list):
                    print(f"     {j+1}. {smiles}")
            
            print("\n3. Temperature sampling:")
            temperatures = [0.5, 1.0, 1.5]
            
            for temp in temperatures:
                temp_results = generator.generate(
                    protein_sequences=[proteins[0]],  # Use first protein
                    max_length=15,
                    temperature=temp,
                    deterministic=False,
                    num_return_sequences=2
                )
                
                print(f"   Temperature {temp}: {temp_results[0]}")
        
        print("\n✓ Generation completed successfully!")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Note: This is a demo with untrained model - results are random")


def demo_tokenizer_functionality():
    """Demonstrate tokenizer functionality"""
    print("=== Tokenizer Functionality Demo ===\n")
    
    tokenizer = SMILESTokenizer()
    
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.SPECIAL_TOKENS}")
    print()
    
    # Test molecules
    test_molecules = [
        "CCO",           # Ethanol
        "c1ccccc1",      # Benzene
        "CC(=O)O",       # Acetic acid
        "CCN(CC)CC",     # Triethylamine
        "c1ccc(O)cc1"    # Phenol
    ]
    
    print("Tokenization examples:")
    for smiles in test_molecules:
        tokens = tokenizer.encode(smiles)
        decoded = tokenizer.decode(tokens)
        
        print(f"  SMILES: {smiles}")
        print(f"  Tokens: {tokens}")
        print(f"  Decoded: {decoded}")
        print(f"  Match: {'✓' if decoded == smiles else '✗'}")
        print()


def demo_model_architecture():
    """Demonstrate model architecture details"""
    print("=== Model Architecture Demo ===\n")
    
    # Create components
    protein_encoder = SimpleProteinEncoder(output_dim=128)
    tokenizer = SMILESTokenizer()
    
    generator = ProteinConditionedGenerator(
        protein_encoder=protein_encoder,
        vocab_size=len(tokenizer),
        d_model=256,
        nhead=8,
        num_layers=4,
        max_length=64
    )
    
    print("Model Architecture:")
    print(f"  Protein Encoder Output Dim: {protein_encoder.output_dim}")
    print(f"  Vocabulary Size: {len(tokenizer)}")
    print(f"  Transformer Dimension: 256")
    print(f"  Attention Heads: 8")
    print(f"  Decoder Layers: 4")
    print(f"  Max Generation Length: 64")
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    
    print("Parameter Count:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print()
    
    # Test forward pass
    test_proteins = ["MKLLVLSLSLVLVAPMAAQAA"]
    
    print("Testing forward pass...")
    try:
        with torch.no_grad():
            # Test protein encoding
            protein_features = generator.encode_protein(test_proteins)
            print(f"  Protein encoding shape: {protein_features.shape}")
            
            # Test generation
            generated = generator.generate(
                protein_sequences=test_proteins,
                max_length=10,
                deterministic=True
            )
            print(f"  Generated molecule: {generated[0]}")
        
        print("  ✓ Forward pass successful")
        
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")


def demo_generation_strategies():
    """Demonstrate different generation strategies"""
    print("=== Generation Strategies Demo ===\n")
    
    # Create model
    protein_encoder = SimpleProteinEncoder(output_dim=32)
    tokenizer = SMILESTokenizer()
    
    generator = ProteinConditionedGenerator(
        protein_encoder=protein_encoder,
        vocab_size=len(tokenizer),
        d_model=32,
        nhead=2,
        num_layers=1,
        max_length=16
    )
    
    test_protein = ["MKLLVLSLSLVLVAPMAAQAA"]
    
    generator.eval()
    
    print("Comparing generation strategies:")
    
    strategies = [
        ("Greedy (deterministic)", {"deterministic": True, "temperature": 1.0}),
        ("Low temperature", {"deterministic": False, "temperature": 0.5}),
        ("Medium temperature", {"deterministic": False, "temperature": 1.0}),
        ("High temperature", {"deterministic": False, "temperature": 1.5}),
    ]
    
    for strategy_name, params in strategies:
        print(f"\n{strategy_name}:")
        
        try:
            with torch.no_grad():
                results = generator.generate(
                    protein_sequences=test_protein,
                    max_length=12,
                    num_return_sequences=3,
                    **params
                )
            
            for i, smiles in enumerate(results[0]):
                print(f"  {i+1}. {smiles}")
                
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Simple Drug Generation Demo")
    parser.add_argument('--demo', type=str, 
                       choices=['basic', 'tokenizer', 'architecture', 'strategies', 'all'],
                       default='all', help='Which demo to run')
    
    args = parser.parse_args()
    
    print("Simple Drug Generation Demo")
    print("=" * 50)
    print("Note: This demo uses untrained models for demonstration purposes.")
    print("Generated molecules are random and not chemically meaningful.\n")
    
    if args.demo in ['basic', 'all']:
        demo_basic_generation()
        print("\n" + "="*60 + "\n")
    
    if args.demo in ['tokenizer', 'all']:
        demo_tokenizer_functionality()
        print("\n" + "="*60 + "\n")
    
    if args.demo in ['architecture', 'all']:
        demo_model_architecture()
        print("\n" + "="*60 + "\n")
    
    if args.demo in ['strategies', 'all']:
        demo_generation_strategies()
        print("\n" + "="*60 + "\n")
    
    print("Demo completed!")
    print("\nNext steps:")
    print("1. Train the model on real drug-protein interaction data")
    print("2. Implement chemical validity checking with RDKit")
    print("3. Add molecular property prediction")
    print("4. Evaluate generation quality with diversity and novelty metrics")


if __name__ == "__main__":
    main()
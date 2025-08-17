"""
Simple test for drug generation capabilities
Tests core functionality without heavy dependencies
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_smiles_tokenizer():
    """Test SMILES tokenizer basic functionality"""
    print("Testing SMILES Tokenizer...")
    
    try:
        from core.drug_generation import SMILESTokenizer
        
        tokenizer = SMILESTokenizer()
        
        # Test basic functionality
        test_smiles = "CCO"
        tokens = tokenizer.encode(test_smiles)
        decoded = tokenizer.decode(tokens)
        
        print(f"  Original: {test_smiles}")
        print(f"  Tokens: {tokens}")
        print(f"  Decoded: {decoded}")
        
        assert decoded == test_smiles, f"Decode mismatch: {decoded} != {test_smiles}"
        assert len(tokenizer) > 0, "Tokenizer vocabulary is empty"
        
        print("  ✓ SMILES Tokenizer test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ SMILES Tokenizer test failed: {e}")
        return False


def test_transformer_decoder():
    """Test transformer decoder basic functionality"""
    print("Testing Transformer Decoder...")
    
    try:
        from core.drug_generation import TransformerDecoder
        
        vocab_size = 50
        d_model = 32
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            max_length=16
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 8
        memory_len = 4
        
        tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
        memory = torch.randn(batch_size, memory_len, d_model)
        
        output = decoder(tgt, memory)
        
        expected_shape = (batch_size, seq_len, vocab_size)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
        
        # Test mask generation
        mask = decoder.generate_square_subsequent_mask(seq_len)
        assert mask.shape == (seq_len, seq_len), f"Mask shape mismatch: {mask.shape}"
        
        print(f"  Output shape: {output.shape}")
        print(f"  Mask shape: {mask.shape}")
        print("  ✓ Transformer Decoder test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Transformer Decoder test failed: {e}")
        return False


def test_protein_encoder():
    """Test protein encoder basic functionality"""
    print("Testing Protein Encoder...")
    
    try:
        from core.models import ESMProteinEncoder
        
        encoder = ESMProteinEncoder(output_dim=32)
        
        # Test with short sequences to avoid memory issues
        test_proteins = ["MKLLVL", "AAAAAA"]
        
        with torch.no_grad():
            output = encoder(test_proteins)
        
        expected_shape = (len(test_proteins), 32)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
        
        print(f"  Input proteins: {len(test_proteins)}")
        print(f"  Output shape: {output.shape}")
        print("  ✓ Protein Encoder test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Protein Encoder test failed: {e}")
        return False


def test_generation_pipeline():
    """Test basic generation pipeline"""
    print("Testing Generation Pipeline...")
    
    try:
        from core.models import ESMProteinEncoder
        from core.drug_generation import ProteinConditionedGenerator, SMILESTokenizer
        
        # Create lightweight components
        protein_encoder = ESMProteinEncoder(output_dim=16)
        tokenizer = SMILESTokenizer()
        
        generator = ProteinConditionedGenerator(
            protein_encoder=protein_encoder,
            vocab_size=len(tokenizer),
            d_model=16,
            nhead=2,
            num_layers=1,
            max_length=8
        )
        
        # Test generation
        test_proteins = ["MKLLVL"]
        
        generator.eval()
        with torch.no_grad():
            generated = generator.generate(
                protein_sequences=test_proteins,
                max_length=6,
                deterministic=True,
                num_return_sequences=1
            )
        
        assert len(generated) == len(test_proteins), f"Output count mismatch: {len(generated)} != {len(test_proteins)}"
        assert isinstance(generated[0], str), f"Output type mismatch: {type(generated[0])}"
        
        print(f"  Input proteins: {len(test_proteins)}")
        print(f"  Generated: {generated}")
        print("  ✓ Generation Pipeline test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Generation Pipeline test failed: {e}")
        return False


def test_chemical_validator():
    """Test chemical validator without RDKit"""
    print("Testing Chemical Validator (basic)...")
    
    try:
        from core.drug_generation import ChemicalValidator
        
        validator = ChemicalValidator()
        
        # Test basic validation (will return False without RDKit, but shouldn't crash)
        test_smiles = ["CCO", "INVALID", "c1ccccc1"]
        
        for smiles in test_smiles:
            is_valid = validator.is_valid_smiles(smiles)
            print(f"  {smiles}: {'Valid' if is_valid else 'Invalid/Unknown'}")
        
        # Test property calculation (should return empty dict without RDKit)
        properties = validator.calculate_properties("CCO")
        print(f"  Properties available: {len(properties) > 0}")
        
        print("  ✓ Chemical Validator test passed (basic)")
        return True
        
    except Exception as e:
        print(f"  ✗ Chemical Validator test failed: {e}")
        return False


def test_scoring_components():
    """Test scoring components basic functionality"""
    print("Testing Scoring Components...")
    
    try:
        from core.generation_scoring import MolecularPropertyCalculator
        
        calculator = MolecularPropertyCalculator()
        
        # Test drug-likeness scoring (should work without RDKit, return 0)
        test_smiles = "CCO"
        drug_score = calculator.calculate_drug_likeness_score(test_smiles)
        
        assert isinstance(drug_score, float), f"Drug score type mismatch: {type(drug_score)}"
        assert 0.0 <= drug_score <= 1.0, f"Drug score out of range: {drug_score}"
        
        print(f"  Drug-likeness score for {test_smiles}: {drug_score}")
        
        # Test synthetic accessibility
        sa_score = calculator.calculate_synthetic_accessibility(test_smiles)
        
        assert isinstance(sa_score, float), f"SA score type mismatch: {type(sa_score)}"
        assert 0.0 <= sa_score <= 1.0, f"SA score out of range: {sa_score}"
        
        print(f"  Synthetic accessibility for {test_smiles}: {sa_score}")
        print("  ✓ Scoring Components test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Scoring Components test failed: {e}")
        return False


def run_simple_tests():
    """Run all simple tests"""
    print("Running Simple Drug Generation Tests")
    print("=" * 50)
    
    tests = [
        test_smiles_tokenizer,
        test_transformer_decoder,
        test_protein_encoder,
        test_generation_pipeline,
        test_chemical_validator,
        test_scoring_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"Test {test_func.__name__} crashed: {e}")
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_simple_tests()
    exit(0 if success else 1)
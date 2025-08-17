"""
Standalone test for drug generation capabilities
Tests core functionality without importing problematic dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict
from abc import ABC, abstractmethod


# ============================================================================
# Standalone implementations for testing
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SMILESTokenizer:
    """Simple SMILES tokenizer"""
    
    SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']
    
    def __init__(self):
        self.chars = list("()[]{}.-=+#%0123456789@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        self.vocab = self.SPECIAL_TOKENS + self.chars
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
    
    def encode(self, smiles: str, max_length: Optional[int] = None) -> List[int]:
        tokens = [self.char_to_idx.get('<sos>', 1)]
        
        for char in smiles:
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                tokens.append(self.char_to_idx.get('<unk>', 3))
        
        tokens.append(self.char_to_idx.get('<eos>', 2))
        
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [self.char_to_idx.get('<eos>', 2)]
            else:
                tokens.extend([self.char_to_idx.get('<pad>', 0)] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        chars = []
        for token in tokens:
            if token == self.char_to_idx.get('<eos>', 2):
                break
            if token not in [self.char_to_idx.get('<pad>', 0), self.char_to_idx.get('<sos>', 1)]:
                chars.append(self.idx_to_char.get(token, '<unk>'))
        
        return ''.join(chars)
    
    def __len__(self):
        return len(self.vocab)
    
    @property
    def pad_token_id(self):
        return self.char_to_idx.get('<pad>', 0)
    
    @property
    def sos_token_id(self):
        return self.char_to_idx.get('<sos>', 1)
    
    @property
    def eos_token_id(self):
        return self.char_to_idx.get('<eos>', 2)


class TransformerDecoder(nn.Module):
    """Simple transformer decoder for SMILES generation"""
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_length: int = 128):
        super().__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_length)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_embedded = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        
        output = self.transformer_decoder(
            tgt=tgt_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        logits = self.output_projection(output)
        return logits
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class SimpleProteinEncoder(nn.Module):
    """Simple protein encoder for testing"""
    
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        
        # Simple embedding-based encoder
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        
        self.embedding = nn.Embedding(len(self.amino_acids) + 1, 64)  # +1 for unknown
        self.lstm = nn.LSTM(64, output_dim // 2, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(output_dim, output_dim)
    
    def forward(self, protein_sequences: List[str]) -> torch.Tensor:
        batch_size = len(protein_sequences)
        max_len = min(max(len(seq) for seq in protein_sequences), 200)
        
        # Convert sequences to indices
        sequences = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for i, seq in enumerate(protein_sequences):
            for j, aa in enumerate(seq[:max_len]):
                sequences[i, j] = self.aa_to_idx.get(aa, len(self.amino_acids))
        
        # Embed and encode
        embedded = self.embedding(sequences)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use final hidden state
        final_hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # Concatenate bidirectional
        
        return self.projection(final_hidden)


class ProteinConditionedGenerator(nn.Module):
    """Simple protein-conditioned generator for testing"""
    
    def __init__(self,
                 protein_encoder,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 max_length: int = 128):
        super().__init__()
        
        self.protein_encoder = protein_encoder
        self.max_length = max_length
        
        self.protein_projection = nn.Linear(protein_encoder.output_dim, d_model)
        
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_length=max_length
        )
        
        self.tokenizer = SMILESTokenizer()
    
    def encode_protein(self, protein_sequences: List[str]) -> torch.Tensor:
        protein_features = self.protein_encoder(protein_sequences)
        protein_encoded = self.protein_projection(protein_features)
        return protein_encoded.unsqueeze(1)
    
    def generate(self,
                 protein_sequences: List[str],
                 max_length: Optional[int] = None,
                 temperature: float = 1.0,
                 deterministic: bool = False,
                 num_return_sequences: int = 1) -> List[str]:
        
        if max_length is None:
            max_length = self.max_length
        
        self.eval()
        device = next(self.parameters()).device
        
        protein_memory = self.encode_protein(protein_sequences)
        batch_size = len(protein_sequences)
        
        if num_return_sequences > 1:
            protein_memory = protein_memory.repeat_interleave(num_return_sequences, dim=0)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size
        
        generated_tokens = torch.full(
            (effective_batch_size, max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device
        )
        generated_tokens[:, 0] = self.tokenizer.sos_token_id
        
        finished = torch.zeros(effective_batch_size, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            for step in range(1, max_length):
                current_tokens = generated_tokens[:, :step]
                tgt_mask = self.decoder.generate_square_subsequent_mask(step).to(device)
                
                logits = self.decoder(
                    tgt=current_tokens,
                    memory=protein_memory,
                    tgt_mask=tgt_mask
                )
                
                next_token_logits = logits[:, -1, :]
                
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                if deterministic:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                generated_tokens[:, step] = next_tokens
                finished |= (next_tokens == self.tokenizer.eos_token_id)
                
                if finished.all():
                    break
        
        results = []
        for i in range(batch_size):
            protein_results = []
            for j in range(num_return_sequences):
                idx = i * num_return_sequences + j if num_return_sequences > 1 else i
                tokens = generated_tokens[idx].cpu().tolist()
                smiles = self.tokenizer.decode(tokens)
                protein_results.append(smiles)
            
            if num_return_sequences == 1:
                results.append(protein_results[0])
            else:
                results.append(protein_results)
        
        return results


# ============================================================================
# Test Functions
# ============================================================================

def test_smiles_tokenizer():
    """Test SMILES tokenizer"""
    print("Testing SMILES Tokenizer...")
    
    try:
        tokenizer = SMILESTokenizer()
        
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
    """Test transformer decoder"""
    print("Testing Transformer Decoder...")
    
    try:
        vocab_size = 50
        d_model = 32
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            max_length=16
        )
        
        batch_size = 2
        seq_len = 8
        memory_len = 4
        
        tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
        memory = torch.randn(batch_size, memory_len, d_model)
        
        output = decoder(tgt, memory)
        
        expected_shape = (batch_size, seq_len, vocab_size)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
        
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
    """Test simple protein encoder"""
    print("Testing Simple Protein Encoder...")
    
    try:
        encoder = SimpleProteinEncoder(output_dim=32)
        
        test_proteins = ["MKLLVL", "AAAAAA"]
        
        with torch.no_grad():
            output = encoder(test_proteins)
        
        expected_shape = (len(test_proteins), 32)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
        
        print(f"  Input proteins: {len(test_proteins)}")
        print(f"  Output shape: {output.shape}")
        print("  ✓ Simple Protein Encoder test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Simple Protein Encoder test failed: {e}")
        return False


def test_generation_pipeline():
    """Test generation pipeline"""
    print("Testing Generation Pipeline...")
    
    try:
        protein_encoder = SimpleProteinEncoder(output_dim=16)
        tokenizer = SMILESTokenizer()
        
        generator = ProteinConditionedGenerator(
            protein_encoder=protein_encoder,
            vocab_size=len(tokenizer),
            d_model=16,
            nhead=2,
            num_layers=1,
            max_length=8
        )
        
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


def test_multiple_generation():
    """Test generating multiple sequences"""
    print("Testing Multiple Sequence Generation...")
    
    try:
        protein_encoder = SimpleProteinEncoder(output_dim=16)
        tokenizer = SMILESTokenizer()
        
        generator = ProteinConditionedGenerator(
            protein_encoder=protein_encoder,
            vocab_size=len(tokenizer),
            d_model=16,
            nhead=2,
            num_layers=1,
            max_length=8
        )
        
        test_proteins = ["MKLLVL"]
        
        generator.eval()
        with torch.no_grad():
            generated = generator.generate(
                protein_sequences=test_proteins,
                max_length=6,
                deterministic=False,
                num_return_sequences=3
            )
        
        assert len(generated) == len(test_proteins), f"Output count mismatch"
        assert isinstance(generated[0], list), f"Output should be list for multiple sequences"
        assert len(generated[0]) == 3, f"Should generate 3 sequences per protein"
        
        print(f"  Input proteins: {len(test_proteins)}")
        print(f"  Generated sequences per protein: {len(generated[0])}")
        print(f"  Sample generations: {generated[0]}")
        print("  ✓ Multiple Sequence Generation test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Multiple Sequence Generation test failed: {e}")
        return False


def test_batch_generation():
    """Test batch generation"""
    print("Testing Batch Generation...")
    
    try:
        protein_encoder = SimpleProteinEncoder(output_dim=16)
        tokenizer = SMILESTokenizer()
        
        generator = ProteinConditionedGenerator(
            protein_encoder=protein_encoder,
            vocab_size=len(tokenizer),
            d_model=16,
            nhead=2,
            num_layers=1,
            max_length=8
        )
        
        test_proteins = ["MKLLVL", "AAAAAA", "CCCCCC"]
        
        generator.eval()
        with torch.no_grad():
            generated = generator.generate(
                protein_sequences=test_proteins,
                max_length=6,
                deterministic=True,
                num_return_sequences=1
            )
        
        assert len(generated) == len(test_proteins), f"Output count mismatch"
        
        for i, smiles in enumerate(generated):
            assert isinstance(smiles, str), f"Output {i} should be string"
        
        print(f"  Input proteins: {len(test_proteins)}")
        print(f"  Generated molecules: {generated}")
        print("  ✓ Batch Generation test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Batch Generation test failed: {e}")
        return False


def run_standalone_tests():
    """Run all standalone tests"""
    print("Running Standalone Drug Generation Tests")
    print("=" * 50)
    
    tests = [
        test_smiles_tokenizer,
        test_transformer_decoder,
        test_protein_encoder,
        test_generation_pipeline,
        test_multiple_generation,
        test_batch_generation
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
    success = run_standalone_tests()
    exit(0 if success else 1)
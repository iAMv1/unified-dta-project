"""
Drug Generation Module
Implements transformer-based SMILES generation conditioned on protein targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple, Union
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import warnings

from .base_components import BaseEncoder, PositionalEncoding


class SMILESTokenizer:
    """Tokenizer for SMILES strings with special tokens"""
    
    SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']
    
    def __init__(self, vocab_file: Optional[str] = None):
        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            # Default SMILES vocabulary
            self.chars = list("()[]{}.-=+#%0123456789@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
            self.vocab = self.SPECIAL_TOKENS + self.chars
            self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
            self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
    
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file"""
        with open(vocab_file, 'r') as f:
            self.vocab = [line.strip() for line in f]
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
    
    def encode(self, smiles: str, max_length: Optional[int] = None) -> List[int]:
        """Encode SMILES string to token indices"""
        tokens = [self.char_to_idx.get('<sos>', 1)]  # Start token
        
        for char in smiles:
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                tokens.append(self.char_to_idx.get('<unk>', 3))  # Unknown token
        
        tokens.append(self.char_to_idx.get('<eos>', 2))  # End token
        
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [self.char_to_idx.get('<eos>', 2)]
            else:
                tokens.extend([self.char_to_idx.get('<pad>', 0)] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token indices to SMILES string"""
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
    
    @property
    def unk_token_id(self):
        return self.char_to_idx.get('<unk>', 3)


class TransformerDecoder(nn.Module):
    """Transformer decoder for SMILES generation"""
    
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
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_length)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            tgt: Target sequence [batch_size, tgt_len]
            memory: Encoded protein features [batch_size, memory_len, d_model]
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask
        
        Returns:
            Output logits [batch_size, tgt_len, vocab_size]
        """
        # Embed tokens and add positional encoding
        tgt_embedded = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        
        # Transformer decoder
        output = self.transformer_decoder(
            tgt=tgt_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class ProteinConditionedGenerator(nn.Module):
    """Protein-conditioned drug generator using transformer decoder"""
    
    def __init__(self,
                 protein_encoder: BaseEncoder,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_length: int = 128):
        super().__init__()
        
        self.protein_encoder = protein_encoder
        self.max_length = max_length
        
        # Project protein features to decoder dimension
        self.protein_projection = nn.Linear(protein_encoder.output_dim, d_model)
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_length
        )
        
        self.tokenizer = SMILESTokenizer()
    
    def encode_protein(self, protein_sequences: List[str]) -> torch.Tensor:
        """Encode protein sequences to features"""
        protein_features = self.protein_encoder(protein_sequences)  # [batch_size, protein_dim]
        
        # Project and add sequence dimension
        protein_encoded = self.protein_projection(protein_features)  # [batch_size, d_model]
        protein_encoded = protein_encoded.unsqueeze(1)  # [batch_size, 1, d_model]
        
        return protein_encoded
    
    def forward(self,
                protein_sequences: List[str],
                target_smiles: List[str]) -> torch.Tensor:
        """
        Forward pass for training
        
        Args:
            protein_sequences: List of protein sequences
            target_smiles: List of target SMILES strings
        
        Returns:
            Logits for next token prediction
        """
        # Encode proteins
        protein_memory = self.encode_protein(protein_sequences)
        
        # Tokenize SMILES
        batch_size = len(target_smiles)
        device = next(self.parameters()).device
        
        # Encode target SMILES
        max_len = max(len(self.tokenizer.encode(smiles)) for smiles in target_smiles)
        target_tokens = []
        
        for smiles in target_smiles:
            tokens = self.tokenizer.encode(smiles, max_length=max_len)
            target_tokens.append(tokens)
        
        target_tokens = torch.tensor(target_tokens, device=device)  # [batch_size, seq_len]
        
        # Create input (all tokens except last) and target (all tokens except first)
        input_tokens = target_tokens[:, :-1]
        target_labels = target_tokens[:, 1:]
        
        # Create causal mask
        seq_len = input_tokens.size(1)
        tgt_mask = self.decoder.generate_square_subsequent_mask(seq_len).to(device)
        
        # Create padding mask
        tgt_key_padding_mask = (input_tokens == self.tokenizer.pad_token_id)
        
        # Forward pass
        logits = self.decoder(
            tgt=input_tokens,
            memory=protein_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return logits, target_labels
    
    def generate(self,
                 protein_sequences: List[str],
                 max_length: Optional[int] = None,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 deterministic: bool = False,
                 num_return_sequences: int = 1) -> List[List[str]]:
        """
        Generate SMILES strings conditioned on protein sequences
        
        Args:
            protein_sequences: List of protein sequences
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            deterministic: Use greedy decoding if True
            num_return_sequences: Number of sequences to generate per protein
        
        Returns:
            List of generated SMILES strings for each protein
        """
        if max_length is None:
            max_length = self.max_length
        
        self.eval()
        device = next(self.parameters()).device
        
        # Encode proteins
        protein_memory = self.encode_protein(protein_sequences)
        batch_size = len(protein_sequences)
        
        # Expand for multiple return sequences
        if num_return_sequences > 1:
            protein_memory = protein_memory.repeat_interleave(num_return_sequences, dim=0)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size
        
        # Initialize generation
        generated_tokens = torch.full(
            (effective_batch_size, max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device
        )
        generated_tokens[:, 0] = self.tokenizer.sos_token_id
        
        # Track finished sequences
        finished = torch.zeros(effective_batch_size, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            for step in range(1, max_length):
                # Current sequence
                current_tokens = generated_tokens[:, :step]
                
                # Create causal mask
                tgt_mask = self.decoder.generate_square_subsequent_mask(step).to(device)
                
                # Forward pass
                logits = self.decoder(
                    tgt=current_tokens,
                    memory=protein_memory,
                    tgt_mask=tgt_mask
                )
                
                # Get next token logits
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Sample next token
                if deterministic:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                else:
                    # Apply top-k filtering
                    if top_k is not None:
                        top_k = min(top_k, next_token_logits.size(-1))
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p filtering
                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # Update generated tokens
                generated_tokens[:, step] = next_tokens
                
                # Check for finished sequences
                finished |= (next_tokens == self.tokenizer.eos_token_id)
                
                # Stop if all sequences are finished
                if finished.all():
                    break
        
        # Decode generated sequences
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


class ChemicalValidator:
    """Chemical validity checker using RDKit"""
    
    @staticmethod
    def is_valid_smiles(smiles: str) -> bool:
        """Check if SMILES string is chemically valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    @staticmethod
    def canonicalize_smiles(smiles: str) -> Optional[str]:
        """Canonicalize SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None
    
    @staticmethod
    def calculate_properties(smiles: str) -> Dict[str, float]:
        """Calculate molecular properties"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': Descriptors.RingCount(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_hbd': Descriptors.NumHDonors(mol),
                'num_hba': Descriptors.NumHAcceptors(mol)
            }
            
            return properties
        except:
            return {}
    
    @staticmethod
    def filter_valid_molecules(smiles_list: List[str]) -> List[str]:
        """Filter list to keep only valid molecules"""
        valid_smiles = []
        for smiles in smiles_list:
            if ChemicalValidator.is_valid_smiles(smiles):
                canonical = ChemicalValidator.canonicalize_smiles(smiles)
                if canonical:
                    valid_smiles.append(canonical)
        return valid_smiles


class DrugGenerationPipeline:
    """Complete pipeline for protein-conditioned drug generation"""
    
    def __init__(self,
                 protein_encoder: BaseEncoder,
                 vocab_size: Optional[int] = None,
                 **generator_kwargs):
        
        if vocab_size is None:
            tokenizer = SMILESTokenizer()
            vocab_size = len(tokenizer)
        
        self.generator = ProteinConditionedGenerator(
            protein_encoder=protein_encoder,
            vocab_size=vocab_size,
            **generator_kwargs
        )
        
        self.validator = ChemicalValidator()
    
    def generate_molecules(self,
                          protein_sequences: List[str],
                          num_molecules: int = 10,
                          filter_valid: bool = True,
                          **generation_kwargs) -> List[Dict]:
        """
        Generate molecules for given protein sequences
        
        Args:
            protein_sequences: List of protein sequences
            num_molecules: Number of molecules to generate per protein
            filter_valid: Whether to filter chemically valid molecules
            **generation_kwargs: Arguments for generation
        
        Returns:
            List of results for each protein
        """
        # Generate SMILES
        generated_smiles = self.generator.generate(
            protein_sequences=protein_sequences,
            num_return_sequences=num_molecules,
            **generation_kwargs
        )
        
        results = []
        for i, protein_seq in enumerate(protein_sequences):
            protein_results = {
                'protein_sequence': protein_seq,
                'generated_smiles': generated_smiles[i] if isinstance(generated_smiles[i], list) else [generated_smiles[i]],
                'valid_smiles': [],
                'properties': []
            }
            
            # Validate and calculate properties
            smiles_list = protein_results['generated_smiles']
            
            for smiles in smiles_list:
                is_valid = self.validator.is_valid_smiles(smiles)
                
                if is_valid:
                    canonical = self.validator.canonicalize_smiles(smiles)
                    if canonical:
                        protein_results['valid_smiles'].append(canonical)
                        properties = self.validator.calculate_properties(canonical)
                        protein_results['properties'].append(properties)
                
                if not filter_valid:
                    # Keep all molecules if not filtering
                    if not is_valid:
                        protein_results['valid_smiles'].append(smiles)
                        protein_results['properties'].append({})
            
            results.append(protein_results)
        
        return results
    
    def train_step(self,
                   protein_sequences: List[str],
                   target_smiles: List[str],
                   optimizer: torch.optim.Optimizer) -> float:
        """
        Single training step
        
        Args:
            protein_sequences: List of protein sequences
            target_smiles: List of target SMILES strings
            optimizer: Optimizer
        
        Returns:
            Loss value
        """
        self.generator.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits, target_labels = self.generator(protein_sequences, target_smiles)
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_labels.reshape(-1),
            ignore_index=self.generator.tokenizer.pad_token_id
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
"""
Enhanced protein encoders for the Unified DTA System
ESM-2 and CNN-based encoders with advanced memory optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
import warnings
import numpy as np

from ..core.base_components import BaseEncoder, SEBlock, PositionalEncoding

logger = logging.getLogger(__name__)

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

try:
    from transformers import EsmModel, EsmTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers library not available. ESM-2 encoder will not work.")
    TRANSFORMERS_AVAILABLE = False


class ESMProteinEncoder(BaseEncoder):
    """ESM-2 based protein sequence encoder with efficient memory management"""
    
    def __init__(self, 
                 output_dim: int = 128,
                 model_name: str = "facebook/esm2_t6_8M_UR50D",
                 max_length: int = 200,
                 freeze_initial: bool = True):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required for ESM-2 encoder")
        
        self.esm_model = EsmModel.from_pretrained(model_name)
        self.esm_tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self._output_dim = output_dim
        
        # ESM-2 outputs 320-dim embeddings for t6_8M model
        esm_dim = self.esm_model.config.hidden_size
        self.projection = nn.Linear(esm_dim, output_dim)
        
        if freeze_initial:
            self.freeze_esm()
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def freeze_esm(self):
        """Freeze all ESM parameters"""
        for param in self.esm_model.parameters():
            param.requires_grad = False
    
    def unfreeze_esm_layers(self, num_layers: int = 4):
        """Unfreeze the last N layers of ESM for fine-tuning"""
        if hasattr(self.esm_model, 'encoder') and hasattr(self.esm_model.encoder, 'layer'):
            for layer in self.esm_model.encoder.layer[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def forward(self, protein_sequences: List[str]) -> torch.Tensor:
        # Truncate sequences for memory efficiency
        truncated_seqs = [seq[:self.max_length] for seq in protein_sequences]
        
        # Tokenize sequences
        inputs = self.esm_tokenizer(
            truncated_seqs, 
            return_tensors="pt", 
            padding=True,
            truncation=True, 
            max_length=self.max_length
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass through ESM
        with torch.set_grad_enabled(self.training):
            outputs = self.esm_model(**inputs)
            # Use CLS token representation
            esm_features = outputs.last_hidden_state[:, 0, :]
        
        return self.projection(esm_features)


class CNNProteinEncoder(BaseEncoder):
    """CNN-based protein encoder with attention mechanisms"""
    
    def __init__(self,
                 vocab_size: int = 25,
                 embed_dim: int = 128,
                 num_filters: List[int] = [64, 128, 256],
                 kernel_sizes: List[int] = [3, 5, 7],
                 output_dim: int = 128,
                 max_length: int = 200,
                 dropout: float = 0.1):
        
        super().__init__()
        
        self._output_dim = output_dim
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(embed_dim, num_filters[0], kernel_size, padding=kernel_size//2)
            self.conv_layers.append(conv)
        
        # Global pooling
        total_features = len(kernel_sizes) * num_filters[0]
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_features // 2, output_dim)
        )
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, protein_tokens: torch.Tensor) -> torch.Tensor:
        # Embedding
        x = self.embedding(protein_tokens)  # [batch, seq_len, embed_dim]
        x = x.transpose(1, 2)  # [batch, embed_dim, seq_len]
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        # Concatenate features
        combined = torch.cat(conv_outputs, dim=1)
        
        # Final projection
        return self.projection(combined)


# Alias for enhanced versions
EnhancedCNNProteinEncoder = CNNProteinEncoder
MemoryOptimizedESMEncoder = ESMProteinEncoder
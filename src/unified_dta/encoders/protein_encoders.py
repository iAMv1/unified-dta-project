"""
Enhanced protein encoders for the Unified DTA System
ESM-2 and CNN-based encoders with advanced memory optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
import math
import warnings
import numpy as np
from abc import ABC, abstractmethod

from .base_components import BaseEncoder, SEBlock, PositionalEncoding

logger = logging.getLogger(__name__)

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

try:
    from transformers import EsmModel, EsmTokenizer, EsmConfig
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers library not available. ESM-2 encoder will not work.")
    TRANSFORMERS_AVAILABLE = False


class MemoryOptimizedESMEncoder(BaseEncoder):
    """
    Memory-optimized ESM-2 protein encoder with advanced features:
    - Gradient checkpointing for memory efficiency
    - Dynamic sequence truncation
    - Batch processing optimization
    - Progressive unfreezing for fine-tuning
    """
    
    def __init__(self,
                 output_dim: int = 128,
                 model_name: str = "facebook/esm2_t6_8M_UR50D",
                 max_length: int = 200,
                 freeze_initial: bool = True,
                 use_gradient_checkpointing: bool = True,
                 pooling_strategy: str = 'cls',  # 'cls', 'mean', 'max', 'attention'
                 attention_pooling_heads: int = 8,
                 dropout: float = 0.1):
        
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required for ESM-2 encoder")
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self._output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        
        # Load ESM-2 model and tokenizer
        logger.info(f"Loading ESM-2 model: {model_name}")
        self.esm_model = EsmModel.from_pretrained(model_name)
        self.esm_tokenizer = EsmTokenizer.from_pretrained(model_name)
        
        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.esm_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for memory optimization")
        
        # Get ESM hidden dimension
        self.esm_dim = self.esm_model.config.hidden_size
        
        # Pooling mechanisms
        if pooling_strategy == 'attention':
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=self.esm_dim,
                num_heads=attention_pooling_heads,
                batch_first=True
            )
            self.pooling_query = nn.Parameter(torch.randn(1, 1, self.esm_dim))
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(self.esm_dim, self.esm_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.esm_dim // 2, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Track frozen/unfrozen state
        self.frozen_layers = set()
        self.unfrozen_layers = set()
        
        # Freeze parameters initially
        if freeze_initial:
            self.freeze_esm()
    
    @property
    def output_dim(self) -> int:
        return self._output_dim    

    def freeze_esm(self) -> None:
        """Freeze all ESM parameters"""
        for name, param in self.esm_model.named_parameters():
            param.requires_grad = False
            self.frozen_layers.add(name)
        logger.info("All ESM-2 parameters frozen")
    
    def unfreeze_esm_layers(self, num_layers: int = 4) -> None:
        """Unfreeze the last N layers of ESM for fine-tuning"""
        if hasattr(self.esm_model, 'encoder') and hasattr(self.esm_model.encoder, 'layer'):
            total_layers = len(self.esm_model.encoder.layer)
            start_layer = max(0, total_layers - num_layers)
            
            unfrozen_count = 0
            for i in range(start_layer, total_layers):
                layer = self.esm_model.encoder.layer[i]
                for name, param in layer.named_parameters():
                    param.requires_grad = True
                    full_name = f"encoder.layer.{i}.{name}"
                    self.unfrozen_layers.add(full_name)
                    if full_name in self.frozen_layers:
                        self.frozen_layers.remove(full_name)
                    unfrozen_count += 1
            
            logger.info(f"Unfroze last {num_layers} layers ({unfrozen_count} parameters)")
        else:
            logger.warning("Could not find ESM encoder layers for unfreezing")
    
    def unfreeze_embeddings(self) -> None:
        """Unfreeze embedding layers"""
        if hasattr(self.esm_model, 'embeddings'):
            for name, param in self.esm_model.embeddings.named_parameters():
                param.requires_grad = True
                full_name = f"embeddings.{name}"
                self.unfrozen_layers.add(full_name)
                if full_name in self.frozen_layers:
                    self.frozen_layers.remove(full_name)
            logger.info("ESM-2 embeddings unfrozen")
    
    def get_frozen_status(self) -> Dict[str, Any]:
        """Get detailed information about frozen/unfrozen parameters"""
        total_params = sum(p.numel() for p in self.esm_model.parameters())
        trainable_params = sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)
        
        return {
            'total_esm_params': total_params,
            'trainable_esm_params': trainable_params,
            'frozen_esm_params': total_params - trainable_params,
            'frozen_percentage': (total_params - trainable_params) / total_params * 100,
            'num_frozen_layers': len(self.frozen_layers),
            'num_unfrozen_layers': len(self.unfrozen_layers)
        } 
   
    def _pool_sequence_features(self, 
                               hidden_states: torch.Tensor,
                               attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to sequence features"""
        
        if self.pooling_strategy == 'cls':
            # Use CLS token (first token)
            return hidden_states[:, 0, :]
        
        elif self.pooling_strategy == 'mean':
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling_strategy == 'max':
            # Max pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states * mask_expanded + (1 - mask_expanded) * -1e9
            return torch.max(hidden_states, 1)[0]
        
        elif self.pooling_strategy == 'attention':
            # Attention-based pooling
            batch_size = hidden_states.size(0)
            query = self.pooling_query.expand(batch_size, -1, -1)
            
            pooled_output, _ = self.attention_pooling(
                query, hidden_states, hidden_states,
                key_padding_mask=~attention_mask.bool()
            )
            return pooled_output.squeeze(1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def _adaptive_truncation(self, sequences: List[str]) -> Tuple[List[str], List[int]]:
        """Adaptively truncate sequences based on batch statistics"""
        lengths = [len(seq) for seq in sequences]
        
        # Use percentile-based truncation for better memory usage
        if max(lengths) > self.max_length:
            # Use 95th percentile or max_length, whichever is smaller
            adaptive_length = min(int(np.percentile(lengths, 95)), self.max_length)
            truncated_seqs = [seq[:adaptive_length] for seq in sequences]
            actual_lengths = [min(len(seq), adaptive_length) for seq in sequences]
        else:
            truncated_seqs = sequences
            actual_lengths = lengths
        
        return truncated_seqs, actual_lengths
    
    def forward(self, protein_sequences: List[str]) -> torch.Tensor:
        """Forward pass with memory optimization"""
        
        # Adaptive truncation
        truncated_seqs, actual_lengths = self._adaptive_truncation(protein_sequences)
        
        # Tokenize sequences
        inputs = self.esm_tokenizer(
            truncated_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass through ESM with gradient management
        if self.training and any(p.requires_grad for p in self.esm_model.parameters()):
            # Training mode with some unfrozen parameters
            outputs = self.esm_model(**inputs)
        else:
            # Inference mode or fully frozen - use no_grad for memory efficiency
            with torch.no_grad():
                outputs = self.esm_model(**inputs)
        
        # Extract hidden states
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Apply pooling strategy
        pooled_features = self._pool_sequence_features(hidden_states, attention_mask)
        
        # Apply projection and normalization
        projected = self.projection(pooled_features)
        output = self.layer_norm(self.dropout(projected))
        
        return output
    
    def get_attention_weights(self, protein_sequences: List[str]) -> torch.Tensor:
        """Get attention weights for interpretability"""
        if not hasattr(self.esm_model, 'encoder'):
            raise ValueError("ESM model does not support attention extraction")
        
        # Tokenize input
        inputs = self.esm_tokenizer(
            protein_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.esm_model(**inputs, output_attentions=True)
            # Return attention weights from last layer
            return outputs.attentions[-1]


class GatedCNNBlock(nn.Module):
    """Gated CNN block with residual connections and SE attention"""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 dropout: float = 0.1,
                 use_se: bool = True,
                 se_reduction: int = 16):
        super().__init__()
        
        # Main convolution path
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.gate = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(out_channels)
        
        # SE attention block
        self.use_se = use_se
        if use_se:
            self.se_block = SEBlock(out_channels, reduction=se_reduction)
        
        # Residual connection
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for residual connection
        residual = x
        
        # Gated convolution
        conv_out = self.conv(x)
        gate_out = torch.sigmoid(self.gate(x))
        gated = conv_out * gate_out
        
        # Batch normalization
        gated = self.bn(gated)
        
        # SE attention
        if self.use_se:
            gated = self.se_block(gated)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        output = gated + residual
        output = F.relu(output)
        output = self.dropout(output)
        
        return output


class EnhancedCNNProteinEncoder(BaseEncoder):
    """
    Enhanced CNN-based protein encoder with:
    - Gated CNN layers with residual connections
    - SE attention blocks for feature enhancement
    - Configurable kernel sizes and filter numbers
    - Efficient embedding and projection layers
    """
    
    def __init__(self,
                 vocab_size: int = 25,
                 embed_dim: int = 128,
                 num_filters: List[int] = [64, 128, 256],
                 kernel_sizes: List[int] = [3, 5, 7],
                 output_dim: int = 128,
                 max_length: int = 200,
                 dropout: float = 0.1,
                 use_se: bool = True,
                 se_reduction: int = 16,
                 use_positional_encoding: bool = True):
        
        super().__init__()
        
        self._output_dim = output_dim
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Embedding layer with padding token
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        
        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(embed_dim, dropout, max_length)
        
        # Multi-scale CNN layers with different kernel sizes
        self.cnn_branches = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            branch = nn.ModuleList()
            in_channels = embed_dim
            
            for i, out_channels in enumerate(num_filters):
                padding = kernel_size // 2  # Same padding
                
                branch.append(GatedCNNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dropout=dropout,
                    use_se=use_se,
                    se_reduction=se_reduction
                ))
                
                in_channels = out_channels
            
            self.cnn_branches.append(branch)
        
        # Global attention pooling
        total_features = sum(num_filters[-1] for _ in kernel_sizes)  # Last layer of each branch
        self.attention_pooling = nn.Sequential(
            nn.Linear(total_features, total_features // 4),
            nn.ReLU(),
            nn.Linear(total_features // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # Final projection layers
        self.projection = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_features // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _create_padding_mask(self, sequences: torch.Tensor) -> torch.Tensor:
        """Create padding mask for attention pooling"""
        return (sequences != 0).float()  # 0 is padding token
    
    def forward(self, protein_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through enhanced CNN encoder
        
        Args:
            protein_tokens: Tensor of shape [batch_size, seq_len] with tokenized sequences
            
        Returns:
            Tensor of shape [batch_size, output_dim] with protein features
        """
        batch_size, seq_len = protein_tokens.shape
        
        # Truncate if necessary
        if seq_len > self.max_length:
            protein_tokens = protein_tokens[:, :self.max_length]
            seq_len = self.max_length
        
        # Embedding
        x = self.embedding(protein_tokens)  # [batch, seq_len, embed_dim]
        
        # Positional encoding
        if self.use_positional_encoding:
            x = x.transpose(0, 1)  # [seq_len, batch, embed_dim] for pos encoding
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)  # Back to [batch, seq_len, embed_dim]
        
        # Transpose for conv1d: [batch, embed_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Multi-scale CNN processing
        branch_outputs = []
        
        for branch in self.cnn_branches:
            branch_x = x
            
            # Pass through all layers in this branch
            for layer in branch:
                branch_x = layer(branch_x)
            
            branch_outputs.append(branch_x)
        
        # Concatenate multi-scale features: [batch, total_features, seq_len]
        combined_features = torch.cat(branch_outputs, dim=1)
        
        # Transpose back for attention pooling: [batch, seq_len, total_features]
        combined_features = combined_features.transpose(1, 2)
        
        # Create padding mask
        padding_mask = self._create_padding_mask(protein_tokens)  # [batch, seq_len]
        
        # Attention-based global pooling
        attention_weights = self.attention_pooling(combined_features)  # [batch, seq_len, 1]
        attention_weights = attention_weights.squeeze(-1)  # [batch, seq_len]
        
        # Apply padding mask to attention weights
        attention_weights = attention_weights * padding_mask
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum pooling
        pooled_features = torch.sum(
            combined_features * attention_weights.unsqueeze(-1), 
            dim=1
        )  # [batch, total_features]
        
        # Final projection
        output = self.projection(pooled_features)
        
        return output
    
    def get_attention_weights(self, protein_tokens: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability"""
        with torch.no_grad():
            batch_size, seq_len = protein_tokens.shape
            
            if seq_len > self.max_length:
                protein_tokens = protein_tokens[:, :self.max_length]
            
            # Forward pass up to attention computation
            x = self.embedding(protein_tokens)
            
            if self.use_positional_encoding:
                x = x.transpose(0, 1)
                x = self.pos_encoding(x)
                x = x.transpose(0, 1)
            
            x = x.transpose(1, 2)
            
            # Multi-scale CNN processing
            branch_outputs = []
            for branch in self.cnn_branches:
                branch_x = x
                for layer in branch:
                    branch_x = layer(branch_x)
                branch_outputs.append(branch_x)
            
            combined_features = torch.cat(branch_outputs, dim=1)
            combined_features = combined_features.transpose(1, 2)
            
            # Get attention weights
            attention_weights = self.attention_pooling(combined_features)
            attention_weights = attention_weights.squeeze(-1)
            
            # Apply padding mask
            padding_mask = self._create_padding_mask(protein_tokens)
            attention_weights = attention_weights * padding_mask
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            return attention_weights

# Backward-compatible alias expected by factory and imports
ESMProteinEncoder = MemoryOptimizedESMEncoder

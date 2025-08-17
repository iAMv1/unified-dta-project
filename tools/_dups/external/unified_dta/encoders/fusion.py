"""
Multi-modal fusion mechanisms for drug-target affinity prediction
Implements cross-attention and various fusion strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiModalFusion(nn.Module):
    """Multi-modal fusion with attention mechanism"""
    
    def __init__(self, 
                 drug_dim: int,
                 protein_dim: int,
                 hidden_dim: int = 512,
                 num_heads: int = 8):
        super().__init__()
        
        # Project to same dimension
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim * 2  # Concatenated features
    
    def forward(self, drug_features: torch.Tensor, protein_features: torch.Tensor) -> torch.Tensor:
        # Project to same dimension
        drug_proj = self.drug_proj(drug_features).unsqueeze(1)  # [batch, 1, hidden]
        protein_proj = self.protein_proj(protein_features).unsqueeze(1)  # [batch, 1, hidden]
        
        # Cross-attention
        drug_attended, _ = self.cross_attention(drug_proj, protein_proj, protein_proj)
        protein_attended, _ = self.cross_attention(protein_proj, drug_proj, drug_proj)
        
        # Layer norm and squeeze
        drug_attended = self.layer_norm(drug_attended).squeeze(1)
        protein_attended = self.layer_norm(protein_attended).squeeze(1)
        
        # Concatenate
        return torch.cat([drug_attended, protein_attended], dim=1)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism between drug and protein features
    with learnable projection layers and residual connections
    """
    
    def __init__(self,
                 drug_dim: int,
                 protein_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_residual: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        
        # Learnable projection layers for dimension alignment
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        
        # Cross-attention layers
        self.drug_to_protein_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.protein_to_drug_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.drug_layer_norm = nn.LayerNorm(hidden_dim)
        self.protein_layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension is concatenated features
        self._output_dim = hidden_dim * 2
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, drug_features: torch.Tensor, protein_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_features: [batch_size, drug_dim]
            protein_features: [batch_size, protein_dim]
        
        Returns:
            fused_features: [batch_size, hidden_dim * 2]
        """
        batch_size = drug_features.size(0)
        
        # Project to same dimension
        drug_proj = self.drug_proj(drug_features).unsqueeze(1)  # [batch, 1, hidden_dim]
        protein_proj = self.protein_proj(protein_features).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Cross-attention: drug attends to protein
        drug_attended, drug_attn_weights = self.drug_to_protein_attention(
            query=drug_proj,
            key=protein_proj,
            value=protein_proj
        )
        
        # Cross-attention: protein attends to drug
        protein_attended, protein_attn_weights = self.protein_to_drug_attention(
            query=protein_proj,
            key=drug_proj,
            value=drug_proj
        )
        
        # Residual connections
        if self.use_residual:
            drug_attended = drug_attended + drug_proj
            protein_attended = protein_attended + protein_proj
        
        # Layer normalization
        drug_attended = self.drug_layer_norm(drug_attended)
        protein_attended = self.protein_layer_norm(protein_attended)
        
        # Dropout
        drug_attended = self.dropout(drug_attended)
        protein_attended = self.dropout(protein_attended)
        
        # Squeeze and concatenate
        drug_attended = drug_attended.squeeze(1)  # [batch, hidden_dim]
        protein_attended = protein_attended.squeeze(1)  # [batch, hidden_dim]
        
        fused_features = torch.cat([drug_attended, protein_attended], dim=1)
        
        return fused_features


class SimpleConcatenationFusion(nn.Module):
    """Simple concatenation fusion with optional projection"""
    
    def __init__(self,
                 drug_dim: int,
                 protein_dim: int,
                 hidden_dim: Optional[int] = None,
                 use_projection: bool = False):
        super().__init__()
        
        self.use_projection = use_projection
        
        if use_projection and hidden_dim is not None:
            self.projection = nn.Linear(drug_dim + protein_dim, hidden_dim)
            self._output_dim = hidden_dim
        else:
            self.projection = None
            self._output_dim = drug_dim + protein_dim
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, drug_features: torch.Tensor, protein_features: torch.Tensor) -> torch.Tensor:
        # Simple concatenation
        fused_features = torch.cat([drug_features, protein_features], dim=1)
        
        # Optional projection
        if self.projection is not None:
            fused_features = self.projection(fused_features)
        
        return fused_features
"""
Multi-modal fusion mechanisms for drug-target affinity prediction
Implements cross-attention and various fusion strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


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


class BilinearFusion(nn.Module):
    """
    Bilinear fusion mechanism for drug and protein features
    """
    
    def __init__(self,
                 drug_dim: int,
                 protein_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        
        # Bilinear transformation
        self.bilinear = nn.Bilinear(drug_dim, protein_dim, hidden_dim)
        
        # Additional projections
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 3)
        
        self._output_dim = hidden_dim * 3
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, drug_features: torch.Tensor, protein_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drug_features: [batch_size, drug_dim]
            protein_features: [batch_size, protein_dim]
        
        Returns:
            fused_features: [batch_size, hidden_dim * 3]
        """
        # Bilinear interaction
        bilinear_out = self.bilinear(drug_features, protein_features)
        
        # Individual projections
        drug_proj = self.drug_proj(drug_features)
        protein_proj = self.protein_proj(protein_features)
        
        # Concatenate all features
        fused_features = torch.cat([bilinear_out, drug_proj, protein_proj], dim=1)
        
        # Layer normalization
        fused_features = self.layer_norm(fused_features)
        
        return fused_features


class SimpleConcatenationFusion(nn.Module):
    """
    Simple concatenation fusion with optional projection
    """
    
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
        """
        Args:
            drug_features: [batch_size, drug_dim]
            protein_features: [batch_size, protein_dim]
        
        Returns:
            fused_features: [batch_size, output_dim]
        """
        # Simple concatenation
        fused_features = torch.cat([drug_features, protein_features], dim=1)
        
        # Optional projection
        if self.projection is not None:
            fused_features = self.projection(fused_features)
        
        return fused_features


class MultiModalFusion(nn.Module):
    """
    Enhanced multi-modal fusion with configurable fusion strategies
    """
    
    def __init__(self,
                 drug_dim: int,
                 protein_dim: int,
                 fusion_type: str = 'cross_attention',
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_residual: bool = True):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'cross_attention':
            self.fusion_module = CrossAttentionFusion(
                drug_dim=drug_dim,
                protein_dim=protein_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_residual=use_residual
            )
        elif fusion_type == 'bilinear':
            self.fusion_module = BilinearFusion(
                drug_dim=drug_dim,
                protein_dim=protein_dim,
                hidden_dim=hidden_dim
            )
        elif fusion_type == 'concatenation':
            self.fusion_module = SimpleConcatenationFusion(
                drug_dim=drug_dim,
                protein_dim=protein_dim,
                hidden_dim=hidden_dim,
                use_projection=True
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    @property
    def output_dim(self) -> int:
        return self.fusion_module.output_dim
    
    def forward(self, drug_features: torch.Tensor, protein_features: torch.Tensor) -> torch.Tensor:
        return self.fusion_module(drug_features, protein_features)


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns to weight different fusion strategies
    """
    
    def __init__(self,
                 drug_dim: int,
                 protein_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        # Different fusion strategies
        self.cross_attention = CrossAttentionFusion(
            drug_dim, protein_dim, hidden_dim, num_heads, dropout
        )
        self.bilinear = BilinearFusion(drug_dim, protein_dim, hidden_dim)
        self.concatenation = SimpleConcatenationFusion(
            drug_dim, protein_dim, hidden_dim, use_projection=True
        )
        
        # Gating mechanism to weight different fusion strategies
        total_dim = (self.cross_attention.output_dim + 
                    self.bilinear.output_dim + 
                    self.concatenation.output_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(drug_dim + protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3 fusion strategies
            nn.Softmax(dim=1)
        )
        
        # Final projection
        self.final_proj = nn.Linear(total_dim, hidden_dim)
        self._output_dim = hidden_dim
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, drug_features: torch.Tensor, protein_features: torch.Tensor) -> torch.Tensor:
        # Compute different fusion strategies
        cross_attn_out = self.cross_attention(drug_features, protein_features)
        bilinear_out = self.bilinear(drug_features, protein_features)
        concat_out = self.concatenation(drug_features, protein_features)
        
        # Compute gating weights
        gate_input = torch.cat([drug_features, protein_features], dim=1)
        weights = self.gate(gate_input)  # [batch_size, 3]
        
        # Weight and combine fusion outputs
        weighted_cross_attn = weights[:, 0:1] * cross_attn_out
        weighted_bilinear = weights[:, 1:2] * bilinear_out
        weighted_concat = weights[:, 2:3] * concat_out
        
        # Concatenate all weighted outputs
        combined = torch.cat([weighted_cross_attn, weighted_bilinear, weighted_concat], dim=1)
        
        # Final projection
        return self.final_proj(combined)
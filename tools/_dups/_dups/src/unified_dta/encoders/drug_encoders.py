"""
Enhanced drug encoders for the Unified DTA System
Advanced GIN-based molecular graph encoders with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
import math

from .base_components import BaseEncoder

logger = logging.getLogger(__name__)


class ConfigurableMLPBlock(nn.Module):
    """Configurable MLP block for GIN layers with various activation options"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 use_batch_norm: bool = True,
                 final_activation: bool = False):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if use_batch_norm and len(hidden_dims) > 0:
            layers.append(nn.BatchNorm1d(output_dim))
        
        # Final activation if requested
        if final_activation:
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            elif activation == 'elu':
                layers.append(nn.ELU())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ResidualGINLayer(nn.Module):
    """Enhanced GIN layer with residual connections and configurable MLP"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 mlp_hidden_dims: List[int] = [128],
                 activation: str = 'relu',
                 dropout: float = 0.1,
                 use_batch_norm: bool = True,
                 residual_connection: bool = True,
                 eps: float = 0.0,
                 train_eps: bool = False):
        super().__init__()
        
        self.residual_connection = residual_connection and (input_dim == hidden_dim)
        
        # Create configurable MLP for GIN
        self.mlp = ConfigurableMLPBlock(
            input_dim=input_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            final_activation=True
        )
        
        # GIN convolution layer
        self.gin_conv = GINConv(self.mlp, eps=eps, train_eps=train_eps)
        
        # Projection layer for residual connection when dimensions don't match
        self.residual_proj = None
        if residual_connection and input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
            self.residual_connection = True
        
        # Layer normalization for residual connection
        if self.residual_connection:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Additional dropout after residual connection
        self.final_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Store input for residual connection
        residual = x
        
        # GIN convolution
        out = self.gin_conv(x, edge_index)
        
        # Residual connection
        if self.residual_connection:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            
            out = out + residual
            out = self.layer_norm(out)
        
        # Final dropout
        out = self.final_dropout(out)
        
        return out


class EnhancedGINDrugEncoder(BaseEncoder):
    """
    Enhanced Graph Isomorphism Network for drug molecular graphs with:
    - Configurable MLP layers with various activation functions
    - Residual connections for training stability
    - Advanced batch normalization and dropout regularization
    - Multiple pooling strategies (mean, max, add, attention)
    """
    
    def __init__(self,
                 node_features: int = 78,
                 hidden_dim: int = 128,
                 num_layers: int = 5,
                 output_dim: int = 128,
                 mlp_hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu',
                 dropout: float = 0.1,
                 use_batch_norm: bool = True,
                 residual_connections: bool = True,
                 pooling_strategy: str = 'dual',  # 'mean', 'max', 'add', 'dual', 'attention'
                 attention_heads: int = 4,
                 eps: float = 0.0,
                 train_eps: bool = False):
        
        super().__init__()
        
        self._output_dim = output_dim
        self.num_layers = num_layers
        self.pooling_strategy = pooling_strategy
        self.hidden_dim = hidden_dim
        
        # Input projection if needed
        self.input_proj = None
        if node_features != hidden_dim:
            self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # Enhanced GIN layers with residual connections
        self.gin_layers = nn.ModuleList()
        
        for i in range(num_layers):
            input_dim = hidden_dim  # After input projection, all layers use hidden_dim
            
            layer = ResidualGINLayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                mlp_hidden_dims=mlp_hidden_dims,
                activation=activation,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                residual_connection=residual_connections,
                eps=eps,
                train_eps=train_eps
            )
            
            self.gin_layers.append(layer)
        
        # Pooling mechanisms
        if pooling_strategy == 'attention':
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                batch_first=True
            )
            self.pooling_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
            pooled_dim = hidden_dim
        elif pooling_strategy == 'dual':
            pooled_dim = hidden_dim * 2  # mean + max
        else:
            pooled_dim = hidden_dim  # single pooling
        
        # Final projection layers
        self.final_projection = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _apply_pooling(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Apply the specified pooling strategy"""
        
        if self.pooling_strategy == 'mean':
            return global_mean_pool(x, batch)
        
        elif self.pooling_strategy == 'max':
            return global_max_pool(x, batch)
        
        elif self.pooling_strategy == 'add':
            return global_add_pool(x, batch)
        
        elif self.pooling_strategy == 'dual':
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            return torch.cat([mean_pool, max_pool], dim=1)
        
        elif self.pooling_strategy == 'attention':
            # Convert to batch format for attention
            batch_size = batch.max().item() + 1
            max_nodes = torch.bincount(batch).max().item()
            
            # Create padded tensor
            padded_x = torch.zeros(batch_size, max_nodes, x.size(1), 
                                 device=x.device, dtype=x.dtype)
            mask = torch.zeros(batch_size, max_nodes, device=x.device, dtype=torch.bool)
            
            for i in range(batch_size):
                nodes_in_graph = (batch == i).sum().item()
                if nodes_in_graph > 0:
                    padded_x[i, :nodes_in_graph] = x[batch == i]
                    mask[i, :nodes_in_graph] = True
            
            # Apply attention pooling
            query = self.pooling_query.expand(batch_size, -1, -1)
            pooled_output, _ = self.attention_pooling(
                query, padded_x, padded_x,
                key_padding_mask=~mask
            )
            
            return pooled_output.squeeze(1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass through enhanced GIN encoder
        
        Args:
            data: PyTorch Geometric Data object with x, edge_index, batch
            
        Returns:
            Tensor of shape [batch_size, output_dim] with molecular features
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Pass through GIN layers with residual connections
        for gin_layer in self.gin_layers:
            x = gin_layer(x, edge_index)
        
        # Apply pooling strategy
        pooled_features = self._apply_pooling(x, batch)
        
        # Final projection
        output = self.final_projection(pooled_features)
        
        return output
    
    def get_node_embeddings(self, data) -> torch.Tensor:
        """Get node-level embeddings before pooling"""
        x, edge_index = data.x, data.edge_index
        
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        for gin_layer in self.gin_layers:
            x = gin_layer(x, edge_index)
        
        return x
    
    def get_layer_outputs(self, data) -> List[torch.Tensor]:
        """Get outputs from each GIN layer for analysis"""
        x, edge_index = data.x, data.edge_index
        layer_outputs = []
        
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        layer_outputs.append(x.clone())
        
        for gin_layer in self.gin_layers:
            x = gin_layer(x, edge_index)
            layer_outputs.append(x.clone())
        
        return layer_outputs


class MultiScaleGINEncoder(BaseEncoder):
    """Multi-scale GIN encoder that combines features from different scales"""
    
    def __init__(self,
                 node_features: int = 78,
                 hidden_dims: List[int] = [64, 128, 256],
                 num_layers_per_scale: int = 3,
                 output_dim: int = 128,
                 **kwargs):
        super().__init__()
        
        self._output_dim = output_dim
        self.encoders = nn.ModuleList()
        
        # Create multiple GIN encoders with different hidden dimensions
        for hidden_dim in hidden_dims:
            encoder = EnhancedGINDrugEncoder(
                node_features=node_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers_per_scale,
                output_dim=hidden_dim,
                **kwargs
            )
            self.encoders.append(encoder)
        
        # Combine features from different scales
        total_features = sum(hidden_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Dropout(kwargs.get('dropout', 0.1)),
            nn.Linear(total_features // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, data) -> torch.Tensor:
        # Get features from each scale
        scale_features = []
        for encoder in self.encoders:
            features = encoder(data)
            scale_features.append(features)
        
        # Concatenate and fuse
        combined_features = torch.cat(scale_features, dim=1)
        return self.fusion(combined_features)
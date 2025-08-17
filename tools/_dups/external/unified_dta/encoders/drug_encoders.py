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

from ..core.base_components import BaseEncoder

logger = logging.getLogger(__name__)


class GINDrugEncoder(BaseEncoder):
    """Graph Isomorphism Network for drug molecular graphs"""
    
    def __init__(self, 
                 node_features: int = 78,
                 hidden_dim: int = 128,
                 num_layers: int = 5,
                 output_dim: int = 128,
                 dropout: float = 0.2,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self._output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # GIN layers
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(num_layers):
            input_dim = node_features if i == 0 else hidden_dim
            
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.gin_layers.append(GINConv(mlp))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim * 2, output_dim)  # *2 for mean+max pooling
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GIN layers with residual connections
        for i, gin_layer in enumerate(self.gin_layers):
            x_new = F.relu(gin_layer(x, edge_index))
            
            if self.batch_norms:
                x_new = self.batch_norms[i](x_new)
            
            x_new = self.dropout(x_new)
            
            # Residual connection (if dimensions match)
            if i > 0 and x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
        
        # Global pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        
        # Combine pooling results
        combined = torch.cat([mean_pool, max_pool], dim=1)
        return self.final_proj(combined)


class EnhancedGINDrugEncoder(BaseEncoder):
    """Enhanced GIN encoder with advanced features"""
    
    def __init__(self,
                 node_features: int = 78,
                 hidden_dim: int = 128,
                 num_layers: int = 5,
                 output_dim: int = 128,
                 dropout: float = 0.1,
                 use_batch_norm: bool = True,
                 residual_connections: bool = True,
                 pooling_strategy: str = 'dual'):
        
        super().__init__()
        
        self._output_dim = output_dim
        self.num_layers = num_layers
        self.pooling_strategy = pooling_strategy
        self.hidden_dim = hidden_dim
        
        # Input projection if needed
        self.input_proj = None
        if node_features != hidden_dim:
            self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # GIN layers
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(num_layers):
            input_dim = hidden_dim
            
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.gin_layers.append(GINConv(mlp))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Pooling dimension
        if pooling_strategy == 'dual':
            pooled_dim = hidden_dim * 2
        else:
            pooled_dim = hidden_dim
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Pass through GIN layers
        for i, gin_layer in enumerate(self.gin_layers):
            x_new = gin_layer(x, edge_index)
            
            if self.batch_norms:
                x_new = self.batch_norms[i](x_new)
            
            x_new = F.relu(x_new)
            
            # Residual connection
            if i > 0:
                x = x + x_new
            else:
                x = x_new
        
        # Apply pooling
        if self.pooling_strategy == 'dual':
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            pooled_features = torch.cat([mean_pool, max_pool], dim=1)
        elif self.pooling_strategy == 'mean':
            pooled_features = global_mean_pool(x, batch)
        elif self.pooling_strategy == 'max':
            pooled_features = global_max_pool(x, batch)
        else:
            pooled_features = global_add_pool(x, batch)
        
        # Final projection
        return self.final_projection(pooled_features)


# Aliases for compatibility
LightweightGINEncoder = GINDrugEncoder
AdvancedGINDrugEncoder = EnhancedGINDrugEncoder
MultiScaleGINEncoder = EnhancedGINDrugEncoder
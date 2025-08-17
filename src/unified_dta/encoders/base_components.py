"""
Base components for encoders and models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional
import math


class BaseEncoder(nn.Module, ABC):
    """Base class for all encoders"""
    
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Get the output dimension of the encoder"""
        pass


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, channels, length)
        y = self.pool(x).squeeze(-1)  # (batch, channels)
        y = self.fc(y).unsqueeze(-1)   # (batch, channels, 1)
        return x * y.expand_as(x)      # (batch, channels, length)


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]


class ResidualBlock(nn.Module):
    """Residual block with optional normalization"""
    
    def __init__(self, dim: int, use_batch_norm: bool = True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(dim)
            self.bn2 = nn.BatchNorm1d(dim)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        out = self.linear1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        
        out = self.linear2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        
        out += residual  # Residual connection
        out = self.activation(out)
        
        return out
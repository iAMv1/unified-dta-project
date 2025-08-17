"""
Base components for the Unified DTA System
Shared classes and utilities used across different modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all encoders"""
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for attention mechanism"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.fc(self.avgpool(x))
        return x * attention
"""
Configurable prediction heads for drug-target affinity prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from abc import ABC, abstractmethod


class BasePredictionHead(nn.Module, ABC):
    """Abstract base class for prediction heads"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass


class ActivationFactory:
    """Factory for creating activation functions"""
    
    @staticmethod
    def create(activation: str) -> nn.Module:
        """Create activation function by name"""
        activation = activation.lower()
        
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'selu':
            return nn.SELU()
        elif activation == 'swish' or activation == 'silu':
            return nn.SiLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {activation}")


class MLPPredictionHead(BasePredictionHead):
    """
    Multi-Layer Perceptron prediction head with configurable architecture
    
    Features:
    - Configurable number of layers and dimensions
    - Multiple activation function options (ReLU, GELU)
    - Dropout and batch normalization options
    - Output layer for affinity prediction
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256],
                 output_dim: int = 1,
                 activation: str = 'relu',
                 dropout: float = 0.3,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 output_activation: Optional[str] = None):
        super().__init__()
        
        self._input_dim = input_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            layers.append(ActivationFactory.create(activation))
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Output activation (optional)
        self.output_activation = None
        if output_activation:
            self.output_activation = ActivationFactory.create(output_activation)
    
    @property
    def input_dim(self) -> int:
        return self._input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through hidden layers
        x = self.hidden_layers(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Output activation
        if self.output_activation:
            x = self.output_activation(x)
        
        return x


class PredictionHeadFactory:
    """Factory for creating different types of prediction heads"""
    
    @staticmethod
    def create(head_type: str, **kwargs) -> BasePredictionHead:
        """Create prediction head by type"""
        head_type = head_type.lower()
        
        if head_type == 'mlp':
            return MLPPredictionHead(**kwargs)
        else:
            raise ValueError(f"Unknown prediction head type: {head_type}")


# Predefined configurations
def get_lightweight_predictor(input_dim: int) -> MLPPredictionHead:
    """Lightweight predictor for development and testing"""
    return MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[128],
        activation='relu',
        dropout=0.2,
        use_batch_norm=True
    )


def get_standard_predictor(input_dim: int) -> MLPPredictionHead:
    """Standard predictor for general use"""
    return MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[512, 256],
        activation='relu',
        dropout=0.3,
        use_batch_norm=True
    )


def get_deep_predictor(input_dim: int) -> MLPPredictionHead:
    """Deep predictor for complex tasks"""
    return MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[1024, 512, 256, 128],
        activation='gelu',
        dropout=0.4,
        use_batch_norm=True
    )


# Backward compatibility - alias for the original AffinityPredictor
AffinityPredictor = MLPPredictionHead
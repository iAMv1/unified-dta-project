"""
Unified Drug-Target Affinity Prediction Models
Integrates ESM-2, GIN, and generation capabilities from multiple repositories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from transformers import EsmModel, EsmTokenizer
from typing import Optional, Dict, Tuple, List
import math
from abc import ABC, abstractmethod


# ============================================================================
# Base Components
# ============================================================================

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


# ============================================================================
# Protein Encoders
# ============================================================================

class ESMProteinEncoder(BaseEncoder):
    """ESM-2 based protein sequence encoder with efficient memory management"""
    
    def __init__(self, 
                 output_dim: int = 128,
                 model_name: str = "facebook/esm2_t6_8M_UR50D",
                 max_length: int = 200,
                 freeze_initial: bool = True):
        super().__init__()
        
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
    """CNN-based protein encoder with gated convolutions"""
    
    def __init__(self, 
                 vocab_size: int = 25,
                 embed_dim: int = 128,
                 num_filters: int = 32,
                 kernel_size: int = 8,
                 output_dim: int = 128,
                 max_length: int = 1000):
        super().__init__()
        
        self._output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        
        # Gated CNN layers
        self.conv1 = nn.Conv1d(max_length, num_filters, kernel_size)
        self.gate1 = nn.Conv1d(max_length, num_filters, kernel_size)
        
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size)
        self.gate2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size)
        
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size)
        self.gate3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size)
        
        # SE attention
        self.se_block = SEBlock(num_filters * 3)
        
        # Calculate output size after convolutions
        conv_output_size = self._calculate_conv_output_size(embed_dim, kernel_size, 3)
        self.fc = nn.Linear(num_filters * 3 * conv_output_size, output_dim)
        
        self.relu = nn.ReLU()
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def _calculate_conv_output_size(self, input_size: int, kernel_size: int, num_layers: int) -> int:
        """Calculate output size after multiple conv layers"""
        size = input_size
        for _ in range(num_layers):
            size = size - kernel_size + 1
        return max(1, size)
    
    def forward(self, protein_tokens: torch.Tensor) -> torch.Tensor:
        # Embedding
        x = self.embedding(protein_tokens)  # [batch, seq_len, embed_dim]
        
        # Gated CNN layers
        conv1 = self.conv1(x)
        gate1 = torch.sigmoid(self.gate1(x))
        x = self.relu(conv1 * gate1)
        
        conv2 = self.conv2(x)
        gate2 = torch.sigmoid(self.gate2(x))
        x = self.relu(conv2 * gate2)
        
        conv3 = self.conv3(x)
        gate3 = torch.sigmoid(self.gate3(x))
        x = self.relu(conv3 * gate3)
        
        # SE attention
        x = x * self.se_block(x)
        
        # Flatten and project
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================================
# Drug Encoders
# ============================================================================

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


# ============================================================================
# Fusion and Prediction Modules
# ============================================================================

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


class AffinityPredictor(nn.Module):
    """Final prediction head for drug-target affinity"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256],
                 dropout: float = 0.3,
                 activation: str = 'relu'):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        activation_fn = getattr(F, activation)
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


# ============================================================================
# Complete Models
# ============================================================================

class UnifiedDTAModel(nn.Module):
    """Unified Drug-Target Affinity prediction model"""
    
    def __init__(self,
                 protein_encoder_type: str = 'esm',
                 drug_encoder_type: str = 'gin',
                 use_fusion: bool = True,
                 **kwargs):
        super().__init__()
        
        # Initialize protein encoder
        if protein_encoder_type == 'esm':
            self.protein_encoder = ESMProteinEncoder(**kwargs.get('protein_config', {}))
        elif protein_encoder_type == 'cnn':
            self.protein_encoder = CNNProteinEncoder(**kwargs.get('protein_config', {}))
        else:
            raise ValueError(f"Unknown protein encoder: {protein_encoder_type}")
        
        # Initialize drug encoder
        if drug_encoder_type == 'gin':
            self.drug_encoder = GINDrugEncoder(**kwargs.get('drug_config', {}))
        else:
            raise ValueError(f"Unknown drug encoder: {drug_encoder_type}")
        
        # Fusion module
        if use_fusion:
            self.fusion = MultiModalFusion(
                drug_dim=self.drug_encoder.output_dim,
                protein_dim=self.protein_encoder.output_dim,
                **kwargs.get('fusion_config', {})
            )
            predictor_input_dim = self.fusion.output_dim
        else:
            self.fusion = None
            predictor_input_dim = self.drug_encoder.output_dim + self.protein_encoder.output_dim
        
        # Prediction head
        self.predictor = AffinityPredictor(
            input_dim=predictor_input_dim,
            **kwargs.get('predictor_config', {})
        )
        
        self.protein_encoder_type = protein_encoder_type
    
    def forward(self, drug_data, protein_data) -> torch.Tensor:
        # Encode drug and protein
        drug_features = self.drug_encoder(drug_data)
        
        if self.protein_encoder_type == 'esm':
            protein_features = self.protein_encoder(protein_data)
        else:
            protein_features = self.protein_encoder(protein_data)
        
        # Fusion or concatenation
        if self.fusion:
            combined_features = self.fusion(drug_features, protein_features)
        else:
            combined_features = torch.cat([drug_features, protein_features], dim=1)
        
        # Prediction
        return self.predictor(combined_features)
    
    def set_training_phase(self, phase: int):
        """Set training phase for progressive training"""
        if phase == 2 and hasattr(self.protein_encoder, 'unfreeze_esm_layers'):
            self.protein_encoder.unfreeze_esm_layers()


# ============================================================================
# Model Factory
# ============================================================================

def create_dta_model(config: Dict) -> UnifiedDTAModel:
    """Factory function to create DTA models with different configurations"""
    
    # Default configurations
    default_config = {
        'protein_encoder_type': 'esm',
        'drug_encoder_type': 'gin',
        'use_fusion': True,
        'protein_config': {
            'output_dim': 128,
            'max_length': 200
        },
        'drug_config': {
            'output_dim': 128,
            'num_layers': 5
        },
        'fusion_config': {
            'hidden_dim': 256,
            'num_heads': 8
        },
        'predictor_config': {
            'hidden_dims': [512, 256],
            'dropout': 0.3
        }
    }
    
    # Update with user config
    default_config.update(config)
    
    return UnifiedDTAModel(**default_config)


# ============================================================================
# Predefined Model Configurations
# ============================================================================

def get_lightweight_model() -> UnifiedDTAModel:
    """Lightweight model for testing and development"""
    config = {
        'protein_encoder_type': 'cnn',
        'drug_encoder_type': 'gin',
        'use_fusion': False,
        'protein_config': {'output_dim': 64},
        'drug_config': {'output_dim': 64, 'num_layers': 3},
        'predictor_config': {'hidden_dims': [128], 'dropout': 0.2}
    }
    return create_dta_model(config)


def get_production_model() -> UnifiedDTAModel:
    """Full-featured model for production use"""
    config = {
        'protein_encoder_type': 'esm',
        'drug_encoder_type': 'gin',
        'use_fusion': True,
        'protein_config': {'output_dim': 128, 'max_length': 200},
        'drug_config': {'output_dim': 128, 'num_layers': 5},
        'fusion_config': {'hidden_dim': 256, 'num_heads': 8},
        'predictor_config': {'hidden_dims': [512, 256], 'dropout': 0.3}
    }
    return create_dta_model(config)
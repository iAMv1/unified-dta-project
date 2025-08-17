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

# Import enhanced encoders
try:
    from ..encoders.protein_encoders import EnhancedCNNProteinEncoder
    from ..encoders.drug_encoders import EnhancedGINDrugEncoder, MultiScaleGINEncoder
except ImportError:
    # Fallback for when encoder modules are not available
    EnhancedCNNProteinEncoder = None
    EnhancedGINDrugEncoder = None
    MultiScaleGINEncoder = None


# Import base components
from .base_components import BaseEncoder, SEBlock, PositionalEncoding

# Import prediction heads
from ..evaluation.prediction_heads import (
    MLPPredictionHead, 
    PredictionHeadFactory,
    get_lightweight_predictor,
    get_standard_predictor,
    get_deep_predictor
)

# Import enhanced encoders
try:
    from ..encoders.drug_encoders import AdvancedGINDrugEncoder, LightweightGINEncoder
    # Also import the basic GIN encoder for fallback
    from ..encoders.drug_encoders import EnhancedGINDrugEncoder as GINDrugEncoder
except ImportError:
    # Fallback for when drug_encoders module is not available
    AdvancedGINDrugEncoder = None
    LightweightGINEncoder = None
    GINDrugEncoder = None

# ============================================================================
# Base Components (imported from base_components.py)
# ============================================================================


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


# Import the enhanced CNN encoder from protein_encoders module
# The CNNProteinEncoder is now replaced by EnhancedCNNProteinEncoder


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


# Use the new configurable prediction head system
# AffinityPredictor is now an alias for MLPPredictionHead in prediction_heads.py
AffinityPredictor = MLPPredictionHead


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
            if EnhancedCNNProteinEncoder is not None:
                self.protein_encoder = EnhancedCNNProteinEncoder(**kwargs.get('protein_config', {}))
            else:
                raise ImportError("EnhancedCNNProteinEncoder not available. Check protein_encoders module.")
        else:
            raise ValueError(f"Unknown protein encoder: {protein_encoder_type}")
        
        # Initialize drug encoder
        if drug_encoder_type == 'gin':
            if EnhancedGINDrugEncoder is not None:
                self.drug_encoder = EnhancedGINDrugEncoder(**kwargs.get('drug_config', {}))
            else:
                # Fallback to basic GIN encoder
                self.drug_encoder = GINDrugEncoder(**kwargs.get('drug_config', {}))
        elif drug_encoder_type == 'multiscale_gin':
            if MultiScaleGINEncoder is not None:
                self.drug_encoder = MultiScaleGINEncoder(**kwargs.get('drug_config', {}))
            else:
                raise ImportError("MultiScaleGINEncoder not available. Check drug_encoders module.")
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
        
        # Prediction head - use configurable prediction head system
        predictor_config = kwargs.get('predictor_config', {})
        predictor_type = predictor_config.pop('type', 'mlp')  # Default to MLP
        
        if predictor_type == 'mlp':
            self.predictor = MLPPredictionHead(
                input_dim=predictor_input_dim,
                **predictor_config
            )
        else:
            # Use factory for other types
            self.predictor = PredictionHeadFactory.create(
                predictor_type,
                input_dim=predictor_input_dim,
                **predictor_config
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
    
    def estimate_uncertainty(self, drug_data, protein_data, n_samples: int = 10) -> Tuple[torch.Tensor, float]:
        """
        Estimate prediction uncertainty using Monte Carlo dropout
        
        Args:
            drug_data: Drug molecular graph data
            protein_data: Protein sequence data
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (prediction, confidence_score)
        """
        self.eval()
        
        # Enable dropout for uncertainty estimation
        def enable_dropout(module):
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        
        self.apply(enable_dropout)
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self(drug_data, protein_data)
                predictions.append(pred)
        
        # Reset to eval mode
        self.eval()
        
        # Calculate statistics
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean()
        std_pred = predictions.std()
        
        # Convert std to confidence score (higher std = lower confidence)
        # Using exponential decay: confidence = exp(-std/scale)
        confidence = torch.exp(-std_pred / (mean_pred + 1e-8)).item()
        
        return mean_pred, confidence
    
    def set_training_phase(self, phase: int):
        """Set training phase for progressive training"""
        if phase == 2 and hasattr(self.protein_encoder, 'unfreeze_esm_layers'):
            self.protein_encoder.unfreeze_esm_layers()


# ============================================================================
# Model Factory (moved to model_factory.py)
# ============================================================================

# Import factory functions for backward compatibility
try:
    from .model_factory import (
        ModelFactory,
        create_dta_model,
        get_lightweight_model,
        get_production_model,
        create_lightweight_model,
        create_standard_model,
        create_high_performance_model
    )
except ImportError:
    # Fallback implementations if model_factory is not available
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

    def get_lightweight_model() -> UnifiedDTAModel:
        """Lightweight model for testing and development"""
        config = {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': False,
            'protein_config': {
                'embed_dim': 64,
                'num_filters': [32, 64],
                'kernel_sizes': [3, 5],
                'output_dim': 64
            },
            'drug_config': {
                'hidden_dim': 64,
                'num_layers': 3,
                'output_dim': 64
            },
            'predictor_config': {
                'type': 'mlp',
                'hidden_dims': [128], 
                'dropout': 0.2,
                'activation': 'relu',
                'use_batch_norm': True
            }
        }
        return create_dta_model(config)

    def get_production_model() -> UnifiedDTAModel:
        """Full-featured model for production use"""
        config = {
            'protein_encoder_type': 'esm',
            'drug_encoder_type': 'gin',
            'use_fusion': True,
            'protein_config': {'output_dim': 128, 'max_length': 200},
            'drug_config': {
                'hidden_dim': 128,
                'num_layers': 5,
                'output_dim': 128
            },
            'fusion_config': {'hidden_dim': 256, 'num_heads': 8},
            'predictor_config': {
                'type': 'mlp',
                'hidden_dims': [512, 256], 
                'dropout': 0.3,
                'activation': 'gelu',
                'use_batch_norm': True
            }
        }
        return create_dta_model(config)
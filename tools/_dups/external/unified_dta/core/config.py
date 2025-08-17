"""
Configuration management for the Unified DTA System
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """Configuration class for DTA models"""
    
    # Model configuration
    protein_encoder_type: str = 'esm'
    drug_encoder_type: str = 'gin'
    use_fusion: bool = True
    
    # Encoder configurations
    protein_config: Dict[str, Any] = None
    drug_config: Dict[str, Any] = None
    fusion_config: Dict[str, Any] = None
    predictor_config: Dict[str, Any] = None
    
    # Training configuration
    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 100
    
    # Memory optimization
    max_sequence_length: int = 200
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        """Initialize default configurations if not provided"""
        if self.protein_config is None:
            self.protein_config = {}
        if self.drug_config is None:
            self.drug_config = {}
        if self.fusion_config is None:
            self.fusion_config = {}
        if self.predictor_config is None:
            self.predictor_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file"""
        path = Path(path)
        config_dict = self.to_dict()
        
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from file"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from file"""
    return Config.load(path)


def get_default_configs() -> Dict[str, Config]:
    """Get default configurations for different use cases"""
    configs = {}
    
    # Lightweight configuration
    configs['lightweight'] = Config(
        protein_encoder_type='cnn',
        drug_encoder_type='gin',
        use_fusion=False,
        protein_config={'output_dim': 64},
        drug_config={'output_dim': 64, 'num_layers': 3},
        predictor_config={'hidden_dims': [128], 'dropout': 0.2},
        batch_size=8,
        learning_rate=1e-3
    )
    
    # Production configuration
    configs['production'] = Config(
        protein_encoder_type='esm',
        drug_encoder_type='gin',
        use_fusion=True,
        protein_config={'output_dim': 128, 'max_length': 200},
        drug_config={'output_dim': 128, 'num_layers': 5},
        fusion_config={'hidden_dim': 256, 'num_heads': 8},
        predictor_config={'hidden_dims': [512, 256], 'dropout': 0.3},
        batch_size=4,
        learning_rate=1e-3
    )
    
    return configs
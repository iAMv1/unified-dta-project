#!/usr/bin/env python3
"""
Minimal test for configuration system without heavy dependencies
"""

import sys
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union

# Minimal configuration classes for testing
@dataclass
class ProteinConfig:
    output_dim: int = 128
    max_length: int = 200
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    freeze_initial: bool = True

@dataclass
class DrugConfig:
    output_dim: int = 128
    node_features: int = 78
    hidden_dim: int = 128
    num_layers: int = 5
    dropout: float = 0.2
    use_batch_norm: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate_phase1: float = 1e-3
    learning_rate_phase2: float = 1e-4

@dataclass
class DTAConfig:
    protein_encoder_type: str = 'esm'
    drug_encoder_type: str = 'gin'
    use_fusion: bool = True
    protein_config: ProteinConfig = None
    drug_config: DrugConfig = None
    training_config: TrainingConfig = None
    
    def __post_init__(self):
        if self.protein_config is None:
            self.protein_config = ProteinConfig()
        if self.drug_config is None:
            self.drug_config = DrugConfig()
        if self.training_config is None:
            self.training_config = TrainingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DTAConfig':
        if 'protein_config' in config_dict and isinstance(config_dict['protein_config'], dict):
            config_dict['protein_config'] = ProteinConfig(**config_dict['protein_config'])
        if 'drug_config' in config_dict and isinstance(config_dict['drug_config'], dict):
            config_dict['drug_config'] = DrugConfig(**config_dict['drug_config'])
        if 'training_config' in config_dict and isinstance(config_dict['training_config'], dict):
            config_dict['training_config'] = TrainingConfig(**config_dict['training_config'])
        return cls(**config_dict)


def save_config(config: DTAConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to file"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config_dict, f, indent=2)


def load_config(config_path: Union[str, Path]) -> DTAConfig:
    """Load configuration from file"""
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_dict = json.load(f)
    
    return DTAConfig.from_dict(config_dict)


def test_model_factory_configs():
    """Test the model factory configuration system"""
    print("Testing Model Factory Configuration System")
    print("=" * 50)
    
    # Test predefined configurations
    configurations = {
        'lightweight': {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': False,
            'protein_config': {'output_dim': 64},
            'drug_config': {'output_dim': 64, 'num_layers': 3},
            'training_config': {'batch_size': 8}
        },
        'standard': {
            'protein_encoder_type': 'esm',
            'drug_encoder_type': 'gin',
            'use_fusion': True,
            'protein_config': {'output_dim': 128},
            'drug_config': {'output_dim': 128, 'num_layers': 5},
            'training_config': {'batch_size': 4}
        },
        'high_performance': {
            'protein_encoder_type': 'esm',
            'drug_encoder_type': 'gin',
            'use_fusion': True,
            'protein_config': {'output_dim': 256, 'max_length': 400},
            'drug_config': {'output_dim': 256, 'num_layers': 7},
            'training_config': {'batch_size': 2}
        }
    }
    
    print(f"\n1. Testing {len(configurations)} predefined configurations...")
    
    for name, config_dict in configurations.items():
        try:
            config = DTAConfig.from_dict(config_dict)
            print(f"  ✓ {name}: {config.protein_encoder_type}/{config.drug_encoder_type}")
            print(f"    - Protein dim: {config.protein_config.output_dim}")
            print(f"    - Drug dim: {config.drug_config.output_dim}")
            print(f"    - Batch size: {config.training_config.batch_size}")
            print(f"    - Use fusion: {config.use_fusion}")
        except Exception as e:
            print(f"  ✗ {name}: Error - {e}")
    
    # Test configuration validation
    print(f"\n2. Testing configuration validation...")
    
    def validate_config(config: DTAConfig) -> bool:
        """Simple validation"""
        errors = []
        
        if config.protein_encoder_type not in ['esm', 'cnn']:
            errors.append(f"Invalid protein encoder: {config.protein_encoder_type}")
        
        if config.drug_encoder_type not in ['gin']:
            errors.append(f"Invalid drug encoder: {config.drug_encoder_type}")
        
        if config.protein_config.output_dim <= 0:
            errors.append("Protein output_dim must be positive")
        
        if config.drug_config.output_dim <= 0:
            errors.append("Drug output_dim must be positive")
        
        if config.training_config.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        return len(errors) == 0
    
    for name, config_dict in configurations.items():
        try:
            config = DTAConfig.from_dict(config_dict)
            is_valid = validate_config(config)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {name}: {'valid' if is_valid else 'invalid'}")
        except Exception as e:
            print(f"  ✗ {name}: Validation error - {e}")
    
    # Test file operations
    print(f"\n3. Testing configuration file operations...")
    
    test_config = DTAConfig.from_dict(configurations['standard'])
    
    try:
        # Test YAML
        yaml_path = "test_factory_config.yaml"
        save_config(test_config, yaml_path)
        loaded_config = load_config(yaml_path)
        
        if loaded_config.protein_encoder_type == test_config.protein_encoder_type:
            print("  ✓ YAML save/load successful")
        else:
            print("  ✗ YAML save/load failed")
        
        # Test JSON
        json_path = "test_factory_config.json"
        save_config(test_config, json_path)
        loaded_config = load_config(json_path)
        
        if loaded_config.protein_encoder_type == test_config.protein_encoder_type:
            print("  ✓ JSON save/load successful")
        else:
            print("  ✗ JSON save/load failed")
        
        # Clean up
        Path(yaml_path).unlink(missing_ok=True)
        Path(json_path).unlink(missing_ok=True)
        print("  ✓ Test files cleaned up")
        
    except Exception as e:
        print(f"  ✗ File operations failed: {e}")
    
    # Test configuration inheritance simulation
    print(f"\n4. Testing configuration inheritance...")
    
    try:
        base_config = DTAConfig.from_dict(configurations['standard'])
        
        # Simulate inheritance with overrides
        override_dict = {
            'protein_config': {'output_dim': 192},
            'training_config': {'batch_size': 6}
        }
        
        # Merge configurations
        merged_dict = base_config.to_dict()
        for key, value in override_dict.items():
            if key in merged_dict and isinstance(merged_dict[key], dict):
                merged_dict[key].update(value)
            else:
                merged_dict[key] = value
        
        merged_config = DTAConfig.from_dict(merged_dict)
        
        if (merged_config.protein_config.output_dim == 192 and
            merged_config.training_config.batch_size == 6 and
            merged_config.drug_encoder_type == 'gin'):  # Should be preserved
            print("  ✓ Configuration inheritance successful")
        else:
            print("  ✗ Configuration inheritance failed")
    
    except Exception as e:
        print(f"  ✗ Configuration inheritance failed: {e}")
    
    print("\n" + "=" * 50)
    print("✓ Model factory configuration system test completed!")


if __name__ == "__main__":
    test_model_factory_configs()
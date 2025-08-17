"""
Configuration management for the Unified DTA System
Supports YAML/JSON configuration files with validation
"""

import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProteinConfig:
    """Configuration for protein encoders"""
    output_dim: int = 128
    max_length: int = 200
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    freeze_initial: bool = True
    vocab_size: int = 25
    embed_dim: int = 128
    num_filters: int = 32
    kernel_size: int = 8


@dataclass
class DrugConfig:
    """Configuration for drug encoders"""
    output_dim: int = 128
    node_features: int = 78
    hidden_dim: int = 128
    num_layers: int = 5
    dropout: float = 0.2
    use_batch_norm: bool = True


@dataclass
class FusionConfig:
    """Configuration for fusion mechanisms"""
    hidden_dim: int = 256
    num_heads: int = 8


@dataclass
class PredictorConfig:
    """Configuration for prediction heads"""
    hidden_dims: list = None
    dropout: float = 0.3
    activation: str = 'relu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    batch_size: int = 4
    learning_rate_phase1: float = 1e-3
    learning_rate_phase2: float = 1e-4
    num_epochs_phase1: int = 50
    num_epochs_phase2: int = 30
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    checkpoint_interval: int = 5
    gradient_clip_norm: float = 1.0
    
    # Memory management settings
    max_memory_mb: float = 4000
    enable_gradient_checkpointing: bool = True
    memory_monitoring_interval: int = 10
    aggressive_memory_cleanup: bool = False


@dataclass
class DataConfig:
    """Configuration for data processing"""
    datasets: list = None
    data_dir: str = "data"
    max_protein_length: int = 200
    validation_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["kiba", "davis", "bindingdb"]


@dataclass
class DTAConfig:
    """Main configuration class for the Unified DTA System"""
    
    # Model architecture
    protein_encoder_type: str = 'esm'
    drug_encoder_type: str = 'gin'
    use_fusion: bool = True
    
    # Component configurations
    protein_config: ProteinConfig = None
    drug_config: DrugConfig = None
    fusion_config: FusionConfig = None
    predictor_config: PredictorConfig = None
    training_config: TrainingConfig = None
    data_config: DataConfig = None
    
    # System settings
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'cuda:0', etc.
    seed: int = 42
    verbose: bool = True
    log_level: str = 'INFO'
    
    def __post_init__(self):
        # Initialize sub-configurations if not provided
        if self.protein_config is None:
            self.protein_config = ProteinConfig()
        if self.drug_config is None:
            self.drug_config = DrugConfig()
        if self.fusion_config is None:
            self.fusion_config = FusionConfig()
        if self.predictor_config is None:
            self.predictor_config = PredictorConfig()
        if self.training_config is None:
            self.training_config = TrainingConfig()
        if self.data_config is None:
            self.data_config = DataConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DTAConfig':
        """Create configuration from dictionary"""
        # Handle nested configurations
        if 'protein_config' in config_dict and isinstance(config_dict['protein_config'], dict):
            config_dict['protein_config'] = ProteinConfig(**config_dict['protein_config'])
        if 'drug_config' in config_dict and isinstance(config_dict['drug_config'], dict):
            config_dict['drug_config'] = DrugConfig(**config_dict['drug_config'])
        if 'fusion_config' in config_dict and isinstance(config_dict['fusion_config'], dict):
            config_dict['fusion_config'] = FusionConfig(**config_dict['fusion_config'])
        if 'predictor_config' in config_dict and isinstance(config_dict['predictor_config'], dict):
            config_dict['predictor_config'] = PredictorConfig(**config_dict['predictor_config'])
        if 'training_config' in config_dict and isinstance(config_dict['training_config'], dict):
            config_dict['training_config'] = TrainingConfig(**config_dict['training_config'])
        if 'data_config' in config_dict and isinstance(config_dict['data_config'], dict):
            config_dict['data_config'] = DataConfig(**config_dict['data_config'])
        
        return cls(**config_dict)


def load_config(config_path: Union[str, Path], 
                base_config: Optional[Union[str, Path, DTAConfig]] = None) -> DTAConfig:
    """
    Load configuration from YAML or JSON file with optional inheritance
    
    Args:
        config_path: Path to configuration file
        base_config: Base configuration to inherit from (file path or DTAConfig object)
    
    Returns:
        DTAConfig object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Handle inheritance
        if 'inherits_from' in config_dict:
            inherit_path = config_dict.pop('inherits_from')
            if not Path(inherit_path).is_absolute():
                inherit_path = config_path.parent / inherit_path
            base_config = load_config(inherit_path)
        
        # Apply base configuration if provided
        if base_config is not None:
            if isinstance(base_config, (str, Path)):
                base_config = load_config(base_config)
            
            # Merge with base configuration
            base_dict = base_config.to_dict()
            config_dict = _deep_merge_dicts(base_dict, config_dict)
        
        return DTAConfig.from_dict(config_dict)
    
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: DTAConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML or JSON file"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        config_dict = config.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        logger.info(f"Configuration saved to {config_path}")
    
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        raise


def validate_config(config: DTAConfig, detailed: bool = False) -> bool:
    """
    Validate configuration parameters
    
    Args:
        config: Configuration to validate
        detailed: Whether to use detailed validation with helpful messages
    
    Returns:
        True if configuration is valid
    """
    if detailed:
        try:
            from .config_validator import validate_config_with_details
            is_valid, report = validate_config_with_details(config)
            if not is_valid:
                logger.error("Configuration validation failed:")
                logger.error(report)
            else:
                logger.info("Configuration validation passed")
            return is_valid
        except ImportError:
            logger.warning("Detailed validation not available, using basic validation")
    
    # Basic validation (fallback)
    errors = []
    
    # Validate encoder types
    valid_protein_encoders = ['esm', 'cnn']
    valid_drug_encoders = ['gin']
    
    if config.protein_encoder_type not in valid_protein_encoders:
        errors.append(f"Invalid protein encoder: {config.protein_encoder_type}. "
                     f"Valid options: {valid_protein_encoders}")
    
    if config.drug_encoder_type not in valid_drug_encoders:
        errors.append(f"Invalid drug encoder: {config.drug_encoder_type}. "
                     f"Valid options: {valid_drug_encoders}")
    
    # Validate dimensions
    if config.protein_config.output_dim <= 0:
        errors.append("Protein output dimension must be positive")
    
    if config.drug_config.output_dim <= 0:
        errors.append("Drug output dimension must be positive")
    
    if config.drug_config.num_layers <= 0:
        errors.append("Number of GIN layers must be positive")
    
    # Validate training parameters
    if config.training_config.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if config.training_config.learning_rate_phase1 <= 0:
        errors.append("Learning rate for phase 1 must be positive")
    
    if config.training_config.learning_rate_phase2 <= 0:
        errors.append("Learning rate for phase 2 must be positive")
    
    # Validate data parameters
    if not (0 < config.data_config.validation_split < 1):
        errors.append("Validation split must be between 0 and 1")
    
    if not (0 < config.data_config.test_split < 1):
        errors.append("Test split must be between 0 and 1")
    
    if config.data_config.validation_split + config.data_config.test_split >= 1:
        errors.append("Validation and test splits combined must be less than 1")
    
    # Log errors
    if errors:
        for error in errors:
            logger.error(f"Configuration validation error: {error}")
        return False
    
    logger.info("Configuration validation passed")
    return True


def get_default_configs() -> Dict[str, DTAConfig]:
    """Get predefined default configurations"""
    
    # Lightweight configuration for development
    lightweight_config = DTAConfig(
        protein_encoder_type='cnn',
        drug_encoder_type='gin',
        use_fusion=False,
        protein_config=ProteinConfig(output_dim=64),
        drug_config=DrugConfig(output_dim=64, num_layers=3),
        predictor_config=PredictorConfig(hidden_dims=[128], dropout=0.2),
        training_config=TrainingConfig(batch_size=8, num_epochs_phase1=20, num_epochs_phase2=10)
    )
    
    # Production configuration with full features
    production_config = DTAConfig(
        protein_encoder_type='esm',
        drug_encoder_type='gin',
        use_fusion=True,
        protein_config=ProteinConfig(output_dim=128, max_length=200),
        drug_config=DrugConfig(output_dim=128, num_layers=5),
        fusion_config=FusionConfig(hidden_dim=256, num_heads=8),
        predictor_config=PredictorConfig(hidden_dims=[512, 256], dropout=0.3),
        training_config=TrainingConfig(batch_size=4, num_epochs_phase1=50, num_epochs_phase2=30)
    )
    
    # High-performance configuration for large datasets
    high_performance_config = DTAConfig(
        protein_encoder_type='esm',
        drug_encoder_type='gin',
        use_fusion=True,
        protein_config=ProteinConfig(output_dim=256, max_length=400),
        drug_config=DrugConfig(output_dim=256, num_layers=7, hidden_dim=256),
        fusion_config=FusionConfig(hidden_dim=512, num_heads=16),
        predictor_config=PredictorConfig(hidden_dims=[1024, 512, 256], dropout=0.3),
        training_config=TrainingConfig(batch_size=2, num_epochs_phase1=100, num_epochs_phase2=50)
    )
    
    return {
        'lightweight': lightweight_config,
        'production': production_config,
        'high_performance': high_performance_config
    }


def create_config_template(config_path: Union[str, Path], config_type: str = 'production') -> None:
    """Create a configuration template file"""
    default_configs = get_default_configs()
    
    if config_type not in default_configs:
        raise ValueError(f"Unknown config type: {config_type}. "
                        f"Available types: {list(default_configs.keys())}")
    
    config = default_configs[config_type]
    save_config(config, config_path)
    logger.info(f"Configuration template '{config_type}' created at {config_path}")


def get_environment_config(env: str = 'development') -> DTAConfig:
    """Get environment-specific configuration"""
    
    base_configs = get_default_configs()
    
    if env == 'development':
        return base_configs['lightweight']
    elif env == 'staging':
        return base_configs['production']
    elif env == 'production':
        return base_configs['high_performance']
    else:
        logger.warning(f"Unknown environment '{env}', using development config")
        return base_configs['lightweight']


def merge_configs(base_config: DTAConfig, override_config: Dict[str, Any]) -> DTAConfig:
    """Merge configuration with overrides"""
    base_dict = base_config.to_dict()
    
    def deep_merge(base: Dict, override: Dict) -> Dict:
        """Recursively merge dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = deep_merge(base_dict, override_config)
    return DTAConfig.from_dict(merged_dict)


def generate_config_documentation(output_path: Union[str, Path]) -> None:
    """Generate configuration documentation"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc_content = """# Configuration Documentation

## Overview

The Unified DTA System uses a hierarchical configuration system that supports:
- YAML and JSON configuration files
- Environment-specific configurations
- Configuration validation and error checking
- Predefined configuration templates

## Configuration Structure

### Main Configuration (DTAConfig)

- `protein_encoder_type`: Type of protein encoder ('esm' or 'cnn')
- `drug_encoder_type`: Type of drug encoder ('gin')
- `use_fusion`: Whether to use multi-modal fusion
- `device`: Device to use ('auto', 'cpu', 'cuda', etc.)
- `seed`: Random seed for reproducibility
- `verbose`: Enable verbose logging
- `log_level`: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

### Protein Configuration (ProteinConfig)

- `output_dim`: Output dimension of protein encoder
- `max_length`: Maximum protein sequence length
- `model_name`: ESM model name (for ESM encoder)
- `freeze_initial`: Whether to freeze ESM weights initially
- `vocab_size`: Vocabulary size (for CNN encoder)
- `embed_dim`: Embedding dimension (for CNN encoder)
- `num_filters`: Number of CNN filters
- `kernel_size`: CNN kernel size

### Drug Configuration (DrugConfig)

- `output_dim`: Output dimension of drug encoder
- `node_features`: Number of node features in molecular graphs
- `hidden_dim`: Hidden dimension of GIN layers
- `num_layers`: Number of GIN layers
- `dropout`: Dropout rate
- `use_batch_norm`: Whether to use batch normalization

### Fusion Configuration (FusionConfig)

- `hidden_dim`: Hidden dimension for fusion
- `num_heads`: Number of attention heads

### Predictor Configuration (PredictorConfig)

- `hidden_dims`: List of hidden layer dimensions
- `dropout`: Dropout rate
- `activation`: Activation function ('relu', 'gelu', etc.)

### Training Configuration (TrainingConfig)

- `batch_size`: Training batch size
- `learning_rate_phase1`: Learning rate for phase 1
- `learning_rate_phase2`: Learning rate for phase 2
- `num_epochs_phase1`: Number of epochs for phase 1
- `num_epochs_phase2`: Number of epochs for phase 2
- `weight_decay`: Weight decay for regularization
- `early_stopping_patience`: Early stopping patience
- `checkpoint_interval`: Checkpoint saving interval
- `gradient_clip_norm`: Gradient clipping norm
- `max_memory_mb`: Maximum memory usage in MB
- `enable_gradient_checkpointing`: Enable gradient checkpointing
- `memory_monitoring_interval`: Memory monitoring interval
- `aggressive_memory_cleanup`: Enable aggressive memory cleanup

### Data Configuration (DataConfig)

- `datasets`: List of datasets to use
- `data_dir`: Data directory path
- `max_protein_length`: Maximum protein sequence length
- `validation_split`: Validation split ratio
- `test_split`: Test split ratio
- `num_workers`: Number of data loader workers
- `pin_memory`: Whether to pin memory for data loading

## Predefined Configurations

### Lightweight Configuration
- Optimized for development and testing
- Uses CNN protein encoder
- No fusion mechanism
- Minimal memory usage (~100MB)

### Production Configuration
- Balanced performance and resource usage
- Uses ESM-2 protein encoder
- Includes fusion mechanism
- Moderate memory usage (~2GB)

### High-Performance Configuration
- Maximum performance for large-scale research
- Uses larger ESM-2 model
- Advanced fusion and prediction heads
- High memory usage (~8GB)

## Usage Examples

### Loading Configuration from File

```python
from core.config import load_config

# Load from YAML file
config = load_config('config.yaml')

# Load from JSON file
config = load_config('config.json')
```

### Creating Configuration Templates

```python
from core.config import create_config_template

# Create lightweight template
create_config_template('lightweight_config.yaml', 'lightweight')

# Create production template
create_config_template('production_config.yaml', 'production')
```

### Environment-Specific Configuration

```python
from core.config import get_environment_config

# Get development configuration
dev_config = get_environment_config('development')

# Get production configuration
prod_config = get_environment_config('production')
```

### Configuration Validation

```python
from core.config import validate_config

# Validate configuration
is_valid = validate_config(config)
if not is_valid:
    print("Configuration validation failed")
```

### Merging Configurations

```python
from core.config import merge_configs

# Merge with overrides
overrides = {
    'training_config': {
        'batch_size': 8,
        'learning_rate_phase1': 2e-3
    }
}
merged_config = merge_configs(base_config, overrides)
```

## Configuration File Examples

### YAML Configuration Example

```yaml
protein_encoder_type: esm
drug_encoder_type: gin
use_fusion: true
device: auto
seed: 42
verbose: true
log_level: INFO

protein_config:
  output_dim: 128
  max_length: 200
  model_name: facebook/esm2_t6_8M_UR50D
  freeze_initial: true

drug_config:
  output_dim: 128
  hidden_dim: 128
  num_layers: 5
  dropout: 0.2
  use_batch_norm: true

fusion_config:
  hidden_dim: 256
  num_heads: 8

predictor_config:
  hidden_dims: [512, 256]
  dropout: 0.3
  activation: relu

training_config:
  batch_size: 4
  learning_rate_phase1: 0.001
  learning_rate_phase2: 0.0001
  num_epochs_phase1: 50
  num_epochs_phase2: 30
  weight_decay: 0.00001
  early_stopping_patience: 10

data_config:
  datasets: [kiba, davis, bindingdb]
  data_dir: data
  max_protein_length: 200
  validation_split: 0.1
  test_split: 0.1
```

### JSON Configuration Example

```json
{
  "protein_encoder_type": "esm",
  "drug_encoder_type": "gin",
  "use_fusion": true,
  "device": "auto",
  "seed": 42,
  "verbose": true,
  "log_level": "INFO",
  "protein_config": {
    "output_dim": 128,
    "max_length": 200,
    "model_name": "facebook/esm2_t6_8M_UR50D",
    "freeze_initial": true
  },
  "drug_config": {
    "output_dim": 128,
    "hidden_dim": 128,
    "num_layers": 5,
    "dropout": 0.2,
    "use_batch_norm": true
  },
  "fusion_config": {
    "hidden_dim": 256,
    "num_heads": 8
  },
  "predictor_config": {
    "hidden_dims": [512, 256],
    "dropout": 0.3,
    "activation": "relu"
  },
  "training_config": {
    "batch_size": 4,
    "learning_rate_phase1": 0.001,
    "learning_rate_phase2": 0.0001,
    "num_epochs_phase1": 50,
    "num_epochs_phase2": 30,
    "weight_decay": 1e-05,
    "early_stopping_patience": 10
  },
  "data_config": {
    "datasets": ["kiba", "davis", "bindingdb"],
    "data_dir": "data",
    "max_protein_length": 200,
    "validation_split": 0.1,
    "test_split": 0.1
  }
}
```
"""
    
    with open(output_path, 'w') as f:
        f.write(doc_content)
    
    logger.info(f"Configuration documentation generated at {output_path}")


class ConfigurationManager:
    """Advanced configuration management utilities"""
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def create_all_templates(self) -> None:
        """Create all predefined configuration templates"""
        default_configs = get_default_configs()
        
        for config_name, config in default_configs.items():
            yaml_path = self.config_dir / f"{config_name}_config.yaml"
            json_path = self.config_dir / f"{config_name}_config.json"
            
            save_config(config, yaml_path)
            save_config(config, json_path)
            
            logger.info(f"Created templates for {config_name} configuration")
    
    def validate_all_configs(self) -> Dict[str, bool]:
        """Validate all configuration files in the config directory"""
        results = {}
        
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                config = load_config(config_file)
                is_valid = validate_config(config)
                results[str(config_file)] = is_valid
            except Exception as e:
                logger.error(f"Error validating {config_file}: {e}")
                results[str(config_file)] = False
        
        for config_file in self.config_dir.glob("*.json"):
            try:
                config = load_config(config_file)
                is_valid = validate_config(config)
                results[str(config_file)] = is_valid
            except Exception as e:
                logger.error(f"Error validating {config_file}: {e}")
                results[str(config_file)] = False
        
        return results
    
    def compare_configs(self, config1_path: Union[str, Path], 
                       config2_path: Union[str, Path]) -> Dict[str, Any]:
        """Compare two configuration files"""
        config1 = load_config(config1_path)
        config2 = load_config(config2_path)
        
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()
        
        def find_differences(d1: Dict, d2: Dict, path: str = "") -> Dict[str, Any]:
            differences = {}
            
            all_keys = set(d1.keys()) | set(d2.keys())
            
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences[current_path] = {"status": "added", "value": d2[key]}
                elif key not in d2:
                    differences[current_path] = {"status": "removed", "value": d1[key]}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    sub_diffs = find_differences(d1[key], d2[key], current_path)
                    differences.update(sub_diffs)
                elif d1[key] != d2[key]:
                    differences[current_path] = {
                        "status": "changed",
                        "old_value": d1[key],
                        "new_value": d2[key]
                    }
            
            return differences
        
        return find_differences(dict1, dict2)
    
    def generate_config_report(self, output_path: Union[str, Path]) -> None:
        """Generate a comprehensive configuration report"""
        output_path = Path(output_path)
        
        # Validate all configs
        validation_results = self.validate_all_configs()
        
        # Get default configs info
        default_configs = get_default_configs()
        
        report_content = f"""# Configuration Report

Generated on: {Path().cwd()}
Config Directory: {self.config_dir}

## Validation Results

"""
        
        for config_file, is_valid in validation_results.items():
            status = "✓ VALID" if is_valid else "✗ INVALID"
            report_content += f"- {config_file}: {status}\n"
        
        report_content += f"""

## Available Default Configurations

"""
        
        for name, config in default_configs.items():
            report_content += f"""
### {name.title()} Configuration

- Protein Encoder: {config.protein_encoder_type}
- Drug Encoder: {config.drug_encoder_type}
- Use Fusion: {config.use_fusion}
- Protein Output Dim: {config.protein_config.output_dim}
- Drug Output Dim: {config.drug_config.output_dim}
- Batch Size: {config.training_config.batch_size}
- Phase 1 LR: {config.training_config.learning_rate_phase1}
- Phase 2 LR: {config.training_config.learning_rate_phase2}
"""
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Configuration report generated at {output_path}")


if __name__ == "__main__":
    # Enhanced command-line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration management utilities")
    parser.add_argument("--create-template", type=str, help="Create configuration template")
    parser.add_argument("--config-type", type=str, default="production", 
                       choices=["lightweight", "production", "high_performance"],
                       help="Type of configuration template to create")
    parser.add_argument("--validate", type=str, help="Validate configuration file")
    parser.add_argument("--create-all-templates", action="store_true", 
                       help="Create all predefined configuration templates")
    parser.add_argument("--config-dir", type=str, default="configs",
                       help="Configuration directory")
    parser.add_argument("--generate-docs", type=str, help="Generate configuration documentation")
    parser.add_argument("--generate-report", type=str, help="Generate configuration report")
    parser.add_argument("--compare", nargs=2, metavar=('CONFIG1', 'CONFIG2'),
                       help="Compare two configuration files")
    parser.add_argument("--environment", type=str, choices=["development", "staging", "production"],
                       help="Get environment-specific configuration")
    
    args = parser.parse_args()
    
    # Initialize configuration manager
    config_manager = ConfigurationManager(args.config_dir)
    
    if args.create_template:
        create_config_template(args.create_template, args.config_type)
    
    if args.validate:
        config = load_config(args.validate)
        is_valid = validate_config(config)
        print(f"Configuration is {'valid' if is_valid else 'invalid'}")
    
    if args.create_all_templates:
        config_manager.create_all_templates()
        print(f"All configuration templates created in {args.config_dir}")
    
    if args.generate_docs:
        generate_config_documentation(args.generate_docs)
        print(f"Configuration documentation generated at {args.generate_docs}")
    
    if args.generate_report:
        config_manager.generate_config_report(args.generate_report)
        print(f"Configuration report generated at {args.generate_report}")
    
    if args.compare:
        differences = config_manager.compare_configs(args.compare[0], args.compare[1])
        if differences:
            print("Configuration differences found:")
            for path, diff in differences.items():
                print(f"  {path}: {diff}")
        else:
            print("Configurations are identical")
    
    if args.environment:
        env_config = get_environment_config(args.environment)
        print(f"Environment configuration for '{args.environment}':")
        print(f"  Protein Encoder: {env_config.protein_encoder_type}")
        print(f"  Drug Encoder: {env_config.drug_encoder_type}")
        print(f"  Use Fusion: {env_config.use_fusion}")
        print(f"  Batch Size: {env_config.training_config.batch_size}")
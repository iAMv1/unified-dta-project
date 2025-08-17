# Configuration Guide

## Overview

The Unified DTA System uses a flexible configuration system that allows you to customize model architecture, training parameters, and system behavior through YAML files, Python dictionaries, or environment variables.

## Configuration Methods

### 1. YAML Configuration Files

```yaml
# configs/production.yaml
model:
  protein_encoder_type: "esm"
  drug_encoder_type: "gin"
  use_fusion: true
  
protein_config:
  model_name: "facebook/esm2_t6_8M_UR50D"
  output_dim: 128
  max_length: 200
  
drug_config:
  input_dim: 78
  hidden_dim: 128
  num_layers: 5
  dropout: 0.2
  
fusion_config:
  hidden_dim: 256
  num_heads: 8
  dropout: 0.1
  
predictor_config:
  hidden_dims: [512, 256]
  dropout: 0.3
  activation: "relu"

training:
  batch_size: 4
  learning_rate: 1e-3
  num_epochs: 50
  use_2phase_training: true
  phase1_epochs: 30
  phase2_lr: 1e-4
  early_stopping_patience: 10
  
system:
  device: "auto"
  memory_efficient: false
  gradient_checkpointing: true
  mixed_precision: true
```

### 2. Python Dictionary Configuration

```python
from unified_dta import UnifiedDTAModel

config = {
    'model': {
        'protein_encoder_type': 'esm',
        'drug_encoder_type': 'gin',
        'use_fusion': True
    },
    'protein_config': {
        'output_dim': 128,
        'max_length': 200
    },
    'drug_config': {
        'hidden_dim': 128,
        'num_layers': 5
    }
}

model = UnifiedDTAModel.from_config(config)
```

### 3. Environment Variables

```bash
export UNIFIED_DTA_CONFIG=production
export UNIFIED_DTA_DEVICE=cuda:0
export UNIFIED_DTA_BATCH_SIZE=8
export UNIFIED_DTA_MEMORY_EFFICIENT=true
```

## Pre-defined Configurations

### Lightweight Configuration
**Use Case**: Development, testing, resource-constrained environments

```yaml
# configs/lightweight.yaml
model:
  protein_encoder_type: "cnn"
  drug_encoder_type: "gin"
  use_fusion: false
  
protein_config:
  vocab_size: 25
  embed_dim: 64
  num_filters: 16
  filter_sizes: [3, 5]
  output_dim: 64
  
drug_config:
  input_dim: 78
  hidden_dim: 64
  num_layers: 3
  dropout: 0.1
  
predictor_config:
  hidden_dims: [128]
  dropout: 0.2
  activation: "relu"

training:
  batch_size: 16
  learning_rate: 1e-3
  num_epochs: 20
  use_2phase_training: false

system:
  memory_efficient: true
  gradient_checkpointing: false
```

**Memory Usage**: ~100MB RAM
**Training Time**: Fast
**Performance**: Good for development

### Production Configuration
**Use Case**: Research, production deployments, maximum performance

```yaml
# configs/production.yaml
model:
  protein_encoder_type: "esm"
  drug_encoder_type: "gin"
  use_fusion: true
  
protein_config:
  model_name: "facebook/esm2_t6_8M_UR50D"
  output_dim: 128
  max_length: 200
  freeze_layers: 8  # Freeze first 8 layers
  
drug_config:
  input_dim: 78
  hidden_dim: 128
  num_layers: 5
  dropout: 0.2
  use_residual: true
  
fusion_config:
  type: "cross_attention"
  hidden_dim: 256
  num_heads: 8
  dropout: 0.1
  
predictor_config:
  hidden_dims: [512, 256, 128]
  dropout: 0.3
  activation: "gelu"
  use_batch_norm: true

training:
  batch_size: 4
  learning_rate: 1e-3
  num_epochs: 100
  use_2phase_training: true
  phase1_epochs: 60
  phase2_lr: 1e-4
  weight_decay: 1e-4
  early_stopping_patience: 15

system:
  device: "auto"
  memory_efficient: false
  gradient_checkpointing: true
  mixed_precision: true
```

**Memory Usage**: ~4GB RAM + GPU
**Training Time**: Slower but thorough
**Performance**: State-of-the-art results

### High-Performance Configuration
**Use Case**: Large-scale experiments, maximum accuracy

```yaml
# configs/high_performance.yaml
model:
  protein_encoder_type: "esm"
  drug_encoder_type: "gin"
  use_fusion: true
  
protein_config:
  model_name: "facebook/esm2_t12_35M_UR50D"  # Larger ESM-2 model
  output_dim: 256
  max_length: 400
  
drug_config:
  hidden_dim: 256
  num_layers: 7
  dropout: 0.1
  
fusion_config:
  hidden_dim: 512
  num_heads: 16
  num_layers: 3
  
predictor_config:
  hidden_dims: [1024, 512, 256]
  dropout: 0.2

training:
  batch_size: 2  # Smaller due to larger model
  learning_rate: 5e-4
  num_epochs: 150
  gradient_accumulation_steps: 4

system:
  gradient_checkpointing: true
  mixed_precision: true
```

## Configuration Parameters

### Model Parameters

#### Protein Encoder Configuration
```yaml
protein_config:
  # ESM-2 Encoder
  model_name: "facebook/esm2_t6_8M_UR50D"  # ESM-2 model variant
  output_dim: 128                          # Output feature dimension
  max_length: 200                          # Maximum sequence length
  freeze_layers: 8                         # Number of layers to freeze
  
  # CNN Encoder
  vocab_size: 25                           # Amino acid vocabulary size
  embed_dim: 128                           # Embedding dimension
  num_filters: 32                          # Number of CNN filters
  filter_sizes: [3, 5, 7]                 # Convolution kernel sizes
  use_attention: true                      # Use SE attention blocks
```

#### Drug Encoder Configuration
```yaml
drug_config:
  input_dim: 78                            # Node feature dimension
  hidden_dim: 128                          # Hidden layer dimension
  num_layers: 5                            # Number of GIN layers
  dropout: 0.2                             # Dropout probability
  use_residual: true                       # Use residual connections
  pooling: "dual"                          # Pooling type: "mean", "max", "dual"
  batch_norm: true                         # Use batch normalization
```

#### Fusion Configuration
```yaml
fusion_config:
  type: "cross_attention"                  # Fusion type: "concat", "cross_attention"
  hidden_dim: 256                          # Hidden dimension
  num_heads: 8                             # Number of attention heads
  num_layers: 2                            # Number of fusion layers
  dropout: 0.1                             # Dropout probability
```

#### Predictor Configuration
```yaml
predictor_config:
  hidden_dims: [512, 256]                  # Hidden layer dimensions
  dropout: 0.3                             # Dropout probability
  activation: "relu"                       # Activation function
  use_batch_norm: true                     # Use batch normalization
  output_activation: null                  # Output activation (null for regression)
```

### Training Parameters

```yaml
training:
  # Basic training parameters
  batch_size: 4                            # Training batch size
  learning_rate: 1e-3                      # Initial learning rate
  num_epochs: 50                           # Number of training epochs
  weight_decay: 1e-4                       # L2 regularization
  
  # 2-phase training
  use_2phase_training: true                # Enable 2-phase training
  phase1_epochs: 30                        # Epochs for phase 1
  phase2_lr: 1e-4                          # Learning rate for phase 2
  
  # Optimization
  optimizer: "adam"                        # Optimizer type
  scheduler: "cosine"                      # Learning rate scheduler
  gradient_accumulation_steps: 1           # Gradient accumulation
  max_grad_norm: 1.0                       # Gradient clipping
  
  # Early stopping
  early_stopping_patience: 10              # Early stopping patience
  early_stopping_metric: "val_loss"       # Metric for early stopping
  
  # Validation
  validation_split: 0.2                    # Validation split ratio
  validation_frequency: 1                  # Validation frequency (epochs)
```

### System Parameters

```yaml
system:
  # Device configuration
  device: "auto"                           # Device: "auto", "cpu", "cuda", "cuda:0"
  
  # Memory optimization
  memory_efficient: false                  # Enable memory efficient mode
  gradient_checkpointing: true             # Enable gradient checkpointing
  mixed_precision: true                    # Enable mixed precision training
  
  # Logging and monitoring
  log_level: "INFO"                        # Logging level
  log_frequency: 100                       # Log frequency (steps)
  save_frequency: 1000                     # Checkpoint save frequency
  
  # Reproducibility
  seed: 42                                 # Random seed
  deterministic: true                      # Deterministic operations
```

## Configuration Validation

### Automatic Validation
```python
from unified_dta.core.config import validate_config

config = load_config('configs/production.yaml')
validated_config = validate_config(config)
```

### Custom Validation Rules
```python
def custom_validation(config):
    # Check memory requirements
    if config['model']['protein_encoder_type'] == 'esm':
        assert config['training']['batch_size'] <= 8, \
            "ESM-2 requires batch_size <= 8"
    
    # Check dimension compatibility
    if config['model']['use_fusion']:
        assert 'fusion_config' in config, \
            "Fusion config required when use_fusion=True"
    
    return config
```

## Environment-Specific Configurations

### Development Environment
```yaml
# configs/development.yaml
extends: "lightweight"

training:
  num_epochs: 5
  validation_frequency: 1

system:
  log_level: "DEBUG"
  save_frequency: 50
```

### Testing Environment
```yaml
# configs/testing.yaml
extends: "lightweight"

training:
  batch_size: 2
  num_epochs: 2

system:
  deterministic: true
  seed: 12345
```

### Production Environment
```yaml
# configs/production.yaml
extends: "production"

system:
  log_level: "WARNING"
  mixed_precision: true
  gradient_checkpointing: true
```

## Configuration Inheritance

```yaml
# Base configuration
base_config: &base
  model:
    drug_encoder_type: "gin"
  training:
    optimizer: "adam"

# Lightweight inherits from base
lightweight_config:
  <<: *base
  model:
    protein_encoder_type: "cnn"
    use_fusion: false

# Production inherits from base
production_config:
  <<: *base
  model:
    protein_encoder_type: "esm"
    use_fusion: true
```

## Dynamic Configuration

### Runtime Configuration Updates
```python
from unified_dta import UnifiedDTAModel

# Load base configuration
model = UnifiedDTAModel.from_config('production')

# Update configuration at runtime
model.update_config({
    'training': {'batch_size': 8},
    'system': {'device': 'cuda:1'}
})
```

### Conditional Configuration
```python
import torch

config = load_config('production')

# Adjust based on available memory
if torch.cuda.get_device_properties(0).total_memory < 8e9:  # 8GB
    config['training']['batch_size'] = 2
    config['system']['gradient_checkpointing'] = True

model = UnifiedDTAModel.from_config(config)
```

## Best Practices

### 1. Start with Pre-defined Configurations
```python
# Good: Start with tested configuration
model = UnifiedDTAModel.from_pretrained('lightweight')

# Then customize as needed
model.update_config({'training': {'batch_size': 8}})
```

### 2. Use Configuration Validation
```python
# Always validate custom configurations
config = {
    'model': {'protein_encoder_type': 'esm'},
    'training': {'batch_size': 32}  # Too large for ESM-2
}

try:
    validated_config = validate_config(config)
except ValueError as e:
    print(f"Configuration error: {e}")
```

### 3. Environment-Specific Configs
```python
import os

# Use different configs for different environments
env = os.getenv('ENVIRONMENT', 'development')
config_file = f'configs/{env}.yaml'
model = UnifiedDTAModel.from_config(config_file)
```

### 4. Monitor Resource Usage
```python
from unified_dta.utils import MemoryMonitor

with MemoryMonitor() as monitor:
    model = UnifiedDTAModel.from_config('production')
    
if monitor.peak_memory_gb > 8:
    print("Consider using lightweight configuration")
```
# API Reference

## Overview

The Unified DTA System provides both Python API and RESTful HTTP API for drug-target affinity prediction and model training.

## Python API

### Core Classes

#### UnifiedDTAModel
Main model class for drug-target affinity prediction.

```python
from unified_dta import UnifiedDTAModel

# Load pre-trained model
model = UnifiedDTAModel.from_pretrained('production')

# Create model from configuration
model = UnifiedDTAModel.from_config(config_dict)

# Load model from checkpoint
model = UnifiedDTAModel.load('path/to/model.pt')
```

**Methods**:
- `predict(smiles, protein_sequence)` - Single prediction
- `predict_batch(smiles_list, protein_list)` - Batch prediction
- `train(dataset, config)` - Train the model
- `evaluate(test_dataset)` - Evaluate model performance
- `save(path)` - Save model checkpoint

#### TrainingConfig
Configuration class for training parameters.

```python
from unified_dta import TrainingConfig

config = TrainingConfig(
    batch_size=4,
    learning_rate=1e-3,
    num_epochs=50,
    use_2phase_training=True
)
```

#### DTADataset
Dataset class for loading and preprocessing DTA data.

```python
from unified_dta.data import DTADataset

# Load from CSV
dataset = DTADataset.from_csv('data/kiba_train.csv')

# Load from DataFrame
dataset = DTADataset.from_dataframe(df)

# Create custom dataset
dataset = DTADataset(smiles_list, protein_list, affinity_list)
```

### Encoders

#### Protein Encoders

##### ESMProteinEncoder
```python
from unified_dta.encoders import ESMProteinEncoder

encoder = ESMProteinEncoder(
    model_name='facebook/esm2_t6_8M_UR50D',
    output_dim=128,
    max_length=200
)

# Forward pass
features = encoder(protein_sequences)  # [batch_size, output_dim]
```

##### CNNProteinEncoder
```python
from unified_dta.encoders import CNNProteinEncoder

encoder = CNNProteinEncoder(
    vocab_size=25,
    embed_dim=128,
    num_filters=32,
    filter_sizes=[3, 5, 7],
    output_dim=128
)

features = encoder(tokenized_sequences)
```

#### Drug Encoders

##### GINDrugEncoder
```python
from unified_dta.encoders import GINDrugEncoder

encoder = GINDrugEncoder(
    input_dim=78,
    hidden_dim=128,
    num_layers=5,
    dropout=0.2
)

features = encoder(graph_batch)  # PyTorch Geometric batch
```

### Fusion Mechanisms

#### MultiModalFusion
```python
from unified_dta.encoders import MultiModalFusion

fusion = MultiModalFusion(
    drug_dim=128,
    protein_dim=128,
    hidden_dim=256,
    num_heads=8
)

fused_features = fusion(drug_features, protein_features)
```

### Utilities

#### Data Processing
```python
from unified_dta.data import smiles_to_graph, tokenize_protein

# Convert SMILES to molecular graph
graph = smiles_to_graph("CCO")

# Tokenize protein sequence
tokens = tokenize_protein("MKTVRQERLK")
```

#### Memory Management
```python
from unified_dta.utils import MemoryMonitor, optimize_memory

# Monitor memory usage
with MemoryMonitor() as monitor:
    predictions = model.predict_batch(smiles_list, protein_list)

print(f"Peak memory: {monitor.peak_memory_mb:.1f} MB")

# Optimize memory usage
optimize_memory(model, enable_checkpointing=True)
```

#### Model Factory
```python
from unified_dta.core import ModelFactory

# Create lightweight model
model = ModelFactory.create_model('lightweight')

# Create production model
model = ModelFactory.create_model('production')

# Create custom model
model = ModelFactory.create_model(custom_config)
```

## RESTful API

### Starting the API Server

```bash
# Start with default settings
python -m unified_dta.api.main

# Start with custom configuration
python -m unified_dta.api.main --config production --port 8080

# Start with specific model
python -m unified_dta.api.main --model-path /path/to/model.pt
```

### API Endpoints

#### Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "smiles": "CCO",
  "protein_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
}
```

**Response**:
```json
{
  "prediction": 5.234,
  "confidence": 0.89,
  "processing_time": 0.045
}
```

#### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "data": [
    {
      "smiles": "CCO",
      "protein_sequence": "MKTV..."
    },
    {
      "smiles": "CC(=O)O",
      "protein_sequence": "MKTV..."
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [5.234, 4.876],
  "confidences": [0.89, 0.92],
  "processing_time": 0.123
}
```

#### Model Information
```http
GET /model/info
```

**Response**:
```json
{
  "model_type": "production",
  "protein_encoder": "esm",
  "drug_encoder": "gin",
  "parameters": 1234567,
  "memory_usage_mb": 2048.5
}
```

#### Upload Custom Model
```http
POST /model/upload
Content-Type: multipart/form-data

model_file: <binary model file>
config_file: <optional config file>
```

**Response**:
```json
{
  "message": "Model uploaded successfully",
  "model_id": "custom_model_123"
}
```

### Error Responses

#### Validation Error
```json
{
  "error": "validation_error",
  "message": "Invalid SMILES string",
  "details": {
    "field": "smiles",
    "value": "invalid_smiles"
  }
}
```

#### Server Error
```json
{
  "error": "server_error",
  "message": "Model prediction failed",
  "details": {
    "error_type": "RuntimeError",
    "traceback": "..."
  }
}
```

### API Client Examples

#### Python Client
```python
import requests

# Single prediction
response = requests.post('http://localhost:8000/predict', json={
    'smiles': 'CCO',
    'protein_sequence': 'MKTV...'
})
result = response.json()
print(f"Prediction: {result['prediction']}")

# Batch prediction
response = requests.post('http://localhost:8000/predict/batch', json={
    'data': [
        {'smiles': 'CCO', 'protein_sequence': 'MKTV...'},
        {'smiles': 'CC(=O)O', 'protein_sequence': 'MKTV...'}
    ]
})
results = response.json()
print(f"Predictions: {results['predictions']}")
```

#### JavaScript Client
```javascript
// Single prediction
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    smiles: 'CCO',
    protein_sequence: 'MKTV...'
  })
});

const result = await response.json();
console.log('Prediction:', result.prediction);
```

#### cURL Examples
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CCO",
    "protein_sequence": "MKTV..."
  }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"smiles": "CCO", "protein_sequence": "MKTV..."},
      {"smiles": "CC(=O)O", "protein_sequence": "MKTV..."}
    ]
  }'
```

## Command Line Interface

### Training
```bash
# Train with default configuration
unified-dta train --data data/kiba_train.csv

# Train with custom configuration
unified-dta train --config configs/production.yaml --data data/kiba_train.csv

# Resume training from checkpoint
unified-dta train --resume checkpoints/model_epoch_10.pt
```

### Prediction
```bash
# Single prediction
unified-dta predict --model my_model.pt --smiles "CCO" --protein "MKTV..."

# Batch prediction from file
unified-dta predict --model my_model.pt --input data/test.csv --output predictions.csv
```

### Evaluation
```bash
# Evaluate model
unified-dta evaluate --model my_model.pt --test-data data/test.csv

# Cross-validation
unified-dta evaluate --model my_model.pt --data data/full_dataset.csv --cv 5
```

### Model Conversion
```bash
# Convert to ONNX
unified-dta convert --model my_model.pt --format onnx --output model.onnx

# Convert to TorchScript
unified-dta convert --model my_model.pt --format torchscript --output model.pt
```

## Configuration API

### Loading Configurations
```python
from unified_dta.core import load_config, validate_config

# Load from YAML file
config = load_config('configs/production.yaml')

# Load from dictionary
config = load_config({
    'model': {'protein_encoder_type': 'esm'},
    'training': {'batch_size': 4}
})

# Validate configuration
validated_config = validate_config(config)
```

### Configuration Schema
```python
from unified_dta.core import ConfigSchema

# Get configuration schema
schema = ConfigSchema.get_schema()

# Validate against schema
is_valid = ConfigSchema.validate(config)
```

## Extension API

### Custom Encoders
```python
from unified_dta.core import BaseEncoder

class CustomProteinEncoder(BaseEncoder):
    def __init__(self, custom_params):
        super().__init__()
        # Custom implementation
    
    def forward(self, sequences):
        # Custom encoding logic
        return encoded_features
    
    @property
    def output_dim(self):
        return self._output_dim

# Register custom encoder
from unified_dta.core import register_encoder
register_encoder('custom_protein', CustomProteinEncoder)
```

### Custom Fusion Mechanisms
```python
from unified_dta.encoders import BaseFusion

class CustomFusion(BaseFusion):
    def __init__(self, drug_dim, protein_dim):
        super().__init__()
        # Custom fusion implementation
    
    def forward(self, drug_features, protein_features):
        # Custom fusion logic
        return fused_features

# Register custom fusion
from unified_dta.core import register_fusion
register_fusion('custom_fusion', CustomFusion)
```

## Type Hints and Annotations

```python
from typing import List, Tuple, Optional, Union
import torch
from torch import Tensor

def predict_batch(
    model: UnifiedDTAModel,
    smiles_list: List[str],
    protein_list: List[str],
    batch_size: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """
    Batch prediction with type hints.
    
    Args:
        model: Trained DTA model
        smiles_list: List of SMILES strings
        protein_list: List of protein sequences
        batch_size: Optional batch size override
    
    Returns:
        Tuple of (predictions, confidences)
    """
    pass
```
# Quick Start Guide

## Basic Usage

### 1. Simple Prediction

```python
from unified_dta import UnifiedDTAModel

# Load pre-trained model
model = UnifiedDTAModel.from_pretrained('lightweight')

# Make prediction
smiles = "CCO"  # Ethanol
protein_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
affinity = model.predict(smiles, protein_seq)
print(f"Predicted affinity: {affinity:.4f}")
```

### 2. Batch Predictions

```python
import pandas as pd
from unified_dta import UnifiedDTAModel

# Load model
model = UnifiedDTAModel.from_pretrained('production')

# Prepare data
data = pd.DataFrame({
    'smiles': ['CCO', 'CC(=O)O', 'CC(C)O'],
    'protein_sequence': [
        'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
        'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
        'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'
    ]
})

# Batch prediction
predictions = model.predict_batch(data['smiles'], data['protein_sequence'])
data['predicted_affinity'] = predictions
print(data)
```

### 3. Training Custom Model

```python
from unified_dta import UnifiedDTAModel, TrainingConfig
from unified_dta.data import DTADataset

# Load dataset
dataset = DTADataset.from_csv('data/kiba_train.csv')

# Configure training
config = TrainingConfig(
    model_type='production',
    batch_size=4,
    learning_rate=1e-3,
    num_epochs=50,
    use_2phase_training=True
)

# Initialize and train model
model = UnifiedDTAModel(config)
model.train(dataset)

# Save trained model
model.save('my_trained_model')
```

## Configuration Examples

### Lightweight Configuration (Development)
```python
from unified_dta import UnifiedDTAModel

model = UnifiedDTAModel.from_config({
    'protein_encoder_type': 'cnn',
    'drug_encoder_type': 'gin',
    'use_fusion': False,
    'protein_config': {'output_dim': 64},
    'drug_config': {'output_dim': 64, 'num_layers': 3}
})
```

### Production Configuration
```python
from unified_dta import UnifiedDTAModel

model = UnifiedDTAModel.from_config({
    'protein_encoder_type': 'esm',
    'drug_encoder_type': 'gin',
    'use_fusion': True,
    'protein_config': {'output_dim': 128, 'max_length': 200},
    'drug_config': {'output_dim': 128, 'num_layers': 5},
    'fusion_config': {'hidden_dim': 256, 'num_heads': 8}
})
```

## API Server

### Start API Server
```bash
python -m unified_dta.api.main
```

### Make API Requests
```python
import requests

# Single prediction
response = requests.post('http://localhost:8000/predict', json={
    'smiles': 'CCO',
    'protein_sequence': 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'
})
print(response.json())

# Batch prediction
response = requests.post('http://localhost:8000/predict/batch', json={
    'data': [
        {'smiles': 'CCO', 'protein_sequence': 'MKTV...'},
        {'smiles': 'CC(=O)O', 'protein_sequence': 'MKTV...'}
    ]
})
print(response.json())
```

## Command Line Interface

### Train Model
```bash
unified-dta train --config configs/production.yaml --data data/kiba_train.csv
```

### Make Predictions
```bash
unified-dta predict --model my_model --smiles "CCO" --protein "MKTV..."
```

### Evaluate Model
```bash
unified-dta evaluate --model my_model --test-data data/kiba_test.csv
```

## Memory Management

### For Limited Memory Systems
```python
from unified_dta import UnifiedDTAModel
import os

# Set memory optimization
os.environ['UNIFIED_DTA_MEMORY_EFFICIENT'] = 'true'

# Use lightweight model
model = UnifiedDTAModel.from_pretrained('lightweight')

# Reduce batch size
model.set_batch_size(2)
```

### Monitor Memory Usage
```python
from unified_dta.utils import MemoryMonitor

with MemoryMonitor() as monitor:
    predictions = model.predict_batch(smiles_list, protein_list)
    
print(f"Peak memory usage: {monitor.peak_memory_mb:.1f} MB")
```

## Data Formats

### CSV Format
```csv
compound_iso_smiles,target_sequence,affinity
CCO,MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,5.2
CC(=O)O,MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,4.8
```

### JSON Format
```json
{
  "data": [
    {
      "smiles": "CCO",
      "protein_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
      "affinity": 5.2
    }
  ]
}
```

## Next Steps

- [API Reference](api/README.md) - Complete API documentation
- [Configuration Guide](configuration.md) - Advanced configuration options
- [Training Guide](training.md) - Detailed training instructions
- [Examples](../examples/README.md) - More code examples
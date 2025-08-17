# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### 1. PyTorch Geometric Installation Fails

**Problem**: Error installing torch-geometric or related packages
```
ERROR: Failed building wheel for torch-scatter
```

**Solutions**:
```bash
# Option 1: Install from PyG wheel repository
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install torch-geometric

# Option 2: Use conda
conda install pyg -c pyg

# Option 3: Install specific CUDA version
pip install torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
```

#### 2. RDKit Installation Issues

**Problem**: RDKit fails to install or import
```
ImportError: No module named 'rdkit'
```

**Solutions**:
```bash
# Option 1: Use conda (recommended)
conda install -c conda-forge rdkit

# Option 2: Use pip
pip install rdkit-pypi

# Option 3: For M1/M2 Macs
conda install -c conda-forge rdkit-pypi
```

#### 3. ESM-2 Model Download Issues

**Problem**: Slow or failed ESM-2 model download
```
ConnectionError: Failed to download model
```

**Solutions**:
```python
# Pre-download models
from transformers import EsmModel, EsmTokenizer

# Download and cache model
model = EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D')
tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

# Or set cache directory
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'
```

### Memory Issues

#### 1. CUDA Out of Memory

**Problem**: GPU runs out of memory during training
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
```python
# Reduce batch size
config['training']['batch_size'] = 2

# Enable gradient checkpointing
config['system']['gradient_checkpointing'] = True

# Use mixed precision
config['system']['mixed_precision'] = True

# Reduce sequence length
config['protein_config']['max_length'] = 100

# Use CPU fallback
config['system']['device'] = 'cpu'
```

#### 2. System RAM Exhaustion

**Problem**: System runs out of RAM
```
MemoryError: Unable to allocate array
```

**Solutions**:
```python
# Use lightweight configuration
model = UnifiedDTAModel.from_pretrained('lightweight')

# Enable memory efficient mode
config['system']['memory_efficient'] = True

# Reduce data loader workers
config['data']['num_workers'] = 0

# Use smaller datasets for testing
dataset = dataset[:1000]  # Use subset
```

#### 3. Memory Leaks During Training

**Problem**: Memory usage increases over time
```python
# Monitor memory usage
from unified_dta.utils import MemoryMonitor

monitor = MemoryMonitor()
for epoch in range(num_epochs):
    train_epoch()
    monitor.log_memory_usage()
    
    # Clear cache periodically
    if epoch % 10 == 0:
        torch.cuda.empty_cache()
```

### Model Training Issues

#### 1. Loss Not Decreasing

**Problem**: Training loss remains high or doesn't decrease

**Diagnostic Steps**:
```python
# Check data preprocessing
print(f"Data range: {data['affinity'].min():.2f} to {data['affinity'].max():.2f}")
print(f"Data mean: {data['affinity'].mean():.2f}")

# Check model gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm():.6f}")
```

**Solutions**:
```python
# Adjust learning rate
config['training']['learning_rate'] = 1e-4

# Check data normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data['affinity'] = scaler.fit_transform(data[['affinity']])

# Use gradient clipping
config['training']['max_grad_norm'] = 1.0

# Try different optimizer
config['training']['optimizer'] = 'adamw'
```

#### 2. Model Overfitting

**Problem**: Validation loss increases while training loss decreases

**Solutions**:
```python
# Increase dropout
config['predictor_config']['dropout'] = 0.5

# Add weight decay
config['training']['weight_decay'] = 1e-3

# Use early stopping
config['training']['early_stopping_patience'] = 5

# Reduce model complexity
config['predictor_config']['hidden_dims'] = [256]  # Smaller network
```

#### 3. NaN Loss During Training

**Problem**: Loss becomes NaN
```
Loss: nan
```

**Solutions**:
```python
# Check for invalid inputs
def check_data_validity(batch):
    for key, value in batch.items():
        if torch.isnan(value).any():
            print(f"NaN found in {key}")
        if torch.isinf(value).any():
            print(f"Inf found in {key}")

# Reduce learning rate
config['training']['learning_rate'] = 1e-5

# Use gradient clipping
config['training']['max_grad_norm'] = 0.5

# Check model initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
```

### Data Issues

#### 1. Invalid SMILES Strings

**Problem**: RDKit fails to parse SMILES
```
ValueError: Could not parse SMILES string
```

**Solutions**:
```python
from rdkit import Chem

def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return True

# Filter invalid SMILES
valid_data = data[data['smiles'].apply(validate_smiles)]

# Or use error handling
def safe_smiles_to_graph(smiles):
    try:
        return smiles_to_graph(smiles)
    except:
        return None  # Skip invalid molecules
```

#### 2. Protein Sequence Issues

**Problem**: Invalid amino acid characters
```
KeyError: 'X' not in vocabulary
```

**Solutions**:
```python
# Clean protein sequences
def clean_protein_sequence(seq):
    # Remove invalid characters
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
    return ''.join(c for c in seq if c in valid_chars)

# Handle unknown amino acids
def replace_unknown_aa(seq):
    return seq.replace('X', 'A').replace('U', 'C')
```

#### 3. Data Loading Errors

**Problem**: DataLoader fails with multiprocessing
```
RuntimeError: DataLoader worker (pid 1234) is killed by signal
```

**Solutions**:
```python
# Disable multiprocessing
config['data']['num_workers'] = 0

# Or reduce number of workers
config['data']['num_workers'] = 2

# Use persistent workers
config['data']['persistent_workers'] = True
```

### Performance Issues

#### 1. Slow Training Speed

**Problem**: Training is very slow

**Solutions**:
```python
# Enable mixed precision
config['system']['mixed_precision'] = True

# Use larger batch size (if memory allows)
config['training']['batch_size'] = 8

# Reduce sequence length
config['protein_config']['max_length'] = 150

# Use gradient accumulation
config['training']['gradient_accumulation_steps'] = 4

# Profile the code
import torch.profiler
with torch.profiler.profile() as prof:
    train_step()
print(prof.key_averages().table())
```

#### 2. High Memory Usage

**Problem**: Model uses too much memory

**Solutions**:
```python
# Use gradient checkpointing
config['system']['gradient_checkpointing'] = True

# Reduce model size
config['protein_config']['output_dim'] = 64
config['drug_config']['hidden_dim'] = 64

# Use CPU for some operations
def move_to_cpu_if_needed(tensor):
    if torch.cuda.memory_allocated() > 0.8 * torch.cuda.max_memory_allocated():
        return tensor.cpu()
    return tensor
```

### API Issues

#### 1. API Server Won't Start

**Problem**: FastAPI server fails to start
```
ImportError: No module named 'fastapi'
```

**Solutions**:
```bash
# Install API dependencies
pip install fastapi uvicorn

# Or install with API extras
pip install unified-dta[api]

# Check port availability
netstat -an | grep 8000
```

#### 2. Prediction Endpoint Errors

**Problem**: API returns 500 errors

**Solutions**:
```python
# Check model loading
try:
    model = UnifiedDTAModel.from_pretrained('production')
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")

# Validate input data
def validate_prediction_input(smiles, protein_seq):
    if not isinstance(smiles, str):
        raise ValueError("SMILES must be string")
    if not isinstance(protein_seq, str):
        raise ValueError("Protein sequence must be string")
    if len(protein_seq) == 0:
        raise ValueError("Protein sequence cannot be empty")
```

### Environment Issues

#### 1. CUDA Version Mismatch

**Problem**: PyTorch CUDA version doesn't match system CUDA
```
RuntimeError: CUDA runtime version mismatch
```

**Solutions**:
```bash
# Check CUDA versions
nvidia-smi
python -c "import torch; print(torch.version.cuda)"

# Install matching PyTorch version
pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
```

#### 2. Package Version Conflicts

**Problem**: Dependency version conflicts
```
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**Solutions**:
```bash
# Create fresh environment
conda create -n unified-dta-clean python=3.9
conda activate unified-dta-clean

# Install with specific versions
pip install torch==1.12.0 torch-geometric==2.1.0

# Use pip-tools for dependency management
pip install pip-tools
pip-compile requirements.in
```

## Debugging Tools

### 1. Memory Profiling
```python
from unified_dta.utils import MemoryProfiler

with MemoryProfiler() as profiler:
    model = UnifiedDTAModel.from_pretrained('production')
    predictions = model.predict_batch(smiles_list, protein_list)

profiler.print_stats()
```

### 2. Model Debugging
```python
# Check model architecture
print(model)

# Check parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Check gradients
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"No gradient for {name}")
        elif torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
```

### 3. Data Debugging
```python
# Inspect batch data
def debug_batch(batch):
    print(f"Batch keys: {batch.keys()}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            print(f"  min={value.min():.4f}, max={value.max():.4f}")
        else:
            print(f"{key}: type={type(value)}")
```

## Getting Help

### 1. Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
import os
os.environ['UNIFIED_DTA_LOG_LEVEL'] = 'DEBUG'
```

### 2. Create Minimal Reproducible Example
```python
# Minimal example for bug reports
from unified_dta import UnifiedDTAModel

model = UnifiedDTAModel.from_pretrained('lightweight')
smiles = "CCO"
protein = "MKTVRQERLK"

try:
    prediction = model.predict(smiles, protein)
    print(f"Prediction: {prediction}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

### 3. System Information
```python
def print_system_info():
    import torch
    import sys
    import platform
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print_system_info()
```

### 4. Report Issues
When reporting issues, please include:
- System information (from above)
- Complete error traceback
- Minimal reproducible example
- Configuration used
- Data sample (if possible)
# Installation Guide

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB disk space

### Recommended Requirements
- Python 3.9+
- 8GB RAM
- GPU with 4GB+ VRAM
- 10GB disk space

## Installation Methods

### Method 1: pip install (Recommended)

```bash
pip install unified-dta
```

### Method 2: From Source

```bash
git clone https://github.com/your-org/unified-dta-system.git
cd unified-dta-system
pip install -e .
```

### Method 3: Development Installation

```bash
git clone https://github.com/your-org/unified-dta-system.git
cd unified-dta-system
pip install -r requirements-dev.txt
pip install -e .
```

## Dependencies

### Core Dependencies
```
torch>=1.12.0
torch-geometric>=2.1.0
transformers>=4.21.0
rdkit-pypi>=2022.3.5
pandas>=1.4.0
scipy>=1.8.0
pyyaml>=6.0
```

### Optional Dependencies
```
fastapi>=0.68.0          # For API server
uvicorn>=0.15.0          # For API server
jupyter>=1.0.0           # For notebooks
matplotlib>=3.5.0        # For visualization
seaborn>=0.11.0          # For visualization
```

## Verification

### Quick Test
```bash
python -c "import unified_dta; print('Installation successful!')"
```

### Run Simple Demo
```bash
python -m unified_dta.examples.simple_demo
```

### Check GPU Support
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Environment Setup

### Using Conda
```bash
conda create -n unified-dta python=3.9
conda activate unified-dta
pip install unified-dta
```

### Using Virtual Environment
```bash
python -m venv unified-dta-env
source unified-dta-env/bin/activate  # On Windows: unified-dta-env\Scripts\activate
pip install unified-dta
```

## Docker Installation

### Pull Pre-built Image
```bash
docker pull unified-dta:latest
docker run -it unified-dta:latest
```

### Build from Source
```bash
git clone https://github.com/your-org/unified-dta-system.git
cd unified-dta-system
docker build -t unified-dta .
docker run -it unified-dta
```

## Troubleshooting Installation

### Common Issues

#### PyTorch Geometric Installation
If you encounter issues with PyTorch Geometric:
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install torch-geometric
```

#### RDKit Installation
If RDKit installation fails:
```bash
conda install -c conda-forge rdkit
```

#### CUDA Issues
For CUDA support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues
For systems with limited memory, use lightweight configuration:
```bash
export UNIFIED_DTA_CONFIG=lightweight
```

### Platform-Specific Notes

#### Windows
- Use Anaconda for easier dependency management
- Some packages may require Visual Studio Build Tools

#### macOS
- Use Homebrew for system dependencies
- M1/M2 Macs: Use conda-forge for optimized packages

#### Linux
- Most straightforward installation
- Ensure CUDA drivers are installed for GPU support

## Next Steps

After installation, see:
- [Quick Start Guide](quickstart.md) for basic usage
- [Configuration Guide](configuration.md) for customization
- [Examples](../examples/README.md) for code examples
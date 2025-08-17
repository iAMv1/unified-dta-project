# Unified DTA System - Project Structure

## Current Project Organization

### 📁 Root Directory
```
unified-dta-system/
├── 📁 core/                    # Core system modules
├── 📁 configs/                 # Configuration files
├── 📁 data/                    # Dataset files
├── 📁 docs/                    # Documentation
├── 📁 examples/                # Usage examples
├── 📁 scripts/                 # Utility scripts
├── 📁 tests/                   # Test suite
├── 📁 unified_dta/             # Main package
├── 📁 apps/                    # Applications (NEW)
├── 📁 external/                # External repositories
└── 📄 Various root files
```

## File Classification

### 🔧 Core System Files
**Location: `core/`**
- `__init__.py` - Package initialization
- `models.py` - Main model implementations
- `base_components.py` - Base classes and utilities
- `data_processing.py` - Data handling and preprocessing
- `training.py` - Training infrastructure
- `evaluation.py` - Evaluation metrics and tools
- `config.py` - Configuration management
- `checkpoint_utils.py` - Model checkpointing
- `memory_optimization.py` - Memory management

### 🧬 Encoder Modules
**Location: `core/`**
- `protein_encoders.py` - ESM-2, CNN protein encoders
- `drug_encoders.py` - GIN, enhanced drug encoders
- `fusion.py` - Multi-modal fusion layers
- `prediction_heads.py` - Output prediction layers

### 💊 Drug Generation System
**Location: `core/`**
- `drug_generation.py` - Transformer-based generation
- `generation_scoring.py` - Quality assessment
- `generation_evaluation.py` - Evaluation pipeline

### 🔬 Data Processing
**Location: `core/`**
- `datasets.py` - Dataset classes
- `graph_preprocessing.py` - Molecular graph processing
- `utils.py` - General utilities

### ⚙️ Configuration Files
**Location: `configs/`**
- `base_config.yaml` - Base configuration
- `lightweight_config.yaml` - Lightweight setup
- `production_config.yaml` - Production settings
- `high_performance_config.yaml` - High-performance setup
- `custom_config.yaml` - Custom configurations

### 📊 Datasets
**Location: `data/`**
- `kiba_train.csv`, `kiba_test.csv` - KIBA dataset
- `davis_train.csv`, `davis_test.csv` - Davis dataset  
- `bindingdb_train.csv`, `bindingdb_test.csv` - BindingDB dataset
- `samples/` - Sample data for testing

### 🧪 Test Suite
**Location: `tests/` and root**
- `test_*.py` - Unit and integration tests
- `test_generation_*.py` - Drug generation tests
- `test_*_integration.py` - Integration tests

### 📖 Documentation
**Location: `docs/`**
- `README.md` - Main documentation
- `installation.md` - Installation guide
- `quickstart.md` - Quick start guide
- `architecture.md` - System architecture
- `configuration.md` - Configuration guide
- `troubleshooting.md` - Troubleshooting guide

### 🎯 Demo Applications
**Location: Root directory**
- `demo_*.py` - Various demonstration scripts
- `simple_demo.py` - Basic functionality demo
- `demo_drug_generation.py` - Generation demo
- `demo_generation_simple.py` - Simplified generation demo

### 🏋️ Training Scripts
**Location: Root directory**
- `train_combined.py` - Main training script
- `train_2phase.py` - Two-phase training
- `train_drug_generation.py` - Generation model training

### 🔧 Utility Scripts
**Location: `scripts/` and root**
- `prepare_data.py` - Data preparation
- `run_api.py` - API server
- `run_tests.py` - Test runner
- `checkpoint_cli.py` - Checkpoint management CLI
- `config_cli.py` - Configuration CLI

### 📦 Package Files
**Location: Root directory**
- `setup.py` - Package setup
- `requirements.txt` - Dependencies
- `requirements-dev.txt` - Development dependencies
- `MANIFEST.in` - Package manifest
- `Makefile` - Build automation

### 🌐 External Repositories
**Location: Root directory**
- `DeepDTAGen/` - Drug generation repository
- `DoubleSG-DTA/` - GIN network implementation
- `deepdta_platform/` - Additional DTA tools

### 📋 Documentation Files
**Location: Root directory**
- `README.md` - Main project README
- `CHANGELOG.md` - Version history
- `*_SUMMARY.md` - Implementation summaries

## Recommended Reorganization

### 1. Create Apps Directory
Move interactive applications to dedicated folder:
```
apps/
├── streamlit_app.py          # Web dashboard
├── cli_app.py               # Command-line interface
└── jupyter_notebooks/       # Interactive notebooks
```

### 2. Consolidate External Repos
```
external/
├── DeepDTAGen/             # Drug generation
├── DoubleSG-DTA/           # GIN networks
└── deepdta_platform/       # Additional tools
```

### 3. Organize Demo Scripts
```
examples/
├── demos/
│   ├── basic_usage.py
│   ├── drug_generation.py
│   └── evaluation_demo.py
├── notebooks/              # Jupyter notebooks
└── tutorials/              # Step-by-step guides
```

### 4. Consolidate Tests
```
tests/
├── unit/                   # Unit tests
├── integration/            # Integration tests
├── performance/            # Performance tests
└── fixtures/               # Test data and fixtures
```

## File Categories Summary

| Category | Count | Location | Purpose |
|----------|-------|----------|---------|
| Core Modules | 15 | `core/` | System implementation |
| Configuration | 5 | `configs/` | System settings |
| Tests | 25+ | `tests/`, root | Quality assurance |
| Demos | 10+ | root | Usage examples |
| Documentation | 10+ | `docs/`, root | User guides |
| Training Scripts | 3 | root | Model training |
| Utilities | 8+ | `scripts/`, root | Helper tools |
| External Repos | 3 | root | Third-party code |
| Package Files | 6 | root | Distribution |
| Data Files | 6+ | `data/` | Datasets |

## Next Steps

1. **Create Streamlit Application** - Interactive web dashboard
2. **Reorganize File Structure** - Move files to appropriate directories
3. **Update Import Paths** - Fix any broken imports after reorganization
4. **Create Application Launcher** - Single entry point for all apps
5. **Update Documentation** - Reflect new structure
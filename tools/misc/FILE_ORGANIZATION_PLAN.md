# File Organization and Cleanup Plan

## 🎯 Proposed Directory Structure

```
unified-dta-system/
├── 📁 src/                          # Main source code
│   ├── 📁 unified_dta/              # Core package
│   │   ├── 📁 core/                 # Core modules
│   │   ├── 📁 encoders/             # Encoder implementations
│   │   ├── 📁 data/                 # Data processing
│   │   ├── 📁 training/             # Training utilities
│   │   ├── 📁 evaluation/           # Evaluation systems
│   │   ├── 📁 generation/           # Drug generation
│   │   ├── 📁 api/                  # API endpoints
│   │   └── 📁 utils/                # Utilities
│   └── 📁 apps/                     # Applications
│       ├── streamlit_app.py         # Web interface
│       └── cli.py                   # Command line interface
├── 📁 configs/                      # Configuration files
├── 📁 data/                         # Datasets
│   ├── 📁 raw/                      # Raw datasets
│   ├── 📁 processed/                # Processed datasets
│   └── 📁 samples/                  # Sample data
├── 📁 tests/                        # Test suite
│   ├── 📁 unit/                     # Unit tests
│   ├── 📁 integration/              # Integration tests
│   └── 📁 performance/              # Performance tests
├── 📁 docs/                         # Documentation
│   ├── 📁 api/                      # API documentation
│   ├── 📁 tutorials/                # Tutorials
│   └── 📁 guides/                   # User guides
├── 📁 examples/                     # Usage examples
│   ├── 📁 basic/                    # Basic examples
│   ├── 📁 advanced/                 # Advanced examples
│   └── 📁 notebooks/                # Jupyter notebooks
├── 📁 scripts/                      # Utility scripts
│   ├── 📁 training/                 # Training scripts
│   ├── 📁 evaluation/               # Evaluation scripts
│   └── 📁 deployment/               # Deployment scripts
├── 📁 external/                     # External repositories
│   ├── 📁 DeepDTAGen/              # Drug generation repo
│   ├── 📁 DoubleSG-DTA/            # Graph DTA repo
│   └── 📁 deepdta_platform/        # Platform repo
├── 📁 models/                       # Saved models
├── 📁 results/                      # Experiment results
└── 📁 temp/                         # Temporary files
```

## 🗂️ File Classification and Movement Plan

### Core Source Code (Keep & Organize)
```
src/unified_dta/core/
├── __init__.py                      ← core/__init__.py
├── base_components.py               ← core/base_components.py
├── models.py                        ← core/models.py
├── config.py                        ← core/config.py
├── model_factory.py                 ← core/model_factory.py
└── utils.py                         ← core/utils.py

src/unified_dta/encoders/
├── __init__.py                      ← NEW
├── protein_encoders.py              ← core/protein_encoders.py
├── drug_encoders.py                 ← core/drug_encoders.py
└── fusion.py                        ← core/fusion.py

src/unified_dta/data/
├── __init__.py                      ← NEW
├── data_processing.py               ← core/data_processing.py
├── datasets.py                      ← core/datasets.py
└── graph_preprocessing.py           ← core/graph_preprocessing.py

src/unified_dta/training/
├── __init__.py                      ← NEW
├── training.py                      ← core/training.py
├── checkpoint_utils.py              ← core/checkpoint_utils.py
└── memory_optimization.py           ← core/memory_optimization.py

src/unified_dta/evaluation/
├── __init__.py                      ← NEW
├── evaluation.py                    ← core/evaluation.py
└── prediction_heads.py              ← core/prediction_heads.py

src/unified_dta/generation/
├── __init__.py                      ← NEW
├── drug_generation.py               ← core/drug_generation.py
├── generation_scoring.py            ← core/generation_scoring.py
└── generation_evaluation.py         ← core/generation_evaluation.py

src/unified_dta/api/
├── __init__.py                      ← unified_dta/api/__init__.py
├── app.py                           ← unified_dta/api/app.py
├── endpoints.py                     ← unified_dta/api/endpoints.py
├── models.py                        ← unified_dta/api/models.py
└── prediction.py                    ← unified_dta/api/prediction.py
```

### Applications
```
src/apps/
├── streamlit_app.py                 ← apps/streamlit_app.py
└── cli.py                           ← unified_dta/cli.py
```

### Training Scripts
```
scripts/training/
├── train_combined.py                ← train_combined.py
├── train_2phase.py                  ← train_2phase.py
├── train_drug_generation.py         ← train_drug_generation.py
└── prepare_data.py                  ← prepare_data.py
```

### Demo Scripts
```
scripts/demos/
├── simple_demo.py                   ← simple_demo.py
├── demo.py                          ← demo.py
├── demo_evaluation_system.py        ← demo_evaluation_system.py
├── demo_enhanced_cnn.py             ← demo_enhanced_cnn.py
├── demo_enhanced_gin.py             ← demo_enhanced_gin.py
├── demo_prediction_heads.py         ← demo_prediction_heads.py
├── demo_graph_preprocessing.py      ← demo_graph_preprocessing.py
├── demo_drug_generation.py          ← demo_drug_generation.py
├── demo_generation_simple.py        ← demo_generation_simple.py
└── demo_checkpoint_system.py        ← demo_checkpoint_system.py
```

### Test Suite Organization
```
tests/
├── unit/
│   ├── test_encoders.py             ← test_esm_encoder.py + test_enhanced_*
│   ├── test_data_processing.py      ← test_data_processing.py
│   ├── test_models.py               ← test_model.py
│   ├── test_training.py             ← test_training.py
│   ├── test_evaluation.py           ← test_evaluation_system.py
│   ├── test_generation.py           ← test_drug_generation.py
│   └── test_config.py               ← test_config_system.py
├── integration/
│   ├── test_pipeline.py             ← test_*_integration.py
│   ├── test_api.py                  ← test_api.py
│   └── test_generation_pipeline.py  ← test_generation_integration.py
└── performance/
    ├── test_memory.py               ← test_memory_optimization.py
    └── test_benchmarks.py           ← NEW
```

### Configuration Files
```
configs/
├── base_config.yaml                 ← configs/base_config.yaml
├── lightweight_config.yaml          ← configs/lightweight_config.yaml
├── production_config.yaml           ← configs/production_config.yaml
├── high_performance_config.yaml     ← configs/high_performance_config.yaml
└── custom_config.yaml               ← configs/custom_config.yaml
```

### Data Organization
```
data/
├── raw/
│   ├── kiba/                        ← data/kiba_*.csv
│   ├── davis/                       ← data/davis_*.csv
│   └── bindingdb/                   ← data/bindingdb_*.csv
├── processed/                       ← NEW (for processed datasets)
└── samples/
    ├── sample_kiba_train.csv        ← data/samples/sample_kiba_train.csv
    └── sample_davis_train.csv       ← data/samples/sample_davis_train.csv
```

## 🗑️ Files to Remove (Waste/Duplicates)

### Duplicate/Legacy Files
```
❌ combined_model.py                 # Legacy - replaced by core/models.py
❌ models.py                         # Legacy - replaced by core/models.py
❌ Data_folded.rar                   # Archive - extract or remove
❌ data.rar                          # Archive - extract or remove
```

### Cache Files
```
❌ __pycache__/                      # Python cache - regenerated
❌ core/__pycache__/                 # Python cache - regenerated
❌ tests/__pycache__/                # Python cache - regenerated
❌ unified_dta/__pycache__/          # Python cache - regenerated
```

### Redundant Test Files
```
❌ test_generation_simple.py         # Merge into test_generation.py
❌ test_generation_standalone.py     # Merge into test_generation.py
❌ test_checkpoint_basic.py          # Merge into test_checkpoint_system.py
❌ simple_config_test.py             # Merge into test_config_system.py
❌ minimal_config_test.py            # Merge into test_config_system.py
```

### External Repository Cleanup
```
❌ DeepDTAGen/.git/                  # Git history - not needed
❌ DoubleSG-DTA/.git/                # Git history - not needed
❌ deepdta_platform/.git/            # Git history - not needed
```

### Verification Scripts (Keep but organize)
```
✅ verify_checkpoint_integration.py  → scripts/verification/
```

## 📋 Organization Steps

### Step 1: Create New Directory Structure
1. Create `src/` directory
2. Create organized subdirectories
3. Create `scripts/` with subdirectories
4. Reorganize `tests/` structure

### Step 2: Move Core Files
1. Move `core/` contents to appropriate `src/unified_dta/` subdirectories
2. Update import statements
3. Move applications to `src/apps/`
4. Move training scripts to `scripts/training/`

### Step 3: Reorganize Tests
1. Consolidate similar test files
2. Move to appropriate test subdirectories
3. Update test imports

### Step 4: Clean Up Waste
1. Remove cache directories
2. Remove duplicate/legacy files
3. Clean up external repositories
4. Remove temporary files

### Step 5: Update Documentation
1. Update import examples in documentation
2. Update file paths in README
3. Update configuration examples

## 🔧 Import Statement Updates

After reorganization, imports will change from:
```python
# Old
from core.models import UnifiedDTAModel
from core.training import train_model

# New
from unified_dta.core.models import UnifiedDTAModel
from unified_dta.training.training import train_model
```

## 📊 Expected Benefits

### Organization Benefits
- ✅ Clear separation of concerns
- ✅ Easier navigation and maintenance
- ✅ Better package structure
- ✅ Reduced clutter

### Performance Benefits
- ✅ Faster imports (better organization)
- ✅ Reduced disk usage (removed waste)
- ✅ Cleaner development environment

### Maintenance Benefits
- ✅ Easier to find files
- ✅ Better version control
- ✅ Cleaner CI/CD pipelines
- ✅ Professional project structure

## 🎯 Final Structure Size Estimate

### Before Cleanup: ~500+ files
### After Cleanup: ~200 organized files
### Space Saved: ~60% reduction in clutter
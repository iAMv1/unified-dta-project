# Unified DTA System Architecture and Phase Connections

## System Overview

The Unified DTA (Drug-Target Affinity) System is organized into distinct phases that work together to provide a complete drug discovery pipeline:

1. **Data Preparation Phase**
2. **Model Training Phase** (2-phase progressive training)
3. **Model Deployment Phase** 
4. **Inference Phase**
5. **Monitoring Phase**

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT APPLICATIONS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                              API LAYER                                      │
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   Models    │    │   Prediction    │    │         Cache               │  │
│  │             │    │                 │    │                             │  │
│  │  - Model    │◄──►│  - Prediction   │◄──►│  - Model Loading            │  │
│  │    Info     │    │    Service      │    │  - Checkpoint Management    │  │
│  │             │    │                 │    │  - Memory Management        │  │
│  └─────────────┘    └─────────────────┘    └─────────────────────────────┘  │
│         ▲                       ▲                       ▲                  │
│         │                       │                       │                  │
│         ▼                       ▼                       ▼                  │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   Health    │    │    Batch        │    │         Metrics             │  │
│  │             │    │   Predict       │    │                             │  │
│  │  - System   │    │                 │    │  - Training Progress        │  │
│  │    Status   │    │  - Batch        │    │  - Model Performance        │  │
│  │             │    │    Processing   │    │                             │  │
│  └─────────────┘    └─────────────────┘    └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                        CORE INFERENCE PIPELINE                              │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │   Data Input    │    │   Data          │    │      Model              │ │
│  │                 │───►│   Processing    │───►│                         │ │
│  │  - SMILES       │    │                 │    │  - Drug Encoder         │ │
│  │  - Protein      │    │  - Validation   │    │    (GIN-based)          │ │
│  │    Sequence     │    │  - Conversion   │    │  - Protein Encoder      │ │
│  │                 │    │  - Graph        │    │    (ESM/CNN)            │ │
│  └─────────────────┘    │    Creation     │    │  - Fusion               │ │
│                         └─────────────────┘    │  - Prediction Head      │ │
│                                ▲               │                         │ │
│                                │               └─────────────────────────┘ │
│                                ▼                            ▲             │
│                         ┌─────────────────┐                 │             │
│                         │   Confidence    │                 │             │
│                         │   Scoring       │◄────────────────┘             │
│                         │                 │                               │
│                         └─────────────────┘                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                           TRAINING PIPELINE                               │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │   Training      │    │   Model         │    │      Checkpoint         │ │
│  │   Data          │───►│   Factory       │───►│      Management         │ │
│  │                 │    │                 │    │                         │ │
│  │  - Datasets     │    │  - Model        │    │  - Save/Load            │ │
│  │  - Preprocessing│    │    Creation     │    │  - Versioning           │ │
│  │                 │    │                 │    │                         │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘ │
│                                ▲                       ▲                 │
│                                │                       │                 │
│                                ▼                       ▼                 │
│                         ┌─────────────────┐    ┌─────────────────────────┐ │
│                         │  Progressive    │    │      Evaluation         │ │
│                         │  Trainer        │    │                         │ │
│                         │                 │    │  - Metrics              │ │
│                         │  - Phase 1:     │    │  - Validation           │ │
│                         │    Frozen ESM   │    │                         │ │
│                         │  - Phase 2:     │    │                         │ │
│                         │    ESM Tuning   │    │                         │ │
│                         └─────────────────┘    └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase-by-Phase Connections

### 1. Data Preparation Phase → Training Phase

**Connection Mechanism**: Data Processing Pipeline
- **Source**: Raw datasets (KIBA, Davis, BindingDB)
- **Processing**: 
  - SMILES validation and cleaning
  - Protein sequence processing
  - Molecular graph conversion using RDKit
- **Destination**: Training data loaders
- **Files**: `src/unified_dta/data/data_processing.py`, `src/unified_dta/data/graph_preprocessing.py`

### 2. Training Phase → Model Deployment Phase

**Connection Mechanism**: Checkpoint System
- **Source**: ProgressiveTrainer saves model checkpoints
- **Processing**:
  - Model state dictionary serialization
  - Training metrics and configuration saving
  - Best model identification
- **Destination**: Checkpoint files in `checkpoints/` directory
- **Files**: `src/unified_dta/training/training.py`, `src/unified_dta/training/checkpoint_utils.py`

### 3. Model Deployment Phase → Inference Phase

**Connection Mechanism**: Model Cache Loading
- **Source**: Checkpoint files
- **Processing**:
  - ModelCache searches for checkpoints
  - Configuration consistency validation
  - Model state loading
- **Destination**: In-memory model instances
- **Files**: `src/unified_dta/api/cache.py`, `src/unified_dta/utils/config_consistency.py`

### 4. Inference Phase → Client Applications

**Connection Mechanism**: RESTful API Endpoints
- **Source**: PredictionService processing
- **Processing**:
  - Input validation and processing
  - Model inference with confidence scoring
  - Response formatting
- **Destination**: API responses to clients
- **Files**: `src/unified_dta/api/endpoints.py`, `src/unified_dta/api/prediction.py`

### 5. Training Phase → Monitoring Phase

**Connection Mechanism**: Metrics Tracking
- **Source**: Training metrics during model training
- **Processing**:
  - Metrics collection and serialization
  - Performance statistics calculation
- **Destination**: Metrics files and API endpoints
- **Files**: `src/unified_dta/utils/model_metrics.py`, `src/unified_dta/training/training.py`

## Detailed Connection Implementation

### A. Data Processing Connection (Fixed)

**Before**: API used mock data processing
**After**: 
```python
# API connects to real data processing pipeline
prediction_service._process_drug_data() 
  → DataProcessor.smiles_to_graph() 
  → MolecularGraphConverter.smiles_to_graph() 
  → RDKit molecular processing
```

### B. Model Loading Connection (Fixed)

**Before**: No checkpoint loading mechanism
**After**:
```python
# Cache system connects training outputs to inference inputs
ModelCache._load_model()
  → _find_model_checkpoint()
  → _load_from_checkpoint()
  → torch.load() + state_dict loading
```

### C. Confidence Scoring Connection (New)

**Before**: Confidence was None/TODO
**After**:
```python
# Model provides uncertainty estimation
UnifiedDTAModel.estimate_uncertainty()
  → Monte Carlo dropout sampling
  → Statistical confidence calculation
  → Confidence score in API response
```

### D. Metrics Connection (New)

**Before**: Training metrics not exposed
**After**:
```python
# Training metrics exposed through API
get_model_metrics_endpoint()
  → ModelMetricsManager.load_training_metrics()
  → Checkpoint/training log parsing
  → Performance statistics API response
```

## Cross-Phase Integration Points

### 1. Configuration Consistency
- **Training**: Model configurations saved with checkpoints
- **Inference**: Configurations validated for consistency
- **Integration**: `ConfigConsistencyChecker` ensures compatibility

### 2. Error Propagation
- **Training**: Errors logged with detailed context
- **Inference**: Errors propagated to API responses with details
- **Integration**: Enhanced error handling throughout pipeline

### 3. Memory Management
- **Training**: Memory optimization during training
- **Inference**: Cache memory management
- **Integration**: Shared memory management utilities

## Two-Phase Training Architecture

```
PHASE 1: FROZEN ESM TRAINING
┌─────────────────────────────────────────┐
│           TRAINING PHASE 1              │
├─────────────────────────────────────────┤
│  Components Trained:                    │
│  - Drug Encoder (GIN)                   │
│  - Fusion Layers                        │
│  - Prediction Head                      │
│                                         │
│  Components Frozen:                     │
│  - Protein Encoder (ESM-2)              │
│                                         │
│  Learning Rate: 1e-3                    │
│  Duration: 50 epochs (configurable)     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
PHASE 2: ESM FINE-TUNING
┌─────────────────────────────────────────┐
│           TRAINING PHASE 2              │
├─────────────────────────────────────────┤
│  Components Trained:                    │
│  - ALL Components                       │
│  - Protein Encoder (ESM-2 unfrozen)     │
│  - Drug Encoder (GIN)                   │
│  - Fusion Layers                        │
│  - Prediction Head                      │
│                                         │
│  Learning Rate: 1e-4                    │
│  Duration: 30 epochs (configurable)     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│           MODEL CHECKPOINT              │
├─────────────────────────────────────────┤
│  Contents:                              │
│  - Model weights                        │
│  - Training configuration               │
│  - Training metrics                     │
│  - Optimizer state                      │
└─────────────────────────────────────────┘
```

This architecture ensures robust connections between all phases of the DTA system, with proper data flow, error handling, and monitoring capabilities.
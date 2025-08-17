# Unified DTA System Architecture

## Overview

The Unified DTA (Drug-Target Affinity) System is a comprehensive platform for predicting binding affinities between drugs and target proteins. The system follows a modular architecture with clear separation between training, inference, and API components.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT APPLICATIONS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                              API LAYER                                      │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │   Models    │    │   Prediction    │    │         Cache               │ │
│  │             │    │                 │    │                             │ │
│  │  - Model    │◄──►│  - Prediction   │◄──►│  - Model Loading            │ │
│  │    Info     │    │    Service      │    │  - Checkpoint Management    │ │
│  │             │    │                 │    │  - Memory Management        │ │
│  └─────────────┘    └─────────────────┘    └─────────────────────────────┘ │
│         ▲                       ▲                       ▲                 │
│         │                       │                       │                 │
│         ▼                       ▼                       ▼                 │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │   Health    │    │    Batch        │    │         Metrics             │ │
│  │             │    │   Predict       │    │                             │ │
│  │  - System   │    │                 │    │  - Training Progress        │ │
│  │    Status   │    │  - Batch        │    │  - Model Performance        │ │
│  │             │    │    Processing   │    │                             │ │
│  └─────────────┘    └─────────────────┘    └─────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                           CORE INFERENCE PIPELINE                           │
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

## Detailed Component Architecture

### 1. API Layer

#### Endpoints Module (`src/unified_dta/api/endpoints.py`)
- **Prediction Endpoints**:
  - `/predict` - Single drug-target affinity prediction
  - `/predict/batch` - Batch predictions for multiple pairs
- **Model Management**:
  - `/models/{model_type}/info` - Model information
  - `/models/{model_type}/load` - Preload model into cache
  - `/models` - List available models
- **System Management**:
  - `/health` - System health status
  - `/cache` - Clear model cache
  - `/models/{model_type}/metrics` - Model training metrics
  - `/models/{model_type}/progress` - Training progress

#### Prediction Service (`src/unified_dta/api/prediction.py`)
- **PredictionService Class**: Main service for handling predictions
  - `predict_single()`: Process single prediction requests
  - `predict_batch()`: Process batch prediction requests
  - `_process_input_data()`: Validate and process input data
  - `_make_prediction()`: Execute model inference with confidence scoring

#### Cache Management (`src/unified_dta/api/cache.py`)
- **ModelCache Class**: Thread-safe model caching
  - `_load_model()`: Load models from checkpoints or create new ones
  - `_find_model_checkpoint()`: Locate trained model checkpoints
  - `_load_from_checkpoint()`: Load trained weights into models

### 2. Core Inference Pipeline

#### Data Flow
1. **Input Validation** (`src/unified_dta/data/data_processor.py`):
   - SMILES validation using RDKit
   - Protein sequence cleaning and tokenization
   - Molecular graph conversion

2. **Model Processing** (`src/unified_dta/core/models.py`):
   - Drug encoding using GIN (Graph Isomorphism Network)
   - Protein encoding using ESM-2 or CNN
   - Feature fusion using attention mechanisms
   - Affinity prediction with confidence scoring

3. **Confidence Scoring**:
   - Monte Carlo dropout for uncertainty estimation
   - Statistical analysis of prediction variance

### 3. Training Pipeline

#### Progressive Training (`src/unified_dta/training/training.py`)
- **Phase 1: Frozen ESM Training**
  - Train drug encoder and fusion layers
  - Keep ESM-2 protein encoder frozen
  - Lower learning rate (default: 1e-3)

- **Phase 2: ESM Fine-tuning**
  - Unfreeze ESM-2 layers for fine-tuning
  - Higher learning rate (default: 1e-4)
  - Continue training other components

#### Model Factory (`src/unified_dta/core/model_factory.py`)
- **Predefined Configurations**:
  - Lightweight (development/testing)
  - Standard (production)
  - High-performance (research)
  - Memory-optimized (resource-constrained)

#### Checkpoint Management
- Save/Load model states
- Training metrics tracking
- Early stopping implementation
- Memory optimization

## Connection Points Between Phases

### 1. Training to Inference Connection
```
Training Phase 1 ──► Training Phase 2 ──► Model Checkpoint ──► API Cache ──► Inference
     │                    │                    │                   │            │
     ▼                    ▼                    ▼                   ▼            ▼
Frozen ESM        ESM Fine-tuning      Saved Weights      Model Loading   Real-time
Training          Training             & Config           from Cache      Prediction
```

### 2. Data Processing Pipeline
```
API Input ──► Data Validation ──► Graph Conversion ──► Model Processing ──► Prediction
    │              │                  │                    │                  │
    ▼              ▼                  ▼                    ▼                  ▼
SMILES       SMILES Validator   Molecular Graph      Drug/Protein      Affinity
Protein      Protein Processor  Converter            Encoders          Score +
Sequence                          & Processor                            Confidence
```

### 3. Model Lifecycle
```
Model Factory ──► Model Creation ──► Training ──► Checkpoint ──► Cache ──► Inference
      │               │               │             │            │          │
      ▼               ▼               ▼             ▼            ▼          ▼
Config        UnifiedDTAModel   Progressive    Model File   ModelCache   Prediction
Creation                        Trainer                     Management   Service
```

## Key Integration Points

1. **Checkpoint System**: Trained models are saved as checkpoints and loaded by the API cache
2. **Configuration Consistency**: Model configurations are validated between training and inference
3. **Data Validation**: Comprehensive input validation ensures data quality throughout the pipeline
4. **Metrics Tracking**: Training metrics are exposed through API endpoints for monitoring
5. **Confidence Scoring**: Uncertainty estimation provides quality indicators for predictions

## Data Flow Through the System

### Training Phase:
1. Dataset loading and preprocessing (`src/unified_dta/data/`)
2. Model creation via ModelFactory (`src/unified_dta/core/model_factory.py`)
3. Phase 1 training with frozen ESM (`src/unified_dta/training/training.py`)
4. Phase 2 training with ESM fine-tuning
5. Checkpoint saving with metrics (`src/unified_dta/training/checkpoint_utils.py`)

### Inference Phase:
1. API request received (`src/unified_dta/api/endpoints.py`)
2. Model loading from cache/checkpoint (`src/unified_dta/api/cache.py`)
3. Input data validation and processing (`src/unified_dta/api/prediction.py`)
4. Molecular graph conversion (`src/unified_dta/data/data_processor.py`)
5. Model inference with confidence scoring (`src/unified_dta/core/models.py`)
6. Response formatting and return

This architecture ensures a clean separation of concerns while maintaining strong connections between all phases of the system lifecycle.
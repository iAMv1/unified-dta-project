# Unified DTA System - Phase Connections

## 1. Training Pipeline Connections

```
┌─────────────────────┐
│   DATA PREPARATION │
│                     │
│  Raw DTA Datasets   │
│  (KIBA, Davis, etc) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   DATA PROCESSING   │
│                     │
│  - SMILES Cleanup   │
│  - Protein Seq Proc │
│  - Graph Conversion │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  MODEL CREATION     │
│                     │
│  ModelFactory       │
│  Creates Model      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  TRAINING PHASE 1   │
│                     │
│  - Frozen ESM       │
│  - Train GIN        │
│  - Fusion Training  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  TRAINING PHASE 2   │
│                     │
│  - Unfreeze ESM     │
│  - Fine-tune All    │
│  - Optimize Jointly │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   CHECKPOINTING     │
│                     │
│  - Model Weights    │
│  - Training Metrics │
│  - Configurations   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   MODEL STORAGE     │
│                     │
│  checkpoints/       │
│  models/            │
└─────────────────────┘
```

## 2. Inference Pipeline Connections

```
┌─────────────────────┐
│    API REQUEST      │
│                     │
│  POST /predict      │
│  {                  │
│    drug_smiles,     │
│    protein_sequence │
│  }                  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   MODEL LOADING     │
│                     │
│  ModelCache         │
│  - Check Cache      │
│  - Load Checkpoint  │
│  - Create New       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   DATA VALIDATION   │
│                     │
│  DataProcessor      │
│  - SMILES Validate  │
│  - Protein Process  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   GRAPH CONVERSION  │
│                     │
│  MolecularGraph     │
│  Converter          │
│  - RDKit Processing │
│  - Feature Extract  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   MODEL INFERENCE   │
│                     │
│  UnifiedDTAModel    │
│  - Drug Encoding    │
│  - Protein Encoding │
│  - Fusion           │
│  - Prediction       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  CONFIDENCE SCORING │
│                     │
│  Monte Carlo        │
│  Dropout            │
│  - Uncertainty      │
│  - Confidence       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   API RESPONSE      │
│                     │
│  PredictionResponse │
│  {                  │
│    predicted_affinity,│
│    confidence,      │
│    processing_time  │
│  }                  │
└─────────────────────┘
```

## 3. Cross-Phase Connections

```
TRAINING PHASE CONNECTIONS:

┌─────────────────────────────────────────────────────────────┐
│                    TRAINING ECOSYSTEM                       │
├─────────────────────────────────────────────────────────────┤
│  Configuration Management                                   │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │   DTAConfig     │◄──────►│ ModelFactory    │            │
│  │                 │        │                 │            │
│  │  - Model Config │        │  - Predefined   │            │
│  │  - Train Config │        │    Templates    │            │
│  └─────────────────┘        └─────────────────┘            │
│                                                             │
│  Progressive Training                                       │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │Phase 1 Trainer  │        │Phase 2 Trainer  │            │
│  │                 │        │                 │            │
│  │- Frozen ESM     │        │- ESM Fine-tune  │            │
│  │- GIN Train      │        │- Joint Optimize │            │
│  └─────────┬───────┘        └─────────┬───────┘            │
│            │                          │                    │
│            └──────────────────────────┘                    │
│                          │                                │
│                          ▼                                │
│  Checkpoint System                                        │
│  ┌─────────────────────────────────────────┐              │
│  │           Model Checkpoint              │              │
│  │                                         │              │
│  │  - Model State Dict                     │              │
│  │  - Optimizer State                      │              │
│  │  - Training Metrics                     │              │
│  │  - Configuration                        │              │
│  │  - Validation Results                   │              │
│  └─────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘

INFERENCE PHASE CONNECTIONS:

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE ECOSYSTEM                      │
├─────────────────────────────────────────────────────────────┤
│  API Layer                                                  │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │  Endpoints      │◄──────►│ Prediction      │            │
│  │                 │        │  Service        │            │
│  │- Request Routes │        │                 │            │
│  │- Response       │        │- Process Data   │            │
│  └─────────────────┘        │- Run Inference  │            │
│                             │- Score Conf.    │            │
│                             └─────────┬───────┘            │
│                                       │                    │
│  Model Management                     │                    │
│  ┌─────────────────┐                 │                    │
│  │   ModelCache    │◄────────────────┘                    │
│  │                 │                                      │
│  │- Load Models    │                                      │
│  │- Cache Mgmt     │                                      │
│  │- Checkpoints    │                                      │
│  └─────────────────┘                                      │
│                                                             │
│  Data Processing                                          │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │ DataProcessor   │◄──────►│ Graph Converter │            │
│  │                 │        │                 │            │
│  │- Validate Input │        │- RDKit Graph    │            │
│  │- Process Seq    │        │- Feature Extract│            │
│  └─────────────────┘        └─────────────────┘            │
└─────────────────────────────────────────────────────────────┘

CROSS-PHASE INTEGRATION:

┌─────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  Model Consistency                                          │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │ConfigValidator  │◄──────►│ ConfigConsistency│           │
│  │                 │        │  Checker        │            │
│  │- Validate Train │        │                 │            │
│  │- Validate Infer │        │- Compare Configs│            │
│  └─────────────────┘        └─────────────────┘            │
│                                                             │
│  Metrics Integration                                        │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │Training Metrics │◄──────►│ Model Metrics   │            │
│  │                 │        │  Manager        │            │
│  │- Training Perf  │        │                 │            │
│  │- Validation     │        │- Expose Metrics │            │
│  └─────────────────┘        └─────────────────┘            │
│                                                             │
│  Checkpoint Integration                                     │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │CheckpointLoader │◄──────►│ CheckpointSaver │            │
│  │                 │        │                 │            │
│  │- Load Weights   │        │- Save Weights   │            │
│  │- Restore State  │        │- Save Metrics   │            │
│  └─────────────────┘        └─────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## 4. Data Flow Summary

### Training Data Flow:
```
Raw Data → Preprocessing → Model Creation → Phase 1 Training → Phase 2 Training → Checkpoint
```

### Inference Data Flow:
```
API Request → Validation → Graph Conversion → Model Loading → Inference → Confidence Scoring → Response
```

### Integration Points:
1. **Model Checkpoints**: Bridge between training and inference
2. **Configuration Management**: Ensures consistency between phases
3. **Metrics Tracking**: Provides feedback from training to monitoring
4. **Cache Management**: Optimizes model loading for inference
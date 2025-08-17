"""
Unified Drug-Target Affinity Prediction System
Core package containing models, data processing, and utilities
"""

from .models import (
    BaseEncoder,
    ESMProteinEncoder,
    GINDrugEncoder,
    MultiModalFusion,
    AffinityPredictor,
    UnifiedDTAModel,
    create_dta_model,
    get_lightweight_model,
    get_production_model
)

from .base_components import (
    BaseEncoder,
    SEBlock,
    PositionalEncoding
)

from .protein_encoders import (
    EnhancedCNNProteinEncoder,
    MemoryOptimizedESMEncoder,
    GatedCNNBlock
)

from .drug_encoders import (
    EnhancedGINDrugEncoder,
    MultiScaleGINEncoder,
    ConfigurableMLPBlock,
    ResidualGINLayer
)

from .drug_encoders import (
    EnhancedGINDrugEncoder,
    MultiScaleGINEncoder,
    ConfigurableMLPBlock,
    ResidualGINLayer
)

from .graph_preprocessing import (
    GraphFeatureConfig,
    MolecularGraphProcessor,
    GraphValidator,
    OptimizedGraphBatcher,
    create_molecular_graph_processor,
    process_smiles_batch
)

from .config import (
    DTAConfig,
    load_config,
    save_config,
    validate_config,
    get_default_configs,
    create_config_template
)

from .utils import (
    set_seed,
    get_device,
    get_memory_usage,
    optimize_batch_size,
    clear_memory,
    count_parameters,
    log_model_info,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
    EarlyStopping
)

from .evaluation import (
    EvaluationMetrics,
    StatisticalTestResults,
    MetricsCalculator,
    ComprehensiveEvaluator,
    BaselineComparator,
    EvaluationReporter,
    CrossValidator,
    BenchmarkSuite,
    AutomatedEvaluationPipeline
)

from .data_processing import (
    SMILESValidator,
    ProteinProcessor,
    MolecularGraphConverter,
    DataValidator,
    load_dta_dataset,
    preprocess_dta_dataset
)

from .datasets import (
    DTASample,
    DTADataset,
    MultiDatasetDTA,
    DTADataLoader,
    collate_dta_batch,
    create_data_splits,
    load_standard_datasets,
    DataAugmentation,
    create_balanced_sampler
)

__version__ = "1.0.0"
__author__ = "Unified DTA Team"

__all__ = [
    # Base Components
    'BaseEncoder',
    'SEBlock',
    'PositionalEncoding',
    
    # Protein Encoders
    'ESMProteinEncoder', 
    'EnhancedCNNProteinEncoder',
    'MemoryOptimizedESMEncoder',
    'GatedCNNBlock',
    
    # Drug Encoders
    'GINDrugEncoder',
    'EnhancedGINDrugEncoder',
    'MultiScaleGINEncoder',
    'ConfigurableMLPBlock',
    'ResidualGINLayer',
    
    # Graph Preprocessing
    'GraphFeatureConfig',
    'MolecularGraphProcessor',
    'GraphValidator',
    'OptimizedGraphBatcher',
    'create_molecular_graph_processor',
    'process_smiles_batch',
    'MultiModalFusion',
    'AffinityPredictor',
    'UnifiedDTAModel',
    'create_dta_model',
    'get_lightweight_model',
    'get_production_model',
    
    # Configuration
    'DTAConfig',
    'load_config',
    'save_config', 
    'validate_config',
    'get_default_configs',
    'create_config_template',
    
    # Utilities
    'set_seed',
    'get_device',
    'get_memory_usage',
    'optimize_batch_size',
    'clear_memory',
    'count_parameters',
    'log_model_info',
    'save_checkpoint',
    'load_checkpoint',
    'setup_logging',
    'EarlyStopping',
    
    # Evaluation
    'EvaluationMetrics',
    'StatisticalTestResults',
    'MetricsCalculator',
    'ComprehensiveEvaluator',
    'BaselineComparator',
    'EvaluationReporter',
    'CrossValidator',
    'BenchmarkSuite',
    'AutomatedEvaluationPipeline',
    
    # Data Processing
    'SMILESValidator',
    'ProteinProcessor',
    'MolecularGraphConverter',
    'DataValidator',
    'load_dta_dataset',
    'preprocess_dta_dataset',
    
    # Datasets
    'DTASample',
    'DTADataset',
    'MultiDatasetDTA',
    'DTADataLoader',
    'collate_dta_batch',
    'create_data_splits',
    'load_standard_datasets',
    'DataAugmentation',
    'create_balanced_sampler'
]
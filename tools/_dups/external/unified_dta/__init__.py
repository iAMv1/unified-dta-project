"""
Unified Drug-Target Affinity (DTA) Prediction System

A comprehensive platform that integrates state-of-the-art protein and drug encoders
for drug-target affinity prediction and target-specific drug generation.

Key Features:
- ESM-2 protein language model integration
- Graph Isomorphism Networks for drug encoding
- Multi-modal fusion mechanisms
- 2-phase progressive training
- Memory-optimized implementations
"""

__version__ = "0.1.0"
__author__ = "Unified DTA Team"
__email__ = "contact@unified-dta.org"

# Core imports for easy access
try:
    from .core.models import UnifiedDTAModel
    from .core.model_factory import ModelFactory
    from .core.config import Config
    from .core.training import DTATrainer
    from .core.evaluation import DTAEvaluator
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import core components: {e}")

# Encoder imports
try:
    from .encoders.protein_encoders import ESMProteinEncoder, CNNProteinEncoder
    from .encoders.drug_encoders import GINDrugEncoder
    from .encoders.fusion import MultiModalFusion
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import encoders: {e}")

# Data processing imports
try:
    from .data.datasets import DTADataset
    from .data.data_processing import DataProcessor
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import data components: {e}")

# Utility imports
try:
    from .utils.checkpoint_utils import CheckpointManager
    from .utils.memory_optimization import MemoryOptimizer
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import utilities: {e}")

__all__ = [
    # Core classes
    'UnifiedDTAModel',
    'ModelFactory', 
    'Config',
    'DTATrainer',
    'DTAEvaluator',
    
    # Encoders
    'ESMProteinEncoder',
    'CNNProteinEncoder', 
    'GINDrugEncoder',
    'MultiModalFusion',
    
    # Data
    'DTADataset',
    'DataProcessor',
    
    # Utils
    'CheckpointManager',
    'MemoryOptimizer',
]
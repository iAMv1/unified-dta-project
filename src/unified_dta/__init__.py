"""
Unified Drug-Target Affinity Prediction System

A comprehensive platform for drug-target affinity prediction and molecular generation
that combines state-of-the-art machine learning models with an intuitive interface.
"""

__version__ = "1.0.0"
__author__ = "Unified DTA Team"

# Core imports
from .core.models import UnifiedDTAModel
from .core.config import load_config
from .core.model_factory import ModelFactory, create_dta_model

# Encoder imports
from .encoders.protein_encoders import ESMProteinEncoder
from .encoders.drug_encoders import EnhancedGINDrugEncoder as GINDrugEncoder
from .encoders.fusion import MultiModalFusion

# Training imports
try:
    from .training.training import train_model, evaluate_model
except ImportError:
    # Provide placeholders if functions are not available
    train_model = None
    evaluate_model = None

try:
    from .training.checkpoint_utils import save_checkpoint, load_checkpoint
except ImportError:
    # Provide placeholders if functions are not available
    save_checkpoint = None
    load_checkpoint = None

# Generation imports (if available)
try:
    from .generation.drug_generation import DrugGenerationPipeline
    from .generation.generation_evaluation import GenerationEvaluationPipeline
    GENERATION_AVAILABLE = True
except ImportError:
    GENERATION_AVAILABLE = False

__all__ = [
    # Core
    "UnifiedDTAModel",
    "load_config", 
    "ModelFactory",
    "create_dta_model",
    
    # Encoders
    "ESMProteinEncoder",
    "GINDrugEncoder", 
    "MultiModalFusion",
    
    # Training (conditional)
    *([
        "train_model",
    ] if train_model is not None else []),
    *([
        "evaluate_model",
    ] if evaluate_model is not None else []),
    *([
        "save_checkpoint",
    ] if save_checkpoint is not None else []),
    *([
        "load_checkpoint",
    ] if load_checkpoint is not None else []),
    
    # Generation (conditional)
    *([
        "DrugGenerationPipeline",
        "GenerationEvaluationPipeline"
    ] if GENERATION_AVAILABLE else []),
    
    # Metadata
    "__version__",
    "__author__",
    "GENERATION_AVAILABLE"
]
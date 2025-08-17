"""
Unified Drug-Target Affinity Prediction System

A comprehensive platform for drug-target affinity prediction and molecular generation.
"""

__version__ = "1.0.0"
__author__ = "Unified DTA Team"

# Core imports
try:
    from .src.unified_dta.core.models import UnifiedDTAModel
    from .src.unified_dta.core.config import load_config
    from .src.unified_dta.training.training import train_model
except ImportError:
    # Handle missing dependencies gracefully
    pass

__all__ = ["UnifiedDTAModel", "load_config", "train_model"]

"""
Configuration classes for training module
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    batch_size: int = 4
    learning_rate_phase1: float = 1e-3
    learning_rate_phase2: float = 1e-4
    num_epochs_phase1: int = 50
    num_epochs_phase2: int = 30
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    checkpoint_interval: int = 5
    gradient_clip_norm: float = 1.0
    
    # Memory management settings
    max_memory_mb: float = 4000
    enable_gradient_checkpointing: bool = True
    memory_monitoring_interval: int = 10
    aggressive_memory_cleanup: bool = False


# For DTAConfig, we'll import it from the core module to avoid duplication
try:
    from ..core.config import DTAConfig
except ImportError:
    # Fallback definition if core config is not available
    @dataclass
    class DTAConfig:
        """Main configuration class for the Unified DTA System"""
        protein_encoder_type: str = 'esm'
        drug_encoder_type: str = 'gin'
        use_fusion: bool = True
        device: str = 'auto'
        seed: int = 42
        verbose: bool = True
        log_level: str = 'INFO'
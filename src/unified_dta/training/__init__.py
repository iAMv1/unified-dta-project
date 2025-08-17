"""
Training infrastructure and utilities
"""

# Import what's actually available
from .training import ProgressiveTrainer
from .checkpoint_utils import save_checkpoint, load_checkpoint, CheckpointManager
from .memory_optimization import optimize_memory, get_optimal_batch_size

# Create aliases for expected names
train_model = None  # Placeholder - would need actual implementation
evaluate_model = None  # Placeholder - would need actual implementation
TwoPhaseTrainer = ProgressiveTrainer

__all__ = [
    # Training
    "train_model",
    "evaluate_model", 
    "TwoPhaseTrainer",
    "ProgressiveTrainer",
    
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointManager",
    
    # Memory optimization
    "optimize_memory",
    "get_optimal_batch_size"
]
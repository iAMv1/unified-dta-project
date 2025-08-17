"""
Utility functions for the Unified DTA System
Memory management, device handling, and common operations
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import psutil
import gc
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import random
import os

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


def get_device(device: str = 'auto') -> torch.device:
    """Get the appropriate device for computation"""
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            logger.info("CUDA not available. Using CPU")
    
    device = torch.device(device)
    
    # Log device information
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        logger.info(f"GPU memory: {gpu_memory:.1f} GB")
    
    return device


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    # System memory
    memory = psutil.virtual_memory()
    system_memory = {
        'total_gb': memory.total / 1e9,
        'available_gb': memory.available / 1e9,
        'used_gb': memory.used / 1e9,
        'percent': memory.percent
    }
    
    # GPU memory (if available)
    gpu_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            
            gpu_memory[f'gpu_{i}'] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - reserved
            }
    
    return {'system': system_memory, 'gpu': gpu_memory}


def optimize_batch_size(model: nn.Module, 
                       sample_input: tuple,
                       max_batch_size: int = 64,
                       device: Optional[torch.device] = None) -> int:
    """Automatically determine optimal batch size based on available memory"""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    optimal_batch_size = 1
    
    for batch_size in [2, 4, 8, 16, 32, 64]:
        if batch_size > max_batch_size:
            break
            
        try:
            # Create batch of sample inputs
            if isinstance(sample_input, tuple):
                batch_input = tuple(
                    inp.repeat(batch_size, *[1] * (inp.dim() - 1)) if hasattr(inp, 'repeat')
                    else [inp] * batch_size
                    for inp in sample_input
                )
            else:
                batch_input = sample_input.repeat(batch_size, *[1] * (sample_input.dim() - 1))
            
            # Test forward pass
            with torch.no_grad():
                _ = model(*batch_input if isinstance(batch_input, tuple) else batch_input)
            
            optimal_batch_size = batch_size
            logger.debug(f"Batch size {batch_size} successful")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.debug(f"Batch size {batch_size} failed: out of memory")
                break
            else:
                raise e
        
        # Clear cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    logger.info(f"Optimal batch size determined: {optimal_batch_size}")
    return optimal_batch_size


def clear_memory() -> None:
    """Clear memory caches and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def log_model_info(model: nn.Module, model_name: str = "Model") -> None:
    """Log comprehensive model information"""
    param_counts = count_parameters(model)
    
    logger.info(f"{model_name} Information:")
    logger.info(f"  Total parameters: {param_counts['total']:,}")
    logger.info(f"  Trainable parameters: {param_counts['trainable']:,}")
    logger.info(f"  Frozen parameters: {param_counts['frozen']:,}")
    
    # Model size estimation
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1e6
    
    logger.info(f"  Estimated model size: {model_size_mb:.1f} MB")


def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   metrics: Dict[str, float],
                   checkpoint_path: Union[str, Path],
                   config: Optional[Dict[str, Any]] = None) -> None:
    """Save model checkpoint with metadata"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'model_info': count_parameters(model)
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: Union[str, Path],
                   model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load model checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Loss: {checkpoint.get('loss', 'unknown')}")
    
    return checkpoint


def create_model_summary(model: nn.Module, input_shapes: List[tuple]) -> str:
    """Create a detailed model summary"""
    from torchsummary import summary
    
    try:
        # Capture summary output
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        summary(model, input_shapes)
        
        sys.stdout = old_stdout
        summary_str = buffer.getvalue()
        
        return summary_str
    
    except ImportError:
        logger.warning("torchsummary not available. Install with: pip install torchsummary")
        return "Model summary not available (torchsummary not installed)"
    except Exception as e:
        logger.warning(f"Could not generate model summary: {e}")
        return f"Model summary generation failed: {e}"


def validate_tensor_shapes(tensors: Dict[str, torch.Tensor], 
                          expected_shapes: Dict[str, tuple]) -> bool:
    """Validate tensor shapes match expected dimensions"""
    for name, tensor in tensors.items():
        if name in expected_shapes:
            expected = expected_shapes[name]
            actual = tensor.shape
            
            # Check if shapes match (allowing for batch dimension flexibility)
            if len(expected) != len(actual):
                logger.error(f"Tensor {name}: dimension mismatch. "
                           f"Expected {len(expected)} dims, got {len(actual)} dims")
                return False
            
            for i, (exp, act) in enumerate(zip(expected, actual)):
                if exp != -1 and exp != act:  # -1 means flexible dimension
                    logger.error(f"Tensor {name}: shape mismatch at dimension {i}. "
                               f"Expected {exp}, got {act}")
                    return False
    
    return True


def setup_logging(log_level: str = 'INFO', 
                 log_file: Optional[Union[str, Path]] = None) -> None:
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")


class EarlyStopping:
    """Early stopping utility for training"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop"""
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best model weights")
            return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save the best model weights"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_lr_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str = 'cosine',
                    **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler"""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    # Example usage and testing
    setup_logging('DEBUG')
    
    # Test memory usage
    memory_info = get_memory_usage()
    logger.info(f"Memory usage: {memory_info}")
    
    # Test device detection
    device = get_device()
    logger.info(f"Selected device: {device}")
    
    # Test seed setting
    set_seed(42)
    logger.info("Random seed set successfully")
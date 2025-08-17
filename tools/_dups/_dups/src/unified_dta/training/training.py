"""
Training Infrastructure for Unified DTA System
Implements 2-phase progressive training with memory management and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import psutil
import gc
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import pearsonr, spearmanr
import warnings

from .config import DTAConfig, TrainingConfig
from .models import UnifiedDTAModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int
    phase: int
    train_loss: float
    val_loss: float
    val_pearson: float
    val_spearman: float
    val_rmse: float
    learning_rate: float
    memory_usage: float
    training_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingState:
    """Container for training state information"""
    current_epoch: int = 0
    current_phase: int = 1
    best_val_loss: float = float('inf')
    best_val_pearson: float = -1.0
    patience_counter: int = 0
    total_training_time: float = 0.0
    metrics_history: Optional[List[TrainingMetrics]] = None
    
    def __post_init__(self):
        if self.metrics_history is None:
            self.metrics_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EarlyStopping:
    """Enhanced early stopping utility with multiple metrics support"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min',
                 restore_best_weights: bool = True, monitor_metric: str = 'val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.monitor_metric = monitor_metric
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        self.best_epoch = 0
        self.best_weights = None
        self.score_history = []
        
        logger.info(f"EarlyStopping initialized: patience={patience}, mode={mode}, monitor={monitor_metric}")
    
    def __call__(self, score: float, epoch: int, model: Optional[nn.Module] = None) -> bool:
        """Check if training should stop early"""
        
        self.score_history.append(score)
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
            # Save best weights if requested
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            logger.debug(f"New best {self.monitor_metric}: {score:.6f} at epoch {epoch}")
        else:
            self.counter += 1
            logger.debug(f"No improvement for {self.counter}/{self.patience} epochs")
        
        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            logger.info(f"Best {self.monitor_metric}: {self.best_score:.6f} at epoch {self.best_epoch}")
            
            # Restore best weights if requested
            if self.restore_best_weights and model is not None and self.best_weights is not None:
                model.load_state_dict({k: v.to(next(model.parameters()).device) 
                                     for k, v in self.best_weights.items()})
                logger.info("Restored best model weights")
        
        return self.early_stop
    
    def get_best_score(self) -> float:
        """Get the best score achieved"""
        return self.best_score
    
    def get_best_epoch(self) -> int:
        """Get the epoch where best score was achieved"""
        return self.best_epoch
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False
        self.best_epoch = 0
        self.best_weights = None
        self.score_history = []
        logger.info("Early stopping state reset")
    
    def get_patience_info(self) -> Dict[str, Any]:
        """Get information about current patience state"""
        return {
            'counter': self.counter,
            'patience': self.patience,
            'remaining_patience': self.patience - self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'early_stop': self.early_stop,
            'score_history': self.score_history[-10:]  # Last 10 scores
        }


class LearningRateScheduler:
    """Custom learning rate scheduler for 2-phase training"""
    
    def __init__(self, optimizer: optim.Optimizer, 
                 phase1_lr: float = 1e-3,
                 phase2_lr: float = 1e-4,
                 warmup_epochs: int = 5,
                 decay_factor: float = 0.95):
        self.optimizer = optimizer
        self.phase1_lr = phase1_lr
        self.phase2_lr = phase2_lr
        self.warmup_epochs = warmup_epochs
        self.decay_factor = decay_factor
        self.current_phase = 1
        self.epoch = 0
    
    def set_phase(self, phase: int):
        """Set training phase and adjust learning rate"""
        self.current_phase = phase
        self.epoch = 0
        
        base_lr = self.phase1_lr if phase == 1 else self.phase2_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr
        
        logger.info(f"Set learning rate to {base_lr} for phase {phase}")
    
    def step(self):
        """Step the scheduler"""
        self.epoch += 1
        
        # Warmup for first few epochs
        if self.epoch <= self.warmup_epochs:
            base_lr = self.phase1_lr if self.current_phase == 1 else self.phase2_lr
            lr = base_lr * (self.epoch / self.warmup_epochs)
        else:
            # Exponential decay
            base_lr = self.phase1_lr if self.current_phase == 1 else self.phase2_lr
            lr = base_lr * (self.decay_factor ** (self.epoch - self.warmup_epochs))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class MemoryManager:
    """Enhanced memory management utilities for training with automatic optimization"""
    
    def __init__(self, max_memory_mb: float = 4000, enable_gradient_checkpointing: bool = True):
        self.max_memory_mb = max_memory_mb
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.memory_history = []
        self.batch_size_history = []
        self.oom_count = 0
        self.adaptive_batch_size = None
        
        # Memory thresholds
        self.warning_threshold = 0.8  # 80% of max memory
        self.critical_threshold = 0.9  # 90% of max memory
        
        logger.info(f"MemoryManager initialized with max memory: {max_memory_mb}MB")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get comprehensive memory usage statistics"""
        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024
        
        # GPU memory
        gpu_memory = None
        gpu_reserved = None
        gpu_cached = None
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024
            gpu_cached = torch.cuda.memory_cached() / 1024 / 1024 if hasattr(torch.cuda, 'memory_cached') else 0
        
        memory_stats = {
            'cpu_memory_mb': cpu_memory,
            'gpu_memory_mb': gpu_memory,
            'gpu_reserved_mb': gpu_reserved,
            'gpu_cached_mb': gpu_cached,
            'total_memory_mb': cpu_memory + (gpu_memory or 0)
        }
        
        # Track memory history
        self.memory_history.append(memory_stats['total_memory_mb'])
        if len(self.memory_history) > 100:  # Keep last 100 measurements
            self.memory_history.pop(0)
        
        return memory_stats
    
    def get_memory_utilization(self) -> float:
        """Get current memory utilization as percentage of max allowed"""
        current_memory = self.get_memory_usage()['total_memory_mb']
        return current_memory / self.max_memory_mb
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is in critical range"""
        return self.get_memory_utilization() > self.critical_threshold
    
    def is_memory_warning(self) -> bool:
        """Check if memory usage is in warning range"""
        return self.get_memory_utilization() > self.warning_threshold
    
    def clear_cache(self, aggressive: bool = False):
        """Clear memory caches with optional aggressive cleanup"""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            if aggressive:
                # More aggressive GPU memory cleanup
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        if aggressive:
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
        
        logger.debug("Memory cache cleared" + (" (aggressive)" if aggressive else ""))
    
    def handle_oom_error(self, current_batch_size: int) -> int:
        """Handle out-of-memory error and suggest new batch size"""
        self.oom_count += 1
        logger.warning(f"OOM error #{self.oom_count} encountered with batch size {current_batch_size}")
        
        # Clear memory aggressively
        self.clear_cache(aggressive=True)
        
        # Reduce batch size
        new_batch_size = max(1, current_batch_size // 2)
        logger.info(f"Reducing batch size from {current_batch_size} to {new_batch_size}")
        
        return new_batch_size
    
    def estimate_optimal_batch_size(self, model: nn.Module, sample_input: Any, 
                                  initial_batch_size: int = 8) -> int:
        """Estimate optimal batch size through binary search"""
        if self.adaptive_batch_size is not None:
            return self.adaptive_batch_size
        
        device = next(model.parameters()).device
        model.eval()
        
        # Binary search for optimal batch size
        min_batch_size = 1
        max_batch_size = initial_batch_size * 4  # Start with higher upper bound
        optimal_batch_size = initial_batch_size
        
        logger.info(f"Estimating optimal batch size (range: {min_batch_size}-{max_batch_size})")
        
        while min_batch_size <= max_batch_size:
            test_batch_size = (min_batch_size + max_batch_size) // 2
            
            try:
                # Test memory usage with this batch size
                memory_before = self.get_memory_usage()['total_memory_mb']
                
                with torch.no_grad():
                    # Create test batch
                    if isinstance(sample_input, tuple):
                        drug_data, protein_data = sample_input
                        
                        # Simulate batch by repeating data
                        if hasattr(drug_data, 'batch'):
                            # Handle PyTorch Geometric data
                            test_drug_data = drug_data
                        else:
                            test_drug_data = drug_data
                        
                        test_protein_data = protein_data * test_batch_size if isinstance(protein_data, list) else protein_data
                        
                        if hasattr(test_drug_data, 'to'):
                            test_drug_data = test_drug_data.to(device)
                        
                        # Forward pass
                        _ = model(test_drug_data, test_protein_data)
                
                memory_after = self.get_memory_usage()['total_memory_mb']
                memory_used = memory_after - memory_before
                
                # Check if memory usage is acceptable
                if memory_after < self.max_memory_mb * self.warning_threshold:
                    optimal_batch_size = test_batch_size
                    min_batch_size = test_batch_size + 1
                    logger.debug(f"Batch size {test_batch_size} OK (memory: {memory_after:.1f}MB)")
                else:
                    max_batch_size = test_batch_size - 1
                    logger.debug(f"Batch size {test_batch_size} too large (memory: {memory_after:.1f}MB)")
                
                # Clear memory after test
                self.clear_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    max_batch_size = test_batch_size - 1
                    logger.debug(f"OOM at batch size {test_batch_size}")
                    self.clear_cache(aggressive=True)
                else:
                    logger.warning(f"Error testing batch size {test_batch_size}: {e}")
                    break
        
        self.adaptive_batch_size = max(1, optimal_batch_size)
        logger.info(f"Estimated optimal batch size: {self.adaptive_batch_size}")
        return self.adaptive_batch_size
    
    def monitor_memory_during_training(self, epoch: int, batch_idx: int, 
                                     current_batch_size: int) -> Dict[str, Any]:
        """Monitor memory usage during training and provide recommendations"""
        memory_stats = self.get_memory_usage()
        utilization = self.get_memory_utilization()
        
        recommendations = []
        
        # Check memory levels
        if self.is_memory_critical():
            recommendations.append("CRITICAL: Memory usage > 90%. Consider reducing batch size.")
        elif self.is_memory_warning():
            recommendations.append("WARNING: Memory usage > 80%. Monitor closely.")
        
        # Check for memory leaks (increasing trend)
        if len(self.memory_history) > 10:
            recent_avg = np.mean(self.memory_history[-5:])
            older_avg = np.mean(self.memory_history[-10:-5])
            if recent_avg > older_avg * 1.1:  # 10% increase
                recommendations.append("Potential memory leak detected. Consider clearing cache more frequently.")
        
        # Adaptive batch size adjustment
        if utilization > self.critical_threshold and current_batch_size > 1:
            new_batch_size = max(1, current_batch_size // 2)
            recommendations.append(f"Reduce batch size to {new_batch_size}")
        elif utilization < 0.5 and current_batch_size < 32:  # Under-utilizing memory
            new_batch_size = min(32, current_batch_size * 2)
            recommendations.append(f"Consider increasing batch size to {new_batch_size}")
        
        monitoring_data = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'memory_stats': memory_stats,
            'utilization': utilization,
            'recommendations': recommendations,
            'oom_count': self.oom_count
        }
        
        # Log warnings if needed
        if recommendations:
            for rec in recommendations:
                if "CRITICAL" in rec:
                    logger.error(rec)
                elif "WARNING" in rec:
                    logger.warning(rec)
                else:
                    logger.info(rec)
        
        return monitoring_data
    
    def enable_gradient_checkpointing_for_model(self, model: nn.Module):
        """Enable gradient checkpointing for memory efficiency"""
        if not self.enable_gradient_checkpointing:
            return
        
        # Enable gradient checkpointing for ESM model if available
        if hasattr(model, 'protein_encoder') and hasattr(model.protein_encoder, 'esm_model'):
            if hasattr(model.protein_encoder.esm_model, 'gradient_checkpointing_enable'):
                model.protein_encoder.esm_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled for ESM model")
        
        # Enable for other transformer-based components
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
    
    def get_memory_report(self) -> str:
        """Generate comprehensive memory usage report"""
        memory_stats = self.get_memory_usage()
        utilization = self.get_memory_utilization()
        
        gpu_memory_str = f"{memory_stats['gpu_memory_mb']:.1f}" if memory_stats['gpu_memory_mb'] is not None else "N/A"
        gpu_reserved_str = f"{memory_stats['gpu_reserved_mb']:.1f}" if memory_stats['gpu_reserved_mb'] is not None else "N/A"
        gpu_cached_str = f"{memory_stats['gpu_cached_mb']:.1f}" if memory_stats['gpu_cached_mb'] is not None else "N/A"
        
        report = f"""
Memory Usage Report:
==================
CPU Memory: {memory_stats['cpu_memory_mb']:.1f} MB
GPU Memory: {gpu_memory_str} MB (allocated)
GPU Reserved: {gpu_reserved_str} MB
GPU Cached: {gpu_cached_str} MB
Total Memory: {memory_stats['total_memory_mb']:.1f} MB
Utilization: {utilization:.1%} of {self.max_memory_mb} MB limit
OOM Errors: {self.oom_count}
Adaptive Batch Size: {self.adaptive_batch_size}

Memory History (last 10):
{self.memory_history[-10:] if self.memory_history else 'No history available'}
"""
        return report
    
    @staticmethod
    def get_system_memory_info() -> Dict[str, float]:
        """Get system-wide memory information"""
        # System memory
        system_memory = psutil.virtual_memory()
        
        info = {
            'total_system_memory_gb': system_memory.total / (1024**3),
            'available_system_memory_gb': system_memory.available / (1024**3),
            'system_memory_percent': system_memory.percent
        }
        
        # GPU memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                info[f'gpu_{i}_total_memory_gb'] = gpu_props.total_memory / (1024**3)
                info[f'gpu_{i}_name'] = gpu_props.name
        
        return info


class ModelCheckpoint:
    """Comprehensive model checkpointing utility with backward compatibility"""
    
    def __init__(self, checkpoint_dir: str, save_best: bool = True, 
                 save_interval: int = 5, max_checkpoints: int = 5,
                 save_optimizer: bool = True, save_metrics: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_metrics = save_metrics
        self.checkpoint_files = []
        self.best_score = float('inf')
        self.checkpoint_version = "1.0"  # For backward compatibility
        
        # Create subdirectories for organization
        (self.checkpoint_dir / "regular").mkdir(exist_ok=True)
        (self.checkpoint_dir / "best").mkdir(exist_ok=True)
        (self.checkpoint_dir / "metrics").mkdir(exist_ok=True)
        
        logger.info(f"ModelCheckpoint initialized at {self.checkpoint_dir}")
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: LearningRateScheduler, training_state: TrainingState,
                       config: DTAConfig, val_loss: float, is_best: bool = False,
                       additional_metrics: Optional[Dict[str, Any]] = None):
        """Save comprehensive model checkpoint with all training state"""
        
        import time
        timestamp = int(time.time())
        
        # Determine filename and path
        if is_best:
            filename = f'best_model_epoch_{training_state.current_epoch}_phase_{training_state.current_phase}.pth'
            filepath = self.checkpoint_dir / "best" / filename
            # Also save as latest best
            latest_best_path = self.checkpoint_dir / "best" / "latest_best.pth"
        else:
            filename = f'checkpoint_epoch_{training_state.current_epoch}_phase_{training_state.current_phase}_{timestamp}.pth'
            filepath = self.checkpoint_dir / "regular" / filename
        
        # Prepare checkpoint data
        checkpoint = {
            'version': self.checkpoint_version,
            'timestamp': timestamp,
            'model_state_dict': model.state_dict(),
            'training_state': training_state.to_dict(),
            'config': config.to_dict(),
            'validation_loss': val_loss,
            'is_best': is_best,
            'epoch': training_state.current_epoch,
            'phase': training_state.current_phase
        }
        
        # Add optimizer state if requested
        if self.save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['scheduler_state'] = {
                'epoch': scheduler.epoch,
                'current_phase': scheduler.current_phase,
                'phase1_lr': scheduler.phase1_lr,
                'phase2_lr': scheduler.phase2_lr,
                'warmup_epochs': scheduler.warmup_epochs,
                'decay_factor': scheduler.decay_factor
            }
        
        # Add training metrics if requested
        if self.save_metrics and training_state.metrics_history:
            checkpoint['metrics_history'] = [m.to_dict() for m in training_state.metrics_history]
            
            # Calculate additional statistics
            recent_metrics = training_state.metrics_history[-10:] if len(training_state.metrics_history) >= 10 else training_state.metrics_history
            if recent_metrics:
                checkpoint['recent_performance'] = {
                    'avg_train_loss': np.mean([m.train_loss for m in recent_metrics]),
                    'avg_val_loss': np.mean([m.val_loss for m in recent_metrics]),
                    'avg_val_pearson': np.mean([m.val_pearson for m in recent_metrics]),
                    'avg_val_spearman': np.mean([m.val_spearman for m in recent_metrics]),
                    'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics])
                }
        
        # Add additional metrics if provided
        if additional_metrics:
            checkpoint['additional_metrics'] = additional_metrics
        
        # Add model architecture info for compatibility checking
        checkpoint['model_info'] = {
            'protein_encoder_type': config.protein_encoder_type,
            'drug_encoder_type': config.drug_encoder_type,
            'use_fusion': config.use_fusion,
            'protein_output_dim': config.protein_config.output_dim,
            'drug_output_dim': config.drug_config.output_dim
        }
        
        try:
            # Save main checkpoint
            torch.save(checkpoint, filepath)
            logger.info(f"Checkpoint saved: {filepath}")
            
            # Save latest best if this is the best model
            if is_best:
                torch.save(checkpoint, latest_best_path)
                logger.info(f"Best model checkpoint saved: {latest_best_path}")
            
            # Save separate metrics file for easy analysis
            if self.save_metrics:
                metrics_filename = f'metrics_epoch_{training_state.current_epoch}_phase_{training_state.current_phase}_{timestamp}.json'
                metrics_filepath = self.checkpoint_dir / "metrics" / metrics_filename
                
                metrics_data = {
                    'timestamp': timestamp,
                    'epoch': training_state.current_epoch,
                    'phase': training_state.current_phase,
                    'validation_loss': val_loss,
                    'is_best': is_best,
                    'training_state': training_state.to_dict(),
                    'config': config.to_dict()
                }
                
                if training_state.metrics_history:
                    metrics_data['metrics_history'] = [m.to_dict() for m in training_state.metrics_history]
                
                if additional_metrics:
                    metrics_data['additional_metrics'] = additional_metrics
                
                with open(metrics_filepath, 'w') as f:
                    json.dump(metrics_data, f, indent=2, default=str)
                
                logger.debug(f"Metrics saved: {metrics_filepath}")
            
            # Manage checkpoint files (keep only recent ones)
            if not is_best:
                self.checkpoint_files.append(filepath)
                if len(self.checkpoint_files) > self.max_checkpoints:
                    old_file = self.checkpoint_files.pop(0)
                    if old_file.exists():
                        old_file.unlink()
                        logger.debug(f"Removed old checkpoint: {old_file}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str, 
                       load_optimizer: bool = True,
                       strict_loading: bool = True) -> Dict[str, Any]:
        """Load model checkpoint with backward compatibility"""
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            # Try to find the checkpoint in subdirectories
            if not checkpoint_path.is_absolute():
                for subdir in ["best", "regular"]:
                    full_path = self.checkpoint_dir / subdir / checkpoint_path
                    if full_path.exists():
                        checkpoint_path = full_path
                        break
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            
            # Check version for backward compatibility
            version = checkpoint.get('version', '0.0')  # Default to old version
            
            if version == '0.0':
                # Handle old checkpoint format
                logger.warning("Loading checkpoint from old format, converting...")
                checkpoint = self._convert_old_checkpoint(checkpoint)
            
            # Validate checkpoint integrity
            required_keys = ['model_state_dict', 'training_state', 'config']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                if strict_loading:
                    raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
                else:
                    logger.warning(f"Checkpoint missing keys (continuing anyway): {missing_keys}")
            
            # Validate model compatibility
            if 'model_info' in checkpoint:
                self._validate_model_compatibility(checkpoint['model_info'])
            
            # Remove optimizer state if not requested
            if not load_optimizer:
                checkpoint.pop('optimizer_state_dict', None)
                checkpoint.pop('scheduler_state', None)
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise
    
    def _convert_old_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Convert old checkpoint format to new format"""
        
        # Map old keys to new keys
        key_mapping = {
            'model_state': 'model_state_dict',
            'optimizer_state': 'optimizer_state_dict',
            'val_loss': 'validation_loss'
        }
        
        converted = {}
        
        for old_key, new_key in key_mapping.items():
            if old_key in checkpoint:
                converted[new_key] = checkpoint[old_key]
        
        # Copy other keys as-is
        for key, value in checkpoint.items():
            if key not in key_mapping:
                converted[key] = value
        
        # Add missing fields with defaults
        converted['version'] = '1.0'
        converted['timestamp'] = int(time.time())
        converted['is_best'] = False
        
        return converted
    
    def _validate_model_compatibility(self, model_info: Dict[str, Any]):
        """Validate that checkpoint is compatible with current model architecture"""
        
        # This is a placeholder for model compatibility checking
        # In a real implementation, you would compare the model_info with current config
        logger.debug(f"Model info from checkpoint: {model_info}")
        
        # Example compatibility checks:
        required_fields = ['protein_encoder_type', 'drug_encoder_type']
        for field in required_fields:
            if field not in model_info:
                logger.warning(f"Checkpoint missing model info field: {field}")
    
    def list_checkpoints(self, checkpoint_type: str = "all") -> List[Dict[str, Any]]:
        """List available checkpoints with metadata"""
        
        checkpoints = []
        
        if checkpoint_type in ["all", "regular"]:
            regular_dir = self.checkpoint_dir / "regular"
            if regular_dir.exists():
                for checkpoint_file in regular_dir.glob("*.pth"):
                    checkpoints.append(self._get_checkpoint_info(checkpoint_file, "regular"))
        
        if checkpoint_type in ["all", "best"]:
            best_dir = self.checkpoint_dir / "best"
            if best_dir.exists():
                for checkpoint_file in best_dir.glob("*.pth"):
                    if checkpoint_file.name != "latest_best.pth":  # Skip the symlink
                        checkpoints.append(self._get_checkpoint_info(checkpoint_file, "best"))
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return checkpoints
    
    def _get_checkpoint_info(self, checkpoint_path: Path, checkpoint_type: str) -> Dict[str, Any]:
        """Get metadata about a checkpoint without fully loading it"""
        
        try:
            # Load only the metadata (not the full model state)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            info = {
                'path': str(checkpoint_path),
                'filename': checkpoint_path.name,
                'type': checkpoint_type,
                'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
                'timestamp': checkpoint.get('timestamp', 0),
                'epoch': checkpoint.get('epoch', 0),
                'phase': checkpoint.get('phase', 0),
                'validation_loss': checkpoint.get('validation_loss', float('inf')),
                'is_best': checkpoint.get('is_best', False),
                'version': checkpoint.get('version', '0.0')
            }
            
            # Add performance metrics if available
            if 'recent_performance' in checkpoint:
                info['recent_performance'] = checkpoint['recent_performance']
            
            return info
            
        except Exception as e:
            logger.warning(f"Could not read checkpoint metadata from {checkpoint_path}: {e}")
            return {
                'path': str(checkpoint_path),
                'filename': checkpoint_path.name,
                'type': checkpoint_type,
                'error': str(e)
            }
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to the best checkpoint"""
        
        latest_best = self.checkpoint_dir / "best" / "latest_best.pth"
        if latest_best.exists():
            return latest_best
        
        # Fallback: find best checkpoint by validation loss
        best_checkpoints = self.list_checkpoints("best")
        if best_checkpoints:
            best_checkpoint = min(best_checkpoints, key=lambda x: x.get('validation_loss', float('inf')))
            return Path(best_checkpoint['path'])
        
        return None
    
    def cleanup_old_checkpoints(self, keep_best: int = 3, keep_regular: int = 5):
        """Clean up old checkpoints, keeping only the most recent ones"""
        
        # Clean up regular checkpoints
        regular_checkpoints = self.list_checkpoints("regular")
        if len(regular_checkpoints) > keep_regular:
            to_remove = regular_checkpoints[keep_regular:]
            for checkpoint in to_remove:
                try:
                    Path(checkpoint['path']).unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint['filename']}")
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint {checkpoint['filename']}: {e}")
        
        # Clean up best checkpoints (keep more of these)
        best_checkpoints = self.list_checkpoints("best")
        if len(best_checkpoints) > keep_best:
            to_remove = best_checkpoints[keep_best:]
            for checkpoint in to_remove:
                if checkpoint['filename'] != 'latest_best.pth':  # Never remove the latest best
                    try:
                        Path(checkpoint['path']).unlink()
                        logger.info(f"Removed old best checkpoint: {checkpoint['filename']}")
                    except Exception as e:
                        logger.warning(f"Could not remove checkpoint {checkpoint['filename']}: {e}")
    
    def export_checkpoint_summary(self, output_path: str):
        """Export a summary of all checkpoints to JSON"""
        
        summary = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'total_checkpoints': 0,
            'best_checkpoints': [],
            'regular_checkpoints': [],
            'summary_generated': time.time()
        }
        
        all_checkpoints = self.list_checkpoints("all")
        summary['total_checkpoints'] = len(all_checkpoints)
        
        for checkpoint in all_checkpoints:
            if checkpoint['type'] == 'best':
                summary['best_checkpoints'].append(checkpoint)
            else:
                summary['regular_checkpoints'].append(checkpoint)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Checkpoint summary exported to {output_path}")


class ProgressiveTrainer:
    """2-phase progressive training system for DTA models"""
    
    def __init__(self, model: UnifiedDTAModel, config: DTAConfig, 
                 training_config: TrainingConfig, checkpoint_dir: str = "checkpoints"):
        self.model = model
        self.config = config
        self.training_config = training_config
        
        # Move model to device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            phase1_lr=self.training_config.learning_rate_phase1,
            phase2_lr=self.training_config.learning_rate_phase2
        )
        self.criterion = nn.MSELoss()
        self.early_stopping = EarlyStopping(
            patience=self.training_config.early_stopping_patience
        )
        
        # Training state
        self.training_state = TrainingState()
        
        # Memory management
        self.memory_manager = MemoryManager(
            max_memory_mb=getattr(training_config, 'max_memory_mb', 4000),
            enable_gradient_checkpointing=getattr(training_config, 'enable_gradient_checkpointing', True)
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.memory_manager.enable_gradient_checkpointing_for_model(self.model)
        
        # Checkpoint manager
        self.checkpoint_manager = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            save_interval=self.training_config.checkpoint_interval
        )
        
        logger.info(f"Progressive trainer initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if self.config.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate_phase1,
            weight_decay=self.training_config.weight_decay
        )
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            pearson_corr, _ = pearsonr(predictions, targets)
            spearman_corr, _ = spearmanr(predictions, targets)
        except:
            pearson_corr = 0.0
            spearman_corr = 0.0
        
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'mse': mse,
            'rmse': rmse
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        """Train for one epoch with enhanced memory management"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        memory_reports = []
        current_batch_size = getattr(train_loader, 'batch_size', 4)
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            # Memory monitoring
            if batch_idx % 10 == 0:  # Monitor every 10 batches
                memory_report = self.memory_manager.monitor_memory_during_training(
                    epoch=self.training_state.current_epoch,
                    batch_idx=batch_idx,
                    current_batch_size=current_batch_size
                )
                memory_reports.append(memory_report)
                
                # Handle critical memory situations
                if self.memory_manager.is_memory_critical():
                    logger.warning("Critical memory usage detected, clearing cache aggressively")
                    self.memory_manager.clear_cache(aggressive=True)
            
            # Move batch to device
            try:
                drug_data = batch['drug_data'].to(self.device)
                protein_sequences = batch['protein_sequences']
                targets = batch['affinities'].to(self.device)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("OOM error while moving batch to device")
                    current_batch_size = self.memory_manager.handle_oom_error(current_batch_size)
                    continue
                else:
                    raise
            
            try:
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(drug_data, protein_sequences)
                loss = self.criterion(predictions.squeeze(), targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.training_config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config.gradient_clip_norm
                    )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Regular memory cleanup
                if batch_idx % 5 == 0:
                    self.memory_manager.clear_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM error in batch {batch_idx}")
                    current_batch_size = self.memory_manager.handle_oom_error(current_batch_size)
                    continue
                else:
                    raise
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compile memory statistics for the epoch
        memory_summary = {
            'avg_memory_utilization': np.mean([r['utilization'] for r in memory_reports]) if memory_reports else 0,
            'max_memory_utilization': max([r['utilization'] for r in memory_reports]) if memory_reports else 0,
            'oom_count': self.memory_manager.oom_count,
            'final_batch_size': current_batch_size
        }
        
        return avg_loss, memory_summary
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                
                # Move batch to device
                drug_data = batch['drug_data'].to(self.device)
                protein_sequences = batch['protein_sequences']
                targets = batch['affinities'].to(self.device)
                
                try:
                    # Forward pass
                    predictions = self.model(drug_data, protein_sequences)
                    loss = self.criterion(predictions.squeeze(), targets)
                    
                    total_loss += loss.item()
                    all_predictions.extend(predictions.cpu().numpy().flatten())
                    all_targets.extend(targets.cpu().numpy().flatten())
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM error in validation batch, skipping...")
                        self.memory_manager.clear_cache()
                        continue
                    else:
                        raise
        
        avg_loss = total_loss / max(num_batches, 1)
        val_metrics = self._calculate_metrics(np.array(all_predictions), np.array(all_targets))
        
        return avg_loss, val_metrics
    
    def train_phase(self, train_loader: DataLoader, val_loader: DataLoader,
                   phase: int, num_epochs: int) -> List[TrainingMetrics]:
        """Train a specific phase"""
        logger.info(f"Starting Phase {phase} training for {num_epochs} epochs")
        
        # Set model training phase
        self.model.set_training_phase(phase)
        self.scheduler.set_phase(phase)
        self.training_state.current_phase = phase
        
        phase_metrics = []
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.training_state.current_epoch += 1
            
            # Training
            train_loss, memory_summary = self._train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start_time
            memory_usage = self.memory_manager.get_memory_usage()
            
            metrics = TrainingMetrics(
                epoch=self.training_state.current_epoch,
                phase=phase,
                train_loss=train_loss,
                val_loss=val_loss,
                val_pearson=val_metrics['pearson'],
                val_spearman=val_metrics['spearman'],
                val_rmse=val_metrics['rmse'],
                learning_rate=self.scheduler.get_lr(),
                memory_usage=memory_usage,
                training_time=epoch_time
            )
            
            phase_metrics.append(metrics)
            self.training_state.metrics_history.append(metrics)
            
            # Logging
            logger.info(
                f"Phase {phase} Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Pearson: {val_metrics['pearson']:.4f}, "
                f"RMSE: {val_metrics['rmse']:.4f}, "
                f"LR: {self.scheduler.get_lr():.2e}, "
                f"Memory: {memory_usage:.1f}MB, Time: {epoch_time:.1f}s"
            )
            
            # Check for improvement and update training state
            is_best = val_loss < self.training_state.best_val_loss
            if is_best:
                self.training_state.best_val_loss = val_loss
                self.training_state.best_val_pearson = val_metrics['pearson']
                self.training_state.patience_counter = 0
            else:
                self.training_state.patience_counter += 1
            
            # Prepare additional metrics for checkpointing
            additional_metrics = {
                'memory_summary': memory_summary,
                'val_metrics': val_metrics,
                'epoch_time': epoch_time,
                'learning_rate': self.scheduler.get_lr(),
                'phase': phase
            }
            
            # Save checkpoint (regular interval or best model)
            if (epoch + 1) % self.training_config.checkpoint_interval == 0 or is_best:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.training_state, self.config,
                    val_loss, is_best, additional_metrics
                )
                
                if is_best:
                    logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
            
            # Early stopping check with enhanced functionality
            if self.early_stopping(val_loss, self.training_state.current_epoch, self.model):
                logger.info(f"Early stopping triggered in phase {phase}")
                logger.info(f"Best validation loss: {self.early_stopping.get_best_score():.6f} at epoch {self.early_stopping.get_best_epoch()}")
                
                # Save final checkpoint before stopping
                final_checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.training_state, self.config,
                    val_loss, False, {**additional_metrics, 'early_stopped': True}
                )
                logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
                break
        
        return phase_metrics
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Complete 2-phase progressive training"""
        
        # Resume from checkpoint if specified
        if resume_from:
            self.resume_training(resume_from)
        
        training_start_time = time.time()
        all_metrics = []
        
        try:
            # Phase 1: Frozen ESM training
            if self.training_state.current_phase < 2:
                logger.info("=" * 50)
                logger.info("PHASE 1: Training with frozen ESM-2 weights")
                logger.info("=" * 50)
                
                phase1_metrics = self.train_phase(
                    train_loader, val_loader, 
                    phase=1, 
                    num_epochs=self.training_config.num_epochs_phase1
                )
                all_metrics.extend(phase1_metrics)
                
                # Early stopping check
                if not self.early_stopping.early_stop:
                    # Phase 2: ESM fine-tuning
                    logger.info("=" * 50)
                    logger.info("PHASE 2: Fine-tuning ESM-2 layers")
                    logger.info("=" * 50)
                    
                    # Reset early stopping for phase 2
                    self.early_stopping = EarlyStopping(
                        patience=self.training_config.early_stopping_patience
                    )
                    
                    phase2_metrics = self.train_phase(
                        train_loader, val_loader,
                        phase=2,
                        num_epochs=self.training_config.num_epochs_phase2
                    )
                    all_metrics.extend(phase2_metrics)
            
            # Final checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.training_state, self.config,
                val_loss=self.training_state.best_val_loss, is_best=True
            )
            
            # Memory cleanup
            self.memory_manager.clear_cache()
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            total_training_time = time.time() - training_start_time
            self.training_state.total_training_time = total_training_time
            
            # Training summary
            training_summary = {
                'total_epochs': self.training_state.current_epoch,
                'total_training_time': total_training_time,
                'best_val_loss': self.training_state.best_val_loss,
                'best_val_pearson': self.training_state.best_val_pearson,
                'final_phase': self.training_state.current_phase,
                'metrics_history': [m.to_dict() for m in all_metrics]
            }
            
            logger.info("=" * 50)
            logger.info("TRAINING COMPLETED")
            logger.info("=" * 50)
            logger.info(f"Total epochs: {self.training_state.current_epoch}")
            logger.info(f"Total time: {total_training_time:.1f}s")
            logger.info(f"Best validation loss: {self.training_state.best_val_loss:.4f}")
            logger.info(f"Best validation Pearson: {self.training_state.best_val_pearson:.4f}")
            logger.info("=" * 50)
            
            return training_summary
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Restore model state
        self.model.load_state_dict(checkpoint['model_state'])
        
        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Restore scheduler state
        scheduler_state = checkpoint['scheduler_state']
        self.scheduler.current_phase = scheduler_state['current_phase']
        self.scheduler.epoch = scheduler_state['epoch']
        
        # Restore training state
        training_state = checkpoint['training_state']
        self.training_state = TrainingState(**training_state)
        
        logger.info(f"Resuming training from epoch {self.training_state.current_epoch}, phase {self.training_state.current_phase}")
    
    def save_metrics(self, filepath: str):
        """Save training metrics to file"""
        metrics_data = {
            'config': self.config.to_dict(),
            'training_state': self.training_state.to_dict(),
            'metrics': [m.to_dict() for m in self.training_state.metrics_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Training metrics saved to {filepath}")
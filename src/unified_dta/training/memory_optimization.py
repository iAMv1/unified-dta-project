"""
Advanced Memory Optimization Utilities for Unified DTA System
Provides automatic batch size adjustment, gradient checkpointing, and memory monitoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import psutil
import gc
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from contextlib import contextmanager
import time
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time"""
    timestamp: float
    cpu_memory_mb: float
    gpu_memory_mb: Optional[float]
    gpu_reserved_mb: Optional[float]
    gpu_cached_mb: Optional[float]
    batch_size: int
    epoch: int
    batch_idx: int


class AdaptiveBatchSizer:
    """Automatically adjusts batch size based on memory constraints"""
    
    def __init__(self, initial_batch_size: int = 4, min_batch_size: int = 1, 
                 max_batch_size: int = 32, memory_threshold: float = 0.85):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.current_batch_size = initial_batch_size
        self.adjustment_history = []
        self.oom_count = 0
        
    def adjust_batch_size(self, memory_utilization: float, had_oom: bool = False) -> int:
        """Adjust batch size based on memory utilization"""
        old_batch_size = self.current_batch_size
        
        if had_oom:
            # Aggressive reduction on OOM
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            self.oom_count += 1
            logger.warning(f"OOM detected, reducing batch size: {old_batch_size} -> {self.current_batch_size}")
        
        elif memory_utilization > self.memory_threshold:
            # Gradual reduction when approaching threshold
            self.current_batch_size = max(self.min_batch_size, 
                                        int(self.current_batch_size * 0.8))
            logger.info(f"High memory usage ({memory_utilization:.1%}), reducing batch size: {old_batch_size} -> {self.current_batch_size}")
        
        elif memory_utilization < 0.5 and self.current_batch_size < self.max_batch_size:
            # Gradual increase when memory is underutilized
            self.current_batch_size = min(self.max_batch_size, 
                                        int(self.current_batch_size * 1.2))
            logger.info(f"Low memory usage ({memory_utilization:.1%}), increasing batch size: {old_batch_size} -> {self.current_batch_size}")
        
        # Record adjustment
        if old_batch_size != self.current_batch_size:
            self.adjustment_history.append({
                'timestamp': time.time(),
                'old_size': old_batch_size,
                'new_size': self.current_batch_size,
                'memory_utilization': memory_utilization,
                'had_oom': had_oom
            })
        
        return self.current_batch_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch size adjustment statistics"""
        return {
            'current_batch_size': self.current_batch_size,
            'initial_batch_size': self.initial_batch_size,
            'oom_count': self.oom_count,
            'total_adjustments': len(self.adjustment_history),
            'adjustment_history': self.adjustment_history[-10:]  # Last 10 adjustments
        }


class GradientCheckpointManager:
    """Manages gradient checkpointing for memory efficiency"""
    
    def __init__(self, enable_by_default: bool = True):
        self.enable_by_default = enable_by_default
        self.checkpointed_modules = []
        
    def enable_for_model(self, model: nn.Module, module_types: Optional[List[type]] = None):
        """Enable gradient checkpointing for specific module types"""
        if module_types is None:
            # Default module types that benefit from checkpointing
            module_types = [nn.TransformerEncoder, nn.TransformerDecoder]
        
        enabled_count = 0
        
        # Enable for ESM model if present
        if hasattr(model, 'protein_encoder') and hasattr(model.protein_encoder, 'esm_model'):
            esm_model = model.protein_encoder.esm_model
            if hasattr(esm_model, 'gradient_checkpointing_enable'):
                esm_model.gradient_checkpointing_enable()
                self.checkpointed_modules.append('esm_model')
                enabled_count += 1
                logger.info("Gradient checkpointing enabled for ESM model")
        
        # Enable for other transformer-based modules
        for name, module in model.named_modules():
            if any(isinstance(module, mt) for mt in module_types):
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
                    self.checkpointed_modules.append(name)
                    enabled_count += 1
        
        logger.info(f"Gradient checkpointing enabled for {enabled_count} modules")
        return enabled_count
    
    def disable_all(self, model: nn.Module):
        """Disable gradient checkpointing for all modules"""
        disabled_count = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'gradient_checkpointing_disable'):
                module.gradient_checkpointing_disable()
                disabled_count += 1
        
        self.checkpointed_modules.clear()
        logger.info(f"Gradient checkpointing disabled for {disabled_count} modules")


class MemoryProfiler:
    """Profiles memory usage during training"""
    
    def __init__(self, profile_interval: int = 10, max_snapshots: int = 1000):
        self.profile_interval = profile_interval
        self.max_snapshots = max_snapshots
        self.snapshots: List[MemorySnapshot] = []
        self.profiling_enabled = True
        
    def take_snapshot(self, batch_size: int, epoch: int, batch_idx: int) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        if not self.profiling_enabled:
            return None
        
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
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            cpu_memory_mb=cpu_memory,
            gpu_memory_mb=gpu_memory,
            gpu_reserved_mb=gpu_reserved,
            gpu_cached_mb=gpu_cached,
            batch_size=batch_size,
            epoch=epoch,
            batch_idx=batch_idx
        )
        
        self.snapshots.append(snapshot)
        
        # Limit snapshot history
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        
        return snapshot
    
    def should_profile(self, batch_idx: int) -> bool:
        """Check if profiling should be done for this batch"""
        return self.profiling_enabled and (batch_idx % self.profile_interval == 0)
    
    def get_memory_trend(self, window_size: int = 50) -> Dict[str, float]:
        """Analyze memory usage trend"""
        if len(self.snapshots) < window_size:
            return {'trend': 0.0, 'confidence': 0.0}
        
        recent_snapshots = self.snapshots[-window_size:]
        timestamps = [s.timestamp for s in recent_snapshots]
        total_memory = [s.cpu_memory_mb + (s.gpu_memory_mb or 0) for s in recent_snapshots]
        
        # Linear regression to find trend
        x = np.array(timestamps) - timestamps[0]  # Normalize timestamps
        y = np.array(total_memory)
        
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            correlation = np.corrcoef(x, y)[0, 1]
            
            return {
                'trend': slope,  # MB per second
                'confidence': abs(correlation),
                'avg_memory': np.mean(total_memory),
                'memory_std': np.std(total_memory)
            }
        
        return {'trend': 0.0, 'confidence': 0.0}
    
    def detect_memory_leak(self, threshold_mb_per_hour: float = 100) -> bool:
        """Detect potential memory leaks"""
        trend_info = self.get_memory_trend()
        
        if trend_info['confidence'] > 0.7:  # High confidence in trend
            trend_per_hour = trend_info['trend'] * 3600  # Convert to MB/hour
            return trend_per_hour > threshold_mb_per_hour
        
        return False
    
    def save_profile(self, filepath: str):
        """Save memory profile to file"""
        profile_data = {
            'snapshots': [
                {
                    'timestamp': s.timestamp,
                    'cpu_memory_mb': s.cpu_memory_mb,
                    'gpu_memory_mb': s.gpu_memory_mb,
                    'gpu_reserved_mb': s.gpu_reserved_mb,
                    'gpu_cached_mb': s.gpu_cached_mb,
                    'batch_size': s.batch_size,
                    'epoch': s.epoch,
                    'batch_idx': s.batch_idx
                }
                for s in self.snapshots
            ],
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"Memory profile saved to {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        if not self.snapshots:
            return {}
        
        total_memory = [s.cpu_memory_mb + (s.gpu_memory_mb or 0) for s in self.snapshots]
        gpu_memory = [s.gpu_memory_mb for s in self.snapshots if s.gpu_memory_mb is not None]
        
        summary = {
            'total_snapshots': len(self.snapshots),
            'duration_hours': (self.snapshots[-1].timestamp - self.snapshots[0].timestamp) / 3600,
            'memory_stats': {
                'avg_total_mb': np.mean(total_memory),
                'max_total_mb': np.max(total_memory),
                'min_total_mb': np.min(total_memory),
                'std_total_mb': np.std(total_memory)
            },
            'trend_analysis': self.get_memory_trend(),
            'potential_leak': self.detect_memory_leak()
        }
        
        if gpu_memory:
            summary['gpu_memory_stats'] = {
                'avg_gpu_mb': np.mean(gpu_memory),
                'max_gpu_mb': np.max(gpu_memory),
                'min_gpu_mb': np.min(gpu_memory),
                'std_gpu_mb': np.std(gpu_memory)
            }
        
        return summary


@contextmanager
def memory_efficient_forward(model: nn.Module, enable_checkpointing: bool = True):
    """Context manager for memory-efficient forward passes"""
    original_training = model.training
    
    try:
        if enable_checkpointing:
            # Enable gradient checkpointing temporarily
            checkpoint_manager = GradientCheckpointManager()
            checkpoint_manager.enable_for_model(model)
        
        yield model
        
    finally:
        # Restore original state
        model.train(original_training)
        
        # Clear cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MemoryOptimizedDataLoader:
    """Wrapper for DataLoader with automatic memory optimization"""
    
    def __init__(self, dataset, initial_batch_size: int = 4, max_memory_mb: float = 4000,
                 **dataloader_kwargs):
        self.dataset = dataset
        self.initial_batch_size = initial_batch_size
        self.max_memory_mb = max_memory_mb
        self.dataloader_kwargs = dataloader_kwargs
        
        # Initialize adaptive batch sizer
        self.batch_sizer = AdaptiveBatchSizer(
            initial_batch_size=initial_batch_size,
            min_batch_size=1,
            max_batch_size=min(32, initial_batch_size * 4)
        )
        
        # Create initial dataloader
        self.current_dataloader = self._create_dataloader(initial_batch_size)
        self.memory_profiler = MemoryProfiler()
        
    def _create_dataloader(self, batch_size: int) -> DataLoader:
        """Create DataLoader with specified batch size"""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            **self.dataloader_kwargs
        )
    
    def _get_memory_utilization(self) -> float:
        """Get current memory utilization"""
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024
        gpu_memory = 0
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        total_memory = cpu_memory + gpu_memory
        return total_memory / self.max_memory_mb
    
    def __iter__(self):
        """Iterate with automatic batch size adjustment"""
        epoch = 0
        
        for batch_idx, batch in enumerate(self.current_dataloader):
            # Profile memory usage
            if self.memory_profiler.should_profile(batch_idx):
                snapshot = self.memory_profiler.take_snapshot(
                    batch_size=self.batch_sizer.current_batch_size,
                    epoch=epoch,
                    batch_idx=batch_idx
                )
                
                # Check for memory optimization opportunities
                memory_utilization = self._get_memory_utilization()
                new_batch_size = self.batch_sizer.adjust_batch_size(memory_utilization)
                
                # Recreate dataloader if batch size changed
                if new_batch_size != self.current_dataloader.batch_size:
                    logger.info(f"Recreating DataLoader with batch size {new_batch_size}")
                    self.current_dataloader = self._create_dataloader(new_batch_size)
                    # Note: This will restart the iteration, which might not be ideal
                    # In practice, batch size changes should be handled at epoch boundaries
            
            yield batch
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            'batch_sizer_stats': self.batch_sizer.get_statistics(),
            'memory_profile_summary': self.memory_profiler.get_summary()
        }


def optimize_model_for_memory(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """Apply various memory optimizations to a model"""
    
    # Enable gradient checkpointing
    if config.get('enable_gradient_checkpointing', True):
        checkpoint_manager = GradientCheckpointManager()
        checkpoint_manager.enable_for_model(model)
    
    # Convert to half precision if requested
    if config.get('use_half_precision', False) and torch.cuda.is_available():
        model = model.half()
        logger.info("Model converted to half precision")
    
    # Enable memory efficient attention if available
    if config.get('enable_memory_efficient_attention', True):
        for module in model.modules():
            if hasattr(module, 'enable_memory_efficient_attention'):
                module.enable_memory_efficient_attention()
    
    return model


def get_memory_recommendations(memory_stats: Dict[str, float], 
                             model_size_mb: float) -> List[str]:
    """Get memory optimization recommendations"""
    recommendations = []
    
    total_memory = memory_stats.get('total_memory_mb', 0)
    gpu_memory = memory_stats.get('gpu_memory_mb', 0)
    
    # High memory usage
    if total_memory > 8000:  # > 8GB
        recommendations.append("Consider reducing batch size or using gradient checkpointing")
    
    # GPU memory specific
    if gpu_memory > 6000:  # > 6GB GPU memory
        recommendations.append("Enable gradient checkpointing for transformer models")
        recommendations.append("Consider using mixed precision training")
    
    # Model size vs available memory
    if model_size_mb > total_memory * 0.5:
        recommendations.append("Model is large relative to available memory - consider model pruning")
    
    # Memory fragmentation
    gpu_reserved = memory_stats.get('gpu_reserved_mb', 0)
    gpu_cached = memory_stats.get('gpu_cached_mb', 0)
    
    if gpu_reserved > gpu_memory * 1.5:
        recommendations.append("High GPU memory fragmentation detected - restart training")
    
    if gpu_cached > 1000:  # > 1GB cached
        recommendations.append("Clear GPU cache regularly to free up memory")
    
    return recommendations
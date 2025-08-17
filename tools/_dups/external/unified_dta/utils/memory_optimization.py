"""
Memory optimization utilities for the Unified DTA System
"""

import torch
import psutil
import gc
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Utilities for memory optimization and monitoring"""
    
    def __init__(self):
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
        }
        
        # Add GPU memory if available
        if torch.cuda.is_available():
            stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return stats
    
    def cleanup_memory(self):
        """Perform memory cleanup"""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Memory cleanup performed")
    
    def get_optimal_batch_size(self, 
                              model: torch.nn.Module,
                              sample_input: Any,
                              max_memory_mb: float = 4000) -> int:
        """Estimate optimal batch size based on memory constraints"""
        
        # Start with batch size 1 and measure memory
        model.eval()
        
        with torch.no_grad():
            initial_memory = self.get_memory_usage()
            
            # Test with batch size 1
            try:
                if hasattr(sample_input, '__len__') and len(sample_input) == 2:
                    # Assume (drug_data, protein_data) tuple
                    _ = model(sample_input[0], sample_input[1])
                else:
                    _ = model(sample_input)
                
                after_memory = self.get_memory_usage()
                memory_per_sample = after_memory['rss_mb'] - initial_memory['rss_mb']
                
                if memory_per_sample > 0:
                    optimal_batch_size = int(max_memory_mb / memory_per_sample)
                    return max(1, min(optimal_batch_size, 32))  # Cap at 32
                
            except Exception as e:
                logger.warning(f"Could not estimate batch size: {e}")
        
        return 4  # Default fallback
    
    def monitor_memory(self, operation_name: str = ""):
        """Context manager for memory monitoring"""
        return MemoryMonitor(self, operation_name)


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations"""
    
    def __init__(self, optimizer: MemoryOptimizer, operation_name: str):
        self.optimizer = optimizer
        self.operation_name = operation_name
        self.start_memory = None
    
    def __enter__(self):
        self.start_memory = self.optimizer.get_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = self.optimizer.get_memory_usage()
        memory_diff = end_memory['rss_mb'] - self.start_memory['rss_mb']
        
        logger.info(f"Memory usage for {self.operation_name}: "
                   f"{memory_diff:.2f} MB (Total: {end_memory['rss_mb']:.2f} MB)")
"""
Model caching and loading mechanisms for the API
"""

import torch
import asyncio
import logging
from typing import Dict, Optional, Any
from pathlib import Path
import psutil
import time
from threading import Lock
import os
import glob

from ..core.model_factory import ModelFactory
from ..core.models import UnifiedDTAModel
from ..utils.config_consistency import ConfigConsistencyChecker

from ..core.model_factory import ModelFactory
from ..core.models import UnifiedDTAModel

logger = logging.getLogger(__name__)


class ModelCache:
    """Thread-safe model cache with lazy loading and memory management"""
    
    def __init__(self, max_models: int = 3, device: str = "auto"):
        self.max_models = max_models
        self.device = self._determine_device(device)
        self.models: Dict[str, UnifiedDTAModel] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = Lock()
        
        logger.info(f"ModelCache initialized with device: {self.device}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    async def get_model(self, model_type: str, config: Optional[Dict[str, Any]] = None) -> UnifiedDTAModel:
        """Get model from cache or load it"""
        with self.lock:
            # Update access time
            self.access_times[model_type] = time.time()
            
            # Return cached model if available
            if model_type in self.models:
                logger.debug(f"Retrieved cached model: {model_type}")
                return self.models[model_type]
            
            # Load new model
            logger.info(f"Loading new model: {model_type}")
            model = await self._load_model(model_type, config)
            
            # Manage cache size
            if len(self.models) >= self.max_models:
                self._evict_oldest_model()
            
            # Cache the model
            self.models[model_type] = model
            self.model_info[model_type] = self._get_model_info(model, model_type)
            
            return model
    
    async def _load_model(self, model_type: str, config: Optional[Dict[str, Any]] = None) -> UnifiedDTAModel:
        """Load model asynchronously"""
        loop = asyncio.get_event_loop()
        
        def _load():
            try:
                # Check if there's a trained model checkpoint available
                checkpoint_path = self._find_model_checkpoint(model_type)
                
                if checkpoint_path and os.path.exists(checkpoint_path):
                    # Load trained model from checkpoint
                    logger.info(f"Loading trained model from checkpoint: {checkpoint_path}")
                    model = self._load_from_checkpoint(checkpoint_path, model_type, config)
                else:
                    # Create model using factory (fallback to untrained model)
                    logger.info(f"No checkpoint found, creating new model: {model_type}")
                    if config:
                        model = ModelFactory.create_model("custom", config)
                    else:
                        model = ModelFactory.create_model(model_type)
                
                # Move to device
                model = model.to(self.device)
                model.eval()
                
                logger.info(f"Model {model_type} loaded successfully on {self.device}")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_type}: {str(e)}")
                raise
        
        return await loop.run_in_executor(None, _load)
    
    def _find_model_checkpoint(self, model_type: str) -> Optional[str]:
        """Find the best checkpoint for a model type"""
        # Define standard checkpoint locations
        checkpoint_dirs = [
            "checkpoints/best",
            "models/checkpoints/best",
            "data/checkpoints/best",
            "checkpoints",
            "models/checkpoints"
        ]
        
        # Standard checkpoint filename patterns
        patterns = [
            f"best_model_*_{model_type}.pth",
            f"best_model_{model_type}.pth",
            f"*{model_type}*.pth",
            "latest_best.pth"
        ]
        
        import os
        import glob
        
        for checkpoint_dir in checkpoint_dirs:
            if os.path.exists(checkpoint_dir):
                for pattern in patterns:
                    full_pattern = os.path.join(checkpoint_dir, pattern)
                    matches = glob.glob(full_pattern)
                    if matches:
                        # Return the most recent match
                        return max(matches, key=os.path.getctime)
        
        return None
    
    def _load_from_checkpoint(self, checkpoint_path: str, model_type: str, config: Optional[Dict[str, Any]] = None) -> UnifiedDTAModel:
        """Load model from checkpoint file"""
        try:
            import torch
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Create model
            if config:
                model = ModelFactory.create_model("custom", config)
            else:
                model = ModelFactory.create_model(model_type)
            
            # Check configuration consistency
            if config:
                checker = ConfigConsistencyChecker()
                training_config = checkpoint.get('config', {})
                if training_config:
                    validation_result = checker.validate_config_consistency(training_config, config)
                    if not validation_result['consistent']:
                        logger.warning("Configuration inconsistencies detected:")
                        for inconsistency in validation_result['inconsistencies']:
                            logger.warning(f"  - {inconsistency}")
            
            # Load state dict
            # Handle potential key mismatches
            model_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            
            if len(filtered_state_dict) == 0:
                logger.warning("No matching keys found in checkpoint, initializing with random weights")
            else:
                model_dict.update(filtered_state_dict)
                model.load_state_dict(model_dict)
                logger.info(f"Loaded {len(filtered_state_dict)}/{len(model_dict)} parameters from checkpoint")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from checkpoint {checkpoint_path}: {str(e)}")
            # Fallback to creating a new model
            if config:
                return ModelFactory.create_model("custom", config)
            else:
                return ModelFactory.create_model(model_type)
    
    def _evict_oldest_model(self):
        """Evict the least recently used model"""
        if not self.models:
            return
        
        oldest_model = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        logger.info(f"Evicting model from cache: {oldest_model}")
        
        # Clean up
        del self.models[oldest_model]
        del self.model_info[oldest_model]
        del self.access_times[oldest_model]
        
        # Force garbage collection
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _get_model_info(self, model: UnifiedDTAModel, model_type: str) -> Dict[str, Any]:
        """Get information about the model"""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate memory usage
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            memory_mb = (param_size + buffer_size) / (1024 * 1024)
            
            return {
                "model_type": model_type,
                "protein_encoder": getattr(model, 'protein_encoder_type', 'unknown'),
                "drug_encoder": "gin",  # Default assumption
                "uses_fusion": hasattr(model, 'fusion') and model.fusion is not None,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "memory_usage_mb": memory_mb,
                "device": str(next(model.parameters()).device)
            }
        except Exception as e:
            logger.warning(f"Could not get model info: {str(e)}")
            return {"model_type": model_type, "error": str(e)}
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models"""
        with self.lock:
            return {
                "cached_models": list(self.models.keys()),
                "model_info": self.model_info.copy(),
                "cache_size": len(self.models),
                "max_cache_size": self.max_models,
                "device": self.device
            }
    
    def clear_cache(self):
        """Clear all cached models"""
        with self.lock:
            logger.info("Clearing model cache")
            self.models.clear()
            self.model_info.clear()
            self.access_times.clear()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory_info = {}
        
        # System memory
        memory = psutil.virtual_memory()
        memory_info["system_memory_used_mb"] = memory.used / (1024 * 1024)
        memory_info["system_memory_available_mb"] = memory.available / (1024 * 1024)
        memory_info["system_memory_percent"] = memory.percent
        
        # GPU memory if available
        if torch.cuda.is_available():
            memory_info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            memory_info["gpu_memory_max_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        return memory_info


# Global model cache instance
_model_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Get the global model cache instance"""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


def initialize_cache(max_models: int = 3, device: str = "auto") -> ModelCache:
    """Initialize the global model cache"""
    global _model_cache
    _model_cache = ModelCache(max_models=max_models, device=device)
    return _model_cache
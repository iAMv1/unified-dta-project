"""
Checkpoint management utilities for the Unified DTA System
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints and training state"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       config: Optional[Dict[str, Any]] = None,
                       filename: Optional[str] = None) -> Path:
        """Save model checkpoint"""
        
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, 
                       checkpoint_path: Path,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Load model checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the latest checkpoint file"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if not checkpoints:
            return None
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return checkpoints[-1]
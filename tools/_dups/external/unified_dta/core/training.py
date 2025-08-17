"""
Training infrastructure for the Unified DTA System
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
import logging
from pathlib import Path
from tqdm import tqdm

from .models import UnifiedDTAModel
from .config import Config
from ..utils.checkpoint_utils import CheckpointManager
from ..utils.memory_optimization import MemoryOptimizer


logger = logging.getLogger(__name__)


class DTATrainer:
    """Trainer class for DTA models with 2-phase progressive training"""
    
    def __init__(self,
                 model: UnifiedDTAModel,
                 config: Config,
                 checkpoint_dir: Optional[Path] = None):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        # Checkpoint management
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        else:
            self.checkpoint_manager = None
        
        # Memory optimization
        self.memory_optimizer = MemoryOptimizer()
        
        # Training state
        self.current_epoch = 0
        self.current_phase = 1
        self.best_loss = float('inf')
        
    def setup_optimizer(self, phase: int = 1):
        """Setup optimizer for training phase"""
        if phase == 1:
            # Phase 1: Frozen ESM, train other components
            lr = self.config.learning_rate
        else:
            # Phase 2: Fine-tune ESM layers
            lr = self.config.learning_rate * 0.1
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=1e-5
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            drug_data = batch['drug_data'].to(self.device)
            protein_data = batch['protein_data']
            targets = batch['affinity'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(drug_data, protein_data)
            loss = self.criterion(predictions.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Memory cleanup
            if num_batches % 10 == 0:
                self.memory_optimizer.cleanup()
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                drug_data = batch['drug_data'].to(self.device)
                protein_data = batch['protein_data']
                targets = batch['affinity'].to(self.device)
                
                predictions = self.model(drug_data, protein_data)
                loss = self.criterion(predictions.squeeze(), targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = None) -> Dict[str, Any]:
        """Main training loop with 2-phase progressive training"""
        
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        training_history = {'train_loss': [], 'val_loss': []}
        
        # Phase 1: Train with frozen ESM
        logger.info("Starting Phase 1: Training with frozen ESM")
        self.current_phase = 1
        self.setup_optimizer(phase=1)
        
        phase1_epochs = num_epochs // 2
        
        for epoch in range(phase1_epochs):
            self.current_epoch = epoch
            
            # Train and validate
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, "
                       f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            training_history['train_loss'].append(train_metrics['train_loss'])
            training_history['val_loss'].append(val_metrics['val_loss'])
            
            # Save checkpoint
            if self.checkpoint_manager and val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, val_metrics['val_loss'])
        
        # Phase 2: Fine-tune ESM layers
        logger.info("Starting Phase 2: Fine-tuning ESM layers")
        self.current_phase = 2
        self.model.set_training_phase(2)
        self.setup_optimizer(phase=2)
        
        for epoch in range(phase1_epochs, num_epochs):
            self.current_epoch = epoch
            
            # Train and validate
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, "
                       f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            training_history['train_loss'].append(train_metrics['train_loss'])
            training_history['val_loss'].append(val_metrics['val_loss'])
            
            # Save checkpoint
            if self.checkpoint_manager and val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, val_metrics['val_loss'])
        
        return training_history
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        if self.checkpoint_manager:
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': epoch,
                'phase': self.current_phase,
                'val_loss': val_loss,
                'config': self.config.to_dict()
            }
            self.checkpoint_manager.save_checkpoint(checkpoint_data, epoch)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        if self.checkpoint_manager:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            if self.optimizer:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            self.current_epoch = checkpoint_data['epoch']
            self.current_phase = checkpoint_data['phase']
            
            logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
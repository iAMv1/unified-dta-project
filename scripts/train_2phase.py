"""
2-Phase Progressive Training Script for Unified DTA System
Demonstrates the complete training workflow with Phase 1 (frozen ESM) and Phase 2 (fine-tuning)
"""

import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader

from core.config import DTAConfig, TrainingConfig
from core.models import get_production_model
from core.training import ProgressiveTrainer
from core.datasets import DTADataset
from core.data_processing import DTADataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    
    # Configuration
    config = DTAConfig(
        device='auto',
        batch_size=4,  # Small batch size for memory efficiency
        num_workers=2
    )
    
    training_config = TrainingConfig(
        num_epochs_phase1=10,  # Phase 1: Frozen ESM training
        num_epochs_phase2=5,   # Phase 2: ESM fine-tuning
        learning_rate_phase1=1e-3,
        learning_rate_phase2=1e-4,
        early_stopping_patience=5,
        checkpoint_interval=2,
        gradient_clip_norm=1.0,
        weight_decay=1e-4
    )
    
    # Create model
    logger.info("Creating production model...")
    model = get_production_model()
    
    # Initialize trainer
    trainer = ProgressiveTrainer(
        model=model,
        config=config,
        training_config=training_config,
        checkpoint_dir="checkpoints/2phase_training"
    )
    
    # Load data (placeholder - replace with actual data loading)
    logger.info("Loading datasets...")
    # train_dataset = DTADataset(...)
    # val_dataset = DTADataset(...)
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # For demonstration, create dummy loaders
    train_loader = None  # Replace with actual data loader
    val_loader = None    # Replace with actual data loader
    
    if train_loader is None:
        logger.warning("No data loaders provided. Please implement data loading.")
        return
    
    # Start training
    logger.info("Starting 2-phase progressive training...")
    training_summary = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Save final metrics
    trainer.save_metrics("training_metrics.json")
    
    logger.info("Training completed successfully!")
    logger.info(f"Final results: {training_summary}")


if __name__ == "__main__":
    main()
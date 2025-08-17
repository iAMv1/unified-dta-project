"""
Test script for 2-phase progressive training system
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from pathlib import Path

from core.config import DTAConfig, TrainingConfig
from core.models import get_production_model, get_lightweight_model
from core.training import ProgressiveTrainer, EarlyStopping, LearningRateScheduler
from core.data_processing import create_dummy_batch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_data_loader(num_samples: int = 100, batch_size: int = 4):
    """Create dummy data loader for testing"""
    
    # Create dummy molecular graphs (simplified)
    dummy_features = torch.randn(num_samples, 10, 78)  # 10 nodes, 78 features
    dummy_edge_indices = torch.randint(0, 10, (num_samples, 2, 20))  # 20 edges
    dummy_batch_indices = torch.arange(num_samples).repeat_interleave(10)
    
    # Create dummy protein sequences
    dummy_proteins = ["MKLLVLSLSLVLVAPMAAQAAEITLVPSVKLQIGDRDNRGYYWDGGHWRDH"] * num_samples
    
    # Create dummy affinities
    dummy_affinities = torch.randn(num_samples)
    
    # Create dataset
    dataset = TensorDataset(dummy_features, dummy_affinities)
    
    def collate_fn(batch):
        """Custom collate function"""
        features, affinities = zip(*batch)
        
        # Create dummy batch structure
        batch_data = {
            'drug_data': create_dummy_batch(len(features)),
            'protein_sequences': dummy_proteins[:len(features)],
            'affinities': torch.stack(affinities)
        }
        return batch_data
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def test_early_stopping():
    """Test early stopping functionality"""
    logger.info("Testing early stopping...")
    
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    # Simulate improving scores
    scores = [1.0, 0.8, 0.6, 0.65, 0.64, 0.63, 0.62]  # Should trigger early stopping
    
    for i, score in enumerate(scores):
        should_stop = early_stopping(score)
        logger.info(f"Epoch {i+1}: Score {score:.3f}, Counter: {early_stopping.counter}, Stop: {should_stop}")
        if should_stop:
            break
    
    assert early_stopping.early_stop, "Early stopping should have been triggered"
    logger.info("✓ Early stopping test passed")


def test_learning_rate_scheduler():
    """Test learning rate scheduler"""
    logger.info("Testing learning rate scheduler...")
    
    # Create dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = LearningRateScheduler(
        optimizer, 
        phase1_lr=1e-3, 
        phase2_lr=1e-4,
        warmup_epochs=3
    )
    
    # Test phase 1
    scheduler.set_phase(1)
    initial_lr = scheduler.get_lr()
    logger.info(f"Phase 1 initial LR: {initial_lr:.2e}")
    
    # Step through warmup
    for epoch in range(5):
        scheduler.step()
        lr = scheduler.get_lr()
        logger.info(f"Phase 1 Epoch {epoch+1}: LR = {lr:.2e}")
    
    # Test phase 2
    scheduler.set_phase(2)
    phase2_lr = scheduler.get_lr()
    logger.info(f"Phase 2 initial LR: {phase2_lr:.2e}")
    
    assert phase2_lr < initial_lr, "Phase 2 LR should be lower than Phase 1"
    logger.info("✓ Learning rate scheduler test passed")


def test_2phase_training():
    """Test complete 2-phase training workflow"""
    logger.info("Testing 2-phase training workflow...")
    
    # Use lightweight model for faster testing
    model = get_lightweight_model()
    
    # Configuration
    config = DTAConfig(device='cpu')
    training_config = TrainingConfig(
        batch_size=2,
        num_epochs_phase1=2,
        num_epochs_phase2=2,
        learning_rate_phase1=1e-3,
        learning_rate_phase2=1e-4,
        early_stopping_patience=10,  # High patience to avoid early stopping in test
        checkpoint_interval=1
    )
    
    # Create trainer
    trainer = ProgressiveTrainer(
        model=model,
        config=config,
        training_config=training_config,
        checkpoint_dir="test_checkpoints"
    )
    
    # Create dummy data
    train_loader = create_dummy_data_loader(num_samples=20, batch_size=2)
    val_loader = create_dummy_data_loader(num_samples=10, batch_size=2)
    
    # Test training
    try:
        training_summary = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Verify training completed
        assert training_summary['total_epochs'] > 0, "Training should complete some epochs"
        assert training_summary['final_phase'] >= 1, "Should complete at least phase 1"
        
        logger.info(f"Training summary: {training_summary}")
        logger.info("✓ 2-phase training test passed")
        
    except Exception as e:
        logger.error(f"Training test failed: {e}")
        raise
    
    # Cleanup
    import shutil
    if Path("test_checkpoints").exists():
        shutil.rmtree("test_checkpoints")


def test_model_phase_switching():
    """Test model phase switching functionality"""
    logger.info("Testing model phase switching...")
    
    model = get_production_model()
    
    # Check initial state (ESM should be frozen)
    esm_encoder = model.protein_encoder
    if hasattr(esm_encoder, 'esm_model'):
        # Check if ESM parameters are frozen initially
        esm_params_frozen = not any(p.requires_grad for p in esm_encoder.esm_model.parameters())
        logger.info(f"ESM parameters frozen initially: {esm_params_frozen}")
        
        # Switch to phase 2 (should unfreeze some ESM layers)
        model.set_training_phase(2)
        
        # Check if some ESM parameters are now unfrozen
        esm_params_unfrozen = any(p.requires_grad for p in esm_encoder.esm_model.parameters())
        logger.info(f"Some ESM parameters unfrozen after phase 2: {esm_params_unfrozen}")
        
        assert esm_params_unfrozen, "Some ESM parameters should be unfrozen in phase 2"
        logger.info("✓ Model phase switching test passed")
    else:
        logger.info("Model doesn't have ESM encoder, skipping phase switching test")


def main():
    """Run all tests"""
    logger.info("Starting 2-phase training system tests...")
    
    try:
        test_early_stopping()
        test_learning_rate_scheduler()
        test_model_phase_switching()
        test_2phase_training()
        
        logger.info("=" * 50)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("2-phase training system is working correctly")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
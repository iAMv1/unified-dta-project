"""
Basic test for checkpoint functionality
Simple verification that the checkpoint system works
"""

import torch
import tempfile
import shutil
from pathlib import Path

from core.config import DTAConfig
from core.models import get_lightweight_model
from core.checkpoint_utils import CheckpointValidator, CheckpointManager
from core.training import ModelCheckpoint, TrainingState


def test_basic_checkpoint_functionality():
    """Test basic checkpoint save/load functionality"""
    print("Testing basic checkpoint functionality...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Setup
        config = DTAConfig()
        model = get_lightweight_model(config)
        
        # Create checkpoint manager
        checkpoint_manager = ModelCheckpoint(str(temp_dir))
        
        # Create mock training state
        training_state = TrainingState(
            current_epoch=5,
            current_phase=1,
            best_val_loss=0.4,
            patience_counter=2
        )
        
        # Create mock optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create mock scheduler
        from core.training import LearningRateScheduler
        scheduler = LearningRateScheduler(optimizer)
        
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            config=config,
            val_loss=0.4,
            is_best=True
        )
        
        print(f"✓ Checkpoint saved to: {checkpoint_path}")
        
        # Verify checkpoint file exists
        assert checkpoint_path.exists(), "Checkpoint file was not created"
        
        # Load and validate checkpoint
        checkpoint = checkpoint_manager.load_checkpoint(str(checkpoint_path))
        
        print("✓ Checkpoint loaded successfully")
        
        # Verify checkpoint contents
        assert 'model_state_dict' in checkpoint, "Missing model state dict"
        assert 'config' in checkpoint, "Missing config"
        assert 'training_state' in checkpoint, "Missing training state"
        assert checkpoint['epoch'] == 5, "Incorrect epoch"
        assert checkpoint['validation_loss'] == 0.4, "Incorrect validation loss"
        
        print("✓ Checkpoint contents validated")
        
        # Test checkpoint validation
        validator = CheckpointValidator()
        validation_result = validator.validate_checkpoint_file(checkpoint_path)
        
        assert validation_result['valid'], "Checkpoint validation failed"
        print("✓ Checkpoint validation passed")
        
        # Test checkpoint manager
        manager = CheckpointManager(str(temp_dir))
        all_checkpoints = manager.list_all_checkpoints()
        
        assert len(all_checkpoints['valid']) >= 1, "No valid checkpoints found"
        print("✓ Checkpoint manager working")
        
        print("✓ All basic checkpoint tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_checkpoint_validation():
    """Test checkpoint validation functionality"""
    print("\nTesting checkpoint validation...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        config = DTAConfig()
        model = get_lightweight_model(config)
        
        # Create valid checkpoint
        valid_checkpoint_path = temp_dir / "valid.pth"
        valid_checkpoint = {
            'version': '1.0',
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'epoch': 10,
            'validation_loss': 0.3
        }
        torch.save(valid_checkpoint, valid_checkpoint_path)
        
        # Create invalid checkpoint
        invalid_checkpoint_path = temp_dir / "invalid.pth"
        with open(invalid_checkpoint_path, 'wb') as f:
            f.write(b"invalid data")
        
        validator = CheckpointValidator()
        
        # Test valid checkpoint
        valid_result = validator.validate_checkpoint_file(valid_checkpoint_path)
        assert valid_result['valid'], "Valid checkpoint marked as invalid"
        print("✓ Valid checkpoint correctly validated")
        
        # Test invalid checkpoint
        invalid_result = validator.validate_checkpoint_file(invalid_checkpoint_path)
        assert not invalid_result['valid'], "Invalid checkpoint marked as valid"
        assert len(invalid_result['errors']) > 0, "No errors reported for invalid checkpoint"
        print("✓ Invalid checkpoint correctly identified")
        
        print("✓ Checkpoint validation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        return False
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run basic checkpoint tests"""
    print("=" * 50)
    print("BASIC CHECKPOINT SYSTEM TESTS")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    # Run tests
    if test_basic_checkpoint_functionality():
        tests_passed += 1
    
    if test_checkpoint_validation():
        tests_passed += 1
    
    # Results
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Checkpoint system is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
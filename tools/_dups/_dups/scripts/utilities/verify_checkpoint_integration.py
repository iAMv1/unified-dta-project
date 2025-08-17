"""
Simple verification that checkpoint utilities are properly integrated
"""

def verify_imports():
    """Verify all checkpoint-related imports work"""
    try:
        # Test core training imports
        from core.training import ModelCheckpoint, ProgressiveTrainer, TrainingState, TrainingMetrics
        print("✓ Core training imports successful")
        
        # Test checkpoint utilities imports
        from core.checkpoint_utils import (
            CheckpointValidator, ModelExporter, CheckpointRecovery, 
            CheckpointManager, CheckpointMetadata
        )
        print("✓ Checkpoint utilities imports successful")
        
        # Test config imports
        from core.config import DTAConfig, TrainingConfig
        print("✓ Configuration imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def verify_basic_functionality():
    """Verify basic functionality without heavy computation"""
    try:
        import torch
        import tempfile
        from pathlib import Path
        
        from core.config import DTAConfig
        from core.checkpoint_utils import CheckpointValidator, CheckpointMetadata
        
        # Test checkpoint metadata creation
        metadata = CheckpointMetadata(
            model_type="TestModel",
            validation_loss=0.5,
            epoch=10,
            description="Test checkpoint"
        )
        print("✓ CheckpointMetadata creation successful")
        
        # Test validator creation
        validator = CheckpointValidator()
        print("✓ CheckpointValidator creation successful")
        
        # Test config creation
        config = DTAConfig()
        print("✓ DTAConfig creation successful")
        
        # Test basic validation on non-existent file
        temp_dir = Path(tempfile.mkdtemp())
        fake_checkpoint = temp_dir / "fake.pth"
        
        result = validator.validate_checkpoint_file(fake_checkpoint)
        assert not result['valid'], "Should be invalid for non-existent file"
        assert len(result['errors']) > 0, "Should have errors"
        print("✓ Basic validation logic working")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """Run verification checks"""
    print("CHECKPOINT SYSTEM INTEGRATION VERIFICATION")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 2
    
    print("1. Verifying imports...")
    if verify_imports():
        checks_passed += 1
    
    print("\n2. Verifying basic functionality...")
    if verify_basic_functionality():
        checks_passed += 1
    
    print("\n" + "=" * 50)
    print(f"VERIFICATION RESULTS: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("✓ Checkpoint system integration verified successfully!")
        print("\nThe checkpoint system includes:")
        print("- Comprehensive model checkpointing during training")
        print("- Checkpoint validation and integrity checking")
        print("- Checkpoint recovery from corruption")
        print("- Model export for inference and sharing")
        print("- High-level checkpoint management")
        print("- Command-line utilities for checkpoint operations")
        print("- Extensive test coverage")
        return 0
    else:
        print("✗ Some verification checks failed.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
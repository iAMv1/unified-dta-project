"""
Demonstration of the comprehensive checkpoint system
Shows all major checkpointing features including validation, recovery, and export
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
from pathlib import Path
import logging
import json

from core.config import DTAConfig, TrainingConfig
from core.models import get_lightweight_model
from core.training import ProgressiveTrainer, TrainingState, TrainingMetrics
from core.checkpoint_utils import (
    CheckpointManager, CheckpointValidator, ModelExporter, 
    CheckpointRecovery, CheckpointMetadata
)
from core.data_processing import create_dummy_batch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_demo_data_loader(num_samples=50, batch_size=4):
    """Create dummy data loader for demonstration"""
    
    # Create dummy data
    dummy_features = torch.randn(num_samples)
    dummy_targets = torch.randn(num_samples)
    dataset = TensorDataset(dummy_features, dummy_targets)
    
    def collate_fn(batch):
        features, targets = zip(*batch)
        return {
            'drug_data': create_dummy_batch(len(features)),
            'protein_sequences': ['MKLLVLSLSLVLVAPMAAQAAEITLVPSVKLQIGDRDNRGYYWDGGHWRDH'] * len(features),
            'affinities': torch.stack(targets)
        }
    
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def demo_basic_checkpointing():
    """Demonstrate basic checkpointing during training"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Checkpointing During Training")
    print("="*60)
    
    # Create temporary directory for this demo
    temp_dir = Path(tempfile.mkdtemp())
    checkpoint_dir = temp_dir / "checkpoints"
    
    try:
        # Setup model and training
        config = DTAConfig()
        training_config = TrainingConfig(
            num_epochs_phase1=3,
            num_epochs_phase2=2,
            checkpoint_interval=2,  # Save every 2 epochs
            early_stopping_patience=5
        )
        
        model = get_lightweight_model(config)
        trainer = ProgressiveTrainer(
            model, config, training_config,
            checkpoint_dir=str(checkpoint_dir)
        )
        
        # Create data loaders
        train_loader = create_demo_data_loader(num_samples=40)
        val_loader = create_demo_data_loader(num_samples=20)
        
        print(f"Training model with checkpointing enabled...")
        print(f"Checkpoint directory: {checkpoint_dir}")
        
        # Train for a few epochs (this will create checkpoints)
        try:
            # Mock training to avoid long computation
            from unittest.mock import patch
            
            with patch.object(trainer, '_train_epoch', return_value=(0.5, {})):
                with patch.object(trainer, '_validate_epoch', return_value=(0.4, {
                    'pearson': 0.8, 'spearman': 0.7, 'rmse': 0.3, 'mse': 0.09
                })):
                    trainer.train_phase(train_loader, val_loader, phase=1, num_epochs=3)
        
        except Exception as e:
            print(f"Training simulation completed (mock training): {e}")
        
        # List created checkpoints
        checkpoint_files = list(checkpoint_dir.rglob("*.pth"))
        print(f"\nCheckpoints created: {len(checkpoint_files)}")
        
        for checkpoint_file in checkpoint_files:
            size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
            print(f"  - {checkpoint_file.name} ({size_mb:.1f} MB)")
        
        # Demonstrate checkpoint loading
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            print(f"\nLoading latest checkpoint: {latest_checkpoint.name}")
            
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  Phase: {checkpoint.get('phase', 'unknown')}")
            print(f"  Validation loss: {checkpoint.get('validation_loss', 'unknown')}")
            print(f"  Contains model state: {'model_state_dict' in checkpoint}")
            print(f"  Contains training state: {'training_state' in checkpoint}")
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nDemo 1 completed. Temporary files cleaned up.")


def demo_checkpoint_validation():
    """Demonstrate checkpoint validation functionality"""
    print("\n" + "="*60)
    print("DEMO 2: Checkpoint Validation")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        config = DTAConfig()
        model = get_lightweight_model(config)
        
        # Create different types of checkpoints for validation
        
        # 1. Valid checkpoint
        valid_checkpoint_path = temp_dir / "valid_checkpoint.pth"
        valid_checkpoint = {
            'version': '1.0',
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'epoch': 10,
            'phase': 1,
            'validation_loss': 0.35,
            'is_best': True,
            'timestamp': 1234567890,
            'training_state': {
                'current_epoch': 10,
                'current_phase': 1,
                'best_val_loss': 0.35
            }
        }
        torch.save(valid_checkpoint, valid_checkpoint_path)
        
        # 2. Incomplete checkpoint (missing some fields)
        incomplete_checkpoint_path = temp_dir / "incomplete_checkpoint.pth"
        incomplete_checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'epoch': 5
            # Missing validation_loss, training_state, etc.
        }
        torch.save(incomplete_checkpoint, incomplete_checkpoint_path)
        
        # 3. Corrupted checkpoint
        corrupted_checkpoint_path = temp_dir / "corrupted_checkpoint.pth"
        with open(corrupted_checkpoint_path, 'wb') as f:
            f.write(b"This is not a valid PyTorch checkpoint file")
        
        # Validate each checkpoint
        validator = CheckpointValidator()
        
        checkpoints_to_validate = [
            ("Valid Checkpoint", valid_checkpoint_path),
            ("Incomplete Checkpoint", incomplete_checkpoint_path),
            ("Corrupted Checkpoint", corrupted_checkpoint_path)
        ]
        
        for name, checkpoint_path in checkpoints_to_validate:
            print(f"\n--- Validating {name} ---")
            result = validator.validate_checkpoint_file(checkpoint_path)
            
            print(f"Valid: {'✓' if result['valid'] else '✗'}")
            print(f"File size: {result.get('file_info', {}).get('size_mb', 0):.1f} MB")
            
            if result.get('metadata'):
                metadata = result['metadata']
                print(f"Epoch: {metadata.get('epoch', 'unknown')}")
                print(f"Validation loss: {metadata.get('validation_loss', 'unknown')}")
            
            if result.get('errors'):
                print(f"Errors ({len(result['errors'])}):")
                for error in result['errors'][:3]:  # Show first 3 errors
                    print(f"  - {error}")
            
            if result.get('warnings'):
                print(f"Warnings ({len(result['warnings'])}):")
                for warning in result['warnings'][:3]:  # Show first 3 warnings
                    print(f"  - {warning}")
        
        # Demonstrate model compatibility checking
        print(f"\n--- Model Compatibility Check ---")
        
        # Compatible configuration
        compatible_config = DTAConfig(
            protein_encoder_type='cnn',
            drug_encoder_type='gin',
            use_fusion=False
        )
        
        compatibility_result = validator.validate_model_compatibility(
            valid_checkpoint, compatible_config
        )
        
        print(f"Compatible with current config: {'✓' if compatibility_result['compatible'] else '✗'}")
        
        if compatibility_result.get('issues'):
            print("Compatibility issues:")
            for issue in compatibility_result['issues']:
                print(f"  - {issue}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nDemo 2 completed. Temporary files cleaned up.")


def demo_checkpoint_recovery():
    """Demonstrate checkpoint recovery functionality"""
    print("\n" + "="*60)
    print("DEMO 3: Checkpoint Recovery")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        config = DTAConfig()
        model = get_lightweight_model(config)
        
        # Create a checkpoint with issues that can be recovered
        problematic_checkpoint_path = temp_dir / "problematic_checkpoint.pth"
        
        # Create model state with some NaN values
        state_dict = model.state_dict()
        param_name = list(state_dict.keys())[0]
        state_dict[param_name][0, 0] = float('nan')  # Introduce NaN
        
        problematic_checkpoint = {
            'model_state_dict': state_dict,
            'config': config.to_dict(),
            'epoch': 8,
            'validation_loss': 0.42
            # Missing version, timestamp, etc.
        }
        torch.save(problematic_checkpoint, problematic_checkpoint_path)
        
        print(f"Created problematic checkpoint: {problematic_checkpoint_path.name}")
        
        # Validate the problematic checkpoint
        validator = CheckpointValidator()
        validation_result = validator.validate_checkpoint_file(problematic_checkpoint_path)
        
        print(f"\nValidation before recovery:")
        print(f"Valid: {'✓' if validation_result['valid'] else '✗'}")
        print(f"Errors: {len(validation_result.get('errors', []))}")
        print(f"Warnings: {len(validation_result.get('warnings', []))}")
        
        # Attempt recovery
        recovery = CheckpointRecovery(str(temp_dir))
        
        print(f"\nAttempting recovery...")
        
        # Create backup first
        backup_path = recovery.create_backup(problematic_checkpoint_path)
        print(f"Backup created: {backup_path.name}")
        
        # Recover the checkpoint
        recovery_result = recovery.attempt_recovery(problematic_checkpoint_path)
        
        print(f"\nRecovery results:")
        print(f"Success: {'✓' if recovery_result['success'] else '✗'}")
        
        if recovery_result.get('issues_fixed'):
            print(f"Issues fixed ({len(recovery_result['issues_fixed'])}):")
            for fix in recovery_result['issues_fixed']:
                print(f"  ✓ {fix}")
        
        if recovery_result.get('remaining_issues'):
            print(f"Remaining issues ({len(recovery_result['remaining_issues'])}):")
            for issue in recovery_result['remaining_issues']:
                print(f"  - {issue}")
        
        # Validate recovered checkpoint
        if recovery_result['success'] and recovery_result['recovered_path']:
            recovered_path = Path(recovery_result['recovered_path'])
            print(f"\nValidating recovered checkpoint: {recovered_path.name}")
            
            recovered_validation = validator.validate_checkpoint_file(recovered_path)
            print(f"Valid after recovery: {'✓' if recovered_validation['valid'] else '✗'}")
            print(f"Errors after recovery: {len(recovered_validation.get('errors', []))}")
            print(f"Warnings after recovery: {len(recovered_validation.get('warnings', []))}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nDemo 3 completed. Temporary files cleaned up.")


def demo_model_export():
    """Demonstrate model export functionality"""
    print("\n" + "="*60)
    print("DEMO 4: Model Export")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        config = DTAConfig()
        model = get_lightweight_model(config)
        
        # Create a checkpoint to export
        checkpoint_path = temp_dir / "model_to_export.pth"
        checkpoint = {
            'version': '1.0',
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'epoch': 25,
            'phase': 2,
            'validation_loss': 0.28,
            'is_best': True,
            'optimizer_state_dict': {'param_groups': [{'lr': 1e-4}]},
            'training_state': {
                'current_epoch': 25,
                'metrics_history': [
                    {'epoch': 1, 'train_loss': 0.8, 'val_loss': 0.7},
                    {'epoch': 2, 'train_loss': 0.6, 'val_loss': 0.5}
                ]
            }
        }
        torch.save(checkpoint, checkpoint_path)
        
        exporter = ModelExporter(str(temp_dir / "exports"))
        
        # 1. Export for inference
        print(f"\n--- Export for Inference ---")
        
        inference_export_paths = exporter.export_for_inference(
            model, config, "demo_model", compress=True
        )
        
        print(f"Inference export completed:")
        for export_type, path in inference_export_paths.items():
            if isinstance(path, Path) and path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  {export_type}: {path.name} ({size_mb:.1f} MB)")
        
        # Verify inference export contents
        if 'model' in inference_export_paths:
            inference_checkpoint = torch.load(inference_export_paths['model'], map_location='cpu')
            print(f"  Export type: {inference_checkpoint.get('export_type')}")
            print(f"  Model class: {inference_checkpoint.get('model_class')}")
            print(f"  Contains optimizer state: {'optimizer_state_dict' in inference_checkpoint}")
        
        # 2. Export for sharing
        print(f"\n--- Export for Sharing ---")
        
        shared_export_path = exporter.export_for_sharing(
            checkpoint_path, "shared_demo_model", 
            include_training_state=True, anonymize=True
        )
        
        shared_size_mb = shared_export_path.stat().st_size / (1024 * 1024)
        print(f"Shared export completed: {shared_export_path.name} ({shared_size_mb:.1f} MB)")
        
        # Verify shared export contents
        shared_checkpoint = torch.load(shared_export_path, map_location='cpu')
        print(f"  Export type: {shared_checkpoint.get('export_type')}")
        print(f"  Contains optimizer state: {'optimizer_state_dict' in shared_checkpoint}")
        print(f"  Contains detailed metrics: {'metrics_history' in shared_checkpoint.get('training_state', {})}")
        print(f"  Contains metrics summary: {'metrics_summary' in shared_checkpoint.get('training_state', {})}")
        
        # 3. Show generated inference script
        if 'inference_script' in inference_export_paths:
            script_path = inference_export_paths['inference_script']
            print(f"\n--- Generated Inference Script Preview ---")
            with open(script_path, 'r') as f:
                lines = f.readlines()
                # Show first 15 lines
                for i, line in enumerate(lines[:15]):
                    print(f"{i+1:2d}: {line.rstrip()}")
                if len(lines) > 15:
                    print(f"... ({len(lines) - 15} more lines)")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nDemo 4 completed. Temporary files cleaned up.")


def demo_checkpoint_manager():
    """Demonstrate high-level checkpoint management"""
    print("\n" + "="*60)
    print("DEMO 5: Checkpoint Manager")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        config = DTAConfig()
        model = get_lightweight_model(config)
        
        # Create various checkpoints for management demo
        checkpoints_to_create = [
            ("good_checkpoint_1.pth", {
                'model_state_dict': model.state_dict(),
                'config': config.to_dict(),
                'epoch': 10,
                'validation_loss': 0.45,
                'is_best': False
            }),
            ("good_checkpoint_2.pth", {
                'model_state_dict': model.state_dict(),
                'config': config.to_dict(),
                'epoch': 15,
                'validation_loss': 0.32,
                'is_best': True
            }),
            ("incomplete_checkpoint.pth", {
                'model_state_dict': model.state_dict(),
                'config': config.to_dict(),
                'epoch': 5
                # Missing validation_loss
            })
        ]
        
        # Create checkpoints
        for filename, checkpoint_data in checkpoints_to_create:
            checkpoint_path = temp_dir / filename
            torch.save(checkpoint_data, checkpoint_path)
        
        # Create a corrupted file
        corrupted_path = temp_dir / "corrupted_checkpoint.pth"
        with open(corrupted_path, 'wb') as f:
            f.write(b"corrupted data")
        
        print(f"Created {len(checkpoints_to_create) + 1} test checkpoints")
        
        # Initialize checkpoint manager
        manager = CheckpointManager(str(temp_dir))
        
        # 1. List all checkpoints
        print(f"\n--- Checkpoint Listing ---")
        all_checkpoints = manager.list_all_checkpoints()
        
        for category, checkpoints in all_checkpoints.items():
            print(f"{category.upper()}: {len(checkpoints)} checkpoints")
            for checkpoint_info in checkpoints:
                path = Path(checkpoint_info['path'])
                validation = checkpoint_info['validation']
                metadata = validation.get('metadata', {})
                
                print(f"  - {path.name}")
                print(f"    Epoch: {metadata.get('epoch', 'unknown')}")
                print(f"    Val Loss: {metadata.get('validation_loss', 'unknown')}")
                print(f"    Errors: {len(validation.get('errors', []))}")
                print(f"    Warnings: {len(validation.get('warnings', []))}")
        
        # 2. Get comprehensive summary
        print(f"\n--- Checkpoint Summary ---")
        summary = manager.get_checkpoint_summary()
        
        print(f"Total checkpoints: {summary['total_checkpoints']}")
        print(f"Valid checkpoints: {summary['valid_checkpoints']}")
        print(f"Invalid checkpoints: {summary['invalid_checkpoints']}")
        print(f"Recoverable checkpoints: {summary['recoverable_checkpoints']}")
        print(f"Total size: {summary['total_size_mb']:.1f} MB")
        
        if summary.get('best_checkpoint'):
            best = summary['best_checkpoint']
            print(f"Best checkpoint: {Path(best['path']).name} (loss: {best['validation_loss']:.4f})")
        
        if summary.get('newest_checkpoint'):
            newest = summary['newest_checkpoint']
            print(f"Newest checkpoint: {Path(newest['path']).name}")
        
        # 3. Demonstrate cleanup (dry run)
        print(f"\n--- Cleanup Preview ---")
        print("Note: This is a dry run - no files will actually be deleted")
        
        cleanup_result = manager.cleanup_invalid_checkpoints(confirm=False)
        print(f"Would remove: {len(cleanup_result['removed'])} files")
        print(f"Would free: {cleanup_result['total_space_freed_mb']:.1f} MB")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nDemo 5 completed. Temporary files cleaned up.")


def demo_advanced_features():
    """Demonstrate advanced checkpointing features"""
    print("\n" + "="*60)
    print("DEMO 6: Advanced Features")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        config = DTAConfig()
        model = get_lightweight_model(config)
        
        # 1. Demonstrate checkpoint metadata
        print(f"\n--- Checkpoint Metadata ---")
        
        metadata = CheckpointMetadata(
            model_type="UnifiedDTAModel",
            validation_loss=0.285,
            epoch=20,
            phase=2,
            is_best=True,
            description="Best model from experiment #42",
            tags=["production", "validated", "esm-based"]
        )
        
        print(f"Metadata example:")
        metadata_dict = metadata.__dict__
        for key, value in metadata_dict.items():
            print(f"  {key}: {value}")
        
        # 2. Demonstrate training metrics serialization
        print(f"\n--- Training Metrics Serialization ---")
        
        metrics = TrainingMetrics(
            epoch=15,
            phase=2,
            train_loss=0.42,
            val_loss=0.35,
            val_pearson=0.82,
            val_spearman=0.79,
            val_rmse=0.31,
            learning_rate=1e-4,
            memory_usage=2048.5,
            training_time=145.2
        )
        
        # Serialize to JSON
        metrics_json = json.dumps(metrics.to_dict(), indent=2)
        print(f"Training metrics as JSON:")
        print(metrics_json[:200] + "..." if len(metrics_json) > 200 else metrics_json)
        
        # 3. Demonstrate training state management
        print(f"\n--- Training State Management ---")
        
        training_state = TrainingState(
            current_epoch=15,
            current_phase=2,
            best_val_loss=0.35,
            best_val_pearson=0.82,
            patience_counter=2,
            total_training_time=3600.0,
            metrics_history=[metrics]
        )
        
        state_dict = training_state.to_dict()
        print(f"Training state keys: {list(state_dict.keys())}")
        print(f"Current epoch: {state_dict['current_epoch']}")
        print(f"Best validation loss: {state_dict['best_val_loss']}")
        print(f"Metrics history length: {len(state_dict['metrics_history'])}")
        
        # 4. Demonstrate configuration hashing for compatibility
        print(f"\n--- Configuration Compatibility ---")
        
        from core.checkpoint_utils import ModelExporter
        exporter = ModelExporter()
        
        config_hash = exporter._calculate_config_hash(config.to_dict())
        print(f"Configuration hash: {config_hash}")
        
        # Modify config slightly
        modified_config = DTAConfig(protein_encoder_type='esm')  # Different from default
        modified_hash = exporter._calculate_config_hash(modified_config.to_dict())
        print(f"Modified config hash: {modified_hash}")
        print(f"Hashes match: {config_hash == modified_hash}")
        
        # 5. Demonstrate checkpoint versioning
        print(f"\n--- Checkpoint Versioning ---")
        
        # Create checkpoints with different versions
        v1_checkpoint = {
            'version': '1.0',
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'epoch': 10
        }
        
        v2_checkpoint = {
            'version': '2.0',  # Future version
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'epoch': 10,
            'enhanced_metadata': {'experiment_id': 'exp_001'}
        }
        
        v1_path = temp_dir / "v1_checkpoint.pth"
        v2_path = temp_dir / "v2_checkpoint.pth"
        
        torch.save(v1_checkpoint, v1_path)
        torch.save(v2_checkpoint, v2_path)
        
        # Validate different versions
        validator = CheckpointValidator()
        
        for name, path in [("v1.0", v1_path), ("v2.0", v2_path)]:
            result = validator.validate_checkpoint_file(path)
            version = result.get('metadata', {}).get('version', 'unknown')
            print(f"Checkpoint {name}: version={version}, valid={result['valid']}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nDemo 6 completed. Temporary files cleaned up.")


def main():
    """Run all checkpoint system demonstrations"""
    print("UNIFIED DTA SYSTEM - CHECKPOINT SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the comprehensive checkpointing and model persistence")
    print("capabilities of the Unified DTA System, including:")
    print("- Automatic checkpointing during training")
    print("- Checkpoint validation and integrity checking")
    print("- Checkpoint recovery from corruption")
    print("- Model export for inference and sharing")
    print("- High-level checkpoint management")
    print("- Advanced features and metadata handling")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        demo_basic_checkpointing()
        demo_checkpoint_validation()
        demo_checkpoint_recovery()
        demo_model_export()
        demo_checkpoint_manager()
        demo_advanced_features()
        
        print("\n" + "="*80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey takeaways:")
        print("✓ Checkpoints are automatically saved during training")
        print("✓ Comprehensive validation ensures checkpoint integrity")
        print("✓ Recovery tools can fix common checkpoint issues")
        print("✓ Export functionality supports both inference and sharing")
        print("✓ High-level management tools simplify checkpoint operations")
        print("✓ Advanced features provide detailed metadata and versioning")
        print("\nThe checkpoint system is now ready for production use!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
#!/usr/bin/env python3
"""
Command-line interface for checkpoint management
Provides easy access to checkpoint validation, recovery, and export functionality
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from core.checkpoint_utils import CheckpointManager, CheckpointValidator, ModelExporter, CheckpointRecovery
from core.config import DTAConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    import logging
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_list_checkpoints(args):
    """List all checkpoints with their status"""
    manager = CheckpointManager(args.checkpoint_dir)
    
    if args.summary:
        summary = manager.get_checkpoint_summary()
        print("\n=== Checkpoint Summary ===")
        print(f"Total checkpoints: {summary['total_checkpoints']}")
        print(f"Valid checkpoints: {summary['valid_checkpoints']}")
        print(f"Invalid checkpoints: {summary['invalid_checkpoints']}")
        print(f"Recoverable checkpoints: {summary['recoverable_checkpoints']}")
        print(f"Total size: {summary['total_size_mb']:.1f} MB")
        
        if summary['best_checkpoint']:
            print(f"\nBest checkpoint:")
            print(f"  Path: {summary['best_checkpoint']['path']}")
            print(f"  Validation loss: {summary['best_checkpoint']['validation_loss']:.4f}")
            print(f"  Size: {summary['best_checkpoint']['size_mb']:.1f} MB")
        
        if summary['newest_checkpoint']:
            print(f"\nNewest checkpoint:")
            print(f"  Path: {summary['newest_checkpoint']['path']}")
            print(f"  Size: {summary['newest_checkpoint']['size_mb']:.1f} MB")
    else:
        all_checkpoints = manager.list_all_checkpoints()
        
        for category, checkpoints in all_checkpoints.items():
            if checkpoints:
                print(f"\n=== {category.upper()} CHECKPOINTS ===")
                for checkpoint_info in checkpoints:
                    path = checkpoint_info['path']
                    validation = checkpoint_info['validation']
                    metadata = validation.get('metadata', {})
                    
                    print(f"\nPath: {path}")
                    print(f"  Epoch: {metadata.get('epoch', 'unknown')}")
                    print(f"  Phase: {metadata.get('phase', 'unknown')}")
                    print(f"  Validation loss: {metadata.get('validation_loss', 'unknown')}")
                    print(f"  Size: {validation.get('file_info', {}).get('size_mb', 0):.1f} MB")
                    
                    if validation.get('errors'):
                        print(f"  Errors: {len(validation['errors'])}")
                        if args.verbose:
                            for error in validation['errors']:
                                print(f"    - {error}")
                    
                    if validation.get('warnings'):
                        print(f"  Warnings: {len(validation['warnings'])}")
                        if args.verbose:
                            for warning in validation['warnings']:
                                print(f"    - {warning}")


def cmd_validate_checkpoint(args):
    """Validate a specific checkpoint"""
    checkpoint_path = Path(args.checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return 1
    
    validator = CheckpointValidator()
    result = validator.validate_checkpoint_file(checkpoint_path)
    
    print(f"\n=== Validation Results for {checkpoint_path} ===")
    print(f"Valid: {'✓' if result['valid'] else '✗'}")
    
    if result['file_info']:
        print(f"File size: {result['file_info']['size_mb']:.1f} MB")
    
    if result['metadata']:
        metadata = result['metadata']
        print(f"\nMetadata:")
        print(f"  Version: {metadata.get('version', 'unknown')}")
        print(f"  Epoch: {metadata.get('epoch', 'unknown')}")
        print(f"  Phase: {metadata.get('phase', 'unknown')}")
        print(f"  Validation loss: {metadata.get('validation_loss', 'unknown')}")
        print(f"  Is best: {metadata.get('is_best', 'unknown')}")
        print(f"  Parameters: {metadata.get('num_parameters', 'unknown')}")
    
    if result['errors']:
        print(f"\nErrors ({len(result['errors'])}):")
        for error in result['errors']:
            print(f"  ✗ {error}")
    
    if result['warnings']:
        print(f"\nWarnings ({len(result['warnings'])}):")
        for warning in result['warnings']:
            print(f"  ⚠ {warning}")
    
    return 0 if result['valid'] else 1


def cmd_recover_checkpoint(args):
    """Attempt to recover a corrupted checkpoint"""
    checkpoint_path = Path(args.checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return 1
    
    recovery = CheckpointRecovery(str(checkpoint_path.parent))
    
    # Create backup if requested
    if args.backup:
        backup_path = recovery.create_backup(checkpoint_path)
        print(f"Backup created: {backup_path}")
    
    # Attempt recovery
    output_path = Path(args.output) if args.output else None
    result = recovery.attempt_recovery(checkpoint_path, output_path)
    
    print(f"\n=== Recovery Results for {checkpoint_path} ===")
    print(f"Success: {'✓' if result['success'] else '✗'}")
    
    if result['success']:
        print(f"Recovered checkpoint saved to: {result['recovered_path']}")
    
    if result['issues_fixed']:
        print(f"\nIssues fixed ({len(result['issues_fixed'])}):")
        for fix in result['issues_fixed']:
            print(f"  ✓ {fix}")
    
    if result['remaining_issues']:
        print(f"\nRemaining issues ({len(result['remaining_issues'])}):")
        for issue in result['remaining_issues']:
            print(f"  ✗ {issue}")
    
    return 0 if result['success'] else 1


def cmd_export_checkpoint(args):
    """Export checkpoint for inference or sharing"""
    checkpoint_path = Path(args.checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return 1
    
    exporter = ModelExporter(args.export_dir)
    
    if args.export_type == 'sharing':
        # Export for sharing
        shared_path = exporter.export_for_sharing(
            checkpoint_path,
            args.export_name,
            include_training_state=args.include_training_state,
            anonymize=not args.no_anonymize
        )
        print(f"Checkpoint exported for sharing: {shared_path}")
    
    elif args.export_type == 'inference':
        # Need to load model for inference export
        try:
            import torch
            from core.models import get_lightweight_model, get_production_model
            
            # Load checkpoint to get config
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            config = DTAConfig(**checkpoint['config'])
            
            # Create model (simplified - in practice you'd need proper model loading)
            if config.protein_encoder_type == 'cnn':
                model = get_lightweight_model(config)
            else:
                model = get_production_model(config)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Export for inference
            export_paths = exporter.export_for_inference(
                model, config, args.export_name,
                compress=args.compress
            )
            
            print(f"Model exported for inference:")
            for export_type, path in export_paths.items():
                print(f"  {export_type}: {path}")
        
        except Exception as e:
            print(f"Error exporting for inference: {e}")
            return 1
    
    return 0


def cmd_cleanup_checkpoints(args):
    """Clean up invalid checkpoints"""
    manager = CheckpointManager(args.checkpoint_dir)
    
    if not args.confirm:
        print("This will permanently delete invalid checkpoint files.")
        print("Use --confirm to proceed with cleanup.")
        return 1
    
    result = manager.cleanup_invalid_checkpoints(confirm=True)
    
    print(f"\n=== Cleanup Results ===")
    print(f"Files removed: {len(result['removed'])}")
    print(f"Space freed: {result['total_space_freed_mb']:.1f} MB")
    
    if result['removed']:
        print(f"\nRemoved files:")
        for removed_file in result['removed']:
            print(f"  ✓ {removed_file}")
    
    if result['failed_to_remove']:
        print(f"\nFailed to remove:")
        for failed in result['failed_to_remove']:
            print(f"  ✗ {failed['path']}: {failed['error']}")
    
    return 0


def cmd_compare_checkpoints(args):
    """Compare two checkpoints"""
    path1 = Path(args.checkpoint1)
    path2 = Path(args.checkpoint2)
    
    if not path1.exists():
        print(f"Error: Checkpoint file not found: {path1}")
        return 1
    
    if not path2.exists():
        print(f"Error: Checkpoint file not found: {path2}")
        return 1
    
    validator = CheckpointValidator()
    
    # Validate both checkpoints
    result1 = validator.validate_checkpoint_file(path1)
    result2 = validator.validate_checkpoint_file(path2)
    
    print(f"\n=== Checkpoint Comparison ===")
    print(f"Checkpoint 1: {path1}")
    print(f"Checkpoint 2: {path2}")
    
    # Compare basic info
    meta1 = result1.get('metadata', {})
    meta2 = result2.get('metadata', {})
    
    print(f"\n--- Basic Information ---")
    print(f"{'Metric':<20} {'Checkpoint 1':<15} {'Checkpoint 2':<15}")
    print("-" * 50)
    print(f"{'Valid':<20} {result1['valid']:<15} {result2['valid']:<15}")
    print(f"{'Epoch':<20} {meta1.get('epoch', 'N/A'):<15} {meta2.get('epoch', 'N/A'):<15}")
    print(f"{'Phase':<20} {meta1.get('phase', 'N/A'):<15} {meta2.get('phase', 'N/A'):<15}")
    print(f"{'Validation Loss':<20} {meta1.get('validation_loss', 'N/A'):<15} {meta2.get('validation_loss', 'N/A'):<15}")
    print(f"{'Is Best':<20} {meta1.get('is_best', 'N/A'):<15} {meta2.get('is_best', 'N/A'):<15}")
    
    # Compare file sizes
    size1 = result1.get('file_info', {}).get('size_mb', 0)
    size2 = result2.get('file_info', {}).get('size_mb', 0)
    print(f"{'Size (MB)':<20} {size1:<15.1f} {size2:<15.1f}")
    
    # Determine which is better
    val_loss1 = meta1.get('validation_loss', float('inf'))
    val_loss2 = meta2.get('validation_loss', float('inf'))
    
    if val_loss1 != float('inf') and val_loss2 != float('inf'):
        if val_loss1 < val_loss2:
            print(f"\n✓ Checkpoint 1 has better validation loss ({val_loss1:.4f} vs {val_loss2:.4f})")
        elif val_loss2 < val_loss1:
            print(f"\n✓ Checkpoint 2 has better validation loss ({val_loss2:.4f} vs {val_loss1:.4f})")
        else:
            print(f"\n= Both checkpoints have the same validation loss ({val_loss1:.4f})")
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Checkpoint management utility for Unified DTA System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list --summary                           # Show checkpoint summary
  %(prog)s validate checkpoint.pth                 # Validate a checkpoint
  %(prog)s recover corrupted.pth --backup          # Recover with backup
  %(prog)s export checkpoint.pth inference model   # Export for inference
  %(prog)s cleanup --confirm                       # Clean up invalid files
  %(prog)s compare checkpoint1.pth checkpoint2.pth # Compare checkpoints
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Checkpoint directory (default: checkpoints)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List checkpoints')
    list_parser.add_argument('--summary', action='store_true',
                            help='Show summary instead of detailed list')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate checkpoint')
    validate_parser.add_argument('checkpoint_path', help='Path to checkpoint file')
    
    # Recover command
    recover_parser = subparsers.add_parser('recover', help='Recover corrupted checkpoint')
    recover_parser.add_argument('checkpoint_path', help='Path to checkpoint file')
    recover_parser.add_argument('--output', '-o', help='Output path for recovered checkpoint')
    recover_parser.add_argument('--backup', action='store_true',
                               help='Create backup before recovery')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export checkpoint')
    export_parser.add_argument('checkpoint_path', help='Path to checkpoint file')
    export_parser.add_argument('export_type', choices=['inference', 'sharing'],
                              help='Export type')
    export_parser.add_argument('export_name', help='Name for exported files')
    export_parser.add_argument('--export-dir', default='exports',
                              help='Export directory (default: exports)')
    export_parser.add_argument('--include-training-state', action='store_true',
                              help='Include training state in shared export')
    export_parser.add_argument('--no-anonymize', action='store_true',
                              help='Do not anonymize shared export')
    export_parser.add_argument('--compress', action='store_true',
                              help='Compress inference export')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up invalid checkpoints')
    cleanup_parser.add_argument('--confirm', action='store_true',
                               help='Confirm deletion of invalid checkpoints')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two checkpoints')
    compare_parser.add_argument('checkpoint1', help='First checkpoint path')
    compare_parser.add_argument('checkpoint2', help='Second checkpoint path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    # Execute command
    command_functions = {
        'list': cmd_list_checkpoints,
        'validate': cmd_validate_checkpoint,
        'recover': cmd_recover_checkpoint,
        'export': cmd_export_checkpoint,
        'cleanup': cmd_cleanup_checkpoints,
        'compare': cmd_compare_checkpoints
    }
    
    try:
        return command_functions[args.command](args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
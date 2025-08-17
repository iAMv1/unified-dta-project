"""
Comprehensive checkpoint utilities for the Unified DTA System
Provides standalone checkpoint management, model export/import, and recovery tools
"""

import torch
import torch.nn as nn
import json
import pickle
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, asdict
import logging
import time
import warnings
from datetime import datetime

from .config import DTAConfig, TrainingConfig
from .models import UnifiedDTAModel

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files"""
    version: str = "1.0"
    created_at: str = ""
    model_type: str = ""
    config_hash: str = ""
    file_size_mb: float = 0.0
    validation_loss: float = float('inf')
    epoch: int = 0
    phase: int = 1
    is_best: bool = False
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []


class CheckpointValidator:
    """Validates checkpoint integrity and compatibility"""
    
    @staticmethod
    def validate_checkpoint_file(checkpoint_path: Path) -> Dict[str, Any]:
        """Validate a checkpoint file and return validation results"""
        
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'metadata': {},
            'file_info': {}
        }
        
        try:
            # Check file existence and size
            if not checkpoint_path.exists():
                validation_result['errors'].append(f"Checkpoint file not found: {checkpoint_path}")
                return validation_result
            
            file_size = checkpoint_path.stat().st_size
            validation_result['file_info']['size_bytes'] = file_size
            validation_result['file_info']['size_mb'] = file_size / (1024 * 1024)
            
            if file_size == 0:
                validation_result['errors'].append("Checkpoint file is empty")
                return validation_result
            
            # Try to load checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            except Exception as e:
                validation_result['errors'].append(f"Failed to load checkpoint: {str(e)}")
                return validation_result
            
            # Validate required keys
            required_keys = ['model_state_dict', 'config', 'epoch']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                validation_result['errors'].extend([f"Missing required key: {key}" for key in missing_keys])
            
            # Validate optional but important keys
            recommended_keys = ['training_state', 'validation_loss', 'timestamp']
            missing_recommended = [key for key in recommended_keys if key not in checkpoint]
            
            if missing_recommended:
                validation_result['warnings'].extend([f"Missing recommended key: {key}" for key in missing_recommended])
            
            # Extract metadata
            validation_result['metadata'] = {
                'version': checkpoint.get('version', 'unknown'),
                'epoch': checkpoint.get('epoch', 0),
                'phase': checkpoint.get('phase', 1),
                'validation_loss': checkpoint.get('validation_loss', float('inf')),
                'is_best': checkpoint.get('is_best', False),
                'timestamp': checkpoint.get('timestamp', 0)
            }
            
            # Validate model state dict
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                if not isinstance(model_state, dict):
                    validation_result['errors'].append("model_state_dict is not a dictionary")
                elif len(model_state) == 0:
                    validation_result['errors'].append("model_state_dict is empty")
                else:
                    validation_result['metadata']['num_parameters'] = len(model_state)
            
            # Validate configuration
            if 'config' in checkpoint:
                config = checkpoint['config']
                if not isinstance(config, dict):
                    validation_result['errors'].append("config is not a dictionary")
                else:
                    validation_result['metadata']['config_keys'] = list(config.keys())
            
            # Check for corruption indicators
            if 'model_state_dict' in checkpoint:
                try:
                    # Try to access a few parameters to check for corruption
                    for i, (key, tensor) in enumerate(checkpoint['model_state_dict'].items()):
                        if i >= 3:  # Check first 3 parameters
                            break
                        if not isinstance(tensor, torch.Tensor):
                            validation_result['warnings'].append(f"Parameter {key} is not a tensor")
                        elif torch.isnan(tensor).any():
                            validation_result['warnings'].append(f"Parameter {key} contains NaN values")
                        elif torch.isinf(tensor).any():
                            validation_result['warnings'].append(f"Parameter {key} contains infinite values")
                except Exception as e:
                    validation_result['warnings'].append(f"Could not validate model parameters: {str(e)}")
            
            # If no errors, mark as valid
            if not validation_result['errors']:
                validation_result['valid'] = True
            
        except Exception as e:
            validation_result['errors'].append(f"Unexpected error during validation: {str(e)}")
        
        return validation_result
    
    @staticmethod
    def validate_model_compatibility(checkpoint: Dict[str, Any], 
                                   target_config: DTAConfig) -> Dict[str, Any]:
        """Validate that a checkpoint is compatible with a target configuration"""
        
        compatibility_result = {
            'compatible': False,
            'issues': [],
            'warnings': []
        }
        
        if 'config' not in checkpoint:
            compatibility_result['issues'].append("Checkpoint missing configuration")
            return compatibility_result
        
        checkpoint_config = checkpoint['config']
        target_dict = target_config.to_dict()
        
        # Check critical compatibility parameters
        critical_params = [
            'protein_encoder_type',
            'drug_encoder_type',
            'use_fusion'
        ]
        
        for param in critical_params:
            if param in checkpoint_config and param in target_dict:
                if checkpoint_config[param] != target_dict[param]:
                    compatibility_result['issues'].append(
                        f"Incompatible {param}: checkpoint={checkpoint_config[param]}, "
                        f"target={target_dict[param]}"
                    )
        
        # Check dimension compatibility
        dimension_params = [
            ('protein_config', 'output_dim'),
            ('drug_config', 'output_dim')
        ]
        
        for config_key, dim_key in dimension_params:
            if (config_key in checkpoint_config and config_key in target_dict and
                dim_key in checkpoint_config[config_key] and dim_key in target_dict[config_key]):
                
                checkpoint_dim = checkpoint_config[config_key][dim_key]
                target_dim = target_dict[config_key][dim_key]
                
                if checkpoint_dim != target_dim:
                    compatibility_result['warnings'].append(
                        f"Dimension mismatch in {config_key}.{dim_key}: "
                        f"checkpoint={checkpoint_dim}, target={target_dim}"
                    )
        
        # If no critical issues, mark as compatible
        if not compatibility_result['issues']:
            compatibility_result['compatible'] = True
        
        return compatibility_result


class ModelExporter:
    """Export models in various formats for deployment and sharing"""
    
    def __init__(self, export_dir: str = "exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_for_inference(self, 
                           model: nn.Module,
                           config: DTAConfig,
                           export_name: str,
                           include_optimizer: bool = False,
                           compress: bool = True) -> Dict[str, Path]:
        """Export model optimized for inference"""
        
        export_paths = {}
        timestamp = int(time.time())
        
        # Create export subdirectory
        export_subdir = self.export_dir / f"{export_name}_{timestamp}"
        export_subdir.mkdir(exist_ok=True)
        
        # Set model to eval mode
        model.eval()
        
        # Export model state dict (lightweight)
        model_path = export_subdir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'export_timestamp': timestamp,
            'export_type': 'inference',
            'model_class': model.__class__.__name__
        }, model_path)
        export_paths['model'] = model_path
        
        # Export configuration separately
        config_path = export_subdir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        export_paths['config'] = config_path
        
        # Export model metadata
        metadata = CheckpointMetadata(
            model_type=model.__class__.__name__,
            config_hash=self._calculate_config_hash(config.to_dict()),
            file_size_mb=model_path.stat().st_size / (1024 * 1024),
            description=f"Inference export of {export_name}",
            tags=['inference', 'exported']
        )
        
        metadata_path = export_subdir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        export_paths['metadata'] = metadata_path
        
        # Create requirements file
        requirements_path = export_subdir / "requirements.txt"
        requirements = [
            "torch>=1.9.0",
            "torch-geometric>=2.0.0",
            "transformers>=4.20.0",
            "numpy>=1.21.0",
            "scipy>=1.7.0"
        ]
        
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
        export_paths['requirements'] = requirements_path
        
        # Create inference script template
        inference_script_path = export_subdir / "inference.py"
        inference_script = self._generate_inference_script(export_name)
        
        with open(inference_script_path, 'w') as f:
            f.write(inference_script)
        export_paths['inference_script'] = inference_script_path
        
        # Compress if requested
        if compress:
            archive_path = self.export_dir / f"{export_name}_{timestamp}.tar.gz"
            shutil.make_archive(str(archive_path).replace('.tar.gz', ''), 'gztar', export_subdir)
            export_paths['archive'] = archive_path
        
        logger.info(f"Model exported for inference to {export_subdir}")
        return export_paths
    
    def export_for_sharing(self,
                          checkpoint_path: Path,
                          export_name: str,
                          include_training_state: bool = False,
                          anonymize: bool = True) -> Path:
        """Export checkpoint for sharing with others"""
        
        # Load original checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create cleaned checkpoint
        shared_checkpoint = {
            'model_state_dict': checkpoint['model_state_dict'],
            'config': checkpoint['config'],
            'validation_loss': checkpoint.get('validation_loss', float('inf')),
            'epoch': checkpoint.get('epoch', 0),
            'phase': checkpoint.get('phase', 1),
            'version': checkpoint.get('version', '1.0'),
            'export_timestamp': int(time.time()),
            'export_type': 'shared'
        }
        
        # Include training state if requested
        if include_training_state and 'training_state' in checkpoint:
            shared_checkpoint['training_state'] = checkpoint['training_state']
        
        # Anonymize if requested (remove potentially sensitive info)
        if anonymize:
            # Remove optimizer state (contains learning rates, etc.)
            shared_checkpoint.pop('optimizer_state_dict', None)
            shared_checkpoint.pop('scheduler_state', None)
            
            # Remove detailed metrics history
            if 'training_state' in shared_checkpoint:
                training_state = shared_checkpoint['training_state']
                if 'metrics_history' in training_state:
                    # Keep only summary statistics
                    metrics = training_state['metrics_history']
                    if metrics:
                        training_state['metrics_summary'] = {
                            'total_epochs': len(metrics),
                            'best_val_loss': min(m.get('val_loss', float('inf')) for m in metrics),
                            'final_val_loss': metrics[-1].get('val_loss', float('inf'))
                        }
                    training_state.pop('metrics_history', None)
        
        # Save shared checkpoint
        timestamp = int(time.time())
        shared_path = self.export_dir / f"{export_name}_shared_{timestamp}.pth"
        torch.save(shared_checkpoint, shared_path)
        
        logger.info(f"Checkpoint exported for sharing to {shared_path}")
        return shared_path
    
    def _calculate_config_hash(self, config_dict: Dict[str, Any]) -> str:
        """Calculate hash of configuration for compatibility checking"""
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _generate_inference_script(self, model_name: str) -> str:
        """Generate a template inference script"""
        
        script_template = f'''"""
Inference script for {model_name}
Generated automatically by ModelExporter
"""

import torch
import json
from pathlib import Path

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load model (you'll need to import your model class)
# from your_module import UnifiedDTAModel
# model = UnifiedDTAModel(config)

# Load checkpoint
checkpoint = torch.load('model.pth', map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

def predict(drug_smiles, protein_sequence):
    """
    Make prediction for drug-protein pair
    
    Args:
        drug_smiles (str): SMILES string of the drug
        protein_sequence (str): Amino acid sequence of the protein
    
    Returns:
        float: Predicted binding affinity
    """
    # Implement your prediction logic here
    # This is a template - you'll need to adapt it to your specific model
    
    with torch.no_grad():
        # Preprocess inputs
        # drug_data = preprocess_drug(drug_smiles)
        # protein_data = preprocess_protein(protein_sequence)
        
        # Make prediction
        # prediction = model(drug_data, protein_data)
        # return prediction.item()
        
        pass

if __name__ == "__main__":
    # Example usage
    drug_smiles = "CCO"  # Ethanol as example
    protein_seq = "MKLLVLSLSLVLVAPMAAQAAEITLVPSVKLQIGDRDNRGYYWDGGHWRDH"
    
    # result = predict(drug_smiles, protein_seq)
    # print(f"Predicted affinity: {{result}}")
    
    print("Inference script template - please implement the prediction logic")
'''
        
        return script_template


class CheckpointRecovery:
    """Tools for recovering from corrupted or incomplete checkpoints"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
    
    def find_recoverable_checkpoints(self) -> List[Dict[str, Any]]:
        """Find checkpoints that might be recoverable"""
        
        recoverable = []
        
        if not self.checkpoint_dir.exists():
            return recoverable
        
        # Check all checkpoint files
        for checkpoint_file in self.checkpoint_dir.rglob("*.pth"):
            validation_result = CheckpointValidator.validate_checkpoint_file(checkpoint_file)
            
            # If there are warnings but no errors, it might be recoverable
            if validation_result['warnings'] and not validation_result['errors']:
                recoverable.append({
                    'path': checkpoint_file,
                    'validation': validation_result,
                    'recovery_potential': 'high'
                })
            elif validation_result['errors']:
                # Check if errors are minor and recoverable
                minor_errors = ['Missing recommended key', 'Parameter contains NaN']
                if all(any(minor in error for minor in minor_errors) for error in validation_result['errors']):
                    recoverable.append({
                        'path': checkpoint_file,
                        'validation': validation_result,
                        'recovery_potential': 'medium'
                    })
        
        return recoverable
    
    def attempt_recovery(self, checkpoint_path: Path, 
                        output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Attempt to recover a corrupted checkpoint"""
        
        recovery_result = {
            'success': False,
            'recovered_path': None,
            'issues_fixed': [],
            'remaining_issues': []
        }
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Fix common issues
            fixed_checkpoint = checkpoint.copy()
            
            # Fix missing version
            if 'version' not in fixed_checkpoint:
                fixed_checkpoint['version'] = '1.0'
                recovery_result['issues_fixed'].append('Added missing version')
            
            # Fix missing timestamp
            if 'timestamp' not in fixed_checkpoint:
                fixed_checkpoint['timestamp'] = int(time.time())
                recovery_result['issues_fixed'].append('Added missing timestamp')
            
            # Fix NaN parameters
            if 'model_state_dict' in fixed_checkpoint:
                model_state = fixed_checkpoint['model_state_dict']
                for key, tensor in model_state.items():
                    if torch.isnan(tensor).any():
                        # Replace NaN with zeros (not ideal, but recoverable)
                        model_state[key] = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
                        recovery_result['issues_fixed'].append(f'Fixed NaN values in {key}')
                        recovery_result['remaining_issues'].append(f'Parameter {key} had NaN values replaced with zeros')
            
            # Set output path
            if output_path is None:
                output_path = checkpoint_path.parent / f"recovered_{checkpoint_path.name}"
            
            # Save recovered checkpoint
            torch.save(fixed_checkpoint, output_path)
            
            # Validate recovered checkpoint
            validation_result = CheckpointValidator.validate_checkpoint_file(output_path)
            
            if validation_result['valid']:
                recovery_result['success'] = True
                recovery_result['recovered_path'] = output_path
                logger.info(f"Successfully recovered checkpoint to {output_path}")
            else:
                recovery_result['remaining_issues'].extend(validation_result['errors'])
                logger.warning(f"Partial recovery - remaining issues: {validation_result['errors']}")
        
        except Exception as e:
            recovery_result['remaining_issues'].append(f"Recovery failed: {str(e)}")
            logger.error(f"Failed to recover checkpoint {checkpoint_path}: {e}")
        
        return recovery_result
    
    def create_backup(self, checkpoint_path: Path, backup_dir: Optional[Path] = None) -> Path:
        """Create a backup of a checkpoint before attempting recovery"""
        
        if backup_dir is None:
            backup_dir = checkpoint_path.parent / "backups"
        
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        backup_path = backup_dir / f"backup_{timestamp}_{checkpoint_path.name}"
        
        shutil.copy2(checkpoint_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
        
        return backup_path


class CheckpointManager:
    """High-level checkpoint management interface"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.validator = CheckpointValidator()
        self.exporter = ModelExporter(str(self.checkpoint_dir / "exports"))
        self.recovery = CheckpointRecovery(str(self.checkpoint_dir))
    
    def list_all_checkpoints(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all checkpoints with their status"""
        
        checkpoints = {
            'valid': [],
            'invalid': [],
            'recoverable': []
        }
        
        for checkpoint_file in self.checkpoint_dir.rglob("*.pth"):
            validation_result = self.validator.validate_checkpoint_file(checkpoint_file)
            
            checkpoint_info = {
                'path': str(checkpoint_file),
                'validation': validation_result
            }
            
            if validation_result['valid']:
                checkpoints['valid'].append(checkpoint_info)
            elif validation_result['warnings'] and not validation_result['errors']:
                checkpoints['recoverable'].append(checkpoint_info)
            else:
                checkpoints['invalid'].append(checkpoint_info)
        
        return checkpoints
    
    def cleanup_invalid_checkpoints(self, confirm: bool = False) -> Dict[str, Any]:
        """Remove invalid checkpoints after confirmation"""
        
        cleanup_result = {
            'removed': [],
            'failed_to_remove': [],
            'total_space_freed_mb': 0.0
        }
        
        if not confirm:
            logger.warning("Cleanup not performed - set confirm=True to proceed")
            return cleanup_result
        
        all_checkpoints = self.list_all_checkpoints()
        
        for checkpoint_info in all_checkpoints['invalid']:
            checkpoint_path = Path(checkpoint_info['path'])
            
            try:
                file_size = checkpoint_path.stat().st_size / (1024 * 1024)
                checkpoint_path.unlink()
                
                cleanup_result['removed'].append(str(checkpoint_path))
                cleanup_result['total_space_freed_mb'] += file_size
                
            except Exception as e:
                cleanup_result['failed_to_remove'].append({
                    'path': str(checkpoint_path),
                    'error': str(e)
                })
        
        logger.info(f"Cleanup completed: {len(cleanup_result['removed'])} files removed, "
                   f"{cleanup_result['total_space_freed_mb']:.1f}MB freed")
        
        return cleanup_result
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all checkpoints"""
        
        all_checkpoints = self.list_all_checkpoints()
        
        summary = {
            'total_checkpoints': sum(len(checkpoints) for checkpoints in all_checkpoints.values()),
            'valid_checkpoints': len(all_checkpoints['valid']),
            'invalid_checkpoints': len(all_checkpoints['invalid']),
            'recoverable_checkpoints': len(all_checkpoints['recoverable']),
            'total_size_mb': 0.0,
            'oldest_checkpoint': None,
            'newest_checkpoint': None,
            'best_checkpoint': None
        }
        
        # Calculate total size and find oldest/newest
        all_checkpoint_files = []
        for category in all_checkpoints.values():
            for checkpoint_info in category:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists():
                    file_size = checkpoint_path.stat().st_size / (1024 * 1024)
                    summary['total_size_mb'] += file_size
                    
                    file_info = {
                        'path': str(checkpoint_path),
                        'size_mb': file_size,
                        'modified_time': checkpoint_path.stat().st_mtime,
                        'validation_loss': checkpoint_info['validation']['metadata'].get('validation_loss', float('inf'))
                    }
                    all_checkpoint_files.append(file_info)
        
        if all_checkpoint_files:
            # Find oldest and newest
            summary['oldest_checkpoint'] = min(all_checkpoint_files, key=lambda x: x['modified_time'])
            summary['newest_checkpoint'] = max(all_checkpoint_files, key=lambda x: x['modified_time'])
            
            # Find best checkpoint (lowest validation loss)
            valid_checkpoints = [f for f in all_checkpoint_files if f['validation_loss'] != float('inf')]
            if valid_checkpoints:
                summary['best_checkpoint'] = min(valid_checkpoints, key=lambda x: x['validation_loss'])
        
        return summary
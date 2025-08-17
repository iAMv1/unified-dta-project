"""
Advanced configuration validation with detailed error reporting
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path

from .config import DTAConfig

logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a configuration validation error"""
    
    def __init__(self, path: str, message: str, severity: str = 'error', 
                 suggestion: Optional[str] = None):
        self.path = path
        self.message = message
        self.severity = severity  # 'error', 'warning', 'info'
        self.suggestion = suggestion
    
    def __str__(self):
        result = f"[{self.severity.upper()}] {self.path}: {self.message}"
        if self.suggestion:
            result += f" (Suggestion: {self.suggestion})"
        return result


class ConfigValidator:
    """Advanced configuration validator with detailed error reporting"""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.info: List[ValidationError] = []
    
    def validate(self, config: DTAConfig) -> Tuple[bool, List[ValidationError]]:
        """
        Validate configuration and return detailed results
        
        Returns:
            Tuple of (is_valid, all_issues)
        """
        self.errors.clear()
        self.warnings.clear()
        self.info.clear()
        
        # Convert to dict for easier validation
        config_dict = config.to_dict()
        
        # Validate main configuration
        self._validate_main_config(config_dict)
        
        # Validate protein configuration
        if 'protein_config' in config_dict:
            self._validate_protein_config(config_dict['protein_config'], 
                                        config_dict.get('protein_encoder_type'))
        
        # Validate drug configuration
        if 'drug_config' in config_dict:
            self._validate_drug_config(config_dict['drug_config'])
        
        # Validate fusion configuration
        if config_dict.get('use_fusion') and 'fusion_config' in config_dict:
            self._validate_fusion_config(config_dict['fusion_config'])
        
        # Validate predictor configuration
        if 'predictor_config' in config_dict:
            self._validate_predictor_config(config_dict['predictor_config'])
        
        # Validate training configuration
        if 'training_config' in config_dict:
            self._validate_training_config(config_dict['training_config'])
        
        # Validate data configuration
        if 'data_config' in config_dict:
            self._validate_data_config(config_dict['data_config'])
        
        # Cross-validation checks
        self._validate_cross_dependencies(config_dict)
        
        # Collect all issues
        all_issues = self.errors + self.warnings + self.info
        is_valid = len(self.errors) == 0
        
        return is_valid, all_issues
    
    def _validate_main_config(self, config: Dict[str, Any]):
        """Validate main configuration parameters"""
        
        # Required fields
        required_fields = ['protein_encoder_type', 'drug_encoder_type', 'use_fusion']
        for field in required_fields:
            if field not in config:
                self.errors.append(ValidationError(
                    field, f"Required field missing",
                    suggestion=f"Add '{field}' to your configuration"
                ))
        
        # Valid encoder types
        valid_protein_encoders = ['esm', 'cnn']
        if config.get('protein_encoder_type') not in valid_protein_encoders:
            self.errors.append(ValidationError(
                'protein_encoder_type', 
                f"Invalid protein encoder: {config.get('protein_encoder_type')}",
                suggestion=f"Use one of: {valid_protein_encoders}"
            ))
        
        valid_drug_encoders = ['gin']
        if config.get('drug_encoder_type') not in valid_drug_encoders:
            self.errors.append(ValidationError(
                'drug_encoder_type',
                f"Invalid drug encoder: {config.get('drug_encoder_type')}",
                suggestion=f"Use one of: {valid_drug_encoders}"
            ))
        
        # Device validation
        device = config.get('device', 'auto')
        if device not in ['auto', 'cpu'] and not device.startswith('cuda'):
            self.warnings.append(ValidationError(
                'device',
                f"Unusual device specification: {device}",
                severity='warning',
                suggestion="Use 'auto', 'cpu', 'cuda', or 'cuda:N'"
            ))
        
        # Seed validation
        seed = config.get('seed')
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            self.errors.append(ValidationError(
                'seed',
                "Seed must be a non-negative integer",
                suggestion="Use a positive integer like 42"
            ))
    
    def _validate_protein_config(self, protein_config: Dict[str, Any], encoder_type: str):
        """Validate protein encoder configuration"""
        
        # Output dimension
        output_dim = protein_config.get('output_dim')
        if output_dim is not None:
            if not isinstance(output_dim, int) or output_dim <= 0:
                self.errors.append(ValidationError(
                    'protein_config.output_dim',
                    "Output dimension must be a positive integer",
                    suggestion="Use a value like 64, 128, or 256"
                ))
            elif output_dim > 1024:
                self.warnings.append(ValidationError(
                    'protein_config.output_dim',
                    f"Very large output dimension: {output_dim}",
                    severity='warning',
                    suggestion="Consider using a smaller dimension for memory efficiency"
                ))
        
        # ESM-specific validation
        if encoder_type == 'esm':
            max_length = protein_config.get('max_length')
            if max_length is not None:
                if not isinstance(max_length, int) or max_length <= 0:
                    self.errors.append(ValidationError(
                        'protein_config.max_length',
                        "Max length must be a positive integer"
                    ))
                elif max_length > 1000:
                    self.warnings.append(ValidationError(
                        'protein_config.max_length',
                        f"Very long sequences may cause memory issues: {max_length}",
                        severity='warning',
                        suggestion="Consider using max_length <= 400 for better performance"
                    ))
            
            model_name = protein_config.get('model_name')
            if model_name and not model_name.startswith('facebook/esm'):
                self.warnings.append(ValidationError(
                    'protein_config.model_name',
                    f"Non-standard ESM model: {model_name}",
                    severity='warning',
                    suggestion="Use facebook/esm2_t6_8M_UR50D or facebook/esm2_t12_35M_UR50D"
                ))
        
        # CNN-specific validation
        elif encoder_type == 'cnn':
            embed_dim = protein_config.get('embed_dim')
            if embed_dim is not None and (not isinstance(embed_dim, int) or embed_dim <= 0):
                self.errors.append(ValidationError(
                    'protein_config.embed_dim',
                    "Embedding dimension must be a positive integer"
                ))
            
            num_filters = protein_config.get('num_filters')
            if num_filters is not None:
                if isinstance(num_filters, list):
                    if not all(isinstance(f, int) and f > 0 for f in num_filters):
                        self.errors.append(ValidationError(
                            'protein_config.num_filters',
                            "All filter counts must be positive integers"
                        ))
                elif not isinstance(num_filters, int) or num_filters <= 0:
                    self.errors.append(ValidationError(
                        'protein_config.num_filters',
                        "Number of filters must be a positive integer or list of positive integers"
                    ))
    
    def _validate_drug_config(self, drug_config: Dict[str, Any]):
        """Validate drug encoder configuration"""
        
        # Output dimension
        output_dim = drug_config.get('output_dim')
        if output_dim is not None:
            if not isinstance(output_dim, int) or output_dim <= 0:
                self.errors.append(ValidationError(
                    'drug_config.output_dim',
                    "Output dimension must be a positive integer"
                ))
        
        # Number of layers
        num_layers = drug_config.get('num_layers')
        if num_layers is not None:
            if not isinstance(num_layers, int) or num_layers <= 0:
                self.errors.append(ValidationError(
                    'drug_config.num_layers',
                    "Number of layers must be a positive integer"
                ))
            elif num_layers > 10:
                self.warnings.append(ValidationError(
                    'drug_config.num_layers',
                    f"Many GIN layers may cause overfitting: {num_layers}",
                    severity='warning',
                    suggestion="Consider using 3-7 layers"
                ))
        
        # Hidden dimension
        hidden_dim = drug_config.get('hidden_dim')
        if hidden_dim is not None:
            if not isinstance(hidden_dim, int) or hidden_dim <= 0:
                self.errors.append(ValidationError(
                    'drug_config.hidden_dim',
                    "Hidden dimension must be a positive integer"
                ))
        
        # Dropout
        dropout = drug_config.get('dropout')
        if dropout is not None:
            if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
                self.errors.append(ValidationError(
                    'drug_config.dropout',
                    "Dropout must be a number between 0 and 1"
                ))
            elif dropout > 0.5:
                self.warnings.append(ValidationError(
                    'drug_config.dropout',
                    f"High dropout rate may hurt performance: {dropout}",
                    severity='warning',
                    suggestion="Consider using dropout <= 0.3"
                ))
    
    def _validate_fusion_config(self, fusion_config: Dict[str, Any]):
        """Validate fusion configuration"""
        
        hidden_dim = fusion_config.get('hidden_dim')
        if hidden_dim is not None:
            if not isinstance(hidden_dim, int) or hidden_dim <= 0:
                self.errors.append(ValidationError(
                    'fusion_config.hidden_dim',
                    "Hidden dimension must be a positive integer"
                ))
        
        num_heads = fusion_config.get('num_heads')
        if num_heads is not None:
            if not isinstance(num_heads, int) or num_heads <= 0:
                self.errors.append(ValidationError(
                    'fusion_config.num_heads',
                    "Number of heads must be a positive integer"
                ))
            elif hidden_dim is not None and hidden_dim % num_heads != 0:
                self.errors.append(ValidationError(
                    'fusion_config.num_heads',
                    f"Hidden dimension ({hidden_dim}) must be divisible by number of heads ({num_heads})"
                ))
    
    def _validate_predictor_config(self, predictor_config: Dict[str, Any]):
        """Validate predictor configuration"""
        
        hidden_dims = predictor_config.get('hidden_dims')
        if hidden_dims is not None:
            if not isinstance(hidden_dims, list):
                self.errors.append(ValidationError(
                    'predictor_config.hidden_dims',
                    "Hidden dimensions must be a list of integers"
                ))
            elif not all(isinstance(dim, int) and dim > 0 for dim in hidden_dims):
                self.errors.append(ValidationError(
                    'predictor_config.hidden_dims',
                    "All hidden dimensions must be positive integers"
                ))
            elif len(hidden_dims) > 5:
                self.warnings.append(ValidationError(
                    'predictor_config.hidden_dims',
                    f"Very deep predictor may overfit: {len(hidden_dims)} layers",
                    severity='warning',
                    suggestion="Consider using 2-4 layers"
                ))
        
        # Activation function
        activation = predictor_config.get('activation')
        if activation is not None:
            valid_activations = ['relu', 'gelu', 'leaky_relu', 'elu', 'selu', 'swish', 'silu', 'tanh', 'sigmoid']
            if activation not in valid_activations:
                self.errors.append(ValidationError(
                    'predictor_config.activation',
                    f"Unknown activation function: {activation}",
                    suggestion=f"Use one of: {valid_activations}"
                ))
    
    def _validate_training_config(self, training_config: Dict[str, Any]):
        """Validate training configuration"""
        
        # Batch size
        batch_size = training_config.get('batch_size')
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size <= 0:
                self.errors.append(ValidationError(
                    'training_config.batch_size',
                    "Batch size must be a positive integer"
                ))
            elif batch_size > 32:
                self.warnings.append(ValidationError(
                    'training_config.batch_size',
                    f"Large batch size may cause memory issues: {batch_size}",
                    severity='warning',
                    suggestion="Consider using batch_size <= 8 for ESM models"
                ))
        
        # Learning rates
        for phase in ['phase1', 'phase2']:
            lr_key = f'learning_rate_{phase}'
            lr = training_config.get(lr_key)
            if lr is not None:
                if not isinstance(lr, (int, float)) or lr <= 0:
                    self.errors.append(ValidationError(
                        f'training_config.{lr_key}',
                        "Learning rate must be a positive number"
                    ))
                elif lr > 0.1:
                    self.warnings.append(ValidationError(
                        f'training_config.{lr_key}',
                        f"Very high learning rate: {lr}",
                        severity='warning',
                        suggestion="Consider using learning rates <= 0.01"
                    ))
        
        # Memory settings
        max_memory = training_config.get('max_memory_mb')
        if max_memory is not None:
            if not isinstance(max_memory, (int, float)) or max_memory <= 0:
                self.errors.append(ValidationError(
                    'training_config.max_memory_mb',
                    "Max memory must be a positive number"
                ))
            elif max_memory < 500:
                self.warnings.append(ValidationError(
                    'training_config.max_memory_mb',
                    f"Very low memory limit may cause issues: {max_memory}MB",
                    severity='warning',
                    suggestion="Consider using at least 1000MB"
                ))
    
    def _validate_data_config(self, data_config: Dict[str, Any]):
        """Validate data configuration"""
        
        # Validation and test splits
        val_split = data_config.get('validation_split')
        test_split = data_config.get('test_split')
        
        if val_split is not None:
            if not isinstance(val_split, (int, float)) or not (0 < val_split < 1):
                self.errors.append(ValidationError(
                    'data_config.validation_split',
                    "Validation split must be between 0 and 1"
                ))
        
        if test_split is not None:
            if not isinstance(test_split, (int, float)) or not (0 < test_split < 1):
                self.errors.append(ValidationError(
                    'data_config.test_split',
                    "Test split must be between 0 and 1"
                ))
        
        if val_split is not None and test_split is not None:
            if val_split + test_split >= 1:
                self.errors.append(ValidationError(
                    'data_config',
                    f"Validation ({val_split}) + test ({test_split}) splits must be < 1"
                ))
        
        # Data directory
        data_dir = data_config.get('data_dir')
        if data_dir is not None:
            data_path = Path(data_dir)
            if not data_path.exists():
                self.warnings.append(ValidationError(
                    'data_config.data_dir',
                    f"Data directory does not exist: {data_dir}",
                    severity='warning',
                    suggestion="Create the directory or update the path"
                ))
        
        # Number of workers
        num_workers = data_config.get('num_workers')
        if num_workers is not None:
            if not isinstance(num_workers, int) or num_workers < 0:
                self.errors.append(ValidationError(
                    'data_config.num_workers',
                    "Number of workers must be a non-negative integer"
                ))
            elif num_workers > 16:
                self.warnings.append(ValidationError(
                    'data_config.num_workers',
                    f"Many workers may not improve performance: {num_workers}",
                    severity='warning',
                    suggestion="Consider using 2-8 workers"
                ))
    
    def _validate_cross_dependencies(self, config: Dict[str, Any]):
        """Validate cross-dependencies between configuration sections"""
        
        # Fusion configuration required when use_fusion=True
        if config.get('use_fusion') and 'fusion_config' not in config:
            self.errors.append(ValidationError(
                'fusion_config',
                "Fusion configuration required when use_fusion=True",
                suggestion="Add fusion_config section or set use_fusion=False"
            ))
        
        # Dimension compatibility
        protein_dim = None
        drug_dim = None
        
        if 'protein_config' in config:
            protein_dim = config['protein_config'].get('output_dim')
        
        if 'drug_config' in config:
            drug_dim = config['drug_config'].get('output_dim')
        
        if protein_dim is not None and drug_dim is not None:
            if abs(protein_dim - drug_dim) > protein_dim * 0.5:
                self.warnings.append(ValidationError(
                    'dimensions',
                    f"Large dimension mismatch: protein={protein_dim}, drug={drug_dim}",
                    severity='warning',
                    suggestion="Consider using similar dimensions for better fusion"
                ))
        
        # Memory vs batch size compatibility
        if 'training_config' in config:
            training_config = config['training_config']
            batch_size = training_config.get('batch_size')
            max_memory = training_config.get('max_memory_mb')
            
            if batch_size is not None and max_memory is not None:
                if config.get('protein_encoder_type') == 'esm' and batch_size > 4 and max_memory < 4000:
                    self.warnings.append(ValidationError(
                        'memory_compatibility',
                        f"Batch size {batch_size} may exceed memory limit {max_memory}MB with ESM",
                        severity='warning',
                        suggestion="Reduce batch size or increase memory limit"
                    ))


def validate_config_with_details(config: DTAConfig) -> Tuple[bool, str]:
    """
    Validate configuration and return detailed report
    
    Returns:
        Tuple of (is_valid, detailed_report)
    """
    validator = ConfigValidator()
    is_valid, issues = validator.validate(config)
    
    if not issues:
        return True, "✓ Configuration is valid with no issues found."
    
    report_lines = []
    
    # Group issues by severity
    errors = [issue for issue in issues if issue.severity == 'error']
    warnings = [issue for issue in issues if issue.severity == 'warning']
    info_items = [issue for issue in issues if issue.severity == 'info']
    
    if errors:
        report_lines.append(f"ERRORS ({len(errors)}):")
        for error in errors:
            report_lines.append(f"  {error}")
        report_lines.append("")
    
    if warnings:
        report_lines.append(f"WARNINGS ({len(warnings)}):")
        for warning in warnings:
            report_lines.append(f"  {warning}")
        report_lines.append("")
    
    if info_items:
        report_lines.append(f"INFO ({len(info_items)}):")
        for info in info_items:
            report_lines.append(f"  {info}")
        report_lines.append("")
    
    summary = f"Configuration validation {'FAILED' if errors else 'PASSED with warnings'}"
    if errors:
        summary += f" - {len(errors)} errors must be fixed"
    if warnings:
        summary += f" - {len(warnings)} warnings should be reviewed"
    
    report_lines.insert(0, summary)
    report_lines.insert(1, "=" * len(summary))
    report_lines.insert(2, "")
    
    return is_valid, "\n".join(report_lines)
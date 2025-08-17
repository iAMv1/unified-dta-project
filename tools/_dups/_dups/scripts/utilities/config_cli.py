#!/usr/bin/env python3
"""
Configuration Management CLI for Unified DTA System
Provides command-line utilities for managing configurations
"""

import argparse
import sys
from pathlib import Path
import logging

# Add core to path for imports
sys.path.append(str(Path(__file__).parent / "core"))

from core.config import (
    DTAConfig, load_config, save_config, validate_config,
    get_default_configs, create_config_template,
    get_environment_config, merge_configs,
    generate_config_documentation, ConfigurationManager
)
from core.model_factory import ModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Configuration Management CLI for Unified DTA System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a production configuration template
  python config_cli.py create-template production_config.yaml --type production
  
  # Validate a configuration file
  python config_cli.py validate my_config.yaml
  
  # Create all predefined templates
  python config_cli.py create-all-templates --output-dir configs/
  
  # Generate configuration documentation
  python config_cli.py generate-docs CONFIG_DOCS.md
  
  # Compare two configurations
  python config_cli.py compare config1.yaml config2.yaml
  
  # List available model configurations
  python config_cli.py list-models
  
  # Test model creation with configuration
  python config_cli.py test-model lightweight_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create template command
    create_parser = subparsers.add_parser('create-template', help='Create configuration template')
    create_parser.add_argument('output_file', help='Output configuration file path')
    create_parser.add_argument('--type', choices=['lightweight', 'production', 'high_performance'],
                              default='production', help='Configuration type')
    create_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                              help='Output format')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('config_file', help='Configuration file to validate')
    validate_parser.add_argument('--verbose', action='store_true', help='Verbose validation output')
    
    # Create all templates command
    create_all_parser = subparsers.add_parser('create-all-templates', 
                                             help='Create all predefined templates')
    create_all_parser.add_argument('--output-dir', default='configs', 
                                  help='Output directory for templates')
    create_all_parser.add_argument('--format', choices=['yaml', 'json', 'both'], 
                                  default='both', help='Output format(s)')
    
    # Generate documentation command
    docs_parser = subparsers.add_parser('generate-docs', 
                                       help='Generate configuration documentation')
    docs_parser.add_argument('output_file', help='Output documentation file')
    
    # Compare configurations command
    compare_parser = subparsers.add_parser('compare', help='Compare two configuration files')
    compare_parser.add_argument('config1', help='First configuration file')
    compare_parser.add_argument('config2', help='Second configuration file')
    compare_parser.add_argument('--output', help='Save comparison to file')
    
    # List model configurations command
    list_parser = subparsers.add_parser('list-models', help='List available model configurations')
    list_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    
    # Test model creation command
    test_parser = subparsers.add_parser('test-model', help='Test model creation with configuration')
    test_parser.add_argument('config_file', help='Configuration file to test')
    test_parser.add_argument('--dry-run', action='store_true', help='Only validate, do not create model')
    
    # Environment configuration command
    env_parser = subparsers.add_parser('env-config', help='Get environment-specific configuration')
    env_parser.add_argument('environment', choices=['development', 'staging', 'production'],
                           help='Environment name')
    env_parser.add_argument('--output', help='Save configuration to file')
    
    # Generate report command
    report_parser = subparsers.add_parser('generate-report', help='Generate configuration report')
    report_parser.add_argument('output_file', help='Output report file')
    report_parser.add_argument('--config-dir', default='configs', help='Configuration directory')
    
    # Convert configuration command
    convert_parser = subparsers.add_parser('convert', help='Convert configuration between formats')
    convert_parser.add_argument('input_file', help='Input configuration file')
    convert_parser.add_argument('output_file', help='Output configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'create-template':
            handle_create_template(args)
        elif args.command == 'validate':
            handle_validate(args)
        elif args.command == 'create-all-templates':
            handle_create_all_templates(args)
        elif args.command == 'generate-docs':
            handle_generate_docs(args)
        elif args.command == 'compare':
            handle_compare(args)
        elif args.command == 'list-models':
            handle_list_models(args)
        elif args.command == 'test-model':
            handle_test_model(args)
        elif args.command == 'env-config':
            handle_env_config(args)
        elif args.command == 'generate-report':
            handle_generate_report(args)
        elif args.command == 'convert':
            handle_convert(args)
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


def handle_create_template(args):
    """Handle create-template command"""
    output_path = Path(args.output_file)
    
    # Ensure correct extension
    if args.format == 'yaml' and not output_path.suffix in ['.yaml', '.yml']:
        output_path = output_path.with_suffix('.yaml')
    elif args.format == 'json' and output_path.suffix != '.json':
        output_path = output_path.with_suffix('.json')
    
    create_config_template(output_path, args.type)
    print(f"✓ Created {args.type} configuration template: {output_path}")


def handle_validate(args):
    """Handle validate command"""
    try:
        config = load_config(args.config_file)
        is_valid = validate_config(config)
        
        if is_valid:
            print(f"✓ Configuration is valid: {args.config_file}")
            
            if args.verbose:
                print("\nConfiguration summary:")
                print(f"  Protein Encoder: {config.protein_encoder_type}")
                print(f"  Drug Encoder: {config.drug_encoder_type}")
                print(f"  Use Fusion: {config.use_fusion}")
                print(f"  Batch Size: {config.training_config.batch_size}")
                print(f"  Device: {config.device}")
        else:
            print(f"✗ Configuration is invalid: {args.config_file}")
            sys.exit(1)
    
    except Exception as e:
        print(f"✗ Error validating configuration: {e}")
        sys.exit(1)


def handle_create_all_templates(args):
    """Handle create-all-templates command"""
    config_manager = ConfigurationManager(args.output_dir)
    
    if args.format in ['yaml', 'both']:
        config_manager.create_all_templates()
    
    if args.format in ['json', 'both']:
        # Create JSON templates
        default_configs = get_default_configs()
        output_dir = Path(args.output_dir)
        
        for config_name, config in default_configs.items():
            json_path = output_dir / f"{config_name}_config.json"
            save_config(config, json_path)
    
    print(f"✓ Created all configuration templates in: {args.output_dir}")


def handle_generate_docs(args):
    """Handle generate-docs command"""
    generate_config_documentation(args.output_file)
    print(f"✓ Generated configuration documentation: {args.output_file}")


def handle_compare(args):
    """Handle compare command"""
    config_manager = ConfigurationManager()
    differences = config_manager.compare_configs(args.config1, args.config2)
    
    if not differences:
        print("✓ Configurations are identical")
        return
    
    print(f"Found {len(differences)} differences:")
    
    comparison_output = []
    for path, diff in differences.items():
        if diff['status'] == 'added':
            line = f"  + {path}: {diff['value']}"
        elif diff['status'] == 'removed':
            line = f"  - {path}: {diff['value']}"
        elif diff['status'] == 'changed':
            line = f"  ~ {path}: {diff['old_value']} → {diff['new_value']}"
        
        print(line)
        comparison_output.append(line)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"Configuration Comparison: {args.config1} vs {args.config2}\n")
            f.write("=" * 60 + "\n\n")
            f.write("\n".join(comparison_output))
        print(f"✓ Comparison saved to: {args.output}")


def handle_list_models(args):
    """Handle list-models command"""
    configurations = ModelFactory.list_configurations()
    
    print("Available Model Configurations:")
    print("=" * 50)
    
    for name, info in configurations.items():
        print(f"\n{name.upper()}:")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Memory Usage: {info['memory_usage']}")
        print(f"  Recommended Use: {info['recommended_use']}")
        
        if args.detailed:
            try:
                details = ModelFactory.get_configuration_details(name)
                config = details['config']
                print(f"  Protein Encoder: {config['protein_encoder_type']}")
                print(f"  Drug Encoder: {config['drug_encoder_type']}")
                print(f"  Use Fusion: {config['use_fusion']}")
                if 'protein_config' in config:
                    print(f"  Protein Output Dim: {config['protein_config'].get('output_dim', 'N/A')}")
                if 'drug_config' in config:
                    print(f"  Drug Output Dim: {config['drug_config'].get('output_dim', 'N/A')}")
            except Exception as e:
                print(f"  Error getting details: {e}")


def handle_test_model(args):
    """Handle test-model command"""
    print(f"Testing model creation with configuration: {args.config_file}")
    
    # First validate the configuration
    try:
        config = load_config(args.config_file)
        is_valid = validate_config(config)
        
        if not is_valid:
            print("✗ Configuration validation failed")
            sys.exit(1)
        
        print("✓ Configuration validation passed")
        
        if args.dry_run:
            print("✓ Dry run completed successfully")
            return
        
        # Try to create model
        model = ModelFactory.create_from_config_file(args.config_file)
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model type: {type(model).__name__}")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        sys.exit(1)


def handle_env_config(args):
    """Handle env-config command"""
    config = get_environment_config(args.environment)
    
    print(f"Environment configuration for '{args.environment}':")
    print(f"  Protein Encoder: {config.protein_encoder_type}")
    print(f"  Drug Encoder: {config.drug_encoder_type}")
    print(f"  Use Fusion: {config.use_fusion}")
    print(f"  Batch Size: {config.training_config.batch_size}")
    print(f"  Phase 1 LR: {config.training_config.learning_rate_phase1}")
    print(f"  Phase 2 LR: {config.training_config.learning_rate_phase2}")
    
    if args.output:
        save_config(config, args.output)
        print(f"✓ Configuration saved to: {args.output}")


def handle_generate_report(args):
    """Handle generate-report command"""
    config_manager = ConfigurationManager(args.config_dir)
    config_manager.generate_config_report(args.output_file)
    print(f"✓ Configuration report generated: {args.output_file}")


def handle_convert(args):
    """Handle convert command"""
    # Load configuration
    config = load_config(args.input_file)
    
    # Save in new format
    save_config(config, args.output_file)
    
    input_format = Path(args.input_file).suffix
    output_format = Path(args.output_file).suffix
    
    print(f"✓ Converted configuration from {input_format} to {output_format}")
    print(f"  Input: {args.input_file}")
    print(f"  Output: {args.output_file}")


if __name__ == "__main__":
    main()
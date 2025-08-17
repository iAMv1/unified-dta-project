# Changelog

All notable changes to the Unified DTA System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package structure with proper Python packaging
- Core model implementations (ESM-2, GIN, CNN encoders)
- Multi-modal fusion mechanisms
- 2-phase progressive training system
- Comprehensive evaluation metrics
- Memory optimization utilities
- Configuration management system
- Command-line interface
- Automated testing infrastructure
- Package building and distribution scripts
- Version management and release automation

### Changed
- Unified codebase from multiple repositories
- Improved memory efficiency for large models
- Enhanced error handling and validation

### Fixed
- Memory leaks in training loops
- Device handling for mixed CPU/GPU environments
- Configuration validation edge cases

## [0.1.0] - 2024-01-XX

### Added
- Initial release of the Unified DTA System
- Integration of DeepDTAGen, DoubleSG-DTA, and ESM-2 capabilities
- Support for KIBA, Davis, and BindingDB datasets
- Lightweight and production model configurations
- RESTful API endpoints for predictions
- Comprehensive documentation and examples

### Features
- **Protein Encoders**: ESM-2 and CNN-based encoders
- **Drug Encoders**: Advanced GIN networks with residual connections
- **Fusion Mechanisms**: Cross-attention and concatenation options
- **Training**: 2-phase progressive training with memory optimization
- **Evaluation**: RMSE, correlation, concordance index metrics
- **Configuration**: YAML/JSON configuration management
- **CLI**: Command-line tools for training, prediction, and evaluation
- **API**: RESTful endpoints for integration
- **Testing**: >90% code coverage with unit and integration tests

### Requirements
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.1+
- Transformers 4.20+
- RDKit 2022.03+
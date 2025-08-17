# Examples and Tutorials

This directory contains comprehensive examples and tutorials for the Unified DTA System.

## Quick Start Examples

- [basic_usage.py](basic_usage.py) - Simple prediction example
- [batch_prediction.py](batch_prediction.py) - Batch processing example
- [custom_configuration.py](custom_configuration.py) - Custom model configuration

## Tutorial Notebooks

- [01_getting_started.ipynb](notebooks/01_getting_started.ipynb) - Introduction and basic usage
- [02_data_preparation.ipynb](notebooks/02_data_preparation.ipynb) - Data loading and preprocessing
- [03_model_training.ipynb](notebooks/03_model_training.ipynb) - Training custom models
- [04_advanced_configuration.ipynb](notebooks/04_advanced_configuration.ipynb) - Advanced configuration options
- [05_performance_optimization.ipynb](notebooks/05_performance_optimization.ipynb) - Memory and speed optimization
- [06_api_integration.ipynb](notebooks/06_api_integration.ipynb) - RESTful API usage

## Advanced Examples

- [custom_encoder.py](advanced/custom_encoder.py) - Creating custom encoders
- [ensemble_models.py](advanced/ensemble_models.py) - Ensemble prediction methods
- [cross_validation.py](advanced/cross_validation.py) - Cross-validation evaluation
- [hyperparameter_tuning.py](advanced/hyperparameter_tuning.py) - Automated hyperparameter optimization

## Comparison Examples

- [baseline_comparison.py](comparisons/baseline_comparison.py) - Compare with baseline models
- [architecture_comparison.py](comparisons/architecture_comparison.py) - Compare different architectures
- [dataset_comparison.py](comparisons/dataset_comparison.py) - Performance across datasets

## Performance Examples

- [memory_optimization.py](performance/memory_optimization.py) - Memory usage optimization
- [speed_benchmarking.py](performance/speed_benchmarking.py) - Speed benchmarking
- [gpu_utilization.py](performance/gpu_utilization.py) - GPU optimization techniques

## Integration Examples

- [flask_integration.py](integration/flask_integration.py) - Flask web application
- [streamlit_app.py](integration/streamlit_app.py) - Streamlit dashboard
- [docker_deployment/](integration/docker_deployment/) - Docker deployment example

## Running Examples

### Prerequisites
```bash
pip install unified-dta[examples]
# or
pip install jupyter matplotlib seaborn plotly streamlit
```

### Basic Examples
```bash
python examples/basic_usage.py
python examples/batch_prediction.py
```

### Jupyter Notebooks
```bash
jupyter notebook examples/notebooks/
```

### Advanced Examples
```bash
python examples/advanced/custom_encoder.py
python examples/performance/memory_optimization.py
```
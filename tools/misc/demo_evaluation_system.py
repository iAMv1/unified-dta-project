#!/usr/bin/env python3
"""
Demonstration of the Comprehensive Evaluation and Metrics System
Shows how to use all evaluation components for DTA model assessment
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import logging
import time

# Import our evaluation system
from core.evaluation import (
    ComprehensiveEvaluator,
    BaselineComparator,
    CrossValidator,
    BenchmarkSuite,
    AutomatedEvaluationPipeline,
    EvaluationReporter
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockDTAModel(nn.Module):
    """Mock DTA model for demonstration purposes"""
    
    def __init__(self, input_dim: int = 100, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.trained = False
    
    def forward(self, x):
        return self.network(x).squeeze()


def generate_mock_dta_data(n_samples: int = 1000, n_features: int = 100, 
                          noise_level: float = 0.5, random_state: int = 42):
    """Generate mock DTA data for demonstration"""
    np.random.seed(random_state)
    
    # Generate features (representing combined drug + protein features)
    X = np.random.randn(n_samples, n_features)
    
    # Generate true affinities with some underlying pattern
    true_weights = np.random.randn(n_features) * 0.1
    y_true = X @ true_weights + np.random.normal(0, noise_level, n_samples)
    
    # Add some non-linear effects
    y_true += 0.1 * np.sin(X[:, 0]) + 0.05 * X[:, 1] * X[:, 2]
    
    # Normalize to typical DTA range
    y_true = (y_true - y_true.mean()) / y_true.std() * 2 + 7  # Mean ~7, std ~2
    
    return X, y_true


def mock_train_function(model, X_train, y_train, epochs: int = 50):
    """Mock training function for demonstration"""
    logger.info(f"Training model on {len(X_train)} samples for {epochs} epochs")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    model.trained = True
    logger.info("Training completed")
    return model


def mock_predict_function(model, X_test):
    """Mock prediction function for demonstration"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        predictions = model(X_tensor)
        return predictions.numpy()


def demonstrate_basic_evaluation():
    """Demonstrate basic evaluation metrics calculation"""
    logger.info("=== Demonstrating Basic Evaluation Metrics ===")
    
    # Generate sample predictions
    np.random.seed(42)
    y_true = np.random.normal(7.0, 2.0, 500)
    y_pred = y_true + np.random.normal(0, 0.5, 500)  # Add prediction noise
    
    # Create evaluator and calculate metrics
    evaluator = ComprehensiveEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    
    print(metrics.to_summary_string())
    
    # Generate evaluation report
    reporter = EvaluationReporter("demo_reports")
    report_path = reporter.generate_report(metrics, "DemoModel", "DemoDataset")
    logger.info(f"Basic evaluation report saved to: {report_path}")


def demonstrate_baseline_comparison():
    """Demonstrate comparison against baseline models"""
    logger.info("=== Demonstrating Baseline Comparison ===")
    
    # Generate training and test data
    X_train, y_train = generate_mock_dta_data(800, random_state=42)
    X_test, y_test = generate_mock_dta_data(200, random_state=123)
    
    # Create and train a mock model
    model = MockDTAModel()
    trained_model = mock_train_function(model, X_train, y_train, epochs=30)
    
    # Get model predictions
    model_predictions = mock_predict_function(trained_model, X_test)
    
    # Compare against baselines
    comparator = BaselineComparator()
    comparison_results = comparator.compare_against_baselines(
        X_train, y_train, X_test, y_test, model_predictions, "MockDTAModel"
    )
    
    print("\nBaseline Comparison Results:")
    print(f"Model rank: {comparison_results['model_rank']} out of {len(comparison_results['metrics'])} models")
    print(f"Best baseline: {comparison_results['best_baseline']}")
    
    print("\nImprovements over baselines:")
    for baseline_name, improvements in comparison_results['improvements'].items():
        rmse_imp = improvements['rmse_improvement']
        pearson_imp = improvements['pearson_improvement']
        ci_imp = improvements['ci_improvement']
        print(f"  vs {baseline_name}:")
        print(f"    RMSE improvement: {rmse_imp:.2%}")
        print(f"    Pearson improvement: {pearson_imp:.4f}")
        print(f"    CI improvement: {ci_imp:.4f}")


def demonstrate_cross_validation():
    """Demonstrate k-fold cross-validation"""
    logger.info("=== Demonstrating Cross-Validation ===")
    
    # Generate data
    X, y = generate_mock_dta_data(1000, random_state=42)
    
    # Setup cross-validator
    cv = CrossValidator(n_splits=5, random_state=42)
    
    # Define model configuration
    model_config = {'input_dim': X.shape[1], 'hidden_dim': 64}
    
    # Run cross-validation
    cv_results = cv.cross_validate_model(
        MockDTAModel, model_config, X, y, 
        lambda m, X_tr, y_tr: mock_train_function(m, X_tr, y_tr, epochs=20),
        mock_predict_function
    )
    
    print("\nCross-Validation Results:")
    cv_stats = cv_results['cv_statistics']
    print(f"Mean RMSE: {cv_stats['rmse_mean']:.4f} ± {cv_stats['rmse_std']:.4f}")
    print(f"Mean Pearson r: {cv_stats['pearson_r_mean']:.4f} ± {cv_stats['pearson_r_std']:.4f}")
    print(f"Mean Concordance Index: {cv_stats['concordance_index_mean']:.4f} ± {cv_stats['concordance_index_std']:.4f}")
    
    print("\nFold-by-fold results:")
    for fold_result in cv_results['fold_results']:
        fold_metrics = fold_result['metrics']
        print(f"  Fold {fold_result['fold']}: RMSE={fold_metrics['rmse']:.4f}, "
              f"Pearson r={fold_metrics['pearson_r']:.4f}")


def demonstrate_benchmarking():
    """Demonstrate comprehensive benchmarking"""
    logger.info("=== Demonstrating Benchmarking Suite ===")
    
    # Generate multiple test datasets
    test_datasets = {
        'KIBA_test': generate_mock_dta_data(300, random_state=42),
        'Davis_test': generate_mock_dta_data(250, random_state=123),
        'BindingDB_test': generate_mock_dta_data(400, random_state=456)
    }
    
    # Generate training data for baseline comparison
    baseline_data = {
        'KIBA_test': generate_mock_dta_data(800, random_state=41),
        'Davis_test': generate_mock_dta_data(750, random_state=122),
        'BindingDB_test': generate_mock_dta_data(900, random_state=455)
    }
    
    # Create and train model
    X_train, y_train = generate_mock_dta_data(1000, random_state=42)
    model = MockDTAModel()
    trained_model = mock_train_function(model, X_train, y_train, epochs=40)
    
    # Run benchmark
    benchmark_suite = BenchmarkSuite("demo_benchmarks")
    benchmark_results = benchmark_suite.benchmark_model(
        trained_model, "MockDTAModel", test_datasets, mock_predict_function, baseline_data
    )
    
    print("\nBenchmark Results:")
    overall_metrics = benchmark_results['overall_summary']['overall_metrics']
    print(f"Overall RMSE: {overall_metrics['rmse']:.4f}")
    print(f"Overall Pearson r: {overall_metrics['pearson_r']:.4f}")
    print(f"Overall Concordance Index: {overall_metrics['concordance_index']:.4f}")
    
    timing_results = benchmark_results['timing_results']
    print(f"Mean prediction time: {timing_results['mean_prediction_time']:.4f}s")
    print(f"Mean throughput: {timing_results['mean_samples_per_second']:.1f} samples/s")
    
    print("\nDataset-specific results:")
    for dataset_name, results in benchmark_results['dataset_results'].items():
        metrics = results['metrics']
        print(f"  {dataset_name}: RMSE={metrics['rmse']:.4f}, "
              f"Speed={results['samples_per_second']:.1f} samples/s")
    
    # Generate benchmark report
    report_path = benchmark_suite.generate_benchmark_report(benchmark_results)
    logger.info(f"Benchmark report saved to: {report_path}")


def demonstrate_model_comparison():
    """Demonstrate comparison between multiple models"""
    logger.info("=== Demonstrating Model Comparison ===")
    
    # Generate test data
    X_test, y_test = generate_mock_dta_data(500, random_state=42)
    
    # Create multiple models with different configurations
    models = {
        'Small_Model': MockDTAModel(input_dim=100, hidden_dim=32),
        'Medium_Model': MockDTAModel(input_dim=100, hidden_dim=64),
        'Large_Model': MockDTAModel(input_dim=100, hidden_dim=128)
    }
    
    # Train all models
    X_train, y_train = generate_mock_dta_data(800, random_state=41)
    trained_models = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}")
        trained_models[model_name] = mock_train_function(model, X_train, y_train, epochs=30)
    
    # Get predictions from all models
    predictions = {}
    for model_name, model in trained_models.items():
        predictions[model_name] = mock_predict_function(model, X_test)
    
    # Compare models
    evaluator = ComprehensiveEvaluator()
    comparison_results = evaluator.compare_models(y_test, predictions)
    
    print("\nModel Comparison Results:")
    sorted_models = sorted(comparison_results.items(), key=lambda x: x[1].rmse)
    
    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        print(f"{rank}. {model_name}:")
        print(f"   RMSE: {metrics.rmse:.4f}")
        print(f"   Pearson r: {metrics.pearson_r:.4f}")
        print(f"   Concordance Index: {metrics.concordance_index:.4f}")
    
    # Generate comparison report
    reporter = EvaluationReporter("demo_reports")
    report_path = reporter.generate_comparison_report(comparison_results, "ModelComparison")
    logger.info(f"Model comparison report saved to: {report_path}")


def demonstrate_complete_pipeline():
    """Demonstrate the complete automated evaluation pipeline"""
    logger.info("=== Demonstrating Complete Evaluation Pipeline ===")
    
    # Generate datasets with train/test splits
    datasets = {
        'KIBA': {
            'train': generate_mock_dta_data(800, random_state=42),
            'test': generate_mock_dta_data(200, random_state=142)
        },
        'Davis': {
            'train': generate_mock_dta_data(600, random_state=43),
            'test': generate_mock_dta_data(150, random_state=143)
        }
    }
    
    # Create model
    model = MockDTAModel()
    model_config = {'input_dim': 100, 'hidden_dim': 64}
    
    # Setup pipeline
    pipeline = AutomatedEvaluationPipeline("demo_pipeline_results")
    
    # Run complete evaluation
    pipeline_results = pipeline.run_complete_evaluation(
        model, "CompleteDemoModel", datasets,
        lambda m, X_tr, y_tr: mock_train_function(m, X_tr, y_tr, epochs=25),
        mock_predict_function, model_config
    )
    
    print("\nComplete Pipeline Results:")
    summary = pipeline_results['summary']
    
    if 'cross_validation_summary' in summary:
        cv_summary = summary['cross_validation_summary']
        print(f"Cross-validation mean RMSE: {cv_summary['mean_cv_rmse']:.4f} ± {cv_summary['std_cv_rmse']:.4f}")
    
    if 'benchmark_summary' in summary:
        bench_summary = summary['benchmark_summary']
        print(f"Benchmark RMSE: {bench_summary['rmse']:.4f}")
        print(f"Benchmark Pearson r: {bench_summary['pearson_r']:.4f}")
    
    print(f"Datasets evaluated: {', '.join(summary['datasets_evaluated'])}")
    
    logger.info("Complete evaluation pipeline finished successfully")


def main():
    """Run all demonstration examples"""
    logger.info("Starting Evaluation System Demonstration")
    
    # Create output directories
    Path("demo_reports").mkdir(exist_ok=True)
    Path("demo_benchmarks").mkdir(exist_ok=True)
    Path("demo_pipeline_results").mkdir(exist_ok=True)
    
    try:
        # Run all demonstrations
        demonstrate_basic_evaluation()
        print("\n" + "="*80 + "\n")
        
        demonstrate_baseline_comparison()
        print("\n" + "="*80 + "\n")
        
        demonstrate_cross_validation()
        print("\n" + "="*80 + "\n")
        
        demonstrate_benchmarking()
        print("\n" + "="*80 + "\n")
        
        demonstrate_model_comparison()
        print("\n" + "="*80 + "\n")
        
        demonstrate_complete_pipeline()
        
        logger.info("All demonstrations completed successfully!")
        
        print("\n" + "="*80)
        print("DEMONSTRATION SUMMARY")
        print("="*80)
        print("✓ Basic evaluation metrics calculation")
        print("✓ Baseline model comparison")
        print("✓ K-fold cross-validation")
        print("✓ Comprehensive benchmarking")
        print("✓ Multi-model comparison")
        print("✓ Complete automated evaluation pipeline")
        print("\nCheck the following directories for generated reports:")
        print("  - demo_reports/: Evaluation reports")
        print("  - demo_benchmarks/: Benchmark results")
        print("  - demo_pipeline_results/: Complete pipeline results")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
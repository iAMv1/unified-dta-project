"""
Comprehensive Evaluation and Metrics System for Unified DTA System
Implements RMSE, MSE, Pearson/Spearman correlations, concordance index, and statistical testing
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
import time
from abc import ABC, abstractmethod
import warnings

# Statistical imports
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for comprehensive evaluation metrics"""
    # Basic regression metrics
    rmse: float
    mse: float
    mae: float
    
    # Correlation metrics
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    
    # Ranking metrics
    concordance_index: float
    
    # Distribution metrics
    r_squared: float
    explained_variance: float
    
    # Sample statistics
    n_samples: int
    mean_true: float
    std_true: float
    mean_pred: float
    std_pred: float
    
    # Additional metrics
    max_error: float
    median_absolute_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_summary_string(self) -> str:
        """Create human-readable summary"""
        return f"""
Evaluation Metrics Summary:
==========================
Regression Metrics:
  RMSE: {self.rmse:.4f}
  MSE: {self.mse:.4f}
  MAE: {self.mae:.4f}
  R²: {self.r_squared:.4f}

Correlation Metrics:
  Pearson r: {self.pearson_r:.4f} (p={self.pearson_p:.4e})
  Spearman ρ: {self.spearman_r:.4f} (p={self.spearman_p:.4e})

Ranking Metrics:
  Concordance Index: {self.concordance_index:.4f}

Sample Statistics:
  N samples: {self.n_samples}
  True values: μ={self.mean_true:.4f}, σ={self.std_true:.4f}
  Predictions: μ={self.mean_pred:.4f}, σ={self.std_pred:.4f}
"""


@dataclass
class StatisticalTestResults:
    """Results from statistical significance tests"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCalculator:
    """Core metrics calculation utilities"""
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Square Error"""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def calculate_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Calculate Pearson correlation coefficient and p-value"""
        try:
            r, p = pearsonr(y_true, y_pred)
            return float(r), float(p)
        except Exception as e:
            logger.warning(f"Error calculating Pearson correlation: {e}")
            return 0.0, 1.0
    
    @staticmethod
    def calculate_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Calculate Spearman rank correlation coefficient and p-value"""
        try:
            rho, p = spearmanr(y_true, y_pred)
            return float(rho), float(p)
        except Exception as e:
            logger.warning(f"Error calculating Spearman correlation: {e}")
            return 0.0, 1.0
    
    @staticmethod
    def calculate_concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Concordance Index (C-index) for ranking evaluation
        
        The concordance index measures the fraction of pairs of samples 
        whose relative ranking is correctly predicted.
        """
        n = len(y_true)
        if n < 2:
            return 0.5  # Random performance for single sample
        
        concordant_pairs = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # Skip tied pairs in true values
                if y_true[i] == y_true[j]:
                    continue
                
                total_pairs += 1
                
                # Check if prediction order matches true order
                true_order = y_true[i] > y_true[j]
                pred_order = y_pred[i] > y_pred[j]
                
                if true_order == pred_order:
                    concordant_pairs += 1
        
        if total_pairs == 0:
            return 0.5  # All pairs are tied
        
        return concordant_pairs / total_pairs
    
    @staticmethod
    def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared (coefficient of determination)"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0  # Perfect prediction when all true values are the same
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def calculate_explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate explained variance score"""
        var_y = np.var(y_true)
        if var_y == 0:
            return 0.0
        
        return 1 - np.var(y_true - y_pred) / var_y
    
    @staticmethod
    def calculate_max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate maximum absolute error"""
        return np.max(np.abs(y_true - y_pred))
    
    @staticmethod
    def calculate_median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate median absolute error"""
        return np.median(np.abs(y_true - y_pred))


class ComprehensiveEvaluator:
    """Main evaluation class that combines all metrics"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.metrics_calculator = MetricsCalculator()
        
    def evaluate(self, y_true: Union[np.ndarray, List, torch.Tensor], 
                y_pred: Union[np.ndarray, List, torch.Tensor]) -> EvaluationMetrics:
        """
        Comprehensive evaluation of predictions against true values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            EvaluationMetrics object with all computed metrics
        """
        # Convert inputs to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        # Validate inputs
        self._validate_inputs(y_true, y_pred)
        
        # Calculate all metrics
        metrics = EvaluationMetrics(
            # Basic regression metrics
            rmse=self.metrics_calculator.calculate_rmse(y_true, y_pred),
            mse=self.metrics_calculator.calculate_mse(y_true, y_pred),
            mae=self.metrics_calculator.calculate_mae(y_true, y_pred),
            
            # Correlation metrics
            pearson_r=0.0, pearson_p=1.0,  # Will be filled below
            spearman_r=0.0, spearman_p=1.0,  # Will be filled below
            
            # Ranking metrics
            concordance_index=self.metrics_calculator.calculate_concordance_index(y_true, y_pred),
            
            # Distribution metrics
            r_squared=self.metrics_calculator.calculate_r_squared(y_true, y_pred),
            explained_variance=self.metrics_calculator.calculate_explained_variance(y_true, y_pred),
            
            # Sample statistics
            n_samples=len(y_true),
            mean_true=float(np.mean(y_true)),
            std_true=float(np.std(y_true)),
            mean_pred=float(np.mean(y_pred)),
            std_pred=float(np.std(y_pred)),
            
            # Additional metrics
            max_error=self.metrics_calculator.calculate_max_error(y_true, y_pred),
            median_absolute_error=self.metrics_calculator.calculate_median_absolute_error(y_true, y_pred)
        )
        
        # Calculate correlations
        metrics.pearson_r, metrics.pearson_p = self.metrics_calculator.calculate_pearson_correlation(y_true, y_pred)
        metrics.spearman_r, metrics.spearman_p = self.metrics_calculator.calculate_spearman_correlation(y_true, y_pred)
        
        return metrics
    
    def compare_models(self, y_true: np.ndarray, 
                      predictions_dict: Dict[str, np.ndarray]) -> Dict[str, EvaluationMetrics]:
        """
        Compare multiple models on the same dataset
        
        Args:
            y_true: True values
            predictions_dict: Dictionary mapping model names to predictions
            
        Returns:
            Dictionary mapping model names to their evaluation metrics
        """
        results = {}
        
        for model_name, y_pred in predictions_dict.items():
            try:
                results[model_name] = self.evaluate(y_true, y_pred)
                logger.info(f"Evaluated model: {model_name}")
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                continue
        
        return results
    
    def statistical_significance_test(self, y_true: np.ndarray, 
                                    y_pred1: np.ndarray, 
                                    y_pred2: np.ndarray,
                                    test_type: str = 'wilcoxon') -> StatisticalTestResults:
        """
        Test statistical significance between two models' predictions
        
        Args:
            y_true: True values
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            test_type: Type of test ('wilcoxon', 'ttest', 'mcnemar')
            
        Returns:
            StatisticalTestResults object
        """
        # Calculate residuals
        residuals1 = np.abs(y_true - y_pred1)
        residuals2 = np.abs(y_true - y_pred2)
        
        if test_type == 'wilcoxon':
            # Wilcoxon signed-rank test for paired samples
            try:
                statistic, p_value = stats.wilcoxon(residuals1, residuals2, alternative='two-sided')
                test_name = "Wilcoxon Signed-Rank Test"
            except Exception as e:
                logger.warning(f"Wilcoxon test failed: {e}")
                statistic, p_value = 0.0, 1.0
                test_name = "Wilcoxon Signed-Rank Test (Failed)"
                
        elif test_type == 'ttest':
            # Paired t-test
            try:
                statistic, p_value = stats.ttest_rel(residuals1, residuals2)
                test_name = "Paired T-Test"
            except Exception as e:
                logger.warning(f"T-test failed: {e}")
                statistic, p_value = 0.0, 1.0
                test_name = "Paired T-Test (Failed)"
                
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        is_significant = p_value < self.significance_level
        
        # Create interpretation
        if is_significant:
            if np.mean(residuals1) < np.mean(residuals2):
                interpretation = "Model 1 significantly outperforms Model 2"
            else:
                interpretation = "Model 2 significantly outperforms Model 1"
        else:
            interpretation = "No significant difference between models"
        
        return StatisticalTestResults(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=self.significance_level,
            interpretation=interpretation
        )
    
    def _to_numpy(self, data: Union[np.ndarray, List, torch.Tensor]) -> np.ndarray:
        """Convert various data types to numpy array"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Validate input arrays"""
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
        
        if len(y_true) == 0:
            raise ValueError("Empty input arrays")
        
        if not np.isfinite(y_true).all():
            raise ValueError("y_true contains non-finite values")
        
        if not np.isfinite(y_pred).all():
            raise ValueError("y_pred contains non-finite values")


class BaselineComparator:
    """Compare model performance against baseline models"""
    
    def __init__(self):
        self.baseline_models = {
            'mean_predictor': self._mean_predictor,
            'median_predictor': self._median_predictor,
            'linear_regression': self._linear_regression_baseline,
            'random_predictor': self._random_predictor
        }
    
    def compare_against_baselines(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                model_predictions: np.ndarray,
                                model_name: str = "Model") -> Dict[str, Any]:
        """
        Compare model against various baseline approaches
        
        Args:
            X_train: Training features (not used for simple baselines)
            y_train: Training targets
            X_test: Test features (not used for simple baselines)
            y_test: Test targets
            model_predictions: Model's predictions on test set
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary with comparison results
        """
        evaluator = ComprehensiveEvaluator()
        results = {}
        
        # Evaluate main model
        results[model_name] = evaluator.evaluate(y_test, model_predictions)
        
        # Evaluate baseline models
        for baseline_name, baseline_func in self.baseline_models.items():
            try:
                baseline_pred = baseline_func(y_train, len(y_test))
                results[baseline_name] = evaluator.evaluate(y_test, baseline_pred)
                logger.info(f"Evaluated baseline: {baseline_name}")
            except Exception as e:
                logger.error(f"Error evaluating baseline {baseline_name}: {e}")
                continue
        
        # Calculate improvement over baselines
        improvements = {}
        main_metrics = results[model_name]
        
        for baseline_name, baseline_metrics in results.items():
            if baseline_name == model_name:
                continue
            
            improvements[baseline_name] = {
                'rmse_improvement': (baseline_metrics.rmse - main_metrics.rmse) / baseline_metrics.rmse,
                'pearson_improvement': main_metrics.pearson_r - baseline_metrics.pearson_r,
                'ci_improvement': main_metrics.concordance_index - baseline_metrics.concordance_index
            }
        
        return {
            'metrics': results,
            'improvements': improvements,
            'best_baseline': min(results.keys(), key=lambda k: results[k].rmse if k != model_name else float('inf')),
            'model_rank': sorted(results.keys(), key=lambda k: results[k].rmse).index(model_name) + 1
        }
    
    def _mean_predictor(self, y_train: np.ndarray, n_samples: int) -> np.ndarray:
        """Predict using training set mean"""
        return np.full(n_samples, np.mean(y_train))
    
    def _median_predictor(self, y_train: np.ndarray, n_samples: int) -> np.ndarray:
        """Predict using training set median"""
        return np.full(n_samples, np.median(y_train))
    
    def _linear_regression_baseline(self, y_train: np.ndarray, n_samples: int) -> np.ndarray:
        """Simple linear trend baseline (placeholder)"""
        # For now, just return mean (would need features for real linear regression)
        return np.full(n_samples, np.mean(y_train))
    
    def _random_predictor(self, y_train: np.ndarray, n_samples: int) -> np.ndarray:
        """Random predictions within training range"""
        min_val, max_val = np.min(y_train), np.max(y_train)
        return np.random.uniform(min_val, max_val, n_samples)


class EvaluationReporter:
    """Generate comprehensive evaluation reports"""
    
    def __init__(self, output_dir: str = "evaluation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self, metrics: EvaluationMetrics, 
                       model_name: str = "Model",
                       dataset_name: str = "Dataset",
                       additional_info: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive evaluation report"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"evaluation_report_{model_name}_{dataset_name}_{timestamp}.txt"
        report_path = self.output_dir / report_filename
        
        report_content = f"""
Evaluation Report
================
Model: {model_name}
Dataset: {dataset_name}
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

{metrics.to_summary_string()}

Detailed Analysis:
-----------------
Performance Level:
  - RMSE: {'Excellent' if metrics.rmse < 0.5 else 'Good' if metrics.rmse < 1.0 else 'Fair' if metrics.rmse < 2.0 else 'Poor'}
  - Pearson r: {'Excellent' if metrics.pearson_r > 0.9 else 'Good' if metrics.pearson_r > 0.7 else 'Fair' if metrics.pearson_r > 0.5 else 'Poor'}
  - Concordance Index: {'Excellent' if metrics.concordance_index > 0.8 else 'Good' if metrics.concordance_index > 0.7 else 'Fair' if metrics.concordance_index > 0.6 else 'Poor'}

Statistical Significance:
  - Pearson correlation: {'Significant' if metrics.pearson_p < 0.05 else 'Not significant'} (p={metrics.pearson_p:.4e})
  - Spearman correlation: {'Significant' if metrics.spearman_p < 0.05 else 'Not significant'} (p={metrics.spearman_p:.4e})

Prediction Quality:
  - Bias: {metrics.mean_pred - metrics.mean_true:.4f} (positive = overestimation)
  - Variance ratio: {(metrics.std_pred / metrics.std_true):.4f} (1.0 = perfect variance matching)
  - Max error: {metrics.max_error:.4f}
"""
        
        if additional_info:
            report_content += "\nAdditional Information:\n"
            for key, value in additional_info.items():
                report_content += f"  {key}: {value}\n"
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Evaluation report saved: {report_path}")
        return str(report_path)
    
    def generate_comparison_report(self, comparison_results: Dict[str, EvaluationMetrics],
                                 dataset_name: str = "Dataset") -> str:
        """Generate model comparison report"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"model_comparison_{dataset_name}_{timestamp}.txt"
        report_path = self.output_dir / report_filename
        
        # Sort models by RMSE
        sorted_models = sorted(comparison_results.items(), key=lambda x: x[1].rmse)
        
        report_content = f"""
Model Comparison Report
======================
Dataset: {dataset_name}
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Models Compared: {len(comparison_results)}

Performance Ranking (by RMSE):
"""
        
        for rank, (model_name, metrics) in enumerate(sorted_models, 1):
            report_content += f"""
{rank}. {model_name}
   RMSE: {metrics.rmse:.4f}
   Pearson r: {metrics.pearson_r:.4f}
   Spearman ρ: {metrics.spearman_r:.4f}
   Concordance Index: {metrics.concordance_index:.4f}
   R²: {metrics.r_squared:.4f}
"""
        
        # Performance comparison table
        report_content += "\nDetailed Comparison:\n"
        report_content += "Model".ljust(20) + "RMSE".ljust(10) + "Pearson".ljust(10) + "Spearman".ljust(10) + "CI".ljust(10) + "R²".ljust(10) + "\n"
        report_content += "-" * 70 + "\n"
        
        for model_name, metrics in sorted_models:
            report_content += (
                model_name[:19].ljust(20) +
                f"{metrics.rmse:.4f}".ljust(10) +
                f"{metrics.pearson_r:.4f}".ljust(10) +
                f"{metrics.spearman_r:.4f}".ljust(10) +
                f"{metrics.concordance_index:.4f}".ljust(10) +
                f"{metrics.r_squared:.4f}".ljust(10) + "\n"
            )
        
        # Best model summary
        best_model_name, best_metrics = sorted_models[0]
        report_content += f"""
Best Performing Model: {best_model_name}
{best_metrics.to_summary_string()}
"""
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Model comparison report saved: {report_path}")
        return str(report_path)


if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.normal(5.0, 2.0, n_samples)
    y_pred = y_true + np.random.normal(0, 0.5, n_samples)  # Add some noise
    
    # Test comprehensive evaluation
    evaluator = ComprehensiveEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    
    print("Evaluation Metrics Test:")
    print(metrics.to_summary_string())
    
    # Test baseline comparison
    comparator = BaselineComparator()
    X_train = np.random.randn(800, 10)  # Dummy features
    y_train = y_true[:800]
    X_test = np.random.randn(200, 10)
    y_test = y_true[800:]
    model_pred = y_pred[800:]
    
    baseline_results = comparator.compare_against_baselines(
        X_train, y_train, X_test, y_test, model_pred, "TestModel"
    )
    
    print("\nBaseline Comparison Test:")
    for model_name, improvement in baseline_results['improvements'].items():
        print(f"vs {model_name}: RMSE improvement = {improvement['rmse_improvement']:.2%}")
    
    # Test report generation
    reporter = EvaluationReporter()
    report_path = reporter.generate_report(metrics, "TestModel", "TestDataset")
    print(f"\nReport generated: {report_path}")


class CrossValidator:
    """K-fold cross-validation with consistent splits for DTA evaluation"""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.evaluator = ComprehensiveEvaluator()
        
    def cross_validate_model(self, model_class, model_config: Dict[str, Any],
                           X: np.ndarray, y: np.ndarray,
                           train_func: Callable,
                           predict_func: Callable,
                           stratify: bool = False) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation on a model
        
        Args:
            model_class: Model class to instantiate
            model_config: Configuration for model initialization
            X: Features (can be complex data structures for DTA)
            y: Target values
            train_func: Function to train model (model, X_train, y_train) -> trained_model
            predict_func: Function to make predictions (model, X_test) -> predictions
            stratify: Whether to use stratified splits
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting {self.n_splits}-fold cross-validation")
        
        # Create cross-validation splits
        if stratify:
            # For stratified splits, bin continuous targets
            y_binned = self._bin_targets_for_stratification(y)
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = list(cv.split(X, y_binned))
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = list(cv.split(X, y))
        
        # Store results for each fold
        fold_results = []
        fold_metrics = []
        all_predictions = []
        all_true_values = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Processing fold {fold_idx + 1}/{self.n_splits}")
            
            # Split data
            X_train, X_val = self._split_data(X, train_idx, val_idx)
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                # Initialize and train model
                model = model_class(**model_config)
                trained_model = train_func(model, X_train, y_train)
                
                # Make predictions
                y_pred = predict_func(trained_model, X_val)
                
                # Evaluate fold
                fold_metrics_obj = self.evaluator.evaluate(y_val, y_pred)
                fold_metrics.append(fold_metrics_obj)
                
                # Store predictions for overall evaluation
                all_predictions.extend(y_pred)
                all_true_values.extend(y_val)
                
                fold_results.append({
                    'fold': fold_idx + 1,
                    'train_size': len(train_idx),
                    'val_size': len(val_idx),
                    'metrics': fold_metrics_obj.to_dict()
                })
                
                logger.info(f"Fold {fold_idx + 1} - RMSE: {fold_metrics_obj.rmse:.4f}, "
                          f"Pearson: {fold_metrics_obj.pearson_r:.4f}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold_idx + 1}: {e}")
                continue
        
        # Calculate overall statistics
        overall_metrics = self.evaluator.evaluate(all_true_values, all_predictions)
        
        # Calculate cross-validation statistics
        cv_stats = self._calculate_cv_statistics(fold_metrics)
        
        results = {
            'n_splits': self.n_splits,
            'fold_results': fold_results,
            'cv_statistics': cv_stats,
            'overall_metrics': overall_metrics.to_dict(),
            'all_predictions': all_predictions,
            'all_true_values': all_true_values
        }
        
        logger.info(f"Cross-validation completed. Mean RMSE: {cv_stats['rmse_mean']:.4f} ± {cv_stats['rmse_std']:.4f}")
        
        return results
    
    def compare_models_cv(self, models_config: Dict[str, Dict[str, Any]],
                         X: np.ndarray, y: np.ndarray,
                         train_func: Callable, predict_func: Callable) -> Dict[str, Any]:
        """
        Compare multiple models using cross-validation
        
        Args:
            models_config: Dictionary mapping model names to (model_class, config) tuples
            X: Features
            y: Target values
            train_func: Training function
            predict_func: Prediction function
            
        Returns:
            Comparison results across all models
        """
        logger.info(f"Comparing {len(models_config)} models using cross-validation")
        
        model_results = {}
        
        for model_name, (model_class, config) in models_config.items():
            logger.info(f"Cross-validating model: {model_name}")
            
            try:
                cv_results = self.cross_validate_model(
                    model_class, config, X, y, train_func, predict_func
                )
                model_results[model_name] = cv_results
                
            except Exception as e:
                logger.error(f"Error cross-validating model {model_name}: {e}")
                continue
        
        # Generate comparison summary
        comparison_summary = self._generate_model_comparison_summary(model_results)
        
        return {
            'model_results': model_results,
            'comparison_summary': comparison_summary,
            'best_model': comparison_summary['best_model_by_rmse']
        }
    
    def _split_data(self, X: Union[np.ndarray, List, Dict], train_idx: np.ndarray, val_idx: np.ndarray):
        """Split data based on indices, handling various data structures"""
        if isinstance(X, np.ndarray):
            return X[train_idx], X[val_idx]
        elif isinstance(X, list):
            return [X[i] for i in train_idx], [X[i] for i in val_idx]
        elif isinstance(X, dict):
            # Handle dictionary of arrays/lists
            X_train, X_val = {}, {}
            for key, value in X.items():
                if isinstance(value, np.ndarray):
                    X_train[key] = value[train_idx]
                    X_val[key] = value[val_idx]
                elif isinstance(value, list):
                    X_train[key] = [value[i] for i in train_idx]
                    X_val[key] = [value[i] for i in val_idx]
                else:
                    X_train[key] = value
                    X_val[key] = value
            return X_train, X_val
        else:
            raise TypeError(f"Unsupported data type for splitting: {type(X)}")
    
    def _bin_targets_for_stratification(self, y: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """Bin continuous targets for stratified cross-validation"""
        return pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
    
    def _calculate_cv_statistics(self, fold_metrics: List[EvaluationMetrics]) -> Dict[str, float]:
        """Calculate statistics across cross-validation folds"""
        if not fold_metrics:
            return {}
        
        metrics_arrays = {}
        for metric_name in fold_metrics[0].to_dict().keys():
            if isinstance(getattr(fold_metrics[0], metric_name), (int, float)):
                metrics_arrays[metric_name] = [getattr(m, metric_name) for m in fold_metrics]
        
        cv_stats = {}
        for metric_name, values in metrics_arrays.items():
            cv_stats[f'{metric_name}_mean'] = np.mean(values)
            cv_stats[f'{metric_name}_std'] = np.std(values)
            cv_stats[f'{metric_name}_min'] = np.min(values)
            cv_stats[f'{metric_name}_max'] = np.max(values)
        
        return cv_stats
    
    def _generate_model_comparison_summary(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary comparing models across cross-validation"""
        if not model_results:
            return {}
        
        summary = {
            'models_compared': list(model_results.keys()),
            'metrics_comparison': {}
        }
        
        # Extract key metrics for comparison
        key_metrics = ['rmse', 'pearson_r', 'spearman_r', 'concordance_index']
        
        for metric in key_metrics:
            metric_mean_key = f'{metric}_mean'
            metric_std_key = f'{metric}_std'
            
            summary['metrics_comparison'][metric] = {}
            
            for model_name, results in model_results.items():
                cv_stats = results['cv_statistics']
                if metric_mean_key in cv_stats:
                    summary['metrics_comparison'][metric][model_name] = {
                        'mean': cv_stats[metric_mean_key],
                        'std': cv_stats[metric_std_key]
                    }
        
        # Determine best models for each metric
        if 'rmse' in summary['metrics_comparison']:
            rmse_results = summary['metrics_comparison']['rmse']
            summary['best_model_by_rmse'] = min(rmse_results.keys(), 
                                              key=lambda k: rmse_results[k]['mean'])
        
        if 'pearson_r' in summary['metrics_comparison']:
            pearson_results = summary['metrics_comparison']['pearson_r']
            summary['best_model_by_pearson'] = max(pearson_results.keys(),
                                                 key=lambda k: pearson_results[k]['mean'])
        
        return summary


class BenchmarkSuite:
    """Comprehensive benchmarking suite for DTA models"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = ComprehensiveEvaluator()
        self.cross_validator = CrossValidator()
        self.baseline_comparator = BaselineComparator()
        
        # Performance tracking
        self.benchmark_results = {}
        self.timing_results = {}
        
    def benchmark_model(self, model, model_name: str,
                       test_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       predict_func: Callable,
                       baseline_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None) -> Dict[str, Any]:
        """
        Comprehensive benchmark of a model across multiple datasets
        
        Args:
            model: Trained model to benchmark
            model_name: Name of the model
            test_datasets: Dict mapping dataset names to (X_test, y_test) tuples
            predict_func: Function to make predictions (model, X) -> predictions
            baseline_data: Optional baseline training data for comparison
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Starting comprehensive benchmark for model: {model_name}")
        
        benchmark_results = {
            'model_name': model_name,
            'datasets_tested': list(test_datasets.keys()),
            'dataset_results': {},
            'overall_summary': {},
            'timing_results': {},
            'baseline_comparisons': {}
        }
        
        all_predictions = []
        all_true_values = []
        
        for dataset_name, (X_test, y_test) in test_datasets.items():
            logger.info(f"Benchmarking on dataset: {dataset_name}")
            
            try:
                # Time the prediction
                start_time = time.time()
                y_pred = predict_func(model, X_test)
                prediction_time = time.time() - start_time
                
                # Evaluate performance
                metrics = self.evaluator.evaluate(y_test, y_pred)
                
                # Store results
                benchmark_results['dataset_results'][dataset_name] = {
                    'metrics': metrics.to_dict(),
                    'prediction_time': prediction_time,
                    'samples_per_second': len(y_test) / prediction_time if prediction_time > 0 else float('inf'),
                    'n_samples': len(y_test)
                }
                
                # Accumulate for overall metrics
                all_predictions.extend(y_pred)
                all_true_values.extend(y_test)
                
                # Baseline comparison if data provided
                if baseline_data and dataset_name in baseline_data:
                    X_train, y_train = baseline_data[dataset_name]
                    baseline_comparison = self.baseline_comparator.compare_against_baselines(
                        X_train, y_train, X_test, y_test, y_pred, model_name
                    )
                    benchmark_results['baseline_comparisons'][dataset_name] = baseline_comparison
                
                logger.info(f"Dataset {dataset_name} - RMSE: {metrics.rmse:.4f}, "
                          f"Time: {prediction_time:.2f}s, "
                          f"Speed: {len(y_test)/prediction_time:.1f} samples/s")
                
            except Exception as e:
                logger.error(f"Error benchmarking dataset {dataset_name}: {e}")
                continue
        
        # Calculate overall performance
        if all_predictions:
            overall_metrics = self.evaluator.evaluate(all_true_values, all_predictions)
            benchmark_results['overall_summary'] = {
                'overall_metrics': overall_metrics.to_dict(),
                'total_samples': len(all_predictions),
                'datasets_count': len(benchmark_results['dataset_results'])
            }
        
        # Calculate timing statistics
        timing_data = [result['prediction_time'] for result in benchmark_results['dataset_results'].values()]
        speed_data = [result['samples_per_second'] for result in benchmark_results['dataset_results'].values()]
        
        if timing_data:
            benchmark_results['timing_results'] = {
                'mean_prediction_time': np.mean(timing_data),
                'std_prediction_time': np.std(timing_data),
                'mean_samples_per_second': np.mean(speed_data),
                'std_samples_per_second': np.std(speed_data),
                'total_prediction_time': sum(timing_data)
            }
        
        # Save results
        self._save_benchmark_results(benchmark_results)
        
        logger.info(f"Benchmark completed for {model_name}")
        return benchmark_results
    
    def compare_models_benchmark(self, models_config: Dict[str, Tuple],
                               test_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               predict_func: Callable) -> Dict[str, Any]:
        """
        Benchmark multiple models and compare their performance
        
        Args:
            models_config: Dict mapping model names to (model, predict_func) tuples
            test_datasets: Test datasets
            predict_func: Prediction function
            
        Returns:
            Comparative benchmark results
        """
        logger.info(f"Benchmarking {len(models_config)} models across {len(test_datasets)} datasets")
        
        model_benchmarks = {}
        
        for model_name, (model, model_predict_func) in models_config.items():
            benchmark_results = self.benchmark_model(
                model, model_name, test_datasets, model_predict_func or predict_func
            )
            model_benchmarks[model_name] = benchmark_results
        
        # Generate comparison
        comparison_results = self._generate_benchmark_comparison(model_benchmarks)
        
        # Save comparison results
        self._save_comparison_results(comparison_results)
        
        return {
            'individual_benchmarks': model_benchmarks,
            'comparison_results': comparison_results
        }
    
    def profile_model_performance(self, model, model_name: str,
                                sample_data: Tuple[np.ndarray, np.ndarray],
                                predict_func: Callable,
                                batch_sizes: List[int] = [1, 4, 8, 16, 32]) -> Dict[str, Any]:
        """
        Profile model performance across different batch sizes and conditions
        
        Args:
            model: Model to profile
            model_name: Name of the model
            sample_data: Sample (X, y) data for profiling
            predict_func: Prediction function
            batch_sizes: List of batch sizes to test
            
        Returns:
            Performance profiling results
        """
        logger.info(f"Profiling model performance: {model_name}")
        
        X_sample, y_sample = sample_data
        profiling_results = {
            'model_name': model_name,
            'batch_size_analysis': {},
            'memory_analysis': {},
            'scalability_analysis': {}
        }
        
        # Test different batch sizes
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            try:
                # Create batches
                n_samples = min(len(X_sample), batch_size * 10)  # Test with reasonable amount
                X_test = X_sample[:n_samples]
                y_test = y_sample[:n_samples]
                
                # Time prediction
                start_time = time.time()
                memory_before = self._get_memory_usage()
                
                y_pred = predict_func(model, X_test)
                
                prediction_time = time.time() - start_time
                memory_after = self._get_memory_usage()
                memory_used = memory_after - memory_before
                
                # Calculate metrics
                metrics = self.evaluator.evaluate(y_test, y_pred)
                
                profiling_results['batch_size_analysis'][batch_size] = {
                    'prediction_time': prediction_time,
                    'samples_per_second': n_samples / prediction_time if prediction_time > 0 else float('inf'),
                    'memory_used_mb': memory_used,
                    'rmse': metrics.rmse,
                    'pearson_r': metrics.pearson_r,
                    'n_samples': n_samples
                }
                
            except Exception as e:
                logger.error(f"Error profiling batch size {batch_size}: {e}")
                continue
        
        # Analyze scalability
        sample_sizes = [100, 500, 1000, min(5000, len(X_sample))]
        for sample_size in sample_sizes:
            if sample_size > len(X_sample):
                continue
                
            try:
                X_test = X_sample[:sample_size]
                y_test = y_sample[:sample_size]
                
                start_time = time.time()
                y_pred = predict_func(model, X_test)
                prediction_time = time.time() - start_time
                
                profiling_results['scalability_analysis'][sample_size] = {
                    'prediction_time': prediction_time,
                    'samples_per_second': sample_size / prediction_time if prediction_time > 0 else float('inf'),
                    'time_per_sample': prediction_time / sample_size if sample_size > 0 else 0
                }
                
            except Exception as e:
                logger.error(f"Error profiling sample size {sample_size}: {e}")
                continue
        
        # Save profiling results
        self._save_profiling_results(profiling_results)
        
        logger.info(f"Performance profiling completed for {model_name}")
        return profiling_results
    
    def generate_benchmark_report(self, benchmark_results: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"benchmark_report_{benchmark_results['model_name']}_{timestamp}.txt"
        report_path = self.output_dir / report_filename
        
        model_name = benchmark_results['model_name']
        overall_summary = benchmark_results.get('overall_summary', {})
        timing_results = benchmark_results.get('timing_results', {})
        
        report_content = f"""
Comprehensive Benchmark Report
=============================
Model: {model_name}
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Datasets Tested: {len(benchmark_results['dataset_results'])}

Overall Performance:
"""
        
        if 'overall_metrics' in overall_summary:
            metrics = overall_summary['overall_metrics']
            report_content += f"""
  RMSE: {metrics['rmse']:.4f}
  Pearson r: {metrics['pearson_r']:.4f}
  Spearman ρ: {metrics['spearman_r']:.4f}
  Concordance Index: {metrics['concordance_index']:.4f}
  R²: {metrics['r_squared']:.4f}
  Total Samples: {overall_summary['total_samples']}
"""
        
        if timing_results:
            report_content += f"""
Performance Metrics:
  Mean Prediction Time: {timing_results['mean_prediction_time']:.4f}s
  Mean Throughput: {timing_results['mean_samples_per_second']:.1f} samples/s
  Total Prediction Time: {timing_results['total_prediction_time']:.2f}s
"""
        
        # Dataset-specific results
        report_content += "\nDataset-Specific Results:\n"
        report_content += "=" * 50 + "\n"
        
        for dataset_name, results in benchmark_results['dataset_results'].items():
            metrics = results['metrics']
            report_content += f"""
{dataset_name}:
  RMSE: {metrics['rmse']:.4f}
  Pearson r: {metrics['pearson_r']:.4f}
  Samples: {results['n_samples']}
  Time: {results['prediction_time']:.4f}s
  Speed: {results['samples_per_second']:.1f} samples/s
"""
        
        # Baseline comparisons
        if 'baseline_comparisons' in benchmark_results:
            report_content += "\nBaseline Comparisons:\n"
            report_content += "=" * 30 + "\n"
            
            for dataset_name, comparison in benchmark_results['baseline_comparisons'].items():
                report_content += f"\n{dataset_name}:\n"
                for baseline_name, improvement in comparison['improvements'].items():
                    rmse_imp = improvement['rmse_improvement']
                    report_content += f"  vs {baseline_name}: {rmse_imp:.2%} RMSE improvement\n"
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Benchmark report saved: {report_path}")
        return str(report_path)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{results['model_name']}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved: {filepath}")
    
    def _save_comparison_results(self, results: Dict[str, Any]):
        """Save model comparison results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Comparison results saved: {filepath}")
    
    def _save_profiling_results(self, results: Dict[str, Any]):
        """Save performance profiling results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"profiling_{results['model_name']}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Profiling results saved: {filepath}")
    
    def _generate_benchmark_comparison(self, model_benchmarks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison analysis across multiple model benchmarks"""
        
        comparison = {
            'models_compared': list(model_benchmarks.keys()),
            'performance_ranking': {},
            'timing_comparison': {},
            'dataset_analysis': {}
        }
        
        # Extract overall metrics for comparison
        model_metrics = {}
        for model_name, benchmark in model_benchmarks.items():
            if 'overall_summary' in benchmark and 'overall_metrics' in benchmark['overall_summary']:
                model_metrics[model_name] = benchmark['overall_summary']['overall_metrics']
        
        # Rank models by different metrics
        if model_metrics:
            for metric in ['rmse', 'pearson_r', 'spearman_r', 'concordance_index']:
                if metric in list(model_metrics.values())[0]:
                    if metric == 'rmse':  # Lower is better
                        ranking = sorted(model_metrics.items(), key=lambda x: x[1][metric])
                    else:  # Higher is better
                        ranking = sorted(model_metrics.items(), key=lambda x: x[1][metric], reverse=True)
                    
                    comparison['performance_ranking'][metric] = [
                        {'rank': i+1, 'model': model, 'value': metrics[metric]}
                        for i, (model, metrics) in enumerate(ranking)
                    ]
        
        # Compare timing performance
        timing_data = {}
        for model_name, benchmark in model_benchmarks.items():
            if 'timing_results' in benchmark:
                timing_data[model_name] = benchmark['timing_results']
        
        if timing_data:
            comparison['timing_comparison'] = {
                'fastest_model': min(timing_data.items(), key=lambda x: x[1]['mean_prediction_time'])[0],
                'highest_throughput': max(timing_data.items(), key=lambda x: x[1]['mean_samples_per_second'])[0],
                'timing_details': timing_data
            }
        
        return comparison


class AutomatedEvaluationPipeline:
    """Automated evaluation pipeline that combines all evaluation components"""
    
    def __init__(self, output_dir: str = "evaluation_pipeline_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = ComprehensiveEvaluator()
        self.cross_validator = CrossValidator()
        self.benchmark_suite = BenchmarkSuite(str(self.output_dir / "benchmarks"))
        self.reporter = EvaluationReporter(str(self.output_dir / "reports"))
        
    def run_complete_evaluation(self, model, model_name: str,
                              datasets: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
                              train_func: Callable, predict_func: Callable,
                              model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline including cross-validation and benchmarking
        
        Args:
            model: Model to evaluate
            model_name: Name of the model
            datasets: Dict with 'train' and 'test' datasets for each dataset name
            train_func: Training function
            predict_func: Prediction function
            model_config: Model configuration
            
        Returns:
            Complete evaluation results
        """
        logger.info(f"Starting complete evaluation pipeline for {model_name}")
        
        pipeline_results = {
            'model_name': model_name,
            'evaluation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'cross_validation_results': {},
            'benchmark_results': {},
            'final_test_results': {},
            'summary': {}
        }
        
        # 1. Cross-validation on training data
        logger.info("Phase 1: Cross-validation evaluation")
        for dataset_name, data_splits in datasets.items():
            if 'train' in data_splits:
                X_train, y_train = data_splits['train']
                
                try:
                    cv_results = self.cross_validator.cross_validate_model(
                        type(model), model_config, X_train, y_train, train_func, predict_func
                    )
                    pipeline_results['cross_validation_results'][dataset_name] = cv_results
                    
                    logger.info(f"CV completed for {dataset_name} - "
                              f"Mean RMSE: {cv_results['cv_statistics']['rmse_mean']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Cross-validation failed for {dataset_name}: {e}")
                    continue
        
        # 2. Benchmark on test data
        logger.info("Phase 2: Benchmark evaluation")
        test_datasets = {}
        baseline_data = {}
        
        for dataset_name, data_splits in datasets.items():
            if 'test' in data_splits:
                test_datasets[dataset_name] = data_splits['test']
            if 'train' in data_splits:
                baseline_data[dataset_name] = data_splits['train']
        
        if test_datasets:
            benchmark_results = self.benchmark_suite.benchmark_model(
                model, model_name, test_datasets, predict_func, baseline_data
            )
            pipeline_results['benchmark_results'] = benchmark_results
        
        # 3. Final test evaluation
        logger.info("Phase 3: Final test evaluation")
        for dataset_name, (X_test, y_test) in test_datasets.items():
            try:
                y_pred = predict_func(model, X_test)
                test_metrics = self.evaluator.evaluate(y_test, y_pred)
                pipeline_results['final_test_results'][dataset_name] = test_metrics.to_dict()
                
            except Exception as e:
                logger.error(f"Final test evaluation failed for {dataset_name}: {e}")
                continue
        
        # 4. Generate summary
        pipeline_results['summary'] = self._generate_pipeline_summary(pipeline_results)
        
        # 5. Generate reports
        self._generate_pipeline_reports(pipeline_results)
        
        # 6. Save complete results
        self._save_pipeline_results(pipeline_results)
        
        logger.info(f"Complete evaluation pipeline finished for {model_name}")
        return pipeline_results
    
    def _generate_pipeline_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of complete evaluation pipeline"""
        
        summary = {
            'datasets_evaluated': [],
            'cross_validation_summary': {},
            'benchmark_summary': {},
            'overall_performance': {}
        }
        
        # Collect dataset names
        all_datasets = set()
        for phase_results in [results['cross_validation_results'], 
                            results['benchmark_results'].get('dataset_results', {}),
                            results['final_test_results']]:
            all_datasets.update(phase_results.keys())
        
        summary['datasets_evaluated'] = list(all_datasets)
        
        # Cross-validation summary
        if results['cross_validation_results']:
            cv_rmse_scores = []
            cv_pearson_scores = []
            
            for dataset_name, cv_results in results['cross_validation_results'].items():
                cv_stats = cv_results['cv_statistics']
                if 'rmse_mean' in cv_stats:
                    cv_rmse_scores.append(cv_stats['rmse_mean'])
                if 'pearson_r_mean' in cv_stats:
                    cv_pearson_scores.append(cv_stats['pearson_r_mean'])
            
            if cv_rmse_scores:
                summary['cross_validation_summary'] = {
                    'mean_cv_rmse': np.mean(cv_rmse_scores),
                    'std_cv_rmse': np.std(cv_rmse_scores),
                    'mean_cv_pearson': np.mean(cv_pearson_scores) if cv_pearson_scores else None,
                    'datasets_count': len(cv_rmse_scores)
                }
        
        # Benchmark summary
        if 'overall_summary' in results['benchmark_results']:
            benchmark_summary = results['benchmark_results']['overall_summary']
            if 'overall_metrics' in benchmark_summary:
                summary['benchmark_summary'] = benchmark_summary['overall_metrics']
        
        return summary
    
    def _generate_pipeline_reports(self, results: Dict[str, Any]):
        """Generate comprehensive reports for the evaluation pipeline"""
        
        model_name = results['model_name']
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Generate main pipeline report
        report_path = self.output_dir / f"complete_evaluation_{model_name}_{timestamp}.txt"
        
        report_content = f"""
Complete Evaluation Pipeline Report
==================================
Model: {model_name}
Generated: {results['evaluation_timestamp']}
Datasets: {', '.join(results['summary']['datasets_evaluated'])}

EXECUTIVE SUMMARY:
"""
        
        # Add cross-validation summary
        if 'cross_validation_summary' in results['summary']:
            cv_summary = results['summary']['cross_validation_summary']
            report_content += f"""
Cross-Validation Results:
  Mean RMSE across datasets: {cv_summary['mean_cv_rmse']:.4f} ± {cv_summary['std_cv_rmse']:.4f}
  Mean Pearson r: {cv_summary.get('mean_cv_pearson', 'N/A')}
  Datasets evaluated: {cv_summary['datasets_count']}
"""
        
        # Add benchmark summary
        if 'benchmark_summary' in results['summary']:
            bench_summary = results['summary']['benchmark_summary']
            report_content += f"""
Benchmark Results:
  Overall RMSE: {bench_summary['rmse']:.4f}
  Overall Pearson r: {bench_summary['pearson_r']:.4f}
  Overall Concordance Index: {bench_summary['concordance_index']:.4f}
"""
        
        # Add detailed results for each dataset
        report_content += "\nDETAILED RESULTS BY DATASET:\n"
        report_content += "=" * 50 + "\n"
        
        for dataset_name in results['summary']['datasets_evaluated']:
            report_content += f"\n{dataset_name.upper()}:\n"
            
            # Cross-validation results
            if dataset_name in results['cross_validation_results']:
                cv_stats = results['cross_validation_results'][dataset_name]['cv_statistics']
                report_content += f"  Cross-Validation: RMSE = {cv_stats.get('rmse_mean', 'N/A'):.4f}"
                if 'rmse_std' in cv_stats:
                    report_content += f" ± {cv_stats['rmse_std']:.4f}"
                report_content += "\n"
            
            # Final test results
            if dataset_name in results['final_test_results']:
                test_metrics = results['final_test_results'][dataset_name]
                report_content += f"  Final Test: RMSE = {test_metrics['rmse']:.4f}, "
                report_content += f"Pearson r = {test_metrics['pearson_r']:.4f}\n"
        
        # Save main report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Complete evaluation report saved: {report_path}")
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save complete pipeline results to JSON"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"complete_evaluation_{results['model_name']}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Complete evaluation results saved: {filepath}")


if __name__ == "__main__":
    # Additional testing for cross-validation and benchmarking
    logger.info("Testing cross-validation and benchmarking components")
    
    # Test cross-validation
    cv = CrossValidator(n_splits=3)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    X_sample = np.random.randn(n_samples, 10)
    y_sample = np.random.normal(5.0, 2.0, n_samples)
    
    # Mock model class and functions for testing
    class MockModel:
        def __init__(self, **kwargs):
            self.trained = False
        
        def predict(self, X):
            return np.random.normal(5.0, 2.0, len(X))
    
    def mock_train_func(model, X_train, y_train):
        model.trained = True
        return model
    
    def mock_predict_func(model, X_test):
        return model.predict(X_test)
    
    # Test cross-validation
    print("Testing cross-validation...")
    cv_results = cv.cross_validate_model(
        MockModel, {}, X_sample, y_sample, mock_train_func, mock_predict_func
    )
    print(f"CV Mean RMSE: {cv_results['cv_statistics']['rmse_mean']:.4f}")
    
    # Test benchmarking
    print("\nTesting benchmarking...")
    benchmark_suite = BenchmarkSuite()
    
    test_datasets = {
        'test_dataset_1': (X_sample[:100], y_sample[:100]),
        'test_dataset_2': (X_sample[100:200], y_sample[100:200])
    }
    
    model = MockModel()
    benchmark_results = benchmark_suite.benchmark_model(
        model, "MockModel", test_datasets, mock_predict_func
    )
    
    print(f"Benchmark completed. Overall RMSE: {benchmark_results['overall_summary']['overall_metrics']['rmse']:.4f}")
    
    print("\nCross-validation and benchmarking tests completed successfully!")
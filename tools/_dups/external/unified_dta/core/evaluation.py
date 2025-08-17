"""
Evaluation metrics and utilities for the Unified DTA System
"""

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

from .models import UnifiedDTAModel


logger = logging.getLogger(__name__)


class DTAEvaluator:
    """Evaluator class for DTA model performance assessment"""
    
    def __init__(self, model: UnifiedDTAModel, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def calculate_metrics(self, 
                         predictions: np.ndarray, 
                         targets: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic regression metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        
        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(predictions, targets)
        spearman_corr, spearman_p = spearmanr(predictions, targets)
        
        # Concordance index (C-index)
        ci = self.concordance_index(targets, predictions)
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'concordance_index': float(ci),
            'r_squared': float(r2)
        }
    
    def concordance_index(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate concordance index (C-index) for ranking evaluation"""
        n = len(y_true)
        concordant = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if y_true[i] != y_true[j]:  # Only consider pairs with different true values
                    total_pairs += 1
                    if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                       (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                        concordant += 1
        
        return concordant / total_pairs if total_pairs > 0 else 0.5
    
    def evaluate_dataset(self, data_loader) -> Dict[str, float]:
        """Evaluate model on a dataset"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                drug_data = batch['drug_data'].to(self.device)
                protein_data = batch['protein_data']
                targets = batch['affinity'].cpu().numpy()
                
                predictions = self.model(drug_data, protein_data)
                predictions = predictions.squeeze().cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        predictions_array = np.array(all_predictions)
        targets_array = np.array(all_targets)
        
        return self.calculate_metrics(predictions_array, targets_array)
    
    def cross_validate(self, 
                      datasets: List,
                      k_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """Perform k-fold cross-validation"""
        fold_results = []
        
        for fold in range(k_folds):
            logger.info(f"Evaluating fold {fold + 1}/{k_folds}")
            
            # Get fold data (assuming datasets are pre-split)
            if fold < len(datasets):
                fold_metrics = self.evaluate_dataset(datasets[fold])
                fold_results.append(fold_metrics)
        
        # Calculate mean and std across folds
        if not fold_results:
            return {}
        
        metrics_summary = {}
        metric_names = fold_results[0].keys()
        
        for metric in metric_names:
            values = [result[metric] for result in fold_results]
            metrics_summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }
        
        return metrics_summary
    
    def compare_models(self, 
                      other_evaluators: List['DTAEvaluator'],
                      test_loader,
                      model_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Compare multiple models on the same test set"""
        
        if model_names is None:
            model_names = [f"Model_{i}" for i in range(len(other_evaluators) + 1)]
        
        results = {}
        
        # Evaluate current model
        results[model_names[0]] = self.evaluate_dataset(test_loader)
        
        # Evaluate other models
        for i, evaluator in enumerate(other_evaluators):
            results[model_names[i + 1]] = evaluator.evaluate_dataset(test_loader)
        
        return results
    
    def generate_report(self, 
                       metrics: Dict[str, float],
                       output_path: Optional[Path] = None) -> str:
        """Generate a formatted evaluation report"""
        
        report_lines = [
            "=" * 60,
            "DTA Model Evaluation Report",
            "=" * 60,
            "",
            "Regression Metrics:",
            f"  RMSE: {metrics['rmse']:.4f}",
            f"  MSE:  {metrics['mse']:.4f}",
            f"  MAE:  {metrics['mae']:.4f}",
            f"  R²:   {metrics['r_squared']:.4f}",
            "",
            "Correlation Metrics:",
            f"  Pearson Correlation:  {metrics['pearson_correlation']:.4f} "
            f"(p={metrics['pearson_p_value']:.2e})",
            f"  Spearman Correlation: {metrics['spearman_correlation']:.4f} "
            f"(p={metrics['spearman_p_value']:.2e})",
            "",
            "Ranking Metrics:",
            f"  Concordance Index: {metrics['concordance_index']:.4f}",
            "",
            "=" * 60
        ]
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            
            # Also save metrics as JSON
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return report
    
    def benchmark_against_baseline(self, 
                                  test_loader,
                                  baseline_predictions: np.ndarray) -> Dict[str, float]:
        """Benchmark model against baseline predictions"""
        
        # Get model predictions
        model_metrics = self.evaluate_dataset(test_loader)
        
        # Get targets for baseline evaluation
        all_targets = []
        with torch.no_grad():
            for batch in test_loader:
                targets = batch['affinity'].cpu().numpy()
                all_targets.extend(targets)
        
        targets_array = np.array(all_targets)
        baseline_metrics = self.calculate_metrics(baseline_predictions, targets_array)
        
        # Calculate improvement
        improvement = {}
        for metric in ['rmse', 'mae', 'pearson_correlation', 'concordance_index']:
            if metric in ['rmse', 'mae']:
                # Lower is better
                improvement[f"{metric}_improvement"] = \
                    (baseline_metrics[metric] - model_metrics[metric]) / baseline_metrics[metric]
            else:
                # Higher is better
                improvement[f"{metric}_improvement"] = \
                    (model_metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric]
        
        return {
            'model_metrics': model_metrics,
            'baseline_metrics': baseline_metrics,
            'improvements': improvement
        }
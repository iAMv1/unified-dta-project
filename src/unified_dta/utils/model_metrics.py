"""
Model metrics utilities for exposing training metrics through the API
"""

import logging
import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelMetricsManager:
    """Manager for model training metrics"""
    
    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
    
    def save_training_metrics(self, model_type: str, metrics: Dict[str, Any]) -> str:
        """Save training metrics to file"""
        metrics_file = self.metrics_dir / f"{model_type}_metrics.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Training metrics saved to {metrics_file}")
            return str(metrics_file)
        except Exception as e:
            logger.error(f"Failed to save training metrics: {e}")
            return None
    
    def load_training_metrics(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Load training metrics from file"""
        metrics_file = self.metrics_dir / f"{model_type}_metrics.json"
        if not metrics_file.exists():
            # Try to find any metrics file for this model type
            pattern = f"*{model_type}*_metrics.json"
            matches = list(self.metrics_dir.glob(pattern))
            if matches:
                metrics_file = matches[0]  # Use the first match
            else:
                logger.warning(f"Training metrics file not found for {model_type}")
                return None
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            logger.info(f"Training metrics loaded from {metrics_file}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to load training metrics: {e}")
            return None
    
    def get_model_performance_summary(self, model_type: str) -> Dict[str, Any]:
        """Get a summary of model performance metrics"""
        metrics = self.load_training_metrics(model_type)
        if not metrics:
            return {"error": "No metrics available for this model"}
        
        # Extract key performance metrics
        summary = {
            "model_type": model_type,
            "training_completed": metrics.get("total_epochs", 0) > 0,
            "total_training_time": metrics.get("total_training_time", 0)
        }
        
        # Get final metrics from history
        metrics_history = metrics.get("metrics_history", [])
        if metrics_history:
            final_metrics = metrics_history[-1]  # Most recent metrics
            summary.update({
                "final_validation_loss": final_metrics.get("val_loss"),
                "final_pearson_correlation": final_metrics.get("val_pearson"),
                "final_spearman_correlation": final_metrics.get("val_spearman"),
                "final_rmse": final_metrics.get("val_rmse")
            })
        
        return summary
    
    def get_training_progress(self, model_type: str) -> Dict[str, Any]:
        """Get training progress information"""
        metrics = self.load_training_metrics(model_type)
        if not metrics:
            return {"error": "No metrics available for this model"}
        
        metrics_history = metrics.get("metrics_history", [])
        if not metrics_history:
            return {"error": "No training history available"}
        
        # Get training progress
        current_epoch = metrics.get("total_epochs", 0)
        best_val_loss = metrics.get("best_val_loss", float('inf'))
        best_val_pearson = metrics.get("best_val_pearson", -1.0)
        
        # Calculate recent improvement
        recent_metrics = metrics_history[-5:] if len(metrics_history) >= 5 else metrics_history
        if len(recent_metrics) >= 2:
            recent_loss_improvement = recent_metrics[0].get("val_loss", 0) - recent_metrics[-1].get("val_loss", 0)
            recent_pearson_improvement = recent_metrics[-1].get("val_pearson", 0) - recent_metrics[0].get("val_pearson", 0)
        else:
            recent_loss_improvement = 0
            recent_pearson_improvement = 0
        
        return {
            "model_type": model_type,
            "current_epoch": current_epoch,
            "best_validation_loss": best_val_loss,
            "best_pearson_correlation": best_val_pearson,
            "recent_loss_improvement": recent_loss_improvement,
            "recent_pearson_improvement": recent_pearson_improvement,
            "total_training_time": metrics.get("total_training_time", 0)
        }


# Global instance
metrics_manager = ModelMetricsManager()


def get_model_metrics(model_type: str) -> Dict[str, Any]:
    """Get model metrics for API exposure"""
    return metrics_manager.get_model_performance_summary(model_type)


def get_training_progress(model_type: str) -> Dict[str, Any]:
    """Get training progress for API exposure"""
    return metrics_manager.get_training_progress(model_type)
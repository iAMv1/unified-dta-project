"""
Evaluation systems and metrics
"""

# Import what's actually available in the evaluation module
from .evaluation import ComprehensiveEvaluator, MetricsCalculator
from .prediction_heads import MLPPredictionHead, PredictionHeadFactory

# Create aliases for the expected names
calculate_metrics = MetricsCalculator
evaluate_model = ComprehensiveEvaluator

__all__ = [
    # Evaluation
    "evaluate_model",
    "calculate_metrics",
    "ComprehensiveEvaluator",
    "MetricsCalculator",
    
    # Prediction heads
    "MLPPredictionHead", 
    "PredictionHeadFactory"
]
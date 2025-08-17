"""
FastAPI endpoints for the Unified DTA System
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
import asyncio
from typing import List, Dict, Any

from .models import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    ErrorResponse, ModelInfo, HealthResponse,
    ModelType
)
from .prediction import get_prediction_service
from .cache import get_model_cache
from ..utils.model_metrics import get_model_metrics, get_training_progress

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post("/predict", 
             response_model=PredictionResponse,
             summary="Single Drug-Target Affinity Prediction",
             description="Predict binding affinity for a single drug-target pair")
async def predict_single(request: PredictionRequest) -> PredictionResponse:
    """
    Predict drug-target affinity for a single compound-protein pair.
    
    - **drug_smiles**: SMILES string representation of the drug compound
    - **protein_sequence**: Amino acid sequence of the target protein
    - **model_type**: Type of model to use (lightweight, production, custom)
    
    Returns the predicted binding affinity with processing time and metadata.
    """
    try:
        prediction_service = get_prediction_service()
        result = await prediction_service.predict_single(request)
        return result
        
    except ValueError as e:
        logger.error(f"Validation error in single prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory error during single prediction")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable due to memory constraints. Try a smaller model.")
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        # Include more detailed error information
        error_detail = {
            "error": "Internal server error during prediction",
            "message": str(e),
            "type": type(e).__name__
        }
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/predict/batch",
             response_model=BatchPredictionResponse,
             summary="Batch Drug-Target Affinity Predictions",
             description="Predict binding affinities for multiple drug-target pairs")
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict drug-target affinities for multiple compound-protein pairs in batch.
    
    - **predictions**: List of prediction requests (max 1000)
    - **model_type**: Type of model to use for all predictions
    
    Returns batch results with success/failure statistics and total processing time.
    """
    try:
        prediction_service = get_prediction_service()
        result = await prediction_service.predict_batch(request)
        return result
        
    except ValueError as e:
        logger.error(f"Validation error in batch prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory error during batch prediction")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable due to memory constraints. Try reducing batch size or using a smaller model.")
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        # Include more detailed error information
        error_detail = {
            "error": "Internal server error during batch prediction",
            "message": str(e),
            "type": type(e).__name__
        }
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/models/{model_type}/info",
            response_model=ModelInfo,
            summary="Get Model Information",
            description="Get detailed information about a specific model")
async def get_model_info(model_type: ModelType) -> ModelInfo:
    """
    Get detailed information about a specific model including architecture and parameters.
    
    - **model_type**: Type of model to get information about
    
    Returns model architecture details, parameter counts, and memory usage.
    """
    try:
        model_cache = get_model_cache()
        
        # Load model to get info (will be cached)
        model = await model_cache.get_model(model_type.value)
        
        # Get cached model info
        cache_info = model_cache.get_cache_info()
        model_info = cache_info["model_info"].get(model_type.value, {})
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_type.value} not found")
        
        return ModelInfo(
            model_type=model_info.get("model_type", model_type.value),
            protein_encoder=model_info.get("protein_encoder", "unknown"),
            drug_encoder=model_info.get("drug_encoder", "gin"),
            uses_fusion=model_info.get("uses_fusion", False),
            parameters=model_info.get("total_parameters", 0),
            memory_usage_mb=model_info.get("memory_usage_mb")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting model info")


@router.get("/models",
            summary="List Available Models",
            description="Get list of available model types and their status")
async def list_models() -> Dict[str, Any]:
    """
    Get list of available model types and their current status.
    
    Returns information about available models and which ones are currently cached.
    """
    try:
        model_cache = get_model_cache()
        cache_info = model_cache.get_cache_info()
        
        available_models = [model_type.value for model_type in ModelType]
        
        return {
            "available_models": available_models,
            "cached_models": cache_info["cached_models"],
            "cache_info": {
                "cache_size": cache_info["cache_size"],
                "max_cache_size": cache_info["max_cache_size"],
                "device": cache_info["device"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error listing models")


@router.post("/models/{model_type}/load",
             summary="Preload Model",
             description="Preload a model into cache for faster predictions")
async def preload_model(model_type: ModelType, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Preload a model into cache for faster subsequent predictions.
    
    - **model_type**: Type of model to preload
    
    Returns confirmation that the model loading has been initiated.
    """
    try:
        async def load_model_task():
            model_cache = get_model_cache()
            await model_cache.get_model(model_type.value)
            logger.info(f"Model {model_type.value} preloaded successfully")
        
        background_tasks.add_task(load_model_task)
        
        return {
            "message": f"Model {model_type.value} loading initiated",
            "status": "loading"
        }
        
    except Exception as e:
        logger.error(f"Error preloading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error preloading model")


@router.delete("/cache",
               summary="Clear Model Cache",
               description="Clear all cached models to free memory")
async def clear_cache() -> Dict[str, str]:
    """
    Clear all cached models to free up memory.
    
    Returns confirmation that the cache has been cleared.
    """
    try:
        model_cache = get_model_cache()
        model_cache.clear_cache()
        
        return {
            "message": "Model cache cleared successfully",
            "status": "cleared"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error clearing cache")


@router.get("/models/{model_type}/metrics",
            summary="Get Model Metrics",
            description="Get training metrics and performance statistics for a specific model")
async def get_model_metrics_endpoint(model_type: ModelType) -> Dict[str, Any]:
    """
    Get training metrics and performance statistics for a specific model.
    
    - **model_type**: Type of model to get metrics for
    
    Returns model performance metrics including validation loss, correlation coefficients, and training time.
    """
    try:
        metrics = get_model_metrics(model_type.value)
        return metrics
    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting model metrics")


@router.get("/models/{model_type}/progress",
            summary="Get Training Progress",
            description="Get training progress information for a specific model")
async def get_training_progress_endpoint(model_type: ModelType) -> Dict[str, Any]:
    """
    Get training progress information for a specific model.
    
    - **model_type**: Type of model to get progress for
    
    Returns training progress including current epoch, best metrics, and recent improvements.
    """
    try:
        progress = get_training_progress(model_type.value)
        return progress
    except Exception as e:
        logger.error(f"Error getting training progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error getting training progress")


@router.get("/health",
            response_model=HealthResponse,
            summary="Health Check",
            description="Get service health status and system information")
async def health_check() -> HealthResponse:
    """
    Get service health status including loaded models and system resources.
    
    Returns comprehensive health information including memory usage and GPU availability.
    """
    try:
        import torch
        from .. import __version__
        
        model_cache = get_model_cache()
        cache_info = model_cache.get_cache_info()
        memory_usage = model_cache.get_memory_usage()
        
        return HealthResponse(
            status="healthy",
            version=__version__,
            models_loaded=cache_info["cached_models"],
            gpu_available=torch.cuda.is_available(),
            memory_usage=memory_usage
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version="unknown",
            models_loaded=[],
            gpu_available=False,
            memory_usage={}
        )


@router.get("/",
            summary="API Information",
            description="Get basic API information and available endpoints")
async def root() -> Dict[str, Any]:
    """
    Get basic API information and available endpoints.
    
    Returns API metadata and endpoint documentation.
    """
    return {
        "name": "Unified DTA System API",
        "description": "RESTful API for drug-target affinity prediction",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Single prediction",
            "POST /predict/batch": "Batch predictions",
            "GET /models": "List available models",
            "GET /models/{model_type}/info": "Get model information",
            "GET /models/{model_type}/metrics": "Get model metrics",
            "GET /models/{model_type}/progress": "Get training progress",
            "POST /models/{model_type}/load": "Preload model",
            "DELETE /cache": "Clear model cache",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "model_types": [model_type.value for model_type in ModelType]
    }
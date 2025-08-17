"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class ModelType(str, Enum):
    """Available model types"""
    lightweight = "lightweight"
    production = "production"
    custom = "custom"


class PredictionRequest(BaseModel):
    """Single prediction request"""
    drug_smiles: str = Field(..., description="SMILES string of the drug compound")
    protein_sequence: str = Field(..., description="Amino acid sequence of the target protein")
    model_type: ModelType = Field(default=ModelType.production, description="Type of model to use")
    
    @validator('drug_smiles')
    def validate_smiles(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("SMILES string cannot be empty")
        return v.strip()
    
    @validator('protein_sequence')
    def validate_protein_sequence(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Protein sequence cannot be empty")
        # Basic validation for amino acid characters
        valid_chars = set('ACDEFGHIKLMNPQRSTVWYXUBZO')
        if not all(c.upper() in valid_chars for c in v.strip()):
            raise ValueError("Protein sequence contains invalid amino acid characters")
        return v.strip().upper()


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")
    model_type: ModelType = Field(default=ModelType.production, description="Type of model to use for all predictions")
    
    @validator('predictions')
    def validate_predictions(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Predictions list cannot be empty")
        if len(v) > 1000:  # Reasonable batch size limit
            raise ValueError("Batch size cannot exceed 1000 predictions")
        return v


class PredictionResponse(BaseModel):
    """Single prediction response"""
    drug_smiles: str = Field(..., description="Input SMILES string")
    protein_sequence: str = Field(..., description="Input protein sequence (truncated if necessary)")
    predicted_affinity: float = Field(..., description="Predicted binding affinity")
    confidence: Optional[float] = Field(None, description="Prediction confidence score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_type: str = Field(..., description="Model type used for prediction")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    model_type: str = Field(..., description="Model type used for predictions")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str = Field(..., description="Type of the model")
    protein_encoder: str = Field(..., description="Type of protein encoder")
    drug_encoder: str = Field(..., description="Type of drug encoder")
    uses_fusion: bool = Field(..., description="Whether the model uses fusion")
    parameters: int = Field(..., description="Number of model parameters")
    memory_usage_mb: Optional[float] = Field(None, description="Estimated memory usage in MB")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")


class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Any = Field(..., description="The invalid value")


class BatchValidationResponse(BaseModel):
    """Batch validation response"""
    valid_requests: List[int] = Field(..., description="Indices of valid requests")
    invalid_requests: List[Dict[str, Any]] = Field(..., description="Details of invalid requests")
    total_requests: int = Field(..., description="Total number of requests")
    validation_errors: List[ValidationError] = Field(..., description="List of validation errors")
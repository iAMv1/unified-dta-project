"""
Prediction service for processing drug-target affinity requests
"""

import torch
import asyncio
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import traceback

from .models import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from .cache import get_model_cache
from ..data.data_processing import DataProcessor
from ..core.models import UnifiedDTAModel

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for handling drug-target affinity predictions"""
    
    def __init__(self, max_workers: int = 4):
        self.data_processor = DataProcessor()
        self.model_cache = get_model_cache()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """Process a single prediction request"""
        start_time = time.time()
        
        try:
            # Get model from cache
            model = await self.model_cache.get_model(request.model_type.value)
            
            # Process input data
            processed_data = await self._process_input_data(
                request.drug_smiles, 
                request.protein_sequence,
                model
            )
            
            # Make prediction
            prediction = await self._make_prediction(model, processed_data)
            
            processing_time = (time.time() - start_time) * 1000
            
            return PredictionResponse(
                drug_smiles=request.drug_smiles,
                protein_sequence=request.protein_sequence[:50] + "..." if len(request.protein_sequence) > 50 else request.protein_sequence,
                predicted_affinity=float(prediction),
                confidence=None,  # TODO: Implement confidence scoring
                processing_time_ms=processing_time,
                model_type=request.model_type.value
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Process a batch of prediction requests"""
        start_time = time.time()
        
        try:
            # Get model from cache
            model = await self.model_cache.get_model(request.model_type.value)
            
            # Process all requests concurrently
            tasks = []
            for pred_request in request.predictions:
                # Override model type with batch model type
                pred_request.model_type = request.model_type
                task = asyncio.create_task(self._process_single_prediction(model, pred_request))
                tasks.append(task)
            
            # Wait for all predictions to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successful and failed predictions
            successful_predictions = []
            failed_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Prediction {i} failed: {str(result)}")
                    failed_count += 1
                else:
                    successful_predictions.append(result)
            
            total_time = (time.time() - start_time) * 1000
            
            return BatchPredictionResponse(
                predictions=successful_predictions,
                total_processing_time_ms=total_time,
                successful_predictions=len(successful_predictions),
                failed_predictions=failed_count,
                model_type=request.model_type.value
            )
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def _process_single_prediction(self, model: UnifiedDTAModel, request: PredictionRequest) -> PredictionResponse:
        """Process a single prediction within a batch"""
        start_time = time.time()
        
        try:
            # Process input data
            processed_data = await self._process_input_data(
                request.drug_smiles,
                request.protein_sequence,
                model
            )
            
            # Make prediction
            prediction = await self._make_prediction(model, processed_data)
            
            processing_time = (time.time() - start_time) * 1000
            
            return PredictionResponse(
                drug_smiles=request.drug_smiles,
                protein_sequence=request.protein_sequence[:50] + "..." if len(request.protein_sequence) > 50 else request.protein_sequence,
                predicted_affinity=float(prediction),
                confidence=None,
                processing_time_ms=processing_time,
                model_type=request.model_type.value
            )
            
        except Exception as e:
            logger.error(f"Single prediction in batch failed: {str(e)}")
            raise
    
    async def _process_input_data(self, smiles: str, protein_sequence: str, model: UnifiedDTAModel) -> Dict[str, Any]:
        """Process input data for prediction"""
        loop = asyncio.get_event_loop()
        
        def _process():
            try:
                # Validate SMILES
                if not self.data_processor.validate_smiles(smiles):
                    raise ValueError(f"Invalid SMILES string: {smiles}")
                
                # Process protein sequence
                if hasattr(model, 'protein_encoder_type') and model.protein_encoder_type == 'esm':
                    # For ESM encoder, use raw sequence
                    protein_data = [protein_sequence]
                else:
                    # For CNN encoder, tokenize
                    protein_data = self.data_processor.process_protein_sequence(protein_sequence)
                
                # Process drug data (create mock graph data for now)
                # In a real implementation, this would convert SMILES to molecular graph
                drug_data = self._create_mock_drug_data(smiles)
                
                return {
                    'drug_data': drug_data,
                    'protein_data': protein_data
                }
                
            except Exception as e:
                logger.error(f"Data processing failed: {str(e)}")
                raise
        
        return await loop.run_in_executor(self.executor, _process)
    
    def _create_mock_drug_data(self, smiles: str):
        """Create mock drug data for prediction (placeholder)"""
        # This is a placeholder - in a real implementation, this would:
        # 1. Convert SMILES to molecular graph using RDKit
        # 2. Extract node and edge features
        # 3. Create PyTorch Geometric Data object
        
        # For now, create a simple mock object
        class MockDrugData:
            def __init__(self):
                # Mock node features (78-dimensional as expected by GIN)
                self.x = torch.randn(10, 78)  # 10 nodes, 78 features each
                # Mock edge indices (simple chain)
                self.edge_index = torch.tensor([[i, i+1] for i in range(9)]).t().contiguous()
                # Mock batch (single molecule)
                self.batch = torch.zeros(10, dtype=torch.long)
        
        return MockDrugData()
    
    async def _make_prediction(self, model: UnifiedDTAModel, processed_data: Dict[str, Any]) -> float:
        """Make prediction using the model"""
        loop = asyncio.get_event_loop()
        
        def _predict():
            try:
                model.eval()
                with torch.no_grad():
                    # Move data to model device
                    device = next(model.parameters()).device
                    
                    drug_data = processed_data['drug_data']
                    protein_data = processed_data['protein_data']
                    
                    # Move drug data to device
                    if hasattr(drug_data, 'x'):
                        drug_data.x = drug_data.x.to(device)
                    if hasattr(drug_data, 'edge_index'):
                        drug_data.edge_index = drug_data.edge_index.to(device)
                    if hasattr(drug_data, 'batch'):
                        drug_data.batch = drug_data.batch.to(device)
                    
                    # Handle protein data based on encoder type
                    if isinstance(protein_data, list):
                        # ESM encoder expects list of strings
                        pass
                    else:
                        # CNN encoder expects tensor
                        protein_data = protein_data.unsqueeze(0).to(device)
                    
                    # Make prediction
                    prediction = model(drug_data, protein_data)
                    
                    # Return scalar prediction
                    if prediction.dim() > 0:
                        return prediction.item()
                    else:
                        return float(prediction)
                        
            except Exception as e:
                logger.error(f"Model prediction failed: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
        return await loop.run_in_executor(self.executor, _predict)


# Global prediction service instance
_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Get the global prediction service instance"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
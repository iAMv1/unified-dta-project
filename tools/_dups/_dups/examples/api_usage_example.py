#!/usr/bin/env python3
"""
Example usage of the Unified DTA API
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any


async def predict_single_affinity(
    drug_smiles: str, 
    protein_sequence: str,
    model_type: str = "production",
    api_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Predict drug-target affinity for a single compound-protein pair
    
    Args:
        drug_smiles: SMILES string of the drug compound
        protein_sequence: Amino acid sequence of the target protein
        model_type: Type of model to use ('lightweight', 'production', 'custom')
        api_url: Base URL of the API server
        
    Returns:
        Prediction result dictionary
    """
    request_data = {
        "drug_smiles": drug_smiles,
        "protein_sequence": protein_sequence,
        "model_type": model_type
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_url}/api/v1/predict",
            json=request_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.json()
                raise Exception(f"API Error {response.status}: {error}")


async def predict_batch_affinities(
    drug_protein_pairs: List[Dict[str, str]],
    model_type: str = "production",
    api_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Predict drug-target affinities for multiple compound-protein pairs
    
    Args:
        drug_protein_pairs: List of dicts with 'drug_smiles' and 'protein_sequence' keys
        model_type: Type of model to use
        api_url: Base URL of the API server
        
    Returns:
        Batch prediction results dictionary
    """
    predictions = []
    for pair in drug_protein_pairs:
        predictions.append({
            "drug_smiles": pair["drug_smiles"],
            "protein_sequence": pair["protein_sequence"]
        })
    
    request_data = {
        "model_type": model_type,
        "predictions": predictions
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_url}/api/v1/predict/batch",
            json=request_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.json()
                raise Exception(f"API Error {response.status}: {error}")


async def get_model_info(
    model_type: str = "production",
    api_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Get information about a specific model
    
    Args:
        model_type: Type of model to get info about
        api_url: Base URL of the API server
        
    Returns:
        Model information dictionary
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{api_url}/api/v1/models/{model_type}/info") as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.json()
                raise Exception(f"API Error {response.status}: {error}")


async def check_api_health(api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Check API health status
    
    Args:
        api_url: Base URL of the API server
        
    Returns:
        Health status dictionary
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{api_url}/api/v1/health") as response:
            return await response.json()


async def main():
    """Example usage of the API"""
    api_url = "http://localhost:8000"
    
    print("Unified DTA API Usage Example")
    print("=" * 40)
    
    try:
        # Check API health
        print("\n1. Checking API health...")
        health = await check_api_health(api_url)
        print(f"API Status: {health['status']}")
        print(f"GPU Available: {health['gpu_available']}")
        print(f"Loaded Models: {health['models_loaded']}")
        
        # Get model information
        print("\n2. Getting model information...")
        model_info = await get_model_info("lightweight", api_url)
        print(f"Model Type: {model_info['model_type']}")
        print(f"Protein Encoder: {model_info['protein_encoder']}")
        print(f"Parameters: {model_info['parameters']:,}")
        print(f"Memory Usage: {model_info.get('memory_usage_mb', 'N/A')} MB")
        
        # Single prediction example
        print("\n3. Single prediction example...")
        drug_smiles = "CCO"  # Ethanol
        protein_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        
        result = await predict_single_affinity(
            drug_smiles=drug_smiles,
            protein_sequence=protein_sequence,
            model_type="lightweight",
            api_url=api_url
        )
        
        print(f"Drug: {result['drug_smiles']}")
        print(f"Protein: {result['protein_sequence']}")
        print(f"Predicted Affinity: {result['predicted_affinity']:.4f}")
        print(f"Processing Time: {result['processing_time_ms']:.2f} ms")
        
        # Batch prediction example
        print("\n4. Batch prediction example...")
        drug_protein_pairs = [
            {
                "drug_smiles": "CCO",  # Ethanol
                "protein_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
            },
            {
                "drug_smiles": "CC(C)O",  # Isopropanol
                "protein_sequence": "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVK"
            },
            {
                "drug_smiles": "CCCCO",  # Butanol
                "protein_sequence": "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPSEECLFLERLEENHYNTYTSKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV"
            }
        ]
        
        batch_result = await predict_batch_affinities(
            drug_protein_pairs=drug_protein_pairs,
            model_type="lightweight",
            api_url=api_url
        )
        
        print(f"Batch Results:")
        print(f"  Total Processing Time: {batch_result['total_processing_time_ms']:.2f} ms")
        print(f"  Successful Predictions: {batch_result['successful_predictions']}")
        print(f"  Failed Predictions: {batch_result['failed_predictions']}")
        
        for i, prediction in enumerate(batch_result['predictions']):
            print(f"  Prediction {i+1}: {prediction['predicted_affinity']:.4f}")
        
        print("\n5. Example completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
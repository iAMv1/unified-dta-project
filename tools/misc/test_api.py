#!/usr/bin/env python3
"""
Test script for the Unified DTA API
"""

import asyncio
import aiohttp
import json
import time
import sys
from typing import Dict, Any


class APITester:
    """Test client for the Unified DTA API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/v1"
    
    async def test_health_check(self) -> Dict[str, Any]:
        """Test the health check endpoint"""
        print("Testing health check...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base}/health") as response:
                result = await response.json()
                print(f"Health check status: {response.status}")
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
    
    async def test_root_endpoint(self) -> Dict[str, Any]:
        """Test the root endpoint"""
        print("\nTesting root endpoint...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base}/") as response:
                result = await response.json()
                print(f"Root endpoint status: {response.status}")
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
    
    async def test_list_models(self) -> Dict[str, Any]:
        """Test the list models endpoint"""
        print("\nTesting list models...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base}/models") as response:
                result = await response.json()
                print(f"List models status: {response.status}")
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
    
    async def test_model_info(self, model_type: str = "lightweight") -> Dict[str, Any]:
        """Test the model info endpoint"""
        print(f"\nTesting model info for {model_type}...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base}/models/{model_type}/info") as response:
                result = await response.json()
                print(f"Model info status: {response.status}")
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
    
    async def test_single_prediction(self) -> Dict[str, Any]:
        """Test single prediction endpoint"""
        print("\nTesting single prediction...")
        
        # Sample data
        request_data = {
            "drug_smiles": "CCO",  # Ethanol (simple molecule)
            "protein_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "model_type": "lightweight"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/predict",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()
                print(f"Single prediction status: {response.status}")
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
    
    async def test_batch_prediction(self) -> Dict[str, Any]:
        """Test batch prediction endpoint"""
        print("\nTesting batch prediction...")
        
        # Sample batch data
        request_data = {
            "model_type": "lightweight",
            "predictions": [
                {
                    "drug_smiles": "CCO",
                    "protein_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
                },
                {
                    "drug_smiles": "CC(C)O",  # Isopropanol
                    "protein_sequence": "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVK"
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/predict/batch",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()
                print(f"Batch prediction status: {response.status}")
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
    
    async def test_preload_model(self, model_type: str = "production") -> Dict[str, Any]:
        """Test model preloading"""
        print(f"\nTesting model preload for {model_type}...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.api_base}/models/{model_type}/load") as response:
                result = await response.json()
                print(f"Preload model status: {response.status}")
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
    
    async def test_clear_cache(self) -> Dict[str, Any]:
        """Test cache clearing"""
        print("\nTesting cache clear...")
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{self.api_base}/cache") as response:
                result = await response.json()
                print(f"Clear cache status: {response.status}")
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid data"""
        print("\nTesting error handling...")
        
        # Invalid request data
        request_data = {
            "drug_smiles": "",  # Empty SMILES
            "protein_sequence": "INVALID123",  # Invalid characters
            "model_type": "lightweight"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/predict",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()
                print(f"Error handling status: {response.status}")
                print(f"Response: {json.dumps(result, indent=2)}")
                return result
    
    async def run_all_tests(self):
        """Run all API tests"""
        print("=" * 60)
        print("UNIFIED DTA API TEST SUITE")
        print("=" * 60)
        
        try:
            # Basic endpoint tests
            await self.test_health_check()
            await self.test_root_endpoint()
            await self.test_list_models()
            
            # Model management tests
            await self.test_model_info("lightweight")
            await self.test_preload_model("lightweight")
            
            # Prediction tests
            await self.test_single_prediction()
            await self.test_batch_prediction()
            
            # Error handling test
            await self.test_error_handling()
            
            # Cache management test
            await self.test_clear_cache()
            
            print("\n" + "=" * 60)
            print("ALL TESTS COMPLETED")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nTest failed with error: {str(e)}")
            import traceback
            traceback.print_exc()


async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Unified DTA API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="Base URL of the API server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--test",
        choices=["all", "health", "predict", "batch", "models"],
        default="all",
        help="Specific test to run (default: all)"
    )
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.test == "all":
        await tester.run_all_tests()
    elif args.test == "health":
        await tester.test_health_check()
    elif args.test == "predict":
        await tester.test_single_prediction()
    elif args.test == "batch":
        await tester.test_batch_prediction()
    elif args.test == "models":
        await tester.test_list_models()
        await tester.test_model_info()


if __name__ == "__main__":
    asyncio.run(main())
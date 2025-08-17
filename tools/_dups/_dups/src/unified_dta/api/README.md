# Unified DTA System API

A RESTful API for drug-target affinity prediction using state-of-the-art machine learning models.

## Features

- **Single & Batch Predictions**: Predict binding affinities for individual or multiple drug-target pairs
- **Multiple Model Types**: Support for lightweight, production, and custom model configurations
- **Async Processing**: High-performance asynchronous request handling
- **Model Caching**: Intelligent model loading and caching for optimal performance
- **Memory Management**: Automatic memory optimization and device handling
- **Comprehensive Error Handling**: Structured error responses with detailed information
- **Health Monitoring**: Built-in health checks and system monitoring

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Development mode (CPU, auto-reload)
python run_api.py --env development --reload

# Production mode (auto device detection)
python run_api.py --env production --host 0.0.0.0 --port 8000

# Custom configuration
python run_api.py --device cuda --max-models 5 --port 8080
```

### 3. Test the API

```bash
# Run comprehensive tests
python test_api.py

# Test specific endpoints
python test_api.py --test health
python test_api.py --test predict
```

## API Endpoints

### Core Prediction Endpoints

#### Single Prediction
```http
POST /api/v1/predict
Content-Type: application/json

{
  "drug_smiles": "CCO",
  "protein_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
  "model_type": "production"
}
```

#### Batch Prediction
```http
POST /api/v1/predict/batch
Content-Type: application/json

{
  "model_type": "production",
  "predictions": [
    {
      "drug_smiles": "CCO",
      "protein_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    },
    {
      "drug_smiles": "CC(C)O",
      "protein_sequence": "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVK"
    }
  ]
}
```

### Model Management Endpoints

#### List Available Models
```http
GET /api/v1/models
```

#### Get Model Information
```http
GET /api/v1/models/{model_type}/info
```

#### Preload Model
```http
POST /api/v1/models/{model_type}/load
```

#### Clear Model Cache
```http
DELETE /api/v1/cache
```

### System Endpoints

#### Health Check
```http
GET /api/v1/health
```

#### API Information
```http
GET /api/v1/
```

## Model Types

### Lightweight
- **Use Case**: Development, testing, resource-constrained environments
- **Memory**: ~100MB RAM
- **Features**: CNN protein encoder, basic GIN drug encoder
- **Performance**: Fast inference, moderate accuracy

### Production
- **Use Case**: Production deployments, high accuracy requirements
- **Memory**: ~4GB RAM + GPU recommended
- **Features**: ESM-2 protein encoder, advanced GIN drug encoder, fusion mechanisms
- **Performance**: High accuracy, moderate inference speed

### Custom
- **Use Case**: Specialized configurations
- **Configuration**: Fully customizable via request parameters
- **Features**: Any combination of available encoders and settings

## Usage Examples

### Python Client Example

```python
import asyncio
import aiohttp

async def predict_affinity(drug_smiles, protein_sequence):
    request_data = {
        "drug_smiles": drug_smiles,
        "protein_sequence": protein_sequence,
        "model_type": "production"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/api/v1/predict",
            json=request_data
        ) as response:
            return await response.json()

# Usage
result = asyncio.run(predict_affinity(
    "CCO", 
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
))
print(f"Predicted affinity: {result['predicted_affinity']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "drug_smiles": "CCO",
    "protein_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "model_type": "lightweight"
  }'
```

## Configuration Options

### Server Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind the server |
| `--port` | `8000` | Port to bind the server |
| `--env` | `development` | Environment configuration |
| `--max-models` | `3` | Maximum cached models |
| `--device` | `auto` | Device for model inference |
| `--workers` | `1` | Number of worker processes |
| `--reload` | `False` | Enable auto-reload |

### Environment Configurations

#### Development
- Debug mode enabled
- CPU-only processing
- Auto-reload enabled
- Verbose logging

#### Production
- Debug mode disabled
- Auto device detection
- CORS disabled (configure for your domain)
- Optimized logging

## Error Handling

The API provides structured error responses with detailed information:

```json
{
  "error": "Request validation failed",
  "error_type": "ValidationError",
  "details": {
    "errors": [
      {
        "field": "drug_smiles",
        "message": "SMILES string cannot be empty"
      }
    ]
  }
}
```

### Common Error Codes

- `400`: Bad Request (validation errors)
- `404`: Not Found (invalid model type)
- `422`: Unprocessable Entity (malformed request)
- `500`: Internal Server Error (processing errors)

## Performance Considerations

### Memory Usage
- **Lightweight Model**: ~100MB RAM
- **Production Model**: ~4GB RAM + GPU memory
- **Batch Processing**: Memory scales with batch size

### Throughput
- **Single Predictions**: ~10-50 requests/second (depending on model and hardware)
- **Batch Processing**: More efficient for multiple predictions
- **Model Caching**: First request loads model, subsequent requests are faster

### Optimization Tips
1. Use batch predictions for multiple requests
2. Preload models during startup
3. Use appropriate model type for your use case
4. Configure cache size based on available memory
5. Use GPU for production models when available

## Monitoring and Logging

### Health Monitoring
The `/health` endpoint provides comprehensive system information:
- Service status
- Loaded models
- GPU availability
- Memory usage statistics

### Logging
- Request/response logging
- Error tracking with stack traces
- Performance metrics
- Model loading/caching events

### Metrics
- Processing time per request
- Memory usage tracking
- Cache hit/miss rates
- Error rates by endpoint

## Development

### Running Tests
```bash
# Full test suite
python test_api.py

# Specific tests
python test_api.py --test health
python test_api.py --test predict
python test_api.py --test batch
```

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Adding Custom Models
1. Implement model in `unified_dta/core/models.py`
2. Add factory method in `unified_dta/core/model_factory.py`
3. Update model type enum in `unified_dta/api/models.py`
4. Test with new model type

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run_api.py", "--env", "production", "--host", "0.0.0.0"]
```

### Production Checklist
- [ ] Configure CORS for your domain
- [ ] Set up proper logging and monitoring
- [ ] Configure resource limits
- [ ] Set up health checks
- [ ] Configure SSL/TLS
- [ ] Set up load balancing if needed
- [ ] Monitor memory usage and cache performance

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check device availability (CPU/GPU)
   - Verify memory requirements
   - Check model dependencies

2. **Memory Issues**
   - Reduce batch size
   - Use lightweight models
   - Clear cache regularly
   - Monitor system memory

3. **Performance Issues**
   - Use GPU for production models
   - Implement model preloading
   - Optimize batch sizes
   - Monitor cache hit rates

4. **Validation Errors**
   - Check SMILES string format
   - Verify protein sequence characters
   - Validate request structure

For more detailed troubleshooting, check the application logs and health endpoint.
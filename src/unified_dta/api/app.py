"""
FastAPI application factory and configuration
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import time
import traceback
from typing import Dict, Any

from .endpoints import router
from .models import ErrorResponse
from .cache import initialize_cache

logger = logging.getLogger(__name__)


def create_app(
    title: str = "Unified DTA System API",
    description: str = "RESTful API for drug-target affinity prediction using state-of-the-art models",
    version: str = "1.0.0",
    max_cached_models: int = 3,
    device: str = "auto",
    enable_cors: bool = True,
    debug: bool = False
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        title: API title
        description: API description
        version: API version
        max_cached_models: Maximum number of models to cache
        device: Device to use for models ('auto', 'cpu', 'cuda')
        enable_cors: Whether to enable CORS middleware
        debug: Enable debug mode
        
    Returns:
        Configured FastAPI application
    """
    
    # Create FastAPI app
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        debug=debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Initialize model cache
    initialize_cache(max_models=max_cached_models, device=device)
    
    # Add CORS middleware
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Add logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} "
                f"({process_time:.3f}s) {request.method} {request.url}"
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url} "
                f"({process_time:.3f}s) - {str(e)}"
            )
            raise
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    # Global exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with structured error response"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail,
                error_type="HTTPException",
                details={"status_code": exc.status_code}
            ).dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="Request validation failed",
                error_type="ValidationError",
                details={
                    "errors": exc.errors(),
                    "body": exc.body
                }
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {str(exc)}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                error_type=type(exc).__name__,
                details={"message": str(exc)} if debug else None
            ).dict()
        )
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup"""
        logger.info(f"Starting {title} v{version}")
        logger.info(f"Model cache initialized with max_models={max_cached_models}, device={device}")
        
        # Optionally preload a lightweight model
        try:
            from .cache import get_model_cache
            model_cache = get_model_cache()
            logger.info("Preloading lightweight model...")
            await model_cache.get_model("lightweight")
            logger.info("Lightweight model preloaded successfully")
        except Exception as e:
            logger.warning(f"Could not preload lightweight model: {str(e)}")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        logger.info("Shutting down API server")
        
        try:
            from .cache import get_model_cache
            model_cache = get_model_cache()
            model_cache.clear_cache()
            logger.info("Model cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing cache on shutdown: {str(e)}")
    
    return app


# Convenience function for development
def create_development_app() -> FastAPI:
    """Create app configured for development"""
    return create_app(
        debug=True,
        max_cached_models=2,
        device="cpu"  # Use CPU for development
    )


# Convenience function for production
def create_production_app() -> FastAPI:
    """Create app configured for production"""
    return create_app(
        debug=False,
        max_cached_models=3,
        device="auto",
        enable_cors=False  # Configure CORS properly for production
    )
"""
Main entry point for the Unified DTA API server
"""

import uvicorn
import logging
import argparse
import sys
from pathlib import Path

from .app import create_app, create_development_app, create_production_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('unified_dta_api.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the API server"""
    parser = argparse.ArgumentParser(description="Unified DTA API Server")
    
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--env", 
        choices=["development", "production", "custom"],
        default="development",
        help="Environment configuration (default: development)"
    )
    parser.add_argument(
        "--max-models", 
        type=int, 
        default=3,
        help="Maximum number of models to cache (default: 3)"
    )
    parser.add_argument(
        "--device", 
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for models (default: auto)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level", 
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Log level (default: info)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create app based on environment
    if args.env == "development":
        app = create_development_app()
    elif args.env == "production":
        app = create_production_app()
    else:
        app = create_app(
            max_cached_models=args.max_models,
            device=args.device,
            debug=(args.env == "development")
        )
    
    logger.info(f"Starting Unified DTA API server")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Max cached models: {args.max_models}")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()
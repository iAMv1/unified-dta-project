"""
RESTful API for Unified DTA System
Provides endpoints for drug-target affinity prediction and batch processing
"""

from .app import create_app
from .models import *
from .endpoints import *

__all__ = ['create_app']
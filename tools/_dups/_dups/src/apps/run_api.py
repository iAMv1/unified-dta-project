#!/usr/bin/env python3
"""
Simple script to run the Unified DTA API server
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from unified_dta.api.main import main

if __name__ == "__main__":
    main()
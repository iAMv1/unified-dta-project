"""
Data processing utilities for the Unified DTA System
SMILES validation, protein processing, and molecular graph conversion
"""

import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. SMILES processing will be limited.")
    RDKIT_AVAILABLE = False

# Amino acid vocabulary for protein sequences
AMINO_ACIDS = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
    'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
    'V': 18, 'W': 19, 'Y': 20, 'X': 21, 'U': 22, 'B': 23, 'Z': 24, 'O': 25
}


class DataProcessor:
    """Main data processing class"""
    
    def __init__(self, max_protein_length: int = 200):
        self.max_protein_length = max_protein_length
        self.smiles_validator = SMILESValidator() if RDKIT_AVAILABLE else None
    
    def process_protein_sequence(self, sequence: str) -> torch.Tensor:
        """Convert protein sequence to tokens"""
        # Truncate sequence
        sequence = sequence[:self.max_protein_length]
        
        # Convert to tokens
        tokens = [AMINO_ACIDS.get(aa, 0) for aa in sequence.upper()]
        
        # Pad to max length
        while len(tokens) < self.max_protein_length:
            tokens.append(0)  # Padding token
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string"""
        if not RDKIT_AVAILABLE:
            return True  # Skip validation if RDKit not available
        
        return self.smiles_validator.validate_smiles(smiles)


class SMILESValidator:
    """Validator for SMILES strings using RDKit"""
    
    def __init__(self):
        self.valid_count = 0
        self.invalid_count = 0
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate a single SMILES string"""
        if not RDKIT_AVAILABLE:
            return True
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                self.valid_count += 1
                return True
            else:
                self.invalid_count += 1
                return False
        except Exception:
            self.invalid_count += 1
            return False
    
    def get_statistics(self) -> Dict[str, int]:
        """Get validation statistics"""
        return {
            'valid_count': self.valid_count,
            'invalid_count': self.invalid_count,
            'total_count': self.valid_count + self.invalid_count
        }
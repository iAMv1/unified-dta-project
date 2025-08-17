"""
Data processing and dataset utilities
"""

# Import what's actually available
from .data_processing import SMILESValidator, ProteinProcessor, MolecularGraphConverter, DataValidator
from .data_processor import DataProcessor
from .datasets import DTADataset
from .graph_preprocessing import MolecularGraphProcessor

# Create convenience functions
def validate_smiles(smiles: str) -> bool:
    """Validate a SMILES string"""
    validator = SMILESValidator()
    return validator.validate_smiles(smiles)

def process_protein_sequence(sequence: str, return_tensor: bool = False):
    """Process a protein sequence"""
    processor = ProteinProcessor()
    return processor.process_sequence(sequence, return_tensor)

def smiles_to_graph(smiles: str):
    """Convert SMILES to molecular graph"""
    processor = MolecularGraphProcessor()
    return processor.smiles_to_graph(smiles)

def create_molecular_graph(smiles: str):
    """Create molecular graph from SMILES"""
    processor = MolecularGraphProcessor()
    return processor.smiles_to_graph(smiles)

# Use DTADataset as UnifiedDTADataset
UnifiedDTADataset = DTADataset

# Placeholder for missing function
create_data_loaders = None

__all__ = [
    # Data processing
    "create_data_loaders",
    "validate_smiles", 
    "process_protein_sequence",
    "DataProcessor",
    "SMILESValidator",
    "ProteinProcessor",
    "MolecularGraphConverter",
    "DataValidator",
    
    # Datasets
    "UnifiedDTADataset",
    "DTADataset",
    
    # Graph processing
    "smiles_to_graph",
    "create_molecular_graph",
    "MolecularGraphProcessor"
]
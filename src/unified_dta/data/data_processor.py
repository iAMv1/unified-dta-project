"""
DataProcessor class that combines functionality from various data processing modules
"""

from .data_processing import SMILESValidator, ProteinProcessor, MolecularGraphConverter, DataValidator
from .graph_preprocessing import MolecularGraphProcessor


class DataProcessor:
    """Unified data processor that combines all data processing functionality"""
    
    def __init__(self):
        self.smiles_validator = SMILESValidator()
        self.protein_processor = ProteinProcessor()
        self.graph_converter = MolecularGraphConverter()
        self.validator = DataValidator()
        self.molecular_processor = MolecularGraphProcessor()
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate a SMILES string"""
        return self.smiles_validator.validate_smiles(smiles)
    
    def process_protein_sequence(self, sequence: str, return_tensor: bool = False):
        """Process a protein sequence"""
        return self.protein_processor.process_sequence(sequence, return_tensor)
    
    def smiles_to_graph(self, smiles: str):
        """Convert SMILES to molecular graph"""
        return self.molecular_processor.smiles_to_graph(smiles)
    
    def validate_dta_sample(self, smiles: str, protein_sequence: str, affinity: float):
        """Validate a DTA sample"""
        return self.validator.validate_dta_sample(smiles, protein_sequence, affinity)
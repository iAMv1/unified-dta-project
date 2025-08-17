"""
Data processing utilities for the Unified DTA System
SMILES validation, protein processing, and molecular graph conversion
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
import re
from pathlib import Path
import warnings

# Suppress RDKit warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. SMILES processing will be limited.")
    RDKIT_AVAILABLE = False


# Amino acid vocabulary for protein sequences
AMINO_ACIDS = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
    'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'V': 17, 'W': 18, 'Y': 19, 'X': 20, 'U': 21, 'B': 22, 'Z': 23, 'O': 24
}

# Reverse mapping
AMINO_ACID_TOKENS = {v: k for k, v in AMINO_ACIDS.items()}


class SMILESValidator:
    """Validator for SMILES strings using RDKit"""
    
    def __init__(self):
        self.valid_count = 0
        self.invalid_count = 0
        self.error_log = []
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate a single SMILES string"""
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available. Cannot validate SMILES.")
            return True  # Assume valid if we can't check
        
        if not isinstance(smiles, str) or not smiles.strip():
            self.invalid_count += 1
            self.error_log.append(f"Empty or invalid SMILES: {smiles}")
            return False
        
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                self.invalid_count += 1
                self.error_log.append(f"Invalid SMILES: {smiles}")
                return False
            
            self.valid_count += 1
            return True
            
        except Exception as e:
            self.invalid_count += 1
            self.error_log.append(f"Error validating SMILES {smiles}: {e}")
            return False
    
    def validate_batch(self, smiles_list: List[str]) -> List[bool]:
        """Validate a batch of SMILES strings"""
        return [self.validate_smiles(smiles) for smiles in smiles_list]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.valid_count + self.invalid_count
        return {
            'total_processed': total,
            'valid_count': self.valid_count,
            'invalid_count': self.invalid_count,
            'valid_percentage': (self.valid_count / total * 100) if total > 0 else 0,
            'recent_errors': self.error_log[-10:]  # Last 10 errors
        }


class ProteinProcessor:
    """Processor for protein sequences"""
    
    def __init__(self, max_length: int = 200, padding_token: int = 25):
        self.max_length = max_length
        self.padding_token = padding_token
        self.processed_count = 0
        self.truncated_count = 0
    
    def clean_sequence(self, sequence: str) -> str:
        """Clean protein sequence by removing invalid characters"""
        if not isinstance(sequence, str):
            return ""
        
        # Remove whitespace and convert to uppercase
        sequence = sequence.strip().upper()
        
        # Remove any non-amino acid characters
        valid_chars = set(AMINO_ACIDS.keys())
        cleaned = ''.join(char if char in valid_chars else 'X' for char in sequence)
        
        return cleaned
    
    def tokenize_sequence(self, sequence: str) -> List[int]:
        """Convert protein sequence to token indices"""
        cleaned_seq = self.clean_sequence(sequence)
        tokens = [AMINO_ACIDS.get(aa, AMINO_ACIDS['X']) for aa in cleaned_seq]
        return tokens
    
    def truncate_sequence(self, sequence: str) -> str:
        """Truncate sequence to maximum length"""
        if len(sequence) > self.max_length:
            self.truncated_count += 1
            return sequence[:self.max_length]
        return sequence
    
    def pad_tokens(self, tokens: List[int]) -> List[int]:
        """Pad token sequence to fixed length"""
        if len(tokens) > self.max_length:
            return tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            return tokens + [self.padding_token] * (self.max_length - len(tokens))
        return tokens
    
    def process_sequence(self, sequence: str, return_tensor: bool = False) -> Union[List[int], torch.Tensor]:
        """Complete processing pipeline for a protein sequence"""
        self.processed_count += 1
        
        # Clean and truncate
        cleaned = self.clean_sequence(sequence)
        truncated = self.truncate_sequence(cleaned)
        
        # Tokenize and pad
        tokens = self.tokenize_sequence(truncated)
        padded_tokens = self.pad_tokens(tokens)
        
        if return_tensor:
            return torch.tensor(padded_tokens, dtype=torch.long)
        return padded_tokens
    
    def process_batch(self, sequences: List[str], return_tensor: bool = False) -> Union[List[List[int]], torch.Tensor]:
        """Process a batch of protein sequences"""
        processed = [self.process_sequence(seq, return_tensor=False) for seq in sequences]
        
        if return_tensor:
            return torch.tensor(processed, dtype=torch.long)
        return processed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'processed_count': self.processed_count,
            'truncated_count': self.truncated_count,
            'truncation_rate': (self.truncated_count / self.processed_count * 100) if self.processed_count > 0 else 0,
            'max_length': self.max_length
        }


class MolecularGraphConverter:
    """Convert SMILES to molecular graphs for PyTorch Geometric"""
    
    def __init__(self):
        self.conversion_count = 0
        self.failed_count = 0
        self.error_log = []
    
    def get_atom_features(self, atom) -> List[float]:
        """Extract atom features for graph nodes"""
        if not RDKIT_AVAILABLE:
            return [0.0] * 78  # Return zero features if RDKit not available
        
        features = []
        
        # Atomic number (one-hot encoded for common elements)
        atomic_nums = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
        atomic_num = atom.GetAtomicNum()
        features.extend([1.0 if atomic_num == num else 0.0 for num in atomic_nums])
        
        # Degree (one-hot encoded)
        degree = atom.GetDegree()
        features.extend([1.0 if degree == d else 0.0 for d in range(6)])
        
        # Formal charge (one-hot encoded)
        formal_charge = atom.GetFormalCharge()
        features.extend([1.0 if formal_charge == c else 0.0 for c in [-2, -1, 0, 1, 2]])
        
        # Hybridization (one-hot encoded)
        hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
        hybridization = atom.GetHybridization()
        features.extend([1.0 if hybridization == h else 0.0 for h in hybridizations])
        
        # Aromaticity
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        
        # Number of hydrogens (one-hot encoded)
        num_hs = atom.GetTotalNumHs()
        features.extend([1.0 if num_hs == h else 0.0 for h in range(5)])
        
        # Chirality
        features.append(1.0 if atom.HasProp('_ChiralityPossible') else 0.0)
        
        # Additional features to reach 78 dimensions
        features.extend([
            atom.GetMass() / 100.0,  # Normalized mass
            atom.GetTotalValence() / 10.0,  # Normalized valence
            1.0 if atom.IsInRing() else 0.0,  # In ring
            1.0 if atom.IsInRingSize(3) else 0.0,  # In 3-ring
            1.0 if atom.IsInRingSize(4) else 0.0,  # In 4-ring
            1.0 if atom.IsInRingSize(5) else 0.0,  # In 5-ring
            1.0 if atom.IsInRingSize(6) else 0.0,  # In 6-ring
            1.0 if atom.IsInRingSize(7) else 0.0,  # In 7-ring
        ])
        
        # Pad to exactly 78 features
        while len(features) < 78:
            features.append(0.0)
        
        return features[:78]
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES string to PyTorch Geometric Data object"""
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available. Cannot convert SMILES to graph.")
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                self.failed_count += 1
                self.error_log.append(f"Invalid SMILES for graph conversion: {smiles}")
                return None
            
            # Add hydrogens for complete structure
            mol = Chem.AddHs(mol)
            
            # Extract atom features
            atom_features = []
            for atom in mol.GetAtoms():
                features = self.get_atom_features(atom)
                atom_features.append(features)
            
            # Extract bonds (edges)
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])  # Undirected graph
            
            # Convert to tensors
            x = torch.tensor(atom_features, dtype=torch.float)
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            else:
                # Handle molecules with no bonds (single atoms)
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index)
            
            # Add molecular properties
            data.num_atoms = mol.GetNumAtoms()
            data.smiles = smiles
            
            self.conversion_count += 1
            return data
            
        except Exception as e:
            self.failed_count += 1
            self.error_log.append(f"Error converting SMILES {smiles}: {e}")
            return None
    
    def convert_batch(self, smiles_list: List[str]) -> List[Optional[Data]]:
        """Convert a batch of SMILES to graphs"""
        return [self.smiles_to_graph(smiles) for smiles in smiles_list]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversion statistics"""
        total = self.conversion_count + self.failed_count
        return {
            'total_processed': total,
            'successful_conversions': self.conversion_count,
            'failed_conversions': self.failed_count,
            'success_rate': (self.conversion_count / total * 100) if total > 0 else 0,
            'recent_errors': self.error_log[-10:]
        }


class DataValidator:
    """Comprehensive data validation for DTA datasets"""
    
    def __init__(self):
        self.smiles_validator = SMILESValidator()
        self.protein_processor = ProteinProcessor()
        self.graph_converter = MolecularGraphConverter()
    
    def validate_dta_sample(self, smiles: str, protein_sequence: str, affinity: float) -> Dict[str, Any]:
        """Validate a single DTA sample"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate SMILES
        if not self.smiles_validator.validate_smiles(smiles):
            validation_result['valid'] = False
            validation_result['errors'].append("Invalid SMILES string")
        
        # Validate protein sequence
        if not isinstance(protein_sequence, str) or len(protein_sequence.strip()) == 0:
            validation_result['valid'] = False
            validation_result['errors'].append("Empty protein sequence")
        elif len(protein_sequence) > 2000:  # Very long sequences might be problematic
            validation_result['warnings'].append("Very long protein sequence (>2000 residues)")
        
        # Validate affinity value
        if not isinstance(affinity, (int, float)) or np.isnan(affinity):
            validation_result['valid'] = False
            validation_result['errors'].append("Invalid affinity value")
        elif affinity < 0:
            validation_result['warnings'].append("Negative affinity value")
        
        return validation_result
    
    def validate_dataset(self, df: pd.DataFrame, 
                        smiles_col: str = 'compound_iso_smiles',
                        protein_col: str = 'target_sequence',
                        affinity_col: str = 'affinity') -> Dict[str, Any]:
        """Validate an entire dataset"""
        
        logger.info(f"Validating dataset with {len(df)} samples...")
        
        validation_stats = {
            'total_samples': len(df),
            'valid_samples': 0,
            'invalid_samples': 0,
            'samples_with_warnings': 0,
            'validation_errors': [],
            'column_stats': {}
        }
        
        # Check required columns
        required_cols = [smiles_col, protein_col, affinity_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            validation_stats['validation_errors'].append(f"Missing columns: {missing_cols}")
            return validation_stats
        
        # Validate each sample
        valid_indices = []
        
        for idx, row in df.iterrows():
            sample_validation = self.validate_dta_sample(
                row[smiles_col], 
                row[protein_col], 
                row[affinity_col]
            )
            
            if sample_validation['valid']:
                validation_stats['valid_samples'] += 1
                valid_indices.append(idx)
            else:
                validation_stats['invalid_samples'] += 1
                validation_stats['validation_errors'].extend(
                    [f"Row {idx}: {error}" for error in sample_validation['errors']]
                )
            
            if sample_validation['warnings']:
                validation_stats['samples_with_warnings'] += 1
        
        # Column statistics
        validation_stats['column_stats'] = {
            smiles_col: {
                'unique_values': df[smiles_col].nunique(),
                'null_values': df[smiles_col].isnull().sum()
            },
            protein_col: {
                'unique_values': df[protein_col].nunique(),
                'null_values': df[protein_col].isnull().sum(),
                'avg_length': df[protein_col].str.len().mean(),
                'max_length': df[protein_col].str.len().max()
            },
            affinity_col: {
                'null_values': df[affinity_col].isnull().sum(),
                'mean': df[affinity_col].mean(),
                'std': df[affinity_col].std(),
                'min': df[affinity_col].min(),
                'max': df[affinity_col].max()
            }
        }
        
        # Overall statistics
        validation_stats['valid_percentage'] = (
            validation_stats['valid_samples'] / validation_stats['total_samples'] * 100
        )
        
        logger.info(f"Validation complete: {validation_stats['valid_samples']}/{validation_stats['total_samples']} "
                   f"samples valid ({validation_stats['valid_percentage']:.1f}%)")
        
        return validation_stats


def load_dta_dataset(file_path: Union[str, Path],
                    smiles_col: str = 'compound_iso_smiles',
                    protein_col: str = 'target_sequence', 
                    affinity_col: str = 'affinity',
                    validate_data: bool = True) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Load and optionally validate a DTA dataset"""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load dataset
    logger.info(f"Loading dataset from {file_path}")
    
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded {len(df)} samples from {file_path}")
    
    # Validate if requested
    validation_stats = None
    if validate_data:
        validator = DataValidator()
        validation_stats = validator.validate_dataset(df, smiles_col, protein_col, affinity_col)
    
    return df, validation_stats


def preprocess_dta_dataset(df: pd.DataFrame,
                          smiles_col: str = 'compound_iso_smiles',
                          protein_col: str = 'target_sequence',
                          affinity_col: str = 'affinity',
                          max_protein_length: int = 200,
                          remove_invalid: bool = True) -> pd.DataFrame:
    """Preprocess a DTA dataset with cleaning and validation"""
    
    logger.info(f"Preprocessing dataset with {len(df)} samples...")
    
    # Create a copy to avoid modifying original
    processed_df = df.copy()
    
    # Initialize processors
    validator = DataValidator()
    protein_processor = ProteinProcessor(max_length=max_protein_length)
    
    # Track processing statistics
    original_count = len(processed_df)
    removed_count = 0
    
    if remove_invalid:
        # Remove samples with invalid data
        valid_indices = []
        
        for idx, row in processed_df.iterrows():
            validation = validator.validate_dta_sample(
                row[smiles_col], 
                row[protein_col], 
                row[affinity_col]
            )
            
            if validation['valid']:
                valid_indices.append(idx)
            else:
                removed_count += 1
        
        processed_df = processed_df.loc[valid_indices].reset_index(drop=True)
        logger.info(f"Removed {removed_count} invalid samples")
    
    # Clean protein sequences
    processed_df[protein_col] = processed_df[protein_col].apply(
        lambda seq: protein_processor.clean_sequence(seq)
    )
    
    # Truncate long protein sequences
    processed_df[protein_col] = processed_df[protein_col].apply(
        lambda seq: protein_processor.truncate_sequence(seq)
    )
    
    # Remove duplicates
    initial_count = len(processed_df)
    processed_df = processed_df.drop_duplicates(subset=[smiles_col, protein_col]).reset_index(drop=True)
    duplicate_count = initial_count - len(processed_df)
    
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate samples")
    
    # Final statistics
    final_count = len(processed_df)
    logger.info(f"Preprocessing complete: {original_count} → {final_count} samples "
               f"({final_count/original_count*100:.1f}% retained)")
    
    return processed_df


if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test SMILES validation
    validator = SMILESValidator()
    test_smiles = ["CCO", "C1=CC=CC=C1", "invalid_smiles", ""]
    
    print("Testing SMILES validation:")
    for smiles in test_smiles:
        is_valid = validator.validate_smiles(smiles)
        print(f"  {smiles}: {'Valid' if is_valid else 'Invalid'}")
    
    print(f"\nValidation statistics: {validator.get_statistics()}")
    
    # Test protein processing
    processor = ProteinProcessor(max_length=10)
    test_sequences = ["ACDEFGHIKLMNPQRSTVWY", "INVALID123", "SHORT"]
    
    print("\nTesting protein processing:")
    for seq in test_sequences:
        tokens = processor.process_sequence(seq)
        print(f"  {seq} → {tokens}")
    
    print(f"\nProcessing statistics: {processor.get_statistics()}")
    
    # Test molecular graph conversion
    if RDKIT_AVAILABLE:
        converter = MolecularGraphConverter()
        test_smiles = ["CCO", "C1=CC=CC=C1"]
        
        print("\nTesting molecular graph conversion:")
        for smiles in test_smiles:
            graph = converter.smiles_to_graph(smiles)
            if graph is not None:
                print(f"  {smiles} → Graph with {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges")
            else:
                print(f"  {smiles} → Conversion failed")
        
        print(f"\nConversion statistics: {converter.get_statistics()}")
    else:
        print("\nSkipping molecular graph conversion (RDKit not available)")
def create_dummy_batch(batch_size: int = 4) -> Data:
    """Create dummy molecular graph batch for testing"""
    
    # Create dummy node features (78-dimensional as per standard)
    num_nodes_per_graph = 10
    total_nodes = batch_size * num_nodes_per_graph
    
    x = torch.randn(total_nodes, 78)  # Node features
    
    # Create dummy edge indices
    edges_per_graph = 15
    edge_index_list = []
    
    for i in range(batch_size):
        node_offset = i * num_nodes_per_graph
        # Create random edges within each graph
        src = torch.randint(0, num_nodes_per_graph, (edges_per_graph,)) + node_offset
        dst = torch.randint(0, num_nodes_per_graph, (edges_per_graph,)) + node_offset
        edges = torch.stack([src, dst], dim=0)
        edge_index_list.append(edges)
    
    edge_index = torch.cat(edge_index_list, dim=1)
    
    # Create batch indices
    batch = torch.arange(batch_size).repeat_interleave(num_nodes_per_graph)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    return data
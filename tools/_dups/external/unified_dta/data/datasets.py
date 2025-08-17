"""
Dataset classes and data loaders for the Unified DTA System
Supports KIBA, Davis, and BindingDB datasets with efficient batching
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DTASample:
    """Data structure for a single DTA sample"""
    smiles: str
    protein_sequence: str
    affinity: float
    dataset_name: str = ""
    sample_id: Optional[str] = None


class DTADataset(Dataset):
    """PyTorch Dataset for Drug-Target Affinity prediction"""
    
    def __init__(self,
                 data_path: Union[str, Path, pd.DataFrame],
                 smiles_col: str = 'compound_iso_smiles',
                 protein_col: str = 'target_sequence',
                 affinity_col: str = 'affinity',
                 max_protein_length: int = 200):
        
        if isinstance(data_path, pd.DataFrame):
            self.data = data_path
        else:
            self.data = pd.read_csv(data_path)
        
        self.smiles_col = smiles_col
        self.protein_col = protein_col
        self.affinity_col = affinity_col
        self.max_protein_length = max_protein_length
        
        # Basic validation
        required_cols = [smiles_col, protein_col, affinity_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove invalid entries
        self.data = self.data.dropna(subset=required_cols)
        
        logger.info(f"Loaded dataset with {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> DTASample:
        row = self.data.iloc[idx]
        
        return DTASample(
            smiles=row[self.smiles_col],
            protein_sequence=row[self.protein_col][:self.max_protein_length],
            affinity=float(row[self.affinity_col]),
            sample_id=str(idx)
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        affinities = self.data[self.affinity_col]
        protein_lengths = self.data[self.protein_col].str.len()
        
        return {
            'num_samples': len(self.data),
            'affinity_mean': float(affinities.mean()),
            'affinity_std': float(affinities.std()),
            'affinity_min': float(affinities.min()),
            'affinity_max': float(affinities.max()),
            'protein_length_mean': float(protein_lengths.mean()),
            'protein_length_std': float(protein_lengths.std()),
            'protein_length_max': int(protein_lengths.max())
        }
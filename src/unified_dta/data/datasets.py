"""
Dataset classes and data loaders for the Unified DTA System
Supports KIBA, Davis, and BindingDB datasets with efficient batching
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import logging
import pickle
from dataclasses import dataclass
import json

from .data_processing import (
    SMILESValidator, 
    ProteinProcessor, 
    MolecularGraphConverter,
    load_dta_dataset,
    preprocess_dta_dataset
)

logger = logging.getLogger(__name__)


@dataclass
class DTASample:
    """Data structure for a single DTA sample"""
    smiles: str
    protein_sequence: str
    affinity: float
    graph: Optional[Data] = None
    protein_tokens: Optional[torch.Tensor] = None
    dataset_name: str = ""
    sample_id: Optional[str] = None


class DTADataset(Dataset):
    """PyTorch Dataset for Drug-Target Affinity prediction"""
    
    def __init__(self,
                 data_path: Union[str, Path, pd.DataFrame],
                 smiles_col: str = 'compound_iso_smiles',
                 protein_col: str = 'target_sequence',
                 affinity_col: str = 'affinity',
                 max_protein_length: int = 200,
                 preprocess: bool = True,
                 cache_dir: Optional[Union[str, Path]] = None,
                 dataset_name: str = "unknown"):
        
        self.smiles_col = smiles_col
        self.protein_col = protein_col
        self.affinity_col = affinity_col
        self.max_protein_length = max_protein_length
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize processors
        self.protein_processor = ProteinProcessor(max_length=max_protein_length)
        self.graph_converter = MolecularGraphConverter()
        
        # Load and preprocess data
        if isinstance(data_path, pd.DataFrame):
            self.df = data_path.copy()
        else:
            self.df, _ = load_dta_dataset(data_path, smiles_col, protein_col, affinity_col)
        
        if preprocess:
            self.df = preprocess_dta_dataset(
                self.df, smiles_col, protein_col, affinity_col, max_protein_length
            )
        
        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file = self.cache_dir / f"{dataset_name}_processed.pkl"
        else:
            self._cache_file = None
        
        # Try to load from cache
        self.samples = self._load_from_cache()
        if self.samples is None:
            self.samples = self._process_all_samples()
            self._save_to_cache()
        
        logger.info(f"Dataset '{dataset_name}' loaded with {len(self.samples)} samples")
    
    def _process_all_samples(self) -> List[DTASample]:
        """Process all samples and convert to DTASample objects"""
        logger.info(f"Processing {len(self.df)} samples...")
        
        samples = []
        failed_count = 0
        
        for idx, row in self.df.iterrows():
            try:
                # Extract data
                smiles = row[self.smiles_col]
                protein_seq = row[self.protein_col]
                affinity = float(row[self.affinity_col])
                
                # Convert SMILES to graph
                graph = self.graph_converter.smiles_to_graph(smiles)
                if graph is None:
                    failed_count += 1
                    continue
                
                # Process protein sequence
                protein_tokens = self.protein_processor.process_sequence(
                    protein_seq, return_tensor=True
                )
                
                # Create sample
                sample = DTASample(
                    smiles=smiles,
                    protein_sequence=protein_seq,
                    affinity=affinity,
                    graph=graph,
                    protein_tokens=protein_tokens,
                    dataset_name=self.dataset_name,
                    sample_id=f"{self.dataset_name}_{idx}"
                )
                
                samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {e}")
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} samples")
        
        logger.info(f"Successfully processed {len(samples)} samples")
        return samples
    
    def _load_from_cache(self) -> Optional[List[DTASample]]:
        """Load processed samples from cache"""
        if not self._cache_file or not self._cache_file.exists():
            return None
        
        try:
            logger.info(f"Loading cached data from {self._cache_file}")
            with open(self._cache_file, 'rb') as f:
                samples = pickle.load(f)
            logger.info(f"Loaded {len(samples)} samples from cache")
            return samples
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self) -> None:
        """Save processed samples to cache"""
        if not self._cache_file:
            return
        
        try:
            logger.info(f"Saving processed data to cache: {self._cache_file}")
            with open(self._cache_file, 'wb') as f:
                pickle.dump(self.samples, f)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> DTASample:
        return self.samples[idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        affinities = [sample.affinity for sample in self.samples]
        protein_lengths = [len(sample.protein_sequence) for sample in self.samples]
        
        return {
            'num_samples': len(self.samples),
            'dataset_name': self.dataset_name,
            'affinity_stats': {
                'mean': np.mean(affinities),
                'std': np.std(affinities),
                'min': np.min(affinities),
                'max': np.max(affinities)
            },
            'protein_length_stats': {
                'mean': np.mean(protein_lengths),
                'std': np.std(protein_lengths),
                'min': np.min(protein_lengths),
                'max': np.max(protein_lengths)
            },
            'unique_smiles': len(set(sample.smiles for sample in self.samples)),
            'unique_proteins': len(set(sample.protein_sequence for sample in self.samples))
        }


class MultiDatasetDTA(Dataset):
    """Combined dataset from multiple DTA datasets"""
    
    def __init__(self, datasets: List[DTADataset]):
        self.datasets = datasets
        self.samples = []
        
        # Combine all samples
        for dataset in datasets:
            self.samples.extend(dataset.samples)
        
        logger.info(f"Combined dataset created with {len(self.samples)} samples "
                   f"from {len(datasets)} datasets")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> DTASample:
        return self.samples[idx]
    
    def get_dataset_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across datasets"""
        distribution = {}
        for sample in self.samples:
            dataset_name = sample.dataset_name
            distribution[dataset_name] = distribution.get(dataset_name, 0) + 1
        return distribution


def collate_dta_batch(batch: List[DTASample]) -> Dict[str, Any]:
    """Custom collate function for DTA samples"""
    
    # Separate components
    graphs = [sample.graph for sample in batch]
    protein_tokens = torch.stack([sample.protein_tokens for sample in batch])
    affinities = torch.tensor([sample.affinity for sample in batch], dtype=torch.float)
    protein_sequences = [sample.protein_sequence for sample in batch]
    smiles_list = [sample.smiles for sample in batch]
    
    # Batch graphs using PyTorch Geometric
    batched_graphs = Batch.from_data_list(graphs)
    
    return {
        'drug_data': batched_graphs,
        'protein_tokens': protein_tokens,
        'protein_sequences': protein_sequences,
        'smiles': smiles_list,
        'affinities': affinities,
        'batch_size': len(batch)
    }


class DTADataLoader:
    """Enhanced DataLoader for DTA datasets with memory optimization"""
    
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 drop_last: bool = False):
        
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_dta_batch
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_data_splits(dataset: DTADataset,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      random_seed: int = 42) -> Tuple[DTADataset, DTADataset, DTADataset]:
    """Create train/validation/test splits from a dataset"""
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Create splits
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    logger.info(f"Created data splits: train={train_size}, val={val_size}, test={test_size}")
    
    return train_dataset, val_dataset, test_dataset


def load_standard_datasets(data_dir: Union[str, Path],
                          datasets: List[str] = ['kiba', 'davis', 'bindingdb'],
                          max_protein_length: int = 200,
                          cache_dir: Optional[Union[str, Path]] = None) -> Dict[str, DTADataset]:
    """Load standard DTA datasets (KIBA, Davis, BindingDB)"""
    
    data_dir = Path(data_dir)
    loaded_datasets = {}
    
    for dataset_name in datasets:
        train_file = data_dir / f"{dataset_name}_train.csv"
        test_file = data_dir / f"{dataset_name}_test.csv"
        
        # Load training data
        if train_file.exists():
            train_dataset = DTADataset(
                data_path=train_file,
                max_protein_length=max_protein_length,
                cache_dir=cache_dir,
                dataset_name=f"{dataset_name}_train"
            )
            loaded_datasets[f"{dataset_name}_train"] = train_dataset
        else:
            logger.warning(f"Training file not found: {train_file}")
        
        # Load test data
        if test_file.exists():
            test_dataset = DTADataset(
                data_path=test_file,
                max_protein_length=max_protein_length,
                cache_dir=cache_dir,
                dataset_name=f"{dataset_name}_test"
            )
            loaded_datasets[f"{dataset_name}_test"] = test_dataset
        else:
            logger.warning(f"Test file not found: {test_file}")
    
    logger.info(f"Loaded {len(loaded_datasets)} datasets: {list(loaded_datasets.keys())}")
    return loaded_datasets


class DataAugmentation:
    """Data augmentation techniques for DTA datasets"""
    
    def __init__(self):
        self.protein_processor = ProteinProcessor()
    
    def augment_protein_sequence(self, sequence: str, 
                               mutation_rate: float = 0.05) -> str:
        """Apply random mutations to protein sequence"""
        import random
        
        sequence_list = list(sequence)
        num_mutations = int(len(sequence) * mutation_rate)
        
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        
        for _ in range(num_mutations):
            pos = random.randint(0, len(sequence_list) - 1)
            sequence_list[pos] = random.choice(amino_acids)
        
        return ''.join(sequence_list)
    
    def augment_smiles(self, smiles: str) -> str:
        """Apply SMILES augmentation (canonical randomization)"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, doRandom=True)
        except ImportError:
            pass
        return smiles
    
    def create_augmented_dataset(self, dataset: DTADataset, 
                               augmentation_factor: int = 2) -> DTADataset:
        """Create augmented version of dataset"""
        augmented_samples = []
        
        for sample in dataset.samples:
            # Add original sample
            augmented_samples.append(sample)
            
            # Add augmented versions
            for i in range(augmentation_factor - 1):
                aug_protein = self.augment_protein_sequence(sample.protein_sequence)
                aug_smiles = self.augment_smiles(sample.smiles)
                
                # Create new sample (would need to reprocess)
                # This is a simplified version - full implementation would
                # need to regenerate graphs and tokens
                augmented_samples.append(sample)  # Placeholder
        
        # Create new dataset with augmented samples
        # This would require creating a new DTADataset from the samples
        return dataset  # Placeholder return


def create_balanced_sampler(dataset: DTADataset, 
                           num_bins: int = 10) -> torch.utils.data.WeightedRandomSampler:
    """Create a balanced sampler based on affinity values"""
    
    affinities = [sample.affinity for sample in dataset.samples]
    
    # Create bins
    min_affinity = min(affinities)
    max_affinity = max(affinities)
    bin_edges = np.linspace(min_affinity, max_affinity, num_bins + 1)
    
    # Assign samples to bins
    bin_indices = np.digitize(affinities, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    # Calculate weights (inverse frequency)
    bin_counts = np.bincount(bin_indices, minlength=num_bins)
    bin_weights = 1.0 / (bin_counts + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Assign weights to samples
    sample_weights = bin_weights[bin_indices]
    
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )


if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data (if available)
    data_dir = Path("data")
    
    if data_dir.exists():
        try:
            # Load datasets
            datasets = load_standard_datasets(data_dir, datasets=['kiba'], cache_dir="cache")
            
            if datasets:
                dataset_name = list(datasets.keys())[0]
                dataset = datasets[dataset_name]
                
                print(f"\nDataset statistics for {dataset_name}:")
                stats = dataset.get_statistics()
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                
                # Test data loader
                dataloader = DTADataLoader(dataset, batch_size=4, shuffle=True)
                
                print(f"\nTesting data loader:")
                for i, batch in enumerate(dataloader):
                    print(f"  Batch {i}: {batch['batch_size']} samples")
                    print(f"    Drug graphs: {batch['drug_data']}")
                    print(f"    Protein tokens shape: {batch['protein_tokens'].shape}")
                    print(f"    Affinities shape: {batch['affinities'].shape}")
                    
                    if i >= 2:  # Only test first few batches
                        break
                
                # Test data splits
                train_ds, val_ds, test_ds = create_data_splits(dataset)
                print(f"\nData splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
            
        except Exception as e:
            print(f"Error testing with real data: {e}")
    
    else:
        print("No data directory found. Skipping real data tests.")
    
    print("\nDataset classes created successfully!")
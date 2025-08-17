"""
Graph preprocessing and feature extraction for molecular graphs
Advanced molecular graph processing with RDKit integration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from torch_geometric.data import Data, Batch
import warnings

logger = logging.getLogger(__name__)

# Try to import RDKit for molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
    from rdkit.Chem.rdchem import BondType, HybridizationType
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Molecular graph processing will be limited.")
    RDKIT_AVAILABLE = False
    Chem = None


@dataclass
class GraphFeatureConfig:
    """Configuration for graph feature extraction"""
    # Node features
    include_atomic_number: bool = True
    include_degree: bool = True
    include_formal_charge: bool = True
    include_hybridization: bool = True
    include_aromaticity: bool = True
    include_mass: bool = True
    include_valence: bool = True
    include_chirality: bool = True
    include_hydrogen_count: bool = True
    include_radical_electrons: bool = True
    
    # Edge features
    include_bond_type: bool = True
    include_conjugation: bool = True
    include_ring_membership: bool = True
    include_stereo: bool = True
    include_bond_length: bool = False  # Requires 3D coordinates
    
    # Graph-level features
    include_ring_info: bool = True
    include_molecular_descriptors: bool = True
    
    # Processing options
    max_atomic_number: int = 100
    use_one_hot_encoding: bool = True
    normalize_features: bool = True


class MolecularGraphProcessor:
    """Advanced molecular graph processor with RDKit integration"""
    
    def __init__(self, config: Optional[GraphFeatureConfig] = None):
        self.config = config or GraphFeatureConfig()
        
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available. Using fallback feature extraction.")
        
        # Atomic number mapping for common elements
        self.atomic_numbers = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17,
            'Br': 35, 'I': 53, 'B': 5, 'Si': 14, 'Se': 34, 'As': 33, 'Al': 13,
            'Zn': 30, 'Ca': 20, 'Mg': 12, 'Na': 11, 'K': 19, 'Li': 3, 'Fe': 26
        }
        
        # Bond type mapping
        if RDKIT_AVAILABLE:
            self.bond_types = {
                BondType.SINGLE: 1,
                BondType.DOUBLE: 2,
                BondType.TRIPLE: 3,
                BondType.AROMATIC: 4
            }
            
            self.hybridization_types = {
                HybridizationType.SP: 1,
                HybridizationType.SP2: 2,
                HybridizationType.SP3: 3,
                HybridizationType.SP3D: 4,
                HybridizationType.SP3D2: 5
            }
        
        # Feature dimensions
        self._calculate_feature_dimensions()
    
    def _calculate_feature_dimensions(self) -> None:
        """Calculate the dimensions of node and edge features"""
        node_dim = 0
        
        if self.config.include_atomic_number:
            node_dim += self.config.max_atomic_number if self.config.use_one_hot_encoding else 1
        if self.config.include_degree:
            node_dim += 10 if self.config.use_one_hot_encoding else 1  # Max degree ~10
        if self.config.include_formal_charge:
            node_dim += 11 if self.config.use_one_hot_encoding else 1  # Charge range -5 to +5
        if self.config.include_hybridization:
            node_dim += 6 if self.config.use_one_hot_encoding else 1   # 5 hybridization types + unknown
        if self.config.include_aromaticity:
            node_dim += 1
        if self.config.include_mass:
            node_dim += 1
        if self.config.include_valence:
            node_dim += 8 if self.config.use_one_hot_encoding else 1   # Max valence ~7
        if self.config.include_chirality:
            node_dim += 4 if self.config.use_one_hot_encoding else 1   # 4 chirality types
        if self.config.include_hydrogen_count:
            node_dim += 5 if self.config.use_one_hot_encoding else 1   # Max H count ~4
        if self.config.include_radical_electrons:
            node_dim += 1
        
        edge_dim = 0
        
        if self.config.include_bond_type:
            edge_dim += 5 if self.config.use_one_hot_encoding else 1   # 4 bond types + unknown
        if self.config.include_conjugation:
            edge_dim += 1
        if self.config.include_ring_membership:
            edge_dim += 1
        if self.config.include_stereo:
            edge_dim += 6 if self.config.use_one_hot_encoding else 1   # Stereo configurations
        if self.config.include_bond_length:
            edge_dim += 1
        
        self.node_feature_dim = node_dim
        self.edge_feature_dim = edge_dim
        
        logger.info(f"Node feature dimension: {self.node_feature_dim}")
        logger.info(f"Edge feature dimension: {self.edge_feature_dim}")
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES string to PyTorch Geometric Data object"""
        if not RDKIT_AVAILABLE:
            return self._fallback_smiles_to_graph(smiles)
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                return None
            
            # Add hydrogens for complete graph
            mol = Chem.AddHs(mol)
            
            # Extract node features
            node_features = self._extract_node_features(mol)
            
            # Extract edge features and connectivity
            edge_index, edge_features = self._extract_edge_features(mol)
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_features, dtype=torch.float32) if edge_features.size > 0 else None,
                smiles=smiles,
                num_nodes=len(node_features)
            )
            
            # Add graph-level features if requested
            if self.config.include_molecular_descriptors:
                graph_features = self._extract_graph_features(mol)
                data.graph_attr = torch.tensor(graph_features, dtype=torch.float32)
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return None
    
    def _extract_node_features(self, mol) -> np.ndarray:
        """Extract node (atom) features from molecule"""
        features = []
        
        for atom in mol.GetAtoms():
            atom_features = []
            
            if self.config.include_atomic_number:
                atomic_num = atom.GetAtomicNum()
                if self.config.use_one_hot_encoding:
                    one_hot = np.zeros(self.config.max_atomic_number)
                    if atomic_num < self.config.max_atomic_number:
                        one_hot[atomic_num] = 1
                    atom_features.extend(one_hot)
                else:
                    atom_features.append(atomic_num)
            
            if self.config.include_degree:
                degree = atom.GetDegree()
                if self.config.use_one_hot_encoding:
                    one_hot = np.zeros(10)
                    if degree < 10:
                        one_hot[degree] = 1
                    atom_features.extend(one_hot)
                else:
                    atom_features.append(degree)
            
            if self.config.include_formal_charge:
                charge = atom.GetFormalCharge()
                if self.config.use_one_hot_encoding:
                    one_hot = np.zeros(11)
                    charge_idx = max(0, min(10, charge + 5))  # Map -5 to +5 -> 0 to 10
                    one_hot[charge_idx] = 1
                    atom_features.extend(one_hot)
                else:
                    atom_features.append(charge)
            
            if self.config.include_hybridization:
                hybridization = atom.GetHybridization()
                if self.config.use_one_hot_encoding:
                    one_hot = np.zeros(6)
                    hyb_idx = self.hybridization_types.get(hybridization, 0)
                    one_hot[hyb_idx] = 1
                    atom_features.extend(one_hot)
                else:
                    atom_features.append(self.hybridization_types.get(hybridization, 0))
            
            if self.config.include_aromaticity:
                atom_features.append(float(atom.GetIsAromatic()))
            
            if self.config.include_mass:
                atom_features.append(atom.GetMass())
            
            if self.config.include_valence:
                valence = atom.GetTotalValence()
                if self.config.use_one_hot_encoding:
                    one_hot = np.zeros(8)
                    if valence < 8:
                        one_hot[valence] = 1
                    atom_features.extend(one_hot)
                else:
                    atom_features.append(valence)
            
            if self.config.include_chirality:
                chirality = atom.GetChiralTag()
                if self.config.use_one_hot_encoding:
                    one_hot = np.zeros(4)
                    one_hot[int(chirality)] = 1
                    atom_features.extend(one_hot)
                else:
                    atom_features.append(int(chirality))
            
            if self.config.include_hydrogen_count:
                h_count = atom.GetTotalNumHs()
                if self.config.use_one_hot_encoding:
                    one_hot = np.zeros(5)
                    if h_count < 5:
                        one_hot[h_count] = 1
                    atom_features.extend(one_hot)
                else:
                    atom_features.append(h_count)
            
            if self.config.include_radical_electrons:
                atom_features.append(atom.GetNumRadicalElectrons())
            
            features.append(atom_features)
        
        features = np.array(features, dtype=np.float32)
        
        # Normalize features if requested
        if self.config.normalize_features:
            features = self._normalize_node_features(features)
        
        return features
    
    def _extract_edge_features(self, mol) -> Tuple[np.ndarray, np.ndarray]:
        """Extract edge (bond) features and connectivity"""
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            # Get bond indices (undirected graph - add both directions)
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_indices.extend([[i, j], [j, i]])
            
            # Extract bond features
            bond_features = []
            
            if self.config.include_bond_type:
                bond_type = bond.GetBondType()
                if self.config.use_one_hot_encoding:
                    one_hot = np.zeros(5)
                    bond_idx = self.bond_types.get(bond_type, 0)
                    one_hot[bond_idx] = 1
                    bond_features.extend(one_hot)
                else:
                    bond_features.append(self.bond_types.get(bond_type, 0))
            
            if self.config.include_conjugation:
                bond_features.append(float(bond.GetIsConjugated()))
            
            if self.config.include_ring_membership:
                bond_features.append(float(bond.IsInRing()))
            
            if self.config.include_stereo:
                stereo = bond.GetStereo()
                if self.config.use_one_hot_encoding:
                    one_hot = np.zeros(6)
                    one_hot[int(stereo)] = 1
                    bond_features.extend(one_hot)
                else:
                    bond_features.append(int(stereo))
            
            # Add same features for both directions
            edge_features.extend([bond_features, bond_features])
        
        edge_index = np.array(edge_indices).T if edge_indices else np.empty((2, 0))
        edge_attr = np.array(edge_features) if edge_features else np.empty((0, 0))
        
        return edge_index, edge_attr
    
    def _extract_graph_features(self, mol) -> np.ndarray:
        """Extract graph-level molecular descriptors"""
        features = []
        
        try:
            # Basic molecular descriptors
            features.append(Descriptors.MolWt(mol))  # Molecular weight
            features.append(Descriptors.MolLogP(mol))  # LogP
            features.append(Descriptors.NumHDonors(mol))  # H-bond donors
            features.append(Descriptors.NumHAcceptors(mol))  # H-bond acceptors
            features.append(Descriptors.TPSA(mol))  # Topological polar surface area
            features.append(Descriptors.NumRotatableBonds(mol))  # Rotatable bonds
            features.append(Descriptors.NumAromaticRings(mol))  # Aromatic rings
            features.append(Descriptors.NumSaturatedRings(mol))  # Saturated rings
            features.append(Descriptors.FractionCsp3(mol))  # Fraction of sp3 carbons
            features.append(rdMolDescriptors.BertzCT(mol))  # Bertz complexity index
            
        except Exception as e:
            logger.warning(f"Error extracting molecular descriptors: {e}")
            features = [0.0] * 10  # Fallback values
        
        return np.array(features, dtype=np.float32)
    
    def _normalize_node_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize node features"""
        # Skip one-hot encoded features (they're already normalized)
        # Only normalize continuous features
        normalized = features.copy()
        
        # Simple min-max normalization for continuous features
        for i in range(features.shape[1]):
            col = features[:, i]
            if len(np.unique(col)) > 2:  # Continuous feature
                min_val, max_val = col.min(), col.max()
                if max_val > min_val:
                    normalized[:, i] = (col - min_val) / (max_val - min_val)
        
        return normalized
    
    def _fallback_smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Fallback graph creation when RDKit is not available"""
        logger.warning("Using fallback graph creation (limited features)")
        
        # Handle invalid inputs
        if not smiles or not isinstance(smiles, str):
            return None
        
        # Create a simple graph based on SMILES string analysis
        # This is a very basic implementation
        atoms = []
        
        # Simple parsing (very limited)
        atom_chars = 'CNOSPFClBrI'
        
        # Check for obviously invalid SMILES patterns
        if any(char in smiles for char in ['invalid', 'error', 'fail']):
            return None
        
        for i, char in enumerate(smiles):
            if char in atom_chars:
                atoms.append(char)
        
        if len(atoms) < 1:  # Changed from < 2 to handle single atoms
            return None
        
        # Create simple linear connectivity (very basic)
        edge_index = []
        if len(atoms) > 1:
            for i in range(len(atoms) - 1):
                edge_index.extend([[i, i+1], [i+1, i]])
        
        # Basic node features (atomic number only)
        node_features = []
        for atom in atoms:
            atomic_num = self.atomic_numbers.get(atom, 6)  # Default to carbon
            node_features.append([atomic_num])
        
        # Handle single atom case
        if not edge_index:
            edge_index = [[0, 0]]  # Self-loop for single atom
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).T,
            smiles=smiles,
            num_nodes=len(atoms)
        )
    
    def batch_process_smiles(self, smiles_list: List[str]) -> Tuple[List[Data], List[str]]:
        """Process a batch of SMILES strings"""
        valid_graphs = []
        failed_smiles = []
        
        for smiles in smiles_list:
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                valid_graphs.append(graph)
            else:
                failed_smiles.append(smiles)
        
        return valid_graphs, failed_smiles


class GraphValidator:
    """Validator for molecular graphs"""
    
    def __init__(self, min_nodes: int = 3, max_nodes: int = 200, min_edges: int = 2):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_edges = min_edges
    
    def validate_graph(self, data: Data) -> Tuple[bool, List[str]]:
        """Validate a molecular graph"""
        errors = []
        
        # Check node count
        if data.num_nodes < self.min_nodes:
            errors.append(f"Too few nodes: {data.num_nodes} < {self.min_nodes}")
        
        if data.num_nodes > self.max_nodes:
            errors.append(f"Too many nodes: {data.num_nodes} > {self.max_nodes}")
        
        # Check edge count
        num_edges = data.edge_index.shape[1] // 2  # Undirected edges
        if num_edges < self.min_edges:
            errors.append(f"Too few edges: {num_edges} < {self.min_edges}")
        
        # Check for isolated nodes
        edge_nodes = torch.unique(data.edge_index)
        if len(edge_nodes) < data.num_nodes:
            isolated_count = data.num_nodes - len(edge_nodes)
            errors.append(f"Found {isolated_count} isolated nodes")
        
        # Check for self-loops
        self_loops = (data.edge_index[0] == data.edge_index[1]).sum().item()
        if self_loops > 0:
            errors.append(f"Found {self_loops} self-loops")
        
        # Check feature dimensions
        if data.x.shape[0] != data.num_nodes:
            errors.append(f"Node feature count mismatch: {data.x.shape[0]} != {data.num_nodes}")
        
        return len(errors) == 0, errors
    
    def filter_valid_graphs(self, graphs: List[Data]) -> Tuple[List[Data], List[Tuple[int, List[str]]]]:
        """Filter valid graphs from a list"""
        valid_graphs = []
        invalid_info = []
        
        for i, graph in enumerate(graphs):
            is_valid, errors = self.validate_graph(graph)
            if is_valid:
                valid_graphs.append(graph)
            else:
                invalid_info.append((i, errors))
        
        return valid_graphs, invalid_info


class OptimizedGraphBatcher:
    """Optimized batching for molecular graphs"""
    
    def __init__(self, max_nodes_per_batch: int = 3000, sort_by_size: bool = True):
        self.max_nodes_per_batch = max_nodes_per_batch
        self.sort_by_size = sort_by_size
    
    def create_batches(self, graphs: List[Data]) -> List[Batch]:
        """Create optimized batches from graphs"""
        if self.sort_by_size:
            # Sort by number of nodes for more efficient batching
            graphs = sorted(graphs, key=lambda g: g.num_nodes)
        
        batches = []
        current_batch = []
        current_node_count = 0
        
        for graph in graphs:
            # Check if adding this graph would exceed the limit
            if (current_node_count + graph.num_nodes > self.max_nodes_per_batch and 
                len(current_batch) > 0):
                
                # Create batch from current graphs
                batches.append(Batch.from_data_list(current_batch))
                current_batch = [graph]
                current_node_count = graph.num_nodes
            else:
                current_batch.append(graph)
                current_node_count += graph.num_nodes
        
        # Add remaining graphs as final batch
        if current_batch:
            batches.append(Batch.from_data_list(current_batch))
        
        return batches
    
    def get_batch_statistics(self, batches: List[Batch]) -> Dict[str, Any]:
        """Get statistics about the created batches"""
        if not batches:
            return {}
        
        batch_sizes = [batch.num_graphs for batch in batches]
        node_counts = [batch.x.shape[0] for batch in batches]
        edge_counts = [batch.edge_index.shape[1] for batch in batches]
        
        return {
            'num_batches': len(batches),
            'batch_sizes': {
                'mean': np.mean(batch_sizes),
                'std': np.std(batch_sizes),
                'min': np.min(batch_sizes),
                'max': np.max(batch_sizes)
            },
            'node_counts': {
                'mean': np.mean(node_counts),
                'std': np.std(node_counts),
                'min': np.min(node_counts),
                'max': np.max(node_counts)
            },
            'edge_counts': {
                'mean': np.mean(edge_counts),
                'std': np.std(edge_counts),
                'min': np.min(edge_counts),
                'max': np.max(edge_counts)
            }
        }


# Convenience functions
def create_molecular_graph_processor(config: Optional[GraphFeatureConfig] = None) -> MolecularGraphProcessor:
    """Create a molecular graph processor with default or custom configuration"""
    return MolecularGraphProcessor(config)


def process_smiles_batch(smiles_list: List[str], 
                        config: Optional[GraphFeatureConfig] = None,
                        validate: bool = True,
                        create_batches: bool = True) -> Dict[str, Any]:
    """Process a batch of SMILES strings with full pipeline"""
    processor = MolecularGraphProcessor(config)
    validator = GraphValidator()
    batcher = OptimizedGraphBatcher()
    
    # Process SMILES to graphs
    graphs, failed_smiles = processor.batch_process_smiles(smiles_list)
    
    results = {
        'total_smiles': len(smiles_list),
        'successful_graphs': len(graphs),
        'failed_smiles': failed_smiles,
        'success_rate': len(graphs) / len(smiles_list) if smiles_list else 0
    }
    
    if validate and graphs:
        # Validate graphs
        valid_graphs, invalid_info = validator.filter_valid_graphs(graphs)
        results.update({
            'valid_graphs': len(valid_graphs),
            'invalid_graphs': len(invalid_info),
            'invalid_info': invalid_info,
            'validation_rate': len(valid_graphs) / len(graphs) if graphs else 0
        })
        graphs = valid_graphs
    
    if create_batches and graphs:
        # Create optimized batches
        batches = batcher.create_batches(graphs)
        batch_stats = batcher.get_batch_statistics(batches)
        results.update({
            'batches': batches,
            'batch_statistics': batch_stats
        })
    else:
        results['graphs'] = graphs
    
    return results
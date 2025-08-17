"""
Generation Scoring and Confidence Module
Provides confidence scoring and quality assessment for generated molecules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from scipy.stats import entropy
import math

from .drug_generation import ChemicalValidator


class ConfidenceScorer(nn.Module):
    """Neural network-based confidence scorer for generated molecules"""
    
    def __init__(self,
                 input_dim: int = 512,
                 hidden_dims: List[int] = [256, 128],
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output confidence score (0-1)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence scores
        
        Args:
            features: Input features [batch_size, input_dim]
        
        Returns:
            Confidence scores [batch_size, 1]
        """
        return self.network(features)


class MolecularPropertyCalculator:
    """Calculate various molecular properties for quality assessment"""
    
    @staticmethod
    def calculate_lipinski_properties(smiles: str) -> Dict[str, float]:
        """Calculate Lipinski's Rule of Five properties"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'num_hbd': Descriptors.NumHDonors(mol),
                'num_hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol)
            }
            
            # Lipinski violations
            violations = 0
            if properties['molecular_weight'] > 500:
                violations += 1
            if properties['logp'] > 5:
                violations += 1
            if properties['num_hbd'] > 5:
                violations += 1
            if properties['num_hba'] > 10:
                violations += 1
            
            properties['lipinski_violations'] = violations
            properties['lipinski_compliant'] = violations <= 1
            
            return properties
        except:
            return {}
    
    @staticmethod
    def calculate_drug_likeness_score(smiles: str) -> float:
        """Calculate drug-likeness score based on multiple criteria"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            # Get basic properties
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable = Descriptors.NumRotatableBonds(mol)
            
            # Scoring based on drug-like ranges
            score = 1.0
            
            # Molecular weight (optimal: 150-500)
            if mw < 150 or mw > 500:
                score *= 0.5
            elif 200 <= mw <= 400:
                score *= 1.2
            
            # LogP (optimal: 0-3)
            if logp < -2 or logp > 5:
                score *= 0.3
            elif 1 <= logp <= 3:
                score *= 1.2
            
            # Hydrogen bond donors (optimal: 0-3)
            if hbd > 5:
                score *= 0.5
            elif hbd <= 3:
                score *= 1.1
            
            # Hydrogen bond acceptors (optimal: 2-8)
            if hba > 10:
                score *= 0.5
            elif 2 <= hba <= 8:
                score *= 1.1
            
            # TPSA (optimal: 20-130)
            if tpsa > 140:
                score *= 0.7
            elif 20 <= tpsa <= 130:
                score *= 1.1
            
            # Rotatable bonds (optimal: 0-7)
            if rotatable > 10:
                score *= 0.6
            elif rotatable <= 7:
                score *= 1.1
            
            return min(score, 1.0)
        except:
            return 0.0
    
    @staticmethod
    def calculate_synthetic_accessibility(smiles: str) -> float:
        """Estimate synthetic accessibility (simplified version)"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            # Simple heuristics for synthetic accessibility
            num_atoms = mol.GetNumAtoms()
            num_rings = Descriptors.RingCount(mol)
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)
            num_heteroatoms = Descriptors.NumHeteroatoms(mol)
            
            # Base score
            score = 0.8
            
            # Penalize complexity
            if num_atoms > 50:
                score *= 0.7
            if num_rings > 4:
                score *= 0.8
            if num_aromatic_rings > 3:
                score *= 0.9
            if num_heteroatoms > 8:
                score *= 0.8
            
            # Bonus for reasonable size
            if 10 <= num_atoms <= 30:
                score *= 1.1
            
            return max(0.0, min(score, 1.0))
        except:
            return 0.0


class GenerationQualityAssessor:
    """Assess the quality of generated molecules"""
    
    def __init__(self):
        self.property_calculator = MolecularPropertyCalculator()
        self.validator = ChemicalValidator()
    
    def assess_molecule(self, smiles: str) -> Dict[str, float]:
        """
        Comprehensive quality assessment of a single molecule
        
        Args:
            smiles: SMILES string
        
        Returns:
            Dictionary of quality metrics
        """
        assessment = {
            'is_valid': 0.0,
            'drug_likeness': 0.0,
            'synthetic_accessibility': 0.0,
            'lipinski_compliant': 0.0,
            'overall_score': 0.0
        }
        
        # Check validity
        if not self.validator.is_valid_smiles(smiles):
            return assessment
        
        assessment['is_valid'] = 1.0
        
        # Calculate drug-likeness
        assessment['drug_likeness'] = self.property_calculator.calculate_drug_likeness_score(smiles)
        
        # Calculate synthetic accessibility
        assessment['synthetic_accessibility'] = self.property_calculator.calculate_synthetic_accessibility(smiles)
        
        # Check Lipinski compliance
        lipinski_props = self.property_calculator.calculate_lipinski_properties(smiles)
        assessment['lipinski_compliant'] = float(lipinski_props.get('lipinski_compliant', False))
        
        # Calculate overall score (weighted average)
        weights = {
            'is_valid': 0.3,
            'drug_likeness': 0.3,
            'synthetic_accessibility': 0.2,
            'lipinski_compliant': 0.2
        }
        
        overall_score = sum(assessment[key] * weights[key] for key in weights.keys())
        assessment['overall_score'] = overall_score
        
        return assessment
    
    def assess_batch(self, smiles_list: List[str]) -> List[Dict[str, float]]:
        """Assess a batch of molecules"""
        return [self.assess_molecule(smiles) for smiles in smiles_list]
    
    def get_batch_statistics(self, smiles_list: List[str]) -> Dict[str, float]:
        """Get statistics for a batch of molecules"""
        assessments = self.assess_batch(smiles_list)
        
        if not assessments:
            return {}
        
        # Calculate statistics
        valid_count = sum(1 for a in assessments if a['is_valid'] > 0)
        drug_like_count = sum(1 for a in assessments if a['drug_likeness'] > 0.7)
        lipinski_count = sum(1 for a in assessments if a['lipinski_compliant'] > 0)
        
        avg_drug_likeness = np.mean([a['drug_likeness'] for a in assessments if a['is_valid'] > 0])
        avg_synthetic_accessibility = np.mean([a['synthetic_accessibility'] for a in assessments if a['is_valid'] > 0])
        avg_overall_score = np.mean([a['overall_score'] for a in assessments])
        
        return {
            'total_molecules': len(smiles_list),
            'valid_molecules': valid_count,
            'validity_rate': valid_count / len(smiles_list),
            'drug_like_molecules': drug_like_count,
            'drug_likeness_rate': drug_like_count / len(smiles_list),
            'lipinski_compliant_molecules': lipinski_count,
            'lipinski_compliance_rate': lipinski_count / len(smiles_list),
            'avg_drug_likeness': avg_drug_likeness if not np.isnan(avg_drug_likeness) else 0.0,
            'avg_synthetic_accessibility': avg_synthetic_accessibility if not np.isnan(avg_synthetic_accessibility) else 0.0,
            'avg_overall_score': avg_overall_score if not np.isnan(avg_overall_score) else 0.0
        }


class DiversityCalculator:
    """Calculate diversity metrics for generated molecules"""
    
    @staticmethod
    def calculate_tanimoto_diversity(smiles_list: List[str]) -> float:
        """Calculate average Tanimoto diversity"""
        try:
            from rdkit.Chem import DataStructs
            from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
            
            # Generate fingerprints
            fps = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(fp)
            
            if len(fps) < 2:
                return 0.0
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    similarities.append(sim)
            
            # Diversity is 1 - average similarity
            avg_similarity = np.mean(similarities)
            return 1.0 - avg_similarity
        except ImportError:
            # Fallback if RDKit fingerprints not available
            return 0.0
        except:
            return 0.0
    
    @staticmethod
    def calculate_scaffold_diversity(smiles_list: List[str]) -> float:
        """Calculate scaffold diversity"""
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            scaffolds = set()
            valid_count = 0
            
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
                    try:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        scaffold_smiles = Chem.MolToSmiles(scaffold)
                        scaffolds.add(scaffold_smiles)
                    except:
                        continue
            
            if valid_count == 0:
                return 0.0
            
            return len(scaffolds) / valid_count
        except ImportError:
            return 0.0
        except:
            return 0.0


class NoveltyCalculator:
    """Calculate novelty metrics for generated molecules"""
    
    def __init__(self, reference_smiles: Optional[List[str]] = None):
        self.reference_smiles = set(reference_smiles) if reference_smiles else set()
    
    def calculate_novelty_rate(self, generated_smiles: List[str]) -> float:
        """Calculate the rate of novel molecules"""
        if not self.reference_smiles:
            return 1.0  # All molecules are novel if no reference set
        
        novel_count = 0
        valid_count = 0
        
        for smiles in generated_smiles:
            if ChemicalValidator.is_valid_smiles(smiles):
                valid_count += 1
                canonical = ChemicalValidator.canonicalize_smiles(smiles)
                if canonical and canonical not in self.reference_smiles:
                    novel_count += 1
        
        if valid_count == 0:
            return 0.0
        
        return novel_count / valid_count
    
    def add_reference_molecules(self, smiles_list: List[str]):
        """Add molecules to the reference set"""
        for smiles in smiles_list:
            canonical = ChemicalValidator.canonicalize_smiles(smiles)
            if canonical:
                self.reference_smiles.add(canonical)


class GenerationMetrics:
    """Comprehensive metrics for evaluating molecular generation"""
    
    def __init__(self, reference_smiles: Optional[List[str]] = None):
        self.quality_assessor = GenerationQualityAssessor()
        self.diversity_calculator = DiversityCalculator()
        self.novelty_calculator = NoveltyCalculator(reference_smiles)
    
    def evaluate_generation(self, generated_smiles: List[str]) -> Dict[str, float]:
        """
        Comprehensive evaluation of generated molecules
        
        Args:
            generated_smiles: List of generated SMILES strings
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Quality metrics
        quality_stats = self.quality_assessor.get_batch_statistics(generated_smiles)
        metrics.update(quality_stats)
        
        # Diversity metrics
        metrics['tanimoto_diversity'] = self.diversity_calculator.calculate_tanimoto_diversity(generated_smiles)
        metrics['scaffold_diversity'] = self.diversity_calculator.calculate_scaffold_diversity(generated_smiles)
        
        # Novelty metrics
        metrics['novelty_rate'] = self.novelty_calculator.calculate_novelty_rate(generated_smiles)
        
        return metrics
    
    def compare_generations(self, 
                          generation_sets: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple generation sets
        
        Args:
            generation_sets: Dictionary mapping set names to SMILES lists
        
        Returns:
            Dictionary mapping set names to their metrics
        """
        results = {}
        
        for set_name, smiles_list in generation_sets.items():
            results[set_name] = self.evaluate_generation(smiles_list)
        
        return results


class ConfidenceScoringPipeline:
    """Complete pipeline for confidence scoring of generated molecules"""
    
    def __init__(self, 
                 confidence_model: Optional[ConfidenceScorer] = None,
                 reference_smiles: Optional[List[str]] = None):
        self.confidence_model = confidence_model
        self.metrics = GenerationMetrics(reference_smiles)
    
    def score_molecules(self, 
                       smiles_list: List[str],
                       protein_features: Optional[torch.Tensor] = None) -> List[Dict[str, float]]:
        """
        Score molecules with confidence and quality metrics
        
        Args:
            smiles_list: List of SMILES strings
            protein_features: Optional protein features for confidence scoring
        
        Returns:
            List of scoring results for each molecule
        """
        results = []
        
        for i, smiles in enumerate(smiles_list):
            # Quality assessment
            quality_scores = self.metrics.quality_assessor.assess_molecule(smiles)
            
            # Neural confidence score (if model available)
            confidence_score = 0.5  # Default
            if self.confidence_model is not None and protein_features is not None:
                with torch.no_grad():
                    if i < len(protein_features):
                        conf_tensor = self.confidence_model(protein_features[i:i+1])
                        confidence_score = conf_tensor.item()
            
            # Combine scores
            result = {
                'smiles': smiles,
                'confidence_score': confidence_score,
                **quality_scores
            }
            
            results.append(result)
        
        return results
    
    def rank_molecules(self, 
                      scoring_results: List[Dict[str, float]],
                      ranking_key: str = 'overall_score') -> List[Dict[str, float]]:
        """
        Rank molecules by a specific scoring metric
        
        Args:
            scoring_results: Results from score_molecules
            ranking_key: Key to rank by
        
        Returns:
            Sorted list of scoring results
        """
        return sorted(scoring_results, key=lambda x: x.get(ranking_key, 0), reverse=True)
"""
Generation Evaluation Module
Implements comprehensive evaluation and validation for drug generation
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import warnings

from .drug_generation import DrugGenerationPipeline, ChemicalValidator
from .generation_scoring import (
    GenerationMetrics, 
    ConfidenceScoringPipeline,
    MolecularPropertyCalculator,
    DiversityCalculator,
    NoveltyCalculator
)


class MolecularPropertyPredictor(nn.Module):
    """Neural network for predicting molecular properties"""
    
    def __init__(self,
                 input_dim: int = 512,
                 hidden_dims: List[int] = [256, 128],
                 output_properties: List[str] = ['logp', 'molecular_weight', 'tpsa'],
                 dropout: float = 0.2):
        super().__init__()
        
        self.output_properties = output_properties
        self.num_properties = len(output_properties)
        
        # Shared layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Property-specific heads
        self.property_heads = nn.ModuleDict({
            prop: nn.Linear(prev_dim, 1) for prop in output_properties
        })
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict molecular properties
        
        Args:
            features: Input features [batch_size, input_dim]
        
        Returns:
            Dictionary mapping property names to predictions
        """
        shared_features = self.shared_layers(features)
        
        predictions = {}
        for prop_name, head in self.property_heads.items():
            predictions[prop_name] = head(shared_features).squeeze(-1)
        
        return predictions


class GenerationBenchmark:
    """Benchmark suite for evaluating drug generation models"""
    
    def __init__(self, 
                 reference_datasets: Optional[Dict[str, List[str]]] = None,
                 property_predictor: Optional[MolecularPropertyPredictor] = None):
        self.reference_datasets = reference_datasets or {}
        self.property_predictor = property_predictor
        self.validator = ChemicalValidator()
        self.property_calculator = MolecularPropertyCalculator()
        
        # Initialize calculators
        self.diversity_calc = DiversityCalculator()
        
        # Initialize novelty calculators for each reference dataset
        self.novelty_calcs = {}
        for dataset_name, smiles_list in self.reference_datasets.items():
            self.novelty_calcs[dataset_name] = NoveltyCalculator(smiles_list)
    
    def evaluate_validity(self, generated_smiles: List[str]) -> Dict[str, float]:
        """Evaluate chemical validity of generated molecules"""
        total_count = len(generated_smiles)
        valid_count = 0
        canonical_smiles = []
        
        for smiles in generated_smiles:
            if self.validator.is_valid_smiles(smiles):
                valid_count += 1
                canonical = self.validator.canonicalize_smiles(smiles)
                if canonical:
                    canonical_smiles.append(canonical)
        
        # Remove duplicates
        unique_smiles = list(set(canonical_smiles))
        
        return {
            'total_generated': total_count,
            'valid_molecules': valid_count,
            'validity_rate': valid_count / total_count if total_count > 0 else 0.0,
            'unique_valid_molecules': len(unique_smiles),
            'uniqueness_rate': len(unique_smiles) / valid_count if valid_count > 0 else 0.0,
            'canonical_smiles': unique_smiles
        }
    
    def evaluate_diversity(self, generated_smiles: List[str]) -> Dict[str, float]:
        """Evaluate diversity of generated molecules"""
        # Filter valid molecules
        valid_smiles = [smiles for smiles in generated_smiles 
                       if self.validator.is_valid_smiles(smiles)]
        
        if len(valid_smiles) < 2:
            return {
                'tanimoto_diversity': 0.0,
                'scaffold_diversity': 0.0,
                'num_valid_for_diversity': len(valid_smiles)
            }
        
        tanimoto_div = self.diversity_calc.calculate_tanimoto_diversity(valid_smiles)
        scaffold_div = self.diversity_calc.calculate_scaffold_diversity(valid_smiles)
        
        return {
            'tanimoto_diversity': tanimoto_div,
            'scaffold_diversity': scaffold_div,
            'num_valid_for_diversity': len(valid_smiles)
        }
    
    def evaluate_novelty(self, generated_smiles: List[str]) -> Dict[str, Dict[str, float]]:
        """Evaluate novelty against reference datasets"""
        novelty_results = {}
        
        for dataset_name, novelty_calc in self.novelty_calcs.items():
            novelty_rate = novelty_calc.calculate_novelty_rate(generated_smiles)
            novelty_results[dataset_name] = {
                'novelty_rate': novelty_rate,
                'reference_size': len(novelty_calc.reference_smiles)
            }
        
        return novelty_results
    
    def evaluate_drug_likeness(self, generated_smiles: List[str]) -> Dict[str, float]:
        """Evaluate drug-likeness of generated molecules"""
        valid_smiles = [smiles for smiles in generated_smiles 
                       if self.validator.is_valid_smiles(smiles)]
        
        if not valid_smiles:
            return {
                'avg_drug_likeness': 0.0,
                'drug_like_molecules': 0,
                'drug_likeness_rate': 0.0,
                'lipinski_compliant': 0,
                'lipinski_compliance_rate': 0.0
            }
        
        drug_likeness_scores = []
        lipinski_compliant_count = 0
        
        for smiles in valid_smiles:
            # Drug-likeness score
            drug_score = self.property_calculator.calculate_drug_likeness_score(smiles)
            drug_likeness_scores.append(drug_score)
            
            # Lipinski compliance
            lipinski_props = self.property_calculator.calculate_lipinski_properties(smiles)
            if lipinski_props.get('lipinski_compliant', False):
                lipinski_compliant_count += 1
        
        avg_drug_likeness = np.mean(drug_likeness_scores)
        drug_like_count = sum(1 for score in drug_likeness_scores if score > 0.7)
        
        return {
            'avg_drug_likeness': avg_drug_likeness,
            'drug_like_molecules': drug_like_count,
            'drug_likeness_rate': drug_like_count / len(valid_smiles),
            'lipinski_compliant': lipinski_compliant_count,
            'lipinski_compliance_rate': lipinski_compliant_count / len(valid_smiles)
        }
    
    def evaluate_property_distribution(self, generated_smiles: List[str]) -> Dict[str, Dict[str, float]]:
        """Evaluate molecular property distributions"""
        valid_smiles = [smiles for smiles in generated_smiles 
                       if self.validator.is_valid_smiles(smiles)]
        
        if not valid_smiles:
            return {}
        
        properties = {
            'molecular_weight': [],
            'logp': [],
            'tpsa': [],
            'num_rotatable_bonds': [],
            'num_hbd': [],
            'num_hba': []
        }
        
        for smiles in valid_smiles:
            mol_props = self.validator.calculate_properties(smiles)
            for prop_name in properties.keys():
                if prop_name in mol_props:
                    properties[prop_name].append(mol_props[prop_name])
        
        # Calculate statistics
        property_stats = {}
        for prop_name, values in properties.items():
            if values:
                property_stats[prop_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return property_stats
    
    def comprehensive_evaluation(self, generated_smiles: List[str]) -> Dict[str, any]:
        """Run comprehensive evaluation suite"""
        results = {}
        
        # Validity evaluation
        results['validity'] = self.evaluate_validity(generated_smiles)
        
        # Diversity evaluation
        results['diversity'] = self.evaluate_diversity(generated_smiles)
        
        # Novelty evaluation
        results['novelty'] = self.evaluate_novelty(generated_smiles)
        
        # Drug-likeness evaluation
        results['drug_likeness'] = self.evaluate_drug_likeness(generated_smiles)
        
        # Property distribution evaluation
        results['property_distributions'] = self.evaluate_property_distribution(generated_smiles)
        
        return results
    
    def compare_models(self, 
                      model_results: Dict[str, List[str]]) -> Dict[str, Dict[str, any]]:
        """Compare multiple generation models"""
        comparison_results = {}
        
        for model_name, generated_smiles in model_results.items():
            comparison_results[model_name] = self.comprehensive_evaluation(generated_smiles)
        
        return comparison_results
    
    def generate_report(self, 
                       evaluation_results: Dict[str, any],
                       output_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report"""
        report_lines = []
        report_lines.append("# Drug Generation Evaluation Report\n")
        
        # Validity section
        if 'validity' in evaluation_results:
            validity = evaluation_results['validity']
            report_lines.append("## Validity Metrics")
            report_lines.append(f"- Total Generated: {validity['total_generated']}")
            report_lines.append(f"- Valid Molecules: {validity['valid_molecules']}")
            report_lines.append(f"- Validity Rate: {validity['validity_rate']:.3f}")
            report_lines.append(f"- Unique Valid Molecules: {validity['unique_valid_molecules']}")
            report_lines.append(f"- Uniqueness Rate: {validity['uniqueness_rate']:.3f}\n")
        
        # Diversity section
        if 'diversity' in evaluation_results:
            diversity = evaluation_results['diversity']
            report_lines.append("## Diversity Metrics")
            report_lines.append(f"- Tanimoto Diversity: {diversity['tanimoto_diversity']:.3f}")
            report_lines.append(f"- Scaffold Diversity: {diversity['scaffold_diversity']:.3f}\n")
        
        # Novelty section
        if 'novelty' in evaluation_results:
            novelty = evaluation_results['novelty']
            report_lines.append("## Novelty Metrics")
            for dataset_name, novelty_data in novelty.items():
                report_lines.append(f"- {dataset_name} Novelty Rate: {novelty_data['novelty_rate']:.3f}")
            report_lines.append("")
        
        # Drug-likeness section
        if 'drug_likeness' in evaluation_results:
            drug_like = evaluation_results['drug_likeness']
            report_lines.append("## Drug-likeness Metrics")
            report_lines.append(f"- Average Drug-likeness Score: {drug_like['avg_drug_likeness']:.3f}")
            report_lines.append(f"- Drug-like Molecules: {drug_like['drug_like_molecules']}")
            report_lines.append(f"- Drug-likeness Rate: {drug_like['drug_likeness_rate']:.3f}")
            report_lines.append(f"- Lipinski Compliant: {drug_like['lipinski_compliant']}")
            report_lines.append(f"- Lipinski Compliance Rate: {drug_like['lipinski_compliance_rate']:.3f}\n")
        
        # Property distributions section
        if 'property_distributions' in evaluation_results:
            prop_dist = evaluation_results['property_distributions']
            report_lines.append("## Property Distributions")
            for prop_name, stats in prop_dist.items():
                report_lines.append(f"### {prop_name.replace('_', ' ').title()}")
                report_lines.append(f"- Mean: {stats['mean']:.3f}")
                report_lines.append(f"- Std: {stats['std']:.3f}")
                report_lines.append(f"- Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                report_lines.append(f"- Median: {stats['median']:.3f}\n")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text


class GenerationVisualizer:
    """Visualization tools for generation evaluation"""
    
    @staticmethod
    def plot_property_distributions(property_data: Dict[str, List[float]], 
                                  save_path: Optional[str] = None):
        """Plot molecular property distributions"""
        n_properties = len(property_data)
        if n_properties == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (prop_name, values) in enumerate(property_data.items()):
            if i >= len(axes):
                break
            
            if values:
                axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{prop_name.replace("_", " ").title()}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(property_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_model_comparison(comparison_results: Dict[str, Dict[str, any]], 
                            metrics: List[str] = None,
                            save_path: Optional[str] = None):
        """Plot comparison between different models"""
        if metrics is None:
            metrics = ['validity_rate', 'uniqueness_rate', 'tanimoto_diversity', 
                      'scaffold_diversity', 'avg_drug_likeness', 'lipinski_compliance_rate']
        
        model_names = list(comparison_results.keys())
        metric_values = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            results = comparison_results[model_name]
            
            for metric in metrics:
                if metric in ['validity_rate', 'uniqueness_rate']:
                    value = results.get('validity', {}).get(metric, 0)
                elif metric in ['tanimoto_diversity', 'scaffold_diversity']:
                    value = results.get('diversity', {}).get(metric, 0)
                elif metric in ['avg_drug_likeness', 'lipinski_compliance_rate']:
                    value = results.get('drug_likeness', {}).get(metric, 0)
                else:
                    value = 0
                
                metric_values[metric].append(value)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, model_name in enumerate(model_names):
            values = [metric_values[metric][i] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Model Comparison - Generation Metrics', size=16, pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class GenerationEvaluationPipeline:
    """Complete pipeline for evaluating drug generation models"""
    
    def __init__(self,
                 reference_datasets: Optional[Dict[str, List[str]]] = None,
                 property_predictor: Optional[MolecularPropertyPredictor] = None):
        
        self.benchmark = GenerationBenchmark(reference_datasets, property_predictor)
        self.visualizer = GenerationVisualizer()
        
        # Initialize metrics
        all_reference_smiles = []
        if reference_datasets:
            for smiles_list in reference_datasets.values():
                all_reference_smiles.extend(smiles_list)
        
        self.metrics = GenerationMetrics(all_reference_smiles)
        self.confidence_pipeline = ConfidenceScoringPipeline(reference_smiles=all_reference_smiles)
    
    def evaluate_single_model(self,
                            generated_smiles: List[str],
                            model_name: str = "model",
                            save_results: bool = True,
                            output_dir: Optional[str] = None) -> Dict[str, any]:
        """
        Evaluate a single generation model
        
        Args:
            generated_smiles: List of generated SMILES
            model_name: Name of the model
            save_results: Whether to save results
            output_dir: Output directory for results
        
        Returns:
            Evaluation results
        """
        # Run comprehensive evaluation
        results = self.benchmark.comprehensive_evaluation(generated_smiles)
        
        # Add timestamp and model info
        results['model_name'] = model_name
        results['num_generated'] = len(generated_smiles)
        
        if save_results and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save results as JSON
            with open(output_path / f"{model_name}_evaluation.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Generate and save report
            report = self.benchmark.generate_report(results)
            with open(output_path / f"{model_name}_report.md", 'w') as f:
                f.write(report)
            
            # Save generated SMILES
            pd.DataFrame({'smiles': generated_smiles}).to_csv(
                output_path / f"{model_name}_generated.csv", index=False
            )
        
        return results
    
    def compare_models(self,
                      model_generations: Dict[str, List[str]],
                      save_results: bool = True,
                      output_dir: Optional[str] = None) -> Dict[str, Dict[str, any]]:
        """
        Compare multiple generation models
        
        Args:
            model_generations: Dictionary mapping model names to generated SMILES
            save_results: Whether to save results
            output_dir: Output directory for results
        
        Returns:
            Comparison results
        """
        # Evaluate each model
        comparison_results = {}
        for model_name, generated_smiles in model_generations.items():
            comparison_results[model_name] = self.evaluate_single_model(
                generated_smiles, model_name, save_results=False
            )
        
        if save_results and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save comparison results
            with open(output_path / "model_comparison.json", 'w') as f:
                json.dump(comparison_results, f, indent=2, default=str)
            
            # Generate comparison visualization
            self.visualizer.plot_model_comparison(
                comparison_results,
                save_path=str(output_path / "model_comparison.png")
            )
        
        return comparison_results
    
    def benchmark_against_baselines(self,
                                  generated_smiles: List[str],
                                  baseline_methods: Dict[str, List[str]],
                                  model_name: str = "test_model") -> Dict[str, any]:
        """
        Benchmark against baseline generation methods
        
        Args:
            generated_smiles: Generated SMILES from test model
            baseline_methods: Dictionary of baseline method results
            model_name: Name of the test model
        
        Returns:
            Benchmarking results
        """
        all_generations = {model_name: generated_smiles}
        all_generations.update(baseline_methods)
        
        return self.compare_models(all_generations)
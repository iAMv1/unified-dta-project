"""
Demo script for drug generation capabilities
Demonstrates transformer-based SMILES generation conditioned on protein targets
"""

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import argparse
import json
from typing import List, Dict

# Import core modules
from core.models import ESMProteinEncoder, UnifiedDTAModel
from core.drug_generation import (
    ProteinConditionedGenerator, 
    DrugGenerationPipeline,
    SMILESTokenizer,
    ChemicalValidator
)
from core.generation_scoring import (
    GenerationMetrics,
    ConfidenceScoringPipeline,
    MolecularPropertyCalculator
)
from core.generation_evaluation import (
    GenerationEvaluationPipeline,
    GenerationBenchmark
)


def load_sample_proteins() -> List[str]:
    """Load sample protein sequences for demonstration"""
    sample_proteins = [
        # Human insulin receptor (shortened for demo)
        "MATGGRRGAAAAPLLVAVAALLLGAAGHLYPGEVCPGMDIRNNLTRLHELENCSVIEGHLQILLMFKTRPEDFRDLSFPKLIMITDYLLLFRVYGLESLKDLFPNLTVIRGSRLFFNYALVIFEMVHLKELGLYNLMNITRGSVRIEKNNELCYLATIDWSRILDSVEDNYIVLNKDDNEECGDICPGTAKGKTNCPATVINGQFVERCWTHSHCQKVCPTICKSHGCTAEGLCCHSECLGNCSQPDDPTKCVACRNFYLDGRCVETCPPPYYHFQDWRCVNFSFCQDLHHKCKNSRRQGCHQYVIHNNKCIPECPSGYTMNSSNLLCTPCLGPCPKVCHLLEGEKTIDSVTSAQELRGCTVINGSLIINIRGGNNLAAELEANLGLIEEISGYLKIRRSYALVSLSFFRKLRLIRGETLEIGNYSFYALDNQNLRQLWDWSKHNLTITQGKLFFHYNPKLCLSEIHKMEEVSGTKGRQERNDIALKTNGDQASCENELLKFSYIRTSFDKILLRWLSLVKKTVEELLANACLCYTPACLLAKAKSATRWLLPSIVPVAAISVPVSLPVVGSSAYALVSCLLLTPGCAAEQWYTLHKDKVSAEVEEAYVLSKTQPLRHHVEALVS",
        
        # Human epidermal growth factor receptor (EGFR) - shortened
        "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA",
        
        # Human tumor protein p53 (shortened)
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    ]
    
    return sample_proteins


def create_lightweight_generator() -> ProteinConditionedGenerator:
    """Create a lightweight generator for demonstration"""
    # Create a simple protein encoder
    protein_encoder = ESMProteinEncoder(output_dim=64)
    
    # Create tokenizer
    tokenizer = SMILESTokenizer()
    
    # Create generator with smaller dimensions for demo
    generator = ProteinConditionedGenerator(
        protein_encoder=protein_encoder,
        vocab_size=len(tokenizer),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_length=64
    )
    
    return generator


def demo_basic_generation():
    """Demonstrate basic drug generation"""
    print("=== Basic Drug Generation Demo ===\n")
    
    # Load sample proteins
    proteins = load_sample_proteins()[:2]  # Use first 2 for demo
    
    print(f"Loaded {len(proteins)} sample protein sequences")
    print(f"First protein length: {len(proteins[0])} residues")
    print(f"Second protein length: {len(proteins[1])} residues\n")
    
    # Create generator
    print("Creating lightweight generator...")
    generator = create_lightweight_generator()
    generator.eval()
    
    # Generate molecules
    print("Generating molecules...")
    try:
        with torch.no_grad():
            generated_smiles = generator.generate(
                protein_sequences=proteins,
                max_length=32,
                temperature=1.0,
                deterministic=False,
                num_return_sequences=3
            )
        
        print(f"Generated {len(generated_smiles)} sets of molecules\n")
        
        # Display results
        validator = ChemicalValidator()
        
        for i, (protein, smiles_list) in enumerate(zip(proteins, generated_smiles)):
            print(f"Protein {i+1} (length: {len(protein)}):")
            print(f"  Sequence preview: {protein[:50]}...")
            print(f"  Generated molecules:")
            
            for j, smiles in enumerate(smiles_list):
                is_valid = validator.is_valid_smiles(smiles)
                validity_status = "✓ Valid" if is_valid else "✗ Invalid"
                print(f"    {j+1}. {smiles} [{validity_status}]")
                
                if is_valid:
                    properties = validator.calculate_properties(smiles)
                    if properties:
                        mw = properties.get('molecular_weight', 0)
                        logp = properties.get('logp', 0)
                        print(f"       MW: {mw:.1f}, LogP: {logp:.2f}")
            print()
    
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Note: This is a demo with untrained model - results are random")


def demo_generation_evaluation():
    """Demonstrate generation evaluation capabilities"""
    print("=== Generation Evaluation Demo ===\n")
    
    # Sample generated molecules (for demonstration)
    sample_generated = [
        "CCO",  # Ethanol - valid
        "CC(C)O",  # Isopropanol - valid
        "c1ccccc1",  # Benzene - valid
        "CC(=O)O",  # Acetic acid - valid
        "INVALID_SMILES",  # Invalid
        "CCN(CC)CC",  # Triethylamine - valid
        "CC(C)(C)O",  # tert-Butanol - valid
    ]
    
    print(f"Evaluating {len(sample_generated)} sample molecules...")
    
    # Create evaluation pipeline
    evaluator = GenerationEvaluationPipeline()
    
    # Run evaluation
    results = evaluator.evaluate_single_model(
        generated_smiles=sample_generated,
        model_name="demo_model",
        save_results=False
    )
    
    # Display results
    print("\n--- Evaluation Results ---")
    
    if 'validity' in results:
        validity = results['validity']
        print(f"Validity Rate: {validity['validity_rate']:.3f}")
        print(f"Valid Molecules: {validity['valid_molecules']}/{validity['total_generated']}")
        print(f"Uniqueness Rate: {validity['uniqueness_rate']:.3f}")
    
    if 'diversity' in results:
        diversity = results['diversity']
        print(f"Tanimoto Diversity: {diversity['tanimoto_diversity']:.3f}")
        print(f"Scaffold Diversity: {diversity['scaffold_diversity']:.3f}")
    
    if 'drug_likeness' in results:
        drug_like = results['drug_likeness']
        print(f"Average Drug-likeness: {drug_like['avg_drug_likeness']:.3f}")
        print(f"Lipinski Compliance Rate: {drug_like['lipinski_compliance_rate']:.3f}")
    
    print("\n--- Property Analysis ---")
    if 'property_distributions' in results:
        prop_dist = results['property_distributions']
        for prop_name, stats in prop_dist.items():
            print(f"{prop_name.replace('_', ' ').title()}:")
            print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
            print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")


def demo_chemical_validation():
    """Demonstrate chemical validation capabilities"""
    print("=== Chemical Validation Demo ===\n")
    
    # Test molecules with various validity levels
    test_molecules = [
        ("CCO", "Ethanol - simple alcohol"),
        ("c1ccccc1O", "Phenol - aromatic alcohol"),
        ("CC(=O)Nc1ccccc1", "Acetanilide - amide"),
        ("INVALID", "Invalid SMILES"),
        ("C[C@H](N)C(=O)O", "Alanine - amino acid"),
        ("CC(C)(C)c1ccc(O)cc1", "4-tert-butylphenol"),
    ]
    
    validator = ChemicalValidator()
    property_calc = MolecularPropertyCalculator()
    
    print("Testing chemical validation and property calculation:\n")
    
    for smiles, description in test_molecules:
        print(f"Molecule: {description}")
        print(f"SMILES: {smiles}")
        
        # Check validity
        is_valid = validator.is_valid_smiles(smiles)
        print(f"Valid: {is_valid}")
        
        if is_valid:
            # Canonicalize
            canonical = validator.canonicalize_smiles(smiles)
            print(f"Canonical: {canonical}")
            
            # Calculate properties
            properties = validator.calculate_properties(smiles)
            if properties:
                print(f"Molecular Weight: {properties.get('molecular_weight', 0):.1f}")
                print(f"LogP: {properties.get('logp', 0):.2f}")
                print(f"TPSA: {properties.get('tpsa', 0):.1f}")
            
            # Drug-likeness score
            drug_score = property_calc.calculate_drug_likeness_score(smiles)
            print(f"Drug-likeness Score: {drug_score:.3f}")
            
            # Lipinski properties
            lipinski = property_calc.calculate_lipinski_properties(smiles)
            if lipinski:
                violations = lipinski.get('lipinski_violations', 0)
                compliant = lipinski.get('lipinski_compliant', False)
                print(f"Lipinski Violations: {violations} ({'Compliant' if compliant else 'Non-compliant'})")
        
        print("-" * 50)


def demo_confidence_scoring():
    """Demonstrate confidence scoring capabilities"""
    print("=== Confidence Scoring Demo ===\n")
    
    # Sample molecules with different quality levels
    molecules = [
        "CCO",  # Simple, drug-like
        "c1ccccc1",  # Simple aromatic
        "CC(=O)Nc1ccccc1",  # More complex, drug-like
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",  # Too large
        "C",  # Too simple
    ]
    
    # Create confidence scoring pipeline
    confidence_pipeline = ConfidenceScoringPipeline()
    
    print("Scoring molecules for quality and confidence:\n")
    
    # Score molecules
    scoring_results = confidence_pipeline.score_molecules(molecules)
    
    for result in scoring_results:
        smiles = result['smiles']
        print(f"SMILES: {smiles}")
        print(f"  Confidence Score: {result['confidence_score']:.3f}")
        print(f"  Overall Quality: {result['overall_score']:.3f}")
        print(f"  Drug-likeness: {result['drug_likeness']:.3f}")
        print(f"  Valid: {'Yes' if result['is_valid'] > 0 else 'No'}")
        print(f"  Lipinski Compliant: {'Yes' if result['lipinski_compliant'] > 0 else 'No'}")
        print()
    
    # Rank molecules
    ranked_results = confidence_pipeline.rank_molecules(scoring_results, 'overall_score')
    
    print("Molecules ranked by overall quality:")
    for i, result in enumerate(ranked_results, 1):
        print(f"{i}. {result['smiles']} (Score: {result['overall_score']:.3f})")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Drug Generation Demo")
    parser.add_argument('--demo', type=str, choices=['basic', 'evaluation', 'validation', 'scoring', 'all'],
                       default='all', help='Which demo to run')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    if args.demo in ['basic', 'all']:
        demo_basic_generation()
        print("\n" + "="*60 + "\n")
    
    if args.demo in ['evaluation', 'all']:
        demo_generation_evaluation()
        print("\n" + "="*60 + "\n")
    
    if args.demo in ['validation', 'all']:
        demo_chemical_validation()
        print("\n" + "="*60 + "\n")
    
    if args.demo in ['scoring', 'all']:
        demo_confidence_scoring()
        print("\n" + "="*60 + "\n")
    
    print("Demo completed!")
    print("\nNote: This demo uses untrained models for demonstration purposes.")
    print("For actual drug generation, models need to be trained on appropriate datasets.")


if __name__ == "__main__":
    main()
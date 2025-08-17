#!/usr/bin/env python3
"""
Demo script for Graph Preprocessing and Feature Extraction
Shows the complete pipeline from SMILES to molecular graph features
"""

import torch
import numpy as np
from core import (
    GraphFeatureConfig,
    MolecularGraphProcessor,
    GraphValidator,
    OptimizedGraphBatcher,
    process_smiles_batch,
    EnhancedGINDrugEncoder,
    create_dta_model
)


def demo_graph_preprocessing_pipeline():
    """Demonstrate the complete graph preprocessing pipeline"""
    print("Graph Preprocessing and Feature Extraction Demo")
    print("=" * 60)
    
    # Sample drug molecules (SMILES format)
    drug_molecules = [
        ('CCO', 'Ethanol - Simple alcohol'),
        ('c1ccccc1', 'Benzene - Aromatic ring'),
        ('CC(C)O', 'Isopropanol - Secondary alcohol'),
        ('CCN(CC)CC', 'Triethylamine - Tertiary amine'),
        ('CC(=O)O', 'Acetic acid - Carboxylic acid'),
        ('c1ccc2c(c1)cccn2', 'Quinoline - Bicyclic aromatic'),
        ('CC(C)(C)c1ccc(O)cc1', 'BHT - Antioxidant'),
        ('invalid_molecule', 'Invalid SMILES for testing'),
        ('C', 'Methane - Simplest hydrocarbon'),
        ('CCCCCCCCCCCCCCCCC', 'Heptadecane - Long alkyl chain')
    ]
    
    smiles_list = [smiles for smiles, _ in drug_molecules]
    descriptions = [desc for _, desc in drug_molecules]
    
    print(f"\n1. Processing {len(smiles_list)} drug molecules...")
    for i, (smiles, desc) in enumerate(drug_molecules):
        print(f"   {i+1:2d}. {smiles:20s} - {desc}")
    
    # Configure feature extraction
    print(f"\n2. Configuring molecular feature extraction...")
    
    # Standard configuration
    standard_config = GraphFeatureConfig(
        include_atomic_number=True,
        include_degree=True,
        include_formal_charge=True,
        include_hybridization=True,
        include_aromaticity=True,
        include_bond_type=True,
        include_conjugation=True,
        use_one_hot_encoding=False,  # Use continuous features for demo
        normalize_features=True,
        include_molecular_descriptors=True
    )
    
    print(f"   - Atomic number: ✓")
    print(f"   - Atom degree: ✓")
    print(f"   - Formal charge: ✓")
    print(f"   - Hybridization: ✓")
    print(f"   - Aromaticity: ✓")
    print(f"   - Bond type: ✓")
    print(f"   - Bond conjugation: ✓")
    print(f"   - Feature normalization: ✓")
    print(f"   - Molecular descriptors: ✓")
    
    # Process molecules with complete pipeline
    print(f"\n3. Processing molecules with validation and batching...")
    
    results = process_smiles_batch(
        smiles_list,
        config=standard_config,
        validate=True,
        create_batches=True
    )
    
    print(f"   - Total molecules: {results['total_smiles']}")
    print(f"   - Successfully processed: {results['successful_graphs']}")
    print(f"   - Success rate: {results['success_rate']*100:.1f}%")
    print(f"   - Failed molecules: {results['failed_smiles']}")
    
    if 'valid_graphs' in results:
        print(f"   - Valid after validation: {results['valid_graphs']}")
        print(f"   - Validation rate: {results['validation_rate']*100:.1f}%")
    
    # Analyze molecular graphs
    if 'batches' in results and results['batches']:
        print(f"\n4. Analyzing molecular graph properties...")
        
        batch = results['batches'][0]
        batch_stats = results['batch_statistics']
        
        print(f"   - Created batches: {len(results['batches'])}")
        print(f"   - Average batch size: {batch_stats['batch_sizes']['mean']:.1f}")
        print(f"   - Average nodes per batch: {batch_stats['node_counts']['mean']:.1f}")
        print(f"   - Average edges per batch: {batch_stats['edge_counts']['mean']:.1f}")
        
        print(f"\n   First batch details:")
        print(f"   - Number of molecules: {batch.num_graphs}")
        print(f"   - Total nodes (atoms): {batch.x.shape[0]}")
        print(f"   - Node feature dimension: {batch.x.shape[1]}")
        print(f"   - Total edges (bonds): {batch.edge_index.shape[1]}")
        
        if batch.edge_attr is not None:
            print(f"   - Edge feature dimension: {batch.edge_attr.shape[1]}")
        
        # Analyze individual molecules in batch
        print(f"\n   Individual molecule analysis:")
        node_counts = torch.bincount(batch.batch)
        for i, count in enumerate(node_counts):
            if i < len(descriptions):
                print(f"   - Molecule {i+1}: {count.item()} atoms - {descriptions[i]}")
    
    # Feature extraction with Enhanced GIN encoder
    if 'batches' in results and results['batches']:
        print(f"\n5. Extracting molecular features with Enhanced GIN encoder...")
        
        batch = results['batches'][0]
        
        # Create Enhanced GIN encoder
        gin_encoder = EnhancedGINDrugEncoder(
            node_features=batch.x.shape[1],
            hidden_dim=128,
            num_layers=4,
            output_dim=256,
            mlp_hidden_dims=[256, 128],
            activation='gelu',
            dropout=0.1,
            pooling_strategy='dual',  # Mean + Max pooling
            residual_connections=True
        )
        
        print(f"   - GIN encoder configuration:")
        print(f"     * Input dimension: {batch.x.shape[1]}")
        print(f"     * Hidden dimension: 128")
        print(f"     * Number of layers: 4")
        print(f"     * Output dimension: 256")
        print(f"     * Pooling strategy: Dual (mean + max)")
        print(f"     * Residual connections: Enabled")
        
        # Extract molecular features
        gin_encoder.eval()
        with torch.no_grad():
            molecular_features = gin_encoder(batch)
            node_embeddings = gin_encoder.get_node_embeddings(batch)
            layer_outputs = gin_encoder.get_layer_outputs(batch)
        
        print(f"\n   Feature extraction results:")
        print(f"   - Molecular features shape: {molecular_features.shape}")
        print(f"   - Node embeddings shape: {node_embeddings.shape}")
        print(f"   - Layer outputs: {len(layer_outputs)} layers")
        
        # Analyze molecular features
        feature_stats = {
            'mean': molecular_features.mean(dim=0),
            'std': molecular_features.std(dim=0),
            'min': molecular_features.min(dim=0)[0],
            'max': molecular_features.max(dim=0)[0]
        }
        
        print(f"   - Feature statistics:")
        print(f"     * Mean: {feature_stats['mean'][:5].tolist()} ... (first 5)")
        print(f"     * Std:  {feature_stats['std'][:5].tolist()} ... (first 5)")
        print(f"     * Range: [{feature_stats['min'][:3].min().item():.3f}, {feature_stats['max'][:3].max().item():.3f}]")
    
    # Compare different configurations
    print(f"\n6. Comparing different feature configurations...")
    
    configs = [
        {
            'name': 'Minimal',
            'config': GraphFeatureConfig(
                include_atomic_number=True,
                include_degree=True,
                use_one_hot_encoding=False,
                include_molecular_descriptors=False
            )
        },
        {
            'name': 'Standard',
            'config': GraphFeatureConfig(
                include_atomic_number=True,
                include_degree=True,
                include_aromaticity=True,
                include_bond_type=True,
                use_one_hot_encoding=False,
                include_molecular_descriptors=True
            )
        },
        {
            'name': 'Comprehensive',
            'config': GraphFeatureConfig(
                include_atomic_number=True,
                include_degree=True,
                include_formal_charge=True,
                include_hybridization=True,
                include_aromaticity=True,
                include_bond_type=True,
                include_conjugation=True,
                include_ring_membership=True,
                use_one_hot_encoding=True,  # One-hot encoding for maximum features
                include_molecular_descriptors=True
            )
        }
    ]
    
    test_smiles = ['CCO', 'c1ccccc1']  # Simple test molecules
    
    for config_info in configs:
        name = config_info['name']
        config = config_info['config']
        
        processor = MolecularGraphProcessor(config)
        graphs, _ = processor.batch_process_smiles(test_smiles)
        
        if graphs:
            from torch_geometric.data import Batch
            test_batch = Batch.from_data_list(graphs)
            
            node_dim = test_batch.x.shape[1]
            edge_dim = test_batch.edge_attr.shape[1] if test_batch.edge_attr is not None else 0
            
            print(f"   - {name:13s}: Node features={node_dim:3d}, Edge features={edge_dim:2d}")
        else:
            print(f"   - {name:13s}: No valid graphs (limited without RDKit)")
    
    # Integration with complete DTA model
    if 'batches' in results and results['batches']:
        print(f"\n7. Integration with complete Drug-Target Affinity model...")
        
        batch = results['batches'][0]
        
        # Create complete DTA model
        dta_config = {
            'protein_encoder_type': 'cnn',
            'drug_encoder_type': 'gin',
            'use_fusion': True,
            'protein_config': {
                'embed_dim': 128,
                'num_filters': [64, 128, 256],
                'kernel_sizes': [3, 5, 7],
                'output_dim': 128,
                'use_se': True
            },
            'drug_config': {
                'node_features': batch.x.shape[1],
                'hidden_dim': 128,
                'num_layers': 4,
                'output_dim': 128,
                'mlp_hidden_dims': [256, 128],
                'pooling_strategy': 'dual',
                'residual_connections': True
            },
            'fusion_config': {
                'hidden_dim': 256,
                'num_heads': 8
            },
            'predictor_config': {
                'hidden_dims': [512, 256, 128],
                'dropout': 0.3
            }
        }
        
        dta_model = create_dta_model(dta_config)
        
        # Mock protein sequences
        protein_tokens = torch.randint(1, 21, (batch.num_graphs, 100))
        
        # Predict drug-target affinities
        dta_model.eval()
        with torch.no_grad():
            affinity_predictions = dta_model(batch, protein_tokens)
        
        print(f"   - DTA model created successfully")
        print(f"   - Drug molecules: {batch.num_graphs}")
        print(f"   - Protein sequences: {protein_tokens.shape}")
        print(f"   - Affinity predictions: {affinity_predictions.shape}")
        print(f"   - Sample predictions: {affinity_predictions.flatten()[:3].tolist()}")
        
        # Model complexity analysis
        total_params = sum(p.numel() for p in dta_model.parameters())
        drug_encoder_params = sum(p.numel() for p in dta_model.drug_encoder.parameters())
        protein_encoder_params = sum(p.numel() for p in dta_model.protein_encoder.parameters())
        
        print(f"\n   Model complexity:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Drug encoder: {drug_encoder_params:,} ({drug_encoder_params/total_params*100:.1f}%)")
        print(f"   - Protein encoder: {protein_encoder_params:,} ({protein_encoder_params/total_params*100:.1f}%)")
    
    print(f"\n✅ Graph Preprocessing Demo Complete!")
    print(f"\nKey Capabilities Demonstrated:")
    print(f"- ✓ SMILES string parsing and validation")
    print(f"- ✓ Comprehensive molecular feature extraction")
    print(f"- ✓ Graph validation and error handling")
    print(f"- ✓ Optimized batch processing")
    print(f"- ✓ Integration with Enhanced GIN encoder")
    print(f"- ✓ Node and edge feature processing")
    print(f"- ✓ Multiple feature configuration options")
    print(f"- ✓ Complete drug-target affinity prediction pipeline")


if __name__ == "__main__":
    demo_graph_preprocessing_pipeline()
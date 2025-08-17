"""
Demonstration of configurable prediction heads for drug-target affinity prediction
"""

import torch
from core.prediction_heads import (
    MLPPredictionHead,
    PredictionHeadFactory,
    get_lightweight_predictor,
    get_standard_predictor,
    get_deep_predictor
)
from core.models import create_dta_model


def demo_basic_usage():
    """Demonstrate basic usage of prediction heads"""
    print("=== Basic Usage Demo ===")
    
    # Create a simple MLP prediction head
    predictor = MLPPredictionHead(
        input_dim=256,
        hidden_dims=[512, 256],
        activation='relu',
        dropout=0.3,
        use_batch_norm=True
    )
    
    # Generate sample input
    batch_size = 4
    features = torch.randn(batch_size, 256)
    
    # Make predictions
    predictions = predictor(features)
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions: {predictions.squeeze().tolist()}")
    print()


def demo_different_activations():
    """Demonstrate different activation functions"""
    print("=== Different Activation Functions Demo ===")
    
    input_dim = 128
    batch_size = 2
    features = torch.randn(batch_size, input_dim)
    
    activations = ['relu', 'gelu', 'leaky_relu', 'elu', 'selu', 'swish']
    
    for activation in activations:
        predictor = MLPPredictionHead(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            activation=activation,
            dropout=0.1
        )
        
        predictions = predictor(features)
        print(f"{activation.upper():12} -> Output: {predictions.squeeze().tolist()}")
    
    print()


def demo_normalization_options():
    """Demonstrate different normalization options"""
    print("=== Normalization Options Demo ===")
    
    input_dim = 128
    batch_size = 4
    features = torch.randn(batch_size, input_dim)
    
    # Batch normalization
    predictor_bn = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        use_batch_norm=True,
        use_layer_norm=False
    )
    
    # Layer normalization
    predictor_ln = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        use_batch_norm=False,
        use_layer_norm=True
    )
    
    # No normalization
    predictor_none = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        use_batch_norm=False,
        use_layer_norm=False
    )
    
    pred_bn = predictor_bn(features)
    pred_ln = predictor_ln(features)
    pred_none = predictor_none(features)
    
    print(f"Batch Norm:  {pred_bn.squeeze().tolist()}")
    print(f"Layer Norm:  {pred_ln.squeeze().tolist()}")
    print(f"No Norm:     {pred_none.squeeze().tolist()}")
    print()


def demo_predefined_configurations():
    """Demonstrate predefined configurations"""
    print("=== Predefined Configurations Demo ===")
    
    input_dim = 256
    batch_size = 2
    features = torch.randn(batch_size, input_dim)
    
    # Lightweight predictor
    lightweight = get_lightweight_predictor(input_dim)
    pred_light = lightweight(features)
    print(f"Lightweight: {pred_light.squeeze().tolist()}")
    
    # Standard predictor
    standard = get_standard_predictor(input_dim)
    pred_std = standard(features)
    print(f"Standard:    {pred_std.squeeze().tolist()}")
    
    # Deep predictor
    deep = get_deep_predictor(input_dim)
    pred_deep = deep(features)
    print(f"Deep:        {pred_deep.squeeze().tolist()}")
    
    print()


def demo_factory_usage():
    """Demonstrate factory pattern usage"""
    print("=== Factory Pattern Demo ===")
    
    input_dim = 128
    
    # Create predictor using factory
    predictor = PredictionHeadFactory.create(
        'mlp',
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        activation='gelu',
        dropout=0.2,
        use_batch_norm=True
    )
    
    features = torch.randn(3, input_dim)
    predictions = predictor(features)
    
    print(f"Factory-created predictor: {type(predictor).__name__}")
    print(f"Predictions: {predictions.squeeze().tolist()}")
    print()


def demo_model_integration():
    """Demonstrate integration with unified DTA model"""
    print("=== Model Integration Demo ===")
    
    # Create model with custom prediction head configuration
    config = {
        'protein_encoder_type': 'esm',
        'drug_encoder_type': 'gin',
        'use_fusion': False,
        'protein_config': {'output_dim': 64, 'max_length': 50},
        'drug_config': {'output_dim': 64, 'num_layers': 3},
        'predictor_config': {
            'type': 'mlp',
            'hidden_dims': [256, 128, 64],
            'activation': 'gelu',
            'dropout': 0.3,
            'use_batch_norm': True,
            'use_layer_norm': False
        }
    }
    
    try:
        model = create_dta_model(config)
        print(f"✓ Model created with predictor: {type(model.predictor).__name__}")
        print(f"  - Hidden dimensions: {model.predictor.hidden_layers}")
        print(f"  - Input dimension: {model.predictor.input_dim}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
    
    print()


def demo_custom_configurations():
    """Demonstrate custom configuration options"""
    print("=== Custom Configuration Demo ===")
    
    input_dim = 200
    
    # Very deep network
    deep_predictor = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64, 32],
        activation='gelu',
        dropout=0.4,
        use_batch_norm=True
    )
    
    # Wide network
    wide_predictor = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[1024, 512],
        activation='relu',
        dropout=0.2,
        use_batch_norm=True
    )
    
    # Minimal network
    minimal_predictor = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[64],
        activation='relu',
        dropout=0.1,
        use_batch_norm=False
    )
    
    features = torch.randn(2, input_dim)
    
    pred_deep = deep_predictor(features)
    pred_wide = wide_predictor(features)
    pred_minimal = minimal_predictor(features)
    
    print(f"Deep (5 layers):    {pred_deep.squeeze().tolist()}")
    print(f"Wide (2 layers):    {pred_wide.squeeze().tolist()}")
    print(f"Minimal (1 layer):  {pred_minimal.squeeze().tolist()}")
    print()


def maiain()    m_main__":
_ == "_if __name_

)
("=" * 70   print
 ed!")ions completonstratem dt("All
    prin)70" * int("=
    pr    tions()
configuraom_ust   demo_c)
 on(egratimo_model_int)
    detory_usage(   demo_facns()
 figuratiofined_con  demo_prede
  _options()lizationma_nor)
    democtivations(different_a   demo_)
 ic_usage(   demo_bas    
 
" * 70)t("=rin")
    p PredictionAffinityDrug-Target n Heads for Predictiofigurable "Con(
    print=" * 70)print("   
 tions"""emonstra"Run all d  ""n():
  
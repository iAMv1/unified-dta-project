"""
Test integration of configurable prediction heads with the unified model system
"""

import torch
from core.models import create_dta_model, get_lightweight_model, get_production_model
from core.prediction_heads import MLPPredictionHead, get_standard_predictor


def test_model_with_different_predictors():
    """Test creating models with different prediction head configurations"""
    print("Testing model integration with different predictors...")
    
    # Test standard MLP predictor
    config_mlp = {
        'protein_encoder_type': 'esm',
        'drug_encoder_type': 'gin',
        'use_fusion': False,
        'protein_config': {'output_dim': 64, 'max_length': 50},
        'drug_config': {'output_dim': 64, 'num_layers': 3},
        'predictor_config': {
            'type': 'mlp',
            'hidden_dims': [128, 64],
            'activation': 'relu',
            'dropout': 0.2
        }
    }
    
    try:
        model_mlp = create_dta_model(config_mlp)
        print(f"✓ MLP Predictor Model: {type(model_mlp.predictor).__name__}")
    except Exception as e:
        print(f"✗ MLP Predictor Model: {e}")
    
    # Test with GELU activation
    config_gelu = {
        'protein_encoder_type': 'esm',
        'drug_encoder_type': 'gin',
        'use_fusion': False,
        'protein_config': {'output_dim': 64, 'max_length': 50},
        'drug_config': {'output_dim': 64, 'num_layers': 3},
        'predictor_config': {
            'type': 'mlp',
            'hidden_dims': [256, 128],
            'activation': 'gelu',
            'dropout': 0.3,
            'use_batch_norm': True
        }
    }
    
    try:
        model_gelu = create_dta_model(config_gelu)
        print(f"✓ GELU Predictor Model: {type(model_gelu.predictor).__name__}")
    except Exception as e:
        print(f"✗ GELU Predictor Model: {e}")
    
    print()


def test_predefined_models():
    """Test predefined model configurations"""
    print("Testing predefined model configurations...")
    
    try:
        lightweight_model = get_lightweight_model()
        print(f"✓ Lightweight Model: {type(lightweight_model.predictor).__name__}")
    except Exception as e:
        print(f"✗ Lightweight Model: {e}")
    
    try:
        production_model = get_production_model()
        print(f"✓ Production Model: {type(production_model.predictor).__name__}")
    except Exception as e:
        print(f"✗ Production Model: {e}")
    
    print()


def test_backward_compatibility():
    """Test that old configurations still work"""
    print("Testing backward compatibility...")
    
    # Old style configuration (should still work)
    old_config = {
        'protein_encoder_type': 'esm',
        'drug_encoder_type': 'gin',
        'use_fusion': False,
        'protein_config': {'output_dim': 64, 'max_length': 50},
        'drug_config': {'output_dim': 64, 'num_layers': 3},
        'predictor_config': {
            'hidden_dims': [128, 64],
            'dropout': 0.2
        }
    }
    
    try:
        old_model = create_dta_model(old_config)
        print(f"✓ Backward Compatible Model: {type(old_model.predictor).__name__}")
    except Exception as e:
        print(f"✗ Backward Compatible Model: {e}")
    
    print()


def test_prediction_head_features():
    """Test specific prediction head features"""
    print("Testing prediction head features...")
    
    input_dim = 128
    batch_size = 2
    
    # Test different activation functions
    activations = ['relu', 'gelu', 'leaky_relu', 'elu']
    
    for activation in activations:
        try:
            predictor = MLPPredictionHead(
                input_dim=input_dim,
                hidden_dims=[64, 32],
                activation=activation,
                dropout=0.1
            )
            
            x = torch.randn(batch_size, input_dim)
            output = predictor(x)
            
            print(f"✓ {activation.upper()} activation: {output.shape}")
        except Exception as e:
            print(f"✗ {activation.upper()} activation: {e}")
    
    # Test batch normalization vs layer normalization
    try:
        predictor_bn = MLPPredictionHead(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            use_batch_norm=True,
            use_layer_norm=False
        )
        
        x = torch.randn(batch_size, input_dim)
        output_bn = predictor_bn(x)
        print(f"✓ Batch Normalization: {output_bn.shape}")
    except Exception as e:
        print(f"✗ Batch Normalization: {e}")
    
    try:
        predictor_ln = MLPPredictionHead(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            use_batch_norm=False,
            use_layer_norm=True
        )
        
        x = torch.randn(batch_size, input_dim)
        output_ln = predictor_ln(x)
        print(f"✓ Layer Normalization: {output_ln.shape}")
    except Exception as e:
        print(f"✗ Layer Normalization: {e}")
    
    print()


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Testing Prediction Head Integration with Unified DTA System")
    print("=" * 60)
    
    test_model_with_different_predictors()
    test_predefined_models()
    test_backward_compatibility()
    test_prediction_head_features()
    
    print("=" * 60)
    print("All integration tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
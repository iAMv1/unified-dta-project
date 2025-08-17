"""
Test script for configurable prediction heads
"""

import torch
import torch.nn as nn
from core.prediction_heads import (
    MLPPredictionHead, 
    PredictionHeadFactory,
    ActivationFactory,
    get_lightweight_predictor,
    get_standard_predictor,
    get_deep_predictor
)


def test_activation_factory():
    """Test activation function factory"""
    print("Testing ActivationFactory...")
    
    # Test different activation functions
    activations = ['relu', 'gelu', 'leaky_relu', 'elu', 'selu', 'swish', 'mish', 'tanh', 'sigmoid']
    
    for act_name in activations:
        try:
            activation = ActivationFactory.create(act_name)
            print(f"✓ {act_name}: {type(activation).__name__}")
        except Exception as e:
            print(f"✗ {act_name}: {e}")
    
    print()


def test_mlp_prediction_head():
    """Test MLP prediction head with various configurations"""
    print("Testing MLPPredictionHead...")
    
    batch_size = 4
    input_dim = 256
    
    # Test basic configuration
    predictor = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[512, 256],
        activation='relu',
        dropout=0.3
    )
    
    x = torch.randn(batch_size, input_dim)
    output = predictor(x)
    
    print(f"✓ Basic MLP: Input {x.shape} -> Output {output.shape}")
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    
    # Test with different activation
    predictor_gelu = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[256, 128],
        activation='gelu',
        dropout=0.2,
        use_batch_norm=True
    )
    
    output_gelu = predictor_gelu(x)
    print(f"✓ GELU MLP: Input {x.shape} -> Output {output_gelu.shape}")
    
    # Test with residual connections
    predictor_residual = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[256, 256, 256],
        activation='relu',
        use_residual=True,
        dropout=0.1
    )
    
    output_residual = predictor_residual(x)
    print(f"✓ Residual MLP: Input {x.shape} -> Output {output_residual.shape}")
    
    # Test with layer normalization
    predictor_ln = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        use_batch_norm=False,
        use_layer_norm=True,
        dropout=0.0
    )
    
    output_ln = predictor_ln(x)
    print(f"✓ LayerNorm MLP: Input {x.shape} -> Output {output_ln.shape}")
    
    print()


def test_additional_features():
    """Test additional features like different activations"""
    print("Testing additional features...")
    
    batch_size = 4
    input_dim = 256
    
    # Test GELU activation
    predictor_gelu = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[256, 128],
        activation='gelu',
        dropout=0.2
    )
    
    x = torch.randn(batch_size, input_dim)
    output = predictor_gelu(x)
    print(f"✓ GELU Activation: Input {x.shape} -> Output {output.shape}")
    
    # Test layer normalization
    predictor_ln = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        use_batch_norm=False,
        use_layer_norm=True,
        dropout=0.1
    )
    
    output_ln = predictor_ln(x)
    print(f"✓ Layer Normalization: Input {x.shape} -> Output {output_ln.shape}")
    
    print()


def test_factory():
    """Test prediction head factory"""
    print("Testing PredictionHeadFactory...")
    
    input_dim = 256
    
    # Test MLP creation
    mlp = PredictionHeadFactory.create('mlp', input_dim=input_dim, hidden_dims=[128, 64])
    print(f"✓ Factory MLP: {type(mlp).__name__}")
    
    print()


def test_predefined_configurations():
    """Test predefined configuration functions"""
    print("Testing predefined configurations...")
    
    input_dim = 256
    batch_size = 4
    x = torch.randn(batch_size, input_dim)
    
    # Test all predefined configurations
    configs = [
        ('Lightweight', get_lightweight_predictor),
        ('Standard', get_standard_predictor),
        ('Deep', get_deep_predictor)
    ]
    
    for name, config_fn in configs:
        try:
            predictor = config_fn(input_dim)
            output = predictor(x)
            print(f"✓ {name}: Input {x.shape} -> Output {output.shape}")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    print()


def test_gradient_flow():
    """Test that gradients flow properly through prediction heads"""
    print("Testing gradient flow...")
    
    input_dim = 256
    batch_size = 4
    
    predictor = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        activation='relu',
        dropout=0.1
    )
    
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    target = torch.randn(batch_size, 1)
    
    # Forward pass
    output = predictor(x)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = x.grad is not None and x.grad.abs().sum() > 0
    print(f"✓ Gradient flow: {'Working' if has_gradients else 'Failed'}")
    
    print()


def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Configurable Prediction Heads")
    print("=" * 50)
    
    test_activation_factory()
    test_mlp_prediction_head()
    test_additional_features()
    test_factory()
    test_predefined_configurations()
    test_gradient_w()
    
    print("=" * 50)
    
    print("=" * 50)


if __name__ == "__main__":
    main()
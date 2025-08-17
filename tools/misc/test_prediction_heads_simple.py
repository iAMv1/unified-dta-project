"""
Simple test script for configurable prediction heads
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


def test_basic_functionality():
    """Test basic prediction head functionality"""
    print("Testing basic functionality...")
    
    batch_size = 4
    input_dim = 256
    
    # Test basic MLP
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
    
    # Test GELU activation
    predictor_gelu = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[256, 128],
        activation='gelu',
        dropout=0.2
    )
    
    output_gelu = predictor_gelu(x)
    print(f"✓ GELU MLP: Input {x.shape} -> Output {output_gelu.shape}")
    
    # Test batch normalization
    predictor_bn = MLPPredictionHead(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        use_batch_norm=True,
        dropout=0.1
    )
    
    output_bn = predictor_bn(x)
    print(f"✓ BatchNorm MLP: Input {x.shape} -> Output {output_bn.shape}")
    
    print()


def test_activation_factory():
    """Test activation function factory"""
    print("Testing ActivationFactory...")
    
    activations = ['relu', 'gelu', 'leaky_relu', 'elu', 'selu', 'swish', 'tanh', 'sigmoid']
    
    for act_name in activations:
        try:
            activation = ActivationFactory.create(act_name)
            print(f"✓ {act_name}: {type(activation).__name__}")
        except Exception as e:
            print(f"✗ {act_name}: {e}")
    
    print()


def test_predefined_configurations():
    """Test predefined configuration functions"""
    print("Testing predefined configurations...")
    
    input_dim = 256
    batch_size = 4
    x = torch.randn(batch_size, input_dim)
    
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


def test_factory():
    """Test prediction head factory"""
    print("Testing PredictionHeadFactory...")
    
    input_dim = 256
    
    mlp = PredictionHeadFactory.create('mlp', input_dim=input_dim, hidden_dims=[128, 64])
    print(f"✓ Factory MLP: {type(mlp).__name__}")
    
    print()


def test_gradient_flow():
    """Test that gradients flow properly"""
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
    test_basic_functionality()
    test_factory()
    test_predefined_configurations()
    test_gradient_flow()
    
    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
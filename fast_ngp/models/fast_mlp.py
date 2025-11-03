"""
Fast MLP Network for Instant-NGP
Lightweight fully connected network optimized for speed.
"""

import torch
import torch.nn as nn


class FastMLP(nn.Module):
    """
    Fast Multi-Layer Perceptron optimized for Instant-NGP.
    
    Uses ReLU activation and optional weight normalization for stability.
    Designed to be small and fast, relying on the hash encoding for
    most of the model capacity.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension (task-specific)
        n_hidden_layers: Number of hidden layers
        hidden_dim: Hidden layer dimension
        use_bias: Whether to use bias in linear layers
        activation: Activation function ('relu', 'leaky_relu', or 'sigmoid')
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden_layers: int = 2,
        hidden_dim: int = 64,
        use_bias: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim, bias=use_bias))
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))
            layers.append(self._get_activation(activation))
        
        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim, bias=use_bias))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features [..., input_dim]
            
        Returns:
            Output [..., output_dim]
        """
        return self.network(x)
    
    def extra_repr(self):
        """Print model info."""
        return (
            f'input_dim={self.input_dim}, '
            f'output_dim={self.output_dim}, '
            f'n_hidden={self.n_hidden_layers}, '
            f'hidden_dim={self.hidden_dim}'
        )





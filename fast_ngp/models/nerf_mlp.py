"""
NeRF MLP Network for Instant-NGP
Lightweight fully connected network optimized for speed.
"""

import torch
import torch.nn as nn
from .fast_mlp import FastMLP

class NeRFMLP(nn.Module):
    """
    MLP specifically designed for NeRF tasks.
    
    Predicts RGB color and volume density from encoded position and
    optionally viewing direction.
    
    Args:
        encoding_dim: Dimension of position encoding
        dir_encoding_dim: Dimension of direction encoding (optional)
        n_hidden_layers: Number of hidden layers
        hidden_dim: Hidden dimension
        use_viewdir: Whether to use viewing direction
    """
    
    def __init__(
        self,
        encoding_dim: int,
        dir_encoding_dim: int = 0,
        n_hidden_layers: int = 2,
        hidden_dim: int = 64,
        use_viewdir: bool = True
    ):
        super().__init__()
        
        self.encoding_dim = encoding_dim
        self.dir_encoding_dim = dir_encoding_dim
        self.use_viewdir = use_viewdir and dir_encoding_dim > 0
        
        # Density network (from position encoding only)
        self.density_net = FastMLP(
            input_dim=encoding_dim,
            output_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
            activation='relu'
        )
        
        # Density head
        self.density_head = nn.Linear(hidden_dim, 1)
        
        # Color network
        if self.use_viewdir:
            # Concatenate density features with direction encoding
            self.color_net = FastMLP(
                input_dim=hidden_dim + dir_encoding_dim,
                output_dim=3,  # RGB
                n_hidden_layers=1,
                hidden_dim=hidden_dim // 2,
                activation='relu'
            )
        else:
            # Color from density features only
            self.color_net = nn.Linear(hidden_dim, 3)
        
        # Activation functions
        self.density_activation = nn.Softplus()  # Ensures positive density
        self.color_activation = nn.Sigmoid()  # RGB in [0, 1]
    
    def forward(self, x, d=None):
        """
        Forward pass for NeRF.
        
        Args:
            x: Position encoding [..., encoding_dim]
            d: Direction encoding [..., dir_encoding_dim] (optional)
            
        Returns:
            rgb: RGB color [..., 3]
            sigma: Volume density [..., 1]
        """
        # Get density features
        density_features = self.density_net(x)
        
        # Predict density
        sigma = self.density_head(density_features)
        sigma = self.density_activation(sigma)
        
        # Predict color
        if self.use_viewdir and d is not None:
            # Concatenate with direction
            color_input = torch.cat([density_features, d], dim=-1)
            rgb = self.color_net(color_input)
        else:
            rgb = self.color_net(density_features)
        
        rgb = self.color_activation(rgb)
        
        return rgb, sigma

"""
GigaPixel MLP Network for Instant-NGP
Lightweight fully connected network optimized for speed.
"""

import torch
import torch.nn as nn 

class GigapixelMLP(nn.Module):
    """
    MLP for gigapixel image representation.
    
    Maps 2D coordinates to RGB color.
    
    Args:
        encoding_dim: Dimension of coordinate encoding
        n_hidden_layers: Number of hidden layers
        hidden_dim: Hidden dimension
    """
    
    def __init__(
        self,
        encoding_dim: int,
        n_hidden_layers: int = 2,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.color_net = FastMLP(
            input_dim=encoding_dim,
            output_dim=3,  # RGB
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
            activation='relu'
        )
        
        self.color_activation = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass for gigapixel image.
        
        Args:
            x: Coordinate encoding [..., encoding_dim]
            
        Returns:
            rgb: RGB color [..., 3]
        """
        rgb = self.color_net(x)
        return self.color_activation(rgb)

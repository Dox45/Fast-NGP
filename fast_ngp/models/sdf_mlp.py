import torch 
import torch.nn as nn 

class SDFMLP(nn.Module):
    """
    MLP for Signed Distance Function (SDF) prediction.
    
    Predicts signed distance from encoded position.
    
    Args:
        encoding_dim: Dimension of position encoding
        n_hidden_layers: Number of hidden layers
        hidden_dim: Hidden dimension
    """
    
    def __init__(
        self,
        encoding_dim: int,
        n_hidden_layers: int = 3,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.sdf_net = FastMLP(
            input_dim=encoding_dim,
            output_dim=1,  # Signed distance
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
            activation='relu'
        )
    
    def forward(self, x):
        """
        Forward pass for SDF.
        
        Args:
            x: Position encoding [..., encoding_dim]
            
        Returns:
            sdf: Signed distance [..., 1]
        """
        return self.sdf_net(x)

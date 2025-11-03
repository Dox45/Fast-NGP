"""
Complete NeRF Model implementation 
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from fast_ngp.encoding.multi_hash_enc import MultiresHashEncoding
from fast_ngp.models.nerf_mlp import NeRFMLP
from fast_ngp.rendering.volume_render import VolumeRenderer

class FastNGP_NeRF(nn.Module):
    """
    Complete Instant-NGP NeRF model.
    
    Combines multiresolution hash encoding with a small MLP for
    fast NeRF training and inference.
    """
    
    def __init__(
        self,
        encoding_config: Optional[Dict] = None,
        mlp_config: Optional[Dict] = None,
        use_viewdir: bool = True,
        bound: float = 1.0
    ):
        super().__init__()
        
        # Default configurations
        if encoding_config is None:
            encoding_config = {
                'n_levels': 16,
                'n_features_per_level': 2,
                'log2_hashmap_size': 19,
                'base_resolution': 16,
                'finest_resolution': 512
            }
        
        if mlp_config is None:
            mlp_config = {
                'n_hidden_layers': 2,
                'hidden_dim': 64
            }
        
        self.use_viewdir = use_viewdir
        self.bound = bound  # Scene bounding box [-bound, bound]^3

        self.position_encoder = MultiresHashEncoding(**encoding_config)
        pos_encoding_dim = self.position_encoder.get_output_dim()
        
        # Direction encoding (simple spherical harmonics)
        if use_viewdir:
            self.dir_encoding_dim = 16
        else:
            self.dir_encoding_dim = 0
                
        self.network = NeRFMLP(
            encoding_dim=pos_encoding_dim,
            dir_encoding_dim=self.dir_encoding_dim,
            use_viewdir=use_viewdir,
            **mlp_config
        )
        
        # Renderer
        self.renderer = VolumeRenderer()
    
    def encode_position(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D position with hash encoding.
        
        Args:
            x: 3D positions [..., 3] in world coordinates
            
        Returns:
            Encoded features [..., encoding_dim]
        """
        # Normalize to [0, 1] based on bounding box
        x_normalized = (x + self.bound) / (2 * self.bound)
        x_normalized = torch.clamp(x_normalized, 0, 1)
        
        return self.position_encoder(x_normalized)
    
    def encode_direction(self, d: torch.Tensor) -> torch.Tensor:
        """
        Encode viewing direction with spherical harmonics.
        
        Args:
            d: Viewing directions [..., 3]
            
        Returns:
            Encoded directions [..., dir_encoding_dim]
        """
        if not self.use_viewdir:
            return None
        
        # Normalize directions
        d = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-8)
        
        # Simple spherical harmonics (degree 2)
        # This is a simplified version
        x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]
        
        features = [
            torch.ones_like(x),  # constant
            x, y, z,             # degree 1
            x*y, x*z, y*z,       # degree 2
            x*x, y*y, z*z        # degree 2
        ]
        
        # Pad to dir_encoding_dim
        while len(features) < self.dir_encoding_dim:
            features.append(torch.zeros_like(x))
        
        return torch.cat(features[:self.dir_encoding_dim], dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        d: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for NeRF.
        
        Args:
            x: 3D positions [..., 3]
            d: Viewing directions [..., 3] (optional)
            
        Returns:
            rgb: RGB color [..., 3]
            sigma: Volume density [..., 1]
        """
        # Encode position
        pos_encoding = self.encode_position(x)
        
        # Encode direction if using viewdir
        dir_encoding = None
        if self.use_viewdir and d is not None:
            # Expand direction to match position shape
            if d.shape[:-1] != x.shape[:-1]:
                # d is per-ray [N, 3], x is per-sample [N, n_samples, 3]
                # Expand d to [N, n_samples, 3]
                n_samples = x.shape[-2]
                d = d.unsqueeze(-2).expand(*d.shape[:-1], n_samples, d.shape[-1])
            dir_encoding = self.encode_direction(d)
        
        # Query network
        rgb, sigma = self.network(pos_encoding, dir_encoding)
        
        return rgb, sigma
    
    def render_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Render a batch of rays.
        
        Args:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            **kwargs: Additional arguments for renderer
            
        Returns:
            Dictionary with rgb, depth, etc.
        """
        return self.renderer(self, rays_o, rays_d, **kwargs)


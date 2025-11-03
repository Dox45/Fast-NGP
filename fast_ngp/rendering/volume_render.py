"""
Volume Renderer component for FastNGP
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class VolumeRenderer(nn.Module):
    """
    Volume rendering for NeRF using ray marching.
    
    Implements the volume rendering equation from the NeRF paper.
    """
    
    def __init__(
        self,
        n_samples: int = 64,
        n_importance: int = 64,
        perturb: bool = True,
        near: float = 0.0,
        far: float = 1.0,
        use_white_bkgd: bool = False
    ):
        super().__init__()
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.perturb = perturb
        self.near = near
        self.far = far
        self.use_white_bkgd = use_white_bkgd
    
    def sample_along_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        n_samples: int,
        perturb: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays.
        
        Args:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            n_samples: Number of samples per ray
            perturb: Add random perturbation to samples
            
        Returns:
            pts: Sampled points [N, n_samples, 3]
            z_vals: Depth values [N, n_samples]
        """
        N = rays_o.shape[0]
        
        # Linearly sample in disparity space
        t_vals = torch.linspace(0., 1., steps=n_samples, device=rays_o.device)
        z_vals = self.near * (1. - t_vals) + self.far * t_vals
        z_vals = z_vals.expand(N, n_samples)
        
        # Perturb sampling along each ray
        if perturb:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=rays_o.device)
            z_vals = lower + (upper - lower) * t_rand
        
        # Points in space: rays_o + z_vals * rays_d
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        return pts, z_vals
    
    def volume_rendering(
        self,
        rgb: torch.Tensor,
        sigma: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        noise_std: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Composite RGB and density along rays.
        
        Args:
            rgb: RGB values [N, n_samples, 3]
            sigma: Density values [N, n_samples, 1]
            z_vals: Depth values [N, n_samples]
            rays_d: Ray directions [N, 3]
            noise_std: Add noise to sigma for regularization
            
        Returns:
            rgb_map: Final RGB image [N, 3]
            depth_map: Depth map [N]
            weights: Sample weights [N, n_samples]
        """
        # Add noise to sigma during training
        if noise_std > 0:
            noise = torch.randn_like(sigma) * noise_std
            sigma = sigma + noise
        
        # Compute distances between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([
            dists,
            torch.full_like(dists[..., :1], 1e10)  # Last distance is infinity
        ], dim=-1)
        
        # Multiply by ray direction norm for proper scaling
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # Compute alpha (probability of ray termination)
        alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)
        
        # Compute transmittance (cumulative product of (1 - alpha))
        # T_i = exp(-sum(sigma * delta) from 0 to i-1)
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones_like(alpha[..., :1]),
                1.0 - alpha[..., :-1] + 1e-10
            ], dim=-1),
            dim=-1
        )
        
        # Weights for each sample
        weights = alpha * transmittance
        
        # Composite RGB
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
        
        # Expected depth
        depth_map = torch.sum(weights * z_vals, dim=-1)
        
        # Add white background if requested
        if self.use_white_bkgd:
            acc_map = torch.sum(weights, dim=-1)
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        
        return rgb_map, depth_map, weights
    
    def forward(
        self,
        model: nn.Module,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        noise_std: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Render rays through the NeRF model.
        
        Args:
            model: NeRF model
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            noise_std: Noise standard deviation
            
        Returns:
            Dictionary containing rgb_map, depth_map, etc.
        """
        # Coarse sampling
        pts, z_vals = self.sample_along_rays(
            rays_o, rays_d, self.n_samples, self.perturb
        )
        
        # Query model
        rgb, sigma = model(pts, rays_d)
        
        # Volume rendering
        rgb_map, depth_map, weights = self.volume_rendering(
            rgb, sigma, z_vals, rays_d, noise_std
        )
        
        return {
            'rgb': rgb_map,
            'depth': depth_map,
            'weights': weights,
            'z_vals': z_vals
        }

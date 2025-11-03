"""
Multiresolution Hash Encoding Module
Core component of Instant-NGP that enables fast training and inference.

Paper: Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
"""

import torch
import torch.nn as nn
import numpy as np


class MultiresHashEncoding(nn.Module):
    """
    Multiresolution hash encoding with learnable feature grids.
    
    This is the core innovation of Instant-NGP that replaces traditional
    positional encoding with a hierarchy of trainable hash tables.
    
    Args:
        n_levels: Number of resolution levels (L)
        n_features_per_level: Feature dimension per level (F)
        log2_hashmap_size: Log2 of hash table size per level
        base_resolution: Coarsest resolution (N_min)
        finest_resolution: Finest resolution (N_max)
        interpolation: Interpolation method ('linear' or 'smoothstep')
    """
    
    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        interpolation: str = 'linear'
    ):
        super().__init__()
        
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.interpolation = interpolation
        
        # Calculate growth factor per level
        self.b = np.exp(
            (np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1)
        )
        
        # Calculate resolution for each level
        self.resolutions = [
            int(np.floor(base_resolution * (self.b ** l)))
            for l in range(n_levels)
        ]
        
        # Hash table size
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # Create hash tables for each level (learnable parameters)
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        
        # Initialize embeddings with uniform distribution
        for embedding in self.embeddings:
            nn.init.uniform_(embedding.weight, -1e-4, 1e-4)
        
        # Precompute prime numbers for hashing
        self.primes = [1, 2654435761, 805459861]  # Large primes for 3D hashing
    
    def hash_function(self, coords, resolution):
        """
        Spatial hash function to map coordinates to hash table indices.
        
        Args:
            coords: Integer coordinates [..., 3]
            resolution: Grid resolution for this level
            
        Returns:
            Hash indices [..., n_vertices] where n_vertices = 8 for 3D
        """
        # Get all 8 corners of the voxel containing the point
        coords = coords.long()
        
        # Compute hash for each corner
        # Use XOR with large prime numbers for better distribution
        hashes = torch.zeros(
            *coords.shape[:-1], 8,
            dtype=torch.long,
            device=coords.device
        )
        
        for i in range(8):
            # Get corner offsets (binary representation of i)
            offset_x = (i >> 0) & 1
            offset_y = (i >> 1) & 1
            offset_z = (i >> 2) & 1
            
            corner = coords.clone()
            corner[..., 0] += offset_x
            corner[..., 1] += offset_y
            corner[..., 2] += offset_z
            
            # Hash function: XOR of coordinate * prime
            h = torch.zeros_like(corner[..., 0])
            h = h ^ (corner[..., 0] * self.primes[0])
            h = h ^ (corner[..., 1] * self.primes[1])
            h = h ^ (corner[..., 2] * self.primes[2])
            
            hashes[..., i] = h % self.hashmap_size
        
        return hashes
    
    def trilinear_interp(self, features, local_coords):
        """
        Trilinear interpolation of features at voxel corners.
        
        Args:
            features: Features at 8 corners [..., 8, F]
            local_coords: Coordinates within voxel [0,1]^3 [..., 3]
            
        Returns:
            Interpolated features [..., F]
        """
        x, y, z = local_coords[..., 0:1], local_coords[..., 1:2], local_coords[..., 2:3]
        
        # Apply smoothstep if requested
        if self.interpolation == 'smoothstep':
            x = x * x * (3 - 2 * x)
            y = y * y * (3 - 2 * y)
            z = z * z * (3 - 2 * z)
        
        # Trilinear interpolation weights
        c000 = (1 - x) * (1 - y) * (1 - z)
        c001 = (1 - x) * (1 - y) * z
        c010 = (1 - x) * y * (1 - z)
        c011 = (1 - x) * y * z
        c100 = x * (1 - y) * (1 - z)
        c101 = x * (1 - y) * z
        c110 = x * y * (1 - z)
        c111 = x * y * z
        
        weights = torch.cat([c000, c001, c010, c011, c100, c101, c110, c111], dim=-1)
        
        # Weighted sum: [..., 8, 1] * [..., 8, F] -> [..., F]
        result = torch.sum(weights.unsqueeze(-1) * features, dim=-2)
        
        return result
    
    def forward(self, x):
        """
        Encode input coordinates with multiresolution hash encoding.
        
        Args:
            x: Input coordinates in [0, 1]^3, shape [..., 3]
            
        Returns:
            Encoded features, shape [..., n_levels * n_features_per_level]
        """
        # Store original shape
        original_shape = x.shape[:-1]
        x = x.reshape(-1, 3)  # Flatten to [N, 3]
        
        encoded_features = []
        
        for level, embedding in enumerate(self.embeddings):
            resolution = self.resolutions[level]
            
            # Scale coordinates to grid resolution
            scaled_coords = x * resolution
            
            # Get integer and fractional parts
            grid_coords = torch.floor(scaled_coords)
            local_coords = scaled_coords - grid_coords
            
            # Get hash indices for 8 corners
            hash_indices = self.hash_function(grid_coords, resolution)
            
            # Look up features from hash table
            corner_features = embedding(hash_indices)  # [N, 8, F]
            
            # Interpolate
            level_features = self.trilinear_interp(corner_features, local_coords)
            
            encoded_features.append(level_features)
        
        # Concatenate all levels
        result = torch.cat(encoded_features, dim=-1)
        
        # Reshape back to original shape
        result = result.reshape(*original_shape, -1)
        
        return result
    
    def get_output_dim(self):
        """Get the output feature dimension."""
        return self.n_levels * self.n_features_per_level
    
    def extra_repr(self):
        """Print model info."""
        return (
            f'n_levels={self.n_levels}, '
            f'n_features_per_level={self.n_features_per_level}, '
            f'hashmap_size={self.hashmap_size}, '
            f'base_res={self.base_resolution}, '
            f'finest_res={self.finest_resolution}, '
            f'output_dim={self.get_output_dim()}'
        )


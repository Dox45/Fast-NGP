"""
Dataset for Fast-NGP NeRF
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from typing import Dict, Optional
import time

class SimpleNeRFDataset(Dataset):
    """
    Simple NeRF dataset for synthetic data (Blender format).
    Loads images and camera poses from a transforms.json file.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_wh: tuple = (800, 800),
        white_back: bool = True
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_w, self.img_h = img_wh
        self.white_back = white_back
        
        # Load transforms
        with open(os.path.join(root_dir, f'transforms_{split}.json'), 'r') as f:
            meta = json.load(f)
        
        # Camera intrinsics
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = 0.5 * self.img_w / np.tan(0.5 * camera_angle_x)
        
        # Load images and poses
        self.images = []
        self.poses = []
        
        for frame in meta['frames']:
            # Load image
            img_path = os.path.join(root_dir, frame['file_path'] + '.png')
            # In real implementation, load actual image
            # For now, create dummy data
            img = np.random.rand(self.img_h, self.img_w, 3)
            self.images.append(img)
            
            # Load pose
            pose = np.array(frame['transform_matrix'])[:3, :4]
            self.poses.append(pose)
        
        self.images = np.stack(self.images, 0)
        self.poses = np.stack(self.poses, 0)
        
        # Generate rays for all images
        self.all_rays, self.all_rgbs = self.generate_rays()
    
    def generate_rays(self):
        """Generate all rays for the dataset."""
        rays_o_list = []
        rays_d_list = []
        rgbs_list = []
        
        for img, pose in zip(self.images, self.poses):
            rays_o, rays_d = self.get_rays(pose)
            
            rays_o_list.append(rays_o)
            rays_d_list.append(rays_d)
            rgbs_list.append(img.reshape(-1, 3))
        
        rays_o = np.concatenate(rays_o_list, 0)
        rays_d = np.concatenate(rays_d_list, 0)
        rgbs = np.concatenate(rgbs_list, 0)
        
        rays = np.concatenate([rays_o, rays_d], 1)  # [N, 6]
        
        return torch.FloatTensor(rays), torch.FloatTensor(rgbs)
    
    def get_rays(self, pose):
        """Get rays for a single image."""
        # Create pixel coordinates
        i, j = np.meshgrid(
            np.arange(self.img_w, dtype=np.float32),
            np.arange(self.img_h, dtype=np.float32),
            indexing='xy'
        )
        
        # Convert to camera coordinates
        dirs = np.stack([
            (i - self.img_w * 0.5) / self.focal,
            -(j - self.img_h * 0.5) / self.focal,
            -np.ones_like(i)
        ], -1)
        
        # Rotate ray directions from camera to world
        rays_d = np.sum(dirs[..., None, :] * pose[:3, :3], -1)
        
        # Origin is same for all rays
        rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
        
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    @classmethod
    def from_tensors(cls, rays_o, rays_d, t_vals, rgbs):
        """
        Utility constructor to build a SimpleNeRFDataset directly from tensors.
        Used for quick testing without loading images or JSON.
        """
        dataset = cls.__new__(cls)
        dataset.rays_o = rays_o
        dataset.rays_d = rays_d
        dataset.t_vals = t_vals
        dataset.all_rays = torch.cat([rays_o, rays_d], dim=-1)
        dataset.all_rgbs = rgbs
        dataset.split = 'train'
        return dataset


    @classmethod
    def from_npz(cls, npz_path):
        data = np.load(npz_path)
        images = data['images']      # [N,H,W,3]
        poses = data['poses']        # [N,4,4]
        focal = float(data['focal'])
        
        # Create an instance without calling __init__
        dataset = cls.__new__(cls)
        dataset.images = images
        dataset.poses = poses
        dataset.focal = focal
        dataset.img_h, dataset.img_w = images.shape[1:3]
        
        # Generate rays using your class logic
        dataset.all_rays, dataset.all_rgbs = dataset.generate_rays()
        
        return dataset


    
    def __len__(self):
        return len(self.all_rays)
    
    def __getitem__(self, idx):
        return {
            'rays': self.all_rays[idx],
            'rgbs': self.all_rgbs[idx]
        }

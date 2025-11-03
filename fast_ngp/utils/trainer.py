"""
Training script for Instant-NGP NeRF
Simple, clean training loop with logging and checkpointing.
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



class NeRFTrainer:
    """
    Trainer for Instant-NGP NeRF.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 8192,
        lr: float = 1e-2,
        num_epochs: int = 20,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_every: int = 100
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_every = log_every
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-15
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.95
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Metrics tracking
        self.train_losses = []
        self.val_psnrs = []
    
    def compute_psnr(self, pred, target):
        """Compute PSNR metric."""
        mse = torch.mean((pred - target) ** 2)
        psnr = -10 * torch.log10(mse)
        return psnr.item()
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs}')
        epoch_loss = 0
        
        for i, batch in enumerate(pbar):
            rays = batch['rays'].to(self.device)  # [B, 6]
            rgbs_gt = batch['rgbs'].to(self.device)  # [B, 3]
            
            # Split rays into origin and direction
            rays_o = rays[:, :3]
            rays_d = rays[:, 3:]
            
            # Render
            outputs = self.model.render_rays(rays_o, rays_d)
            rgbs_pred = outputs['rgb']
            
            # Compute loss
            loss = self.criterion(rgbs_pred, rgbs_gt)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            
            if i % self.log_every == 0:
                psnr = self.compute_psnr(rgbs_pred, rgbs_gt)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr:.2f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Run validation."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        
        total_psnr = 0
        n_samples = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            rays = batch['rays'].to(self.device)
            rgbs_gt = batch['rgbs'].to(self.device)
            
            rays_o = rays[:, :3]
            rays_d = rays[:, 3:]
            
            outputs = self.model.render_rays(rays_o, rays_d)
            rgbs_pred = outputs['rgb']
            
            psnr = self.compute_psnr(rgbs_pred, rgbs_gt)
            total_psnr += psnr * len(rays)
            n_samples += len(rays)
        
        avg_psnr = total_psnr / n_samples
        self.val_psnrs.append(avg_psnr)
        
        return avg_psnr
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_psnrs': self.val_psnrs
        }
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        
        best_psnr = 0
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            if self.val_loader is not None:
                val_psnr = self.validate()
                print(f"Epoch {epoch}: Loss = {train_loss:.4f}, Val PSNR = {val_psnr:.2f} dB")
                
                # Save best model
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    self.save_checkpoint(epoch, is_best=True)
            else:
                print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
            
            # Save regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch)
            
            # Update learning rate
            self.scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/60:.2f} minutes")
        print(f"Best validation PSNR: {best_psnr:.2f} dB")


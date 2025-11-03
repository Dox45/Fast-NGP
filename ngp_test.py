import os
import torch
from fast_ngp.models.fast_nerf import FastNGP_NeRF
from fast_ngp.utils.trainer import NeRFTrainer
from fast_ngp.utils.dataset import SimpleNeRFDataset  # import your class

# Download NPZ if missing
if not os.path.exists('tiny_nerf_data.npz'):
    os.system("wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz")

# Create dataset using your class
dataset = SimpleNeRFDataset.from_npz('tiny_nerf_data.npz')
print(f"✅ Dataset ready with {len(dataset)} rays")

# Model
model = FastNGP_NeRF(
    encoding_config={'n_levels': 8, 'log2_hashmap_size': 15},
    mlp_config={'n_hidden_layers': 2, 'hidden_dim': 32}
)
print("✅ Model initialized")

# Trainer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = NeRFTrainer(
    model=model,
    train_dataset=dataset,
    val_dataset=None,
    batch_size=64,
    lr=1e-2,
    num_epochs=1,
    device=device
)

# Train
trainer.train()
print("🎉 Training complete using your SimpleNeRFDataset on tiny_nerf_data.npz")

# Fast-NGP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/pyinstant-ngp.svg)](https://badge.fury.io/py/pyinstant-ngp)

A clean, pure PyTorch implementation of NVIDIA's **Instant Neural Graphics Primitives** with multiresolution hash encoding.

## Features

✨ **Pure PyTorch** - No CUDA extensions required, easy to install and modify  
📦 **Easy Installation** - `pip install pyinstant-ngp`  
🎯 **Modular Design** - Each component usable independently  
🚀 **Multiple Tasks** - NeRF, SDF, and Gigapixel image support  
📊 **Visualization Tools** - Built-in rendering and metrics  
🧪 **Well Tested** - Comprehensive test suite  
📚 **Great Docs** - Extensive documentation and examples  

## Installation

### From PyPI (recommended)
```bash
pip install fast-ngp
```

### From source
```bash
git clone https://github.com/Dox45/fast-ngp.git
cd fast-ngp
pip install -e .
```

## Quick Start

### Train a NeRF

```python
import torch
from fast_ngp.models.fast_nerf import FastNGP_NeRF
from fast_ngp.utils.dataset import SimpleNeRFDataset
from fast_ngp.utils.trainer import NeRFTrainer

dataset_path = 'path/to/data'  # directory containing transforms_train.json or tiny_nerf_data.npz
dataset = SimpleNeRFDataset(root_dir=dataset_path, split='train', img_wh=(128, 128))
print(f"Dataset ready with {len(dataset)} rays")

# inti model
model = FastNGP_NeRF(
    encoding_config={
        'n_levels': 16,
        'n_features_per_level': 2,
        'log2_hashmap_size': 19,
        'base_resolution': 16,
        'finest_resolution': 512
    },
    mlp_config={
        'n_hidden_layers': 2,
        'hidden_dim': 64
    }
)
print("✅ Model initialized")

# Train

device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = NeRFTrainer(
    model=model,
    train_dataset=dataset,
    val_dataset=None,  # you can add a validation dataset if available
    batch_size=64,
    lr=1e-2,
    num_epochs=5,
    device=device
)

trainer.train()
print("🎉 Training complete!")

# Provide a camera pose as a 3x4 matrix (numpy or torch)
import numpy as np

camera_pose = np.eye(4)[:3, :4]  # identity pose for testing
H, W = 128, 128

# The render_image method should be implemented in FastNGP_NeRF
with torch.no_grad():
    image = model.render_image(camera_pose, H=H, W=W)
print("Rendered image shape:", image.shape)
```

### Command Line Interface

```bash
# Train NeRF
fast-ngp-train --task nerf --data path/to/data --config configs/nerf.yaml

# Render from trained model
fast-ngp-render --checkpoint model.pth --output renders/
```

## Architecture

```
pyinstant-ngp/
├── encoding/          # Multiresolution hash encoding
├── models/          # MLP networks
├── rendering/         # Ray marching and rendering
├── example/              # Dataset loaders
└── utils/             # Dataset loaders and trainer
```

## Citation

If you use this code in your research, please cite both the original paper and this implementation:

```bibtex
@article{mueller2022instant,
    title={Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    author={M\"uller, Thomas and Evans, Alex and Schied, Christoph and Keller, Alexander},
    journal={ACM Transactions on Graphics (ToG)},
    volume={41},
    number={4},
    pages={1--15},
    year={2022},
    publisher={ACM}
}

@software{fast_ngp,
    author = {Chima Emmanuel},
    title = {Fast-NGP: Pure PyTorch Implementation of Instant-NGP},
    year = {2025},
    url = {https://github.com/Dox45/fast-ngp}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original [Instant-NGP](https://github.com/NVlabs/instant-ngp) by NVIDIA
- Inspired by [torch-ngp](https://github.com/ashawkey/torch-ngp) and [HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

- 📖 [Documentation](https://fast-ngp.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/Dox45/fast-ngp/issues)
- 💬 [Discussions](https://github.com/Dox45/fast-ngp/discussions)

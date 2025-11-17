# H-JEPA: Hierarchical Joint-Embedding Predictive Architecture

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A PyTorch implementation of Hierarchical Joint-Embedding Predictive Architecture for visual self-supervised learning**

## Overview

H-JEPA (Hierarchical Joint-Embedding Predictive Architecture) extends the [I-JEPA](https://github.com/facebookresearch/ijepa) framework with hierarchical processing capabilities for multi-scale feature learning.

### Key Features

- **Multi-Scale Hierarchical Learning** - Learns representations at multiple levels (fine to coarse)
- **Advanced Components**:
  - Feature Pyramid Networks (FPN) for multi-scale fusion
  - RoPE (Rotary Position Embeddings) for positional encoding
  - LayerScale for training stability
  - Flash Attention support (CUDA only)
  - SigReg loss for training stability
  - Multi-crop data augmentation
  - Combined loss functions (VICReg + H-JEPA)
- **Vision Transformer Backbone** - Built on ViT architectures from `timm`
- **Efficient Training** - Mixed precision support, gradient checkpointing
- **Apple Silicon Support** - Optimized for MPS devices

## Installation

### Prerequisites

- Python 3.11 or higher
- PyTorch 2.0+
- CUDA 11.7+ (optional, for GPU) or Apple Silicon Mac

### Setup

```bash
# Clone the repository
git clone https://github.com/jonwiggins/H-JEPA.git
cd H-JEPA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('Device:', 'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU')"
```

## Quick Start

### Training

```bash
# Train on CIFAR-10 (auto-downloads)
python scripts/train.py --config configs/default.yaml

# Train on ImageNet-100
python scripts/train.py --config configs/imagenet100.yaml

# Debug/test configuration (minimal)
python scripts/train.py --config configs/debug_minimal.yaml

# Apple Silicon optimized
python scripts/train.py --config configs/mps_optimized.yaml
```

### Evaluation

```bash
# Linear probe evaluation
python scripts/eval_linear_probe.py --checkpoint path/to/checkpoint.pth

# k-NN evaluation
python scripts/eval_knn.py --checkpoint path/to/checkpoint.pth
```

### Visualization

```bash
# Visualize attention maps
python scripts/visualize_attention.py --checkpoint path/to/checkpoint.pth

# Visualize features
python scripts/visualize_features.py --checkpoint path/to/checkpoint.pth
```

## Project Structure

```
H-JEPA/
├── src/
│   ├── models/         # Model architectures (encoder, predictor, H-JEPA)
│   ├── losses/         # Loss functions (VICReg, SigReg, combined)
│   ├── masks/          # Masking strategies
│   ├── data/           # Datasets and transforms
│   ├── trainers/       # Training loops
│   ├── evaluation/     # Evaluation protocols
│   └── utils/          # Utilities (logging, checkpointing)
├── configs/            # YAML configuration files
├── scripts/            # Training and evaluation scripts
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Configuration

Training is configured via YAML files in `configs/`. Key parameters:

```yaml
model:
  encoder_type: "vit_base_patch16_224"
  num_hierarchies: 3      # Number of hierarchy levels
  use_fpn: true           # Feature Pyramid Network
  use_rope: true          # Rotary Position Embeddings

training:
  epochs: 100
  batch_size: 256
  learning_rate: 1.5e-4
  use_amp: true          # Mixed precision (not on MPS)

loss:
  type: "combined"       # or "vicreg", "sigreg", "mse"
  hierarchy_weights: [1.0, 0.7, 0.5]
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_ijepa_compliance.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Docker Support

```bash
# Build Docker image
docker build -t hjepa .

# Run training in container
docker run --gpus all -v $(pwd)/data:/app/data hjepa python scripts/train.py

# Run with Docker Compose
docker-compose up
```

## Known Issues

- Flash Attention is disabled on Apple Silicon (MPS) due to performance issues
- Mixed precision training not supported on MPS
- SVD operations fall back to CPU on MPS

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Run tests and ensure they pass
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Citation

If you use this code, please cite:

```bibtex
@software{hjepa2024,
  title={H-JEPA: Hierarchical Joint-Embedding Predictive Architecture},
  author={Wiggins, Jon and Contributors},
  year={2024},
  url={https://github.com/jonwiggins/H-JEPA}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on [I-JEPA](https://github.com/facebookresearch/ijepa) by Meta AI
- Vision Transformers from [timm](https://github.com/rwightman/pytorch-image-models)
- Thanks to all contributors

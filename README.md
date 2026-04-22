# H-JEPA: Hierarchical Joint-Embedding Predictive Architecture

[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20|%203.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/jonwiggins/H-JEPA/branch/main/graph/badge.svg)](https://codecov.io/gh/jonwiggins/H-JEPA)

**A hierarchical JEPA framework supporting masked-patch SSL, end-to-end SIGReg-only training (LeWM-style), and action-conditioned world modeling**

## Overview

H-JEPA is a hierarchical Joint-Embedding Predictive Architecture framework with three opt-in training regimes that share the same encoder and SIGReg primitives:

1. **Masked-patch SSL** (default) — extends Meta's [I-JEPA](https://github.com/facebookresearch/ijepa) with a Feature Pyramid Network that fuses representations across multiple hierarchy levels for both fine-grained and coarse semantic features.
2. **End-to-end SIGReg-only training** — drops the EMA target encoder and relies on [LeJEPA](https://arxiv.org/abs/2511.08544) / [LeWorldModel](https://arxiv.org/abs/2603.19312)-style SIGReg regularization to prevent collapse, halving trainable parameters and reducing tunable loss hyperparameters from six to one.
3. **Action-conditioned world modeling** — adds an AdaLN-Zero predictor and CEM latent-space planner so the same encoder backbone can be trained on action-annotated trajectories for control.

All three regimes are selectable via config — defaults preserve the original H-JEPA behavior. The architecture uses Rotary Position Embeddings (RoPE) for spatial awareness, multi-crop augmentation for scale invariance, and supports CUDA, Apple Silicon (MPS), and CPU backends.

The implementation is validated end-to-end with 1400+ tests. No pretrained weights are published yet — this is a research codebase for training from scratch.

## Key Features

- **Multi-scale hierarchy** — learn representations at multiple levels with configurable depth
- **Feature Pyramid Network** — fuse features across hierarchy levels
- **Rotary Position Embeddings** — 2D spatial encoding without learned position parameters
- **Flash Attention** — fused attention kernels on CUDA
- **SIGReg loss** — sketched isotropic Gaussian regularization with two Epps–Pulley variants (reference points and characteristic-function quadrature)
- **End-to-end mode** — opt-out of the EMA target encoder per LeWorldModel
- **BatchNorm projection heads** — preserves the variance signal SIGReg shapes
- **AdaLN-Zero action predictor** — zero-initialized action conditioning for stable world-model training
- **CEM latent-space planner** — MPC + Cross-Entropy Method for downstream control
- **Multi-crop augmentation** — multiple views at different scales per image
- **VICReg + prediction loss** — combined objective for the classic SSL regime
- **Apple Silicon support** — runs on MPS with automatic fallbacks
- **Mixed precision & gradient checkpointing** — efficient training on limited hardware

## Training regimes at a glance

| Config | EMA target encoder | Loss | Use case |
|---|---|---|---|
| `configs/default.yaml` | Yes | VICReg + prediction (current default) | Hierarchical masked-patch SSL |
| `configs/lewm_tier1.yaml` | Yes | SIGReg + prediction | SSL with vectorized SIGReg, char-function test, BatchNorm projector |
| `configs/lewm_tier2.yaml` | **No** | SIGReg-only, no detach | SSL without EMA — LeWM-style end-to-end training of the hierarchical stack |
| `configs/lewm_world_model.yaml` | **No** | SIGReg + next-latent prediction | Action-conditioned world model with AdaLN-Zero predictor + CEM planner |

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

> **Note (MPS):** Flash Attention and mixed precision (AMP) are unavailable on Apple Silicon — the code falls back to standard attention and full precision automatically. SVD operations also fall back to CPU due to PyTorch MPS limitations.

## Quick Start

### Training

```bash
# Train on CIFAR-10 (auto-downloads)
python scripts/train.py --config configs/default.yaml

# Train on ImageNet-100
python scripts/train.py --config configs/imagenet100.yaml

# Apple Silicon optimized
python scripts/train.py --config configs/mps_optimized.yaml

# Debug/test configuration (minimal)
python scripts/train.py --config configs/debug_minimal.yaml
```

### Evaluation

```bash
# Linear probe evaluation
python scripts/eval_linear_probe.py --checkpoint path/to/checkpoint.pth

# k-NN evaluation
python scripts/eval_knn.py --checkpoint path/to/checkpoint.pth
```

See [docs/TRAINING.md](docs/TRAINING.md) for the full training guide and [docs/EVALUATION.md](docs/EVALUATION.md) for evaluation details.

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
│   ├── visualization/  # Attention and feature visualization
│   ├── serving/        # Model serving utilities
│   ├── inference/      # Inference pipelines
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
  use_amp: true           # Mixed precision (CUDA only)

loss:
  type: "combined"        # or "vicreg", "sigreg", "mse"
  hierarchy_weights: [1.0, 0.7, 0.5]
```

## Testing

The test suite contains **1400+ tests** across 30+ test modules, using pytest.

```bash
# Quick subset (skip slow tests)
pytest tests/ -m "not slow" -v

# Full suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

Tests cover all core modules — models, losses, masks, data, trainers, evaluation, and utilities — with mocked hardware backends so the full suite runs on CPU, CUDA, or MPS.

**CI pipeline** (GitHub Actions, every push/PR):
1. **black** — code formatting
2. **ruff** — linting
3. **mypy** — static type checking
4. **pytest** — full test suite with coverage

**Pre-commit hooks** (install with `pre-commit install`): black, ruff, mypy.

See [docs/testing.md](docs/testing.md) for the full testing guide.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ViT-Tiny (5.5M params context + 5.5M target EMA) |
| **Predictor** | 4-layer transformer (2.8M params) |
| **Total** | ~13.8M parameters (8.3M trainable) |
| **Hierarchies** | 3 levels with FPN fusion (128-ch) |
| **Embed dim** | 192 |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a detailed architecture description.

## Docker

```bash
# Build Docker image
docker build -t hjepa .

# Run training in container
docker run --gpus all -v $(pwd)/data:/app/data hjepa python scripts/train.py

# Run with Docker Compose
docker-compose up
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{hjepa2025,
  title={H-JEPA: Hierarchical Joint-Embedding Predictive Architecture},
  author={Wiggins, Jon and Contributors},
  year={2025},
  url={https://github.com/jonwiggins/H-JEPA}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Based on [I-JEPA](https://github.com/facebookresearch/ijepa) by Meta AI
- Vision Transformers from [timm](https://github.com/rwightman/pytorch-image-models)

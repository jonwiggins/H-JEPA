# H-JEPA: Hierarchical Joint-Embedding Predictive Architecture

A PyTorch implementation of Hierarchical Joint-Embedding Predictive Architecture (H-JEPA), a self-supervised learning method for visual representation learning.

## Overview

H-JEPA (Hierarchical Joint-Embedding Predictive Architecture) is an advanced self-supervised learning approach that learns visual representations by predicting masked regions in image embeddings across multiple hierarchical levels. This implementation extends the original I-JEPA framework with hierarchical processing capabilities.

### Key Features

- **Hierarchical Multi-Scale Processing**: Learns representations at multiple scales simultaneously
- **Multi-Block Masking Strategy**: Advanced masking technique for robust feature learning
- **Vision Transformer Backbone**: Leverages state-of-the-art ViT architectures
- **Efficient Training**: Optimized for modern GPUs with mixed precision support
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Comprehensive Logging**: Integration with W&B and TensorBoard for experiment tracking

## Architecture

H-JEPA consists of several key components:

1. **Context Encoder**: Processes visible (unmasked) image patches to generate context representations
2. **Target Encoder**: Processes all image patches with EMA updates from the context encoder
3. **Hierarchical Predictors**: Multiple predictors operating at different scales to predict target representations from context
4. **Multi-Block Masking**: Strategic masking of image patches to encourage learning of robust features

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- PyTorch 2.0 or higher

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/H-JEPA.git
   cd H-JEPA
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   Option A: Install as a package (recommended)
   ```bash
   pip install -e .
   ```

   Option B: Install requirements only
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; import timm; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
   ```

## Quick Start

### Training

1. **Prepare your configuration:**

   Create or modify a config file in `configs/`. Example:
   ```bash
   cp configs/default.yaml configs/my_experiment.yaml
   # Edit configs/my_experiment.yaml with your settings
   ```

2. **Start training:**
   ```bash
   python scripts/train.py --config configs/my_experiment.yaml
   ```

3. **Monitor training:**

   - W&B dashboard: Check your W&B project for real-time metrics
   - TensorBoard: `tensorboard --logdir results/logs`

### Evaluation

```bash
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pth --config configs/my_experiment.yaml
```

## Project Structure

```
H-JEPA/
├── configs/                    # Configuration files
│   └── *.yaml                 # YAML config files for experiments
├── src/                       # Source code
│   ├── models/                # Model architectures
│   │   ├── encoder.py        # Context and target encoders
│   │   ├── predictor.py      # Hierarchical predictors
│   │   └── hjepa.py          # Main H-JEPA model
│   ├── data/                  # Data loading and preprocessing
│   │   ├── datasets.py       # Dataset classes
│   │   └── transforms.py     # Data augmentation
│   ├── masks/                 # Masking strategies
│   │   └── multi_block.py    # Multi-block masking implementation
│   ├── losses/                # Loss functions
│   │   └── hjepa_loss.py     # H-JEPA loss with hierarchical consistency
│   ├── trainers/              # Training logic
│   │   └── trainer.py        # Main training loop
│   └── utils/                 # Utility functions
│       ├── checkpoint.py     # Checkpointing utilities
│       ├── logging.py        # Logging setup
│       └── scheduler.py      # Learning rate schedulers
├── scripts/                   # Executable scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── visualize.py          # Visualization utilities
├── notebooks/                 # Jupyter notebooks
│   └── *.ipynb               # Analysis and visualization notebooks
├── tests/                     # Unit tests
│   └── test_*.py             # Test files
├── results/                   # Training outputs
│   ├── checkpoints/          # Model checkpoints
│   └── logs/                 # Training logs
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Package configuration
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Configuration

H-JEPA uses YAML files for configuration. Key configuration sections:

- **model**: Architecture settings (encoder type, embedding dimension, hierarchy levels)
- **data**: Dataset and dataloader settings
- **training**: Training hyperparameters (learning rate, batch size, epochs)
- **masking**: Masking strategy parameters
- **logging**: W&B and TensorBoard settings

Example configuration structure:
```yaml
model:
  encoder_type: "vit_base_patch16_224"
  embed_dim: 768
  num_hierarchies: 3

data:
  dataset: "imagenet"
  batch_size: 256
  num_workers: 8

training:
  epochs: 300
  lr: 1.5e-4
  weight_decay: 0.05

masking:
  num_masks: 4
  mask_scale: [0.15, 0.2]
  aspect_ratio: [0.75, 1.5]
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

### Code Formatting

This project uses Black and isort for code formatting:

```bash
# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Check formatting
black --check src/ scripts/ tests/
```

## Advanced Usage

### Custom Datasets

To use a custom dataset, create a new dataset class in `src/data/datasets.py`:

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        # Your initialization
        pass

    def __getitem__(self, idx):
        # Your data loading logic
        pass

    def __len__(self):
        # Return dataset size
        pass
```

### Custom Masking Strategies

Implement custom masking in `src/masks/`:

```python
class CustomMaskingStrategy:
    def __init__(self, **kwargs):
        # Your initialization
        pass

    def __call__(self, batch_size, device):
        # Return masks
        pass
```

## Performance Tips

1. **Batch Size**: Use the largest batch size that fits in GPU memory
2. **Mixed Precision**: Enable automatic mixed precision training for faster training
3. **Data Loading**: Increase `num_workers` for faster data loading (typically set to number of CPU cores)
4. **Gradient Accumulation**: Use gradient accumulation for effective larger batch sizes
5. **Distributed Training**: Use multiple GPUs with DistributedDataParallel

## Troubleshooting

### Common Issues

**Out of Memory Errors:**
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision training

**Slow Training:**
- Increase number of data workers
- Enable pin_memory in dataloader
- Use faster data augmentation

**Diverging Loss:**
- Reduce learning rate
- Check EMA decay rate
- Verify masking strategy parameters

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{hjepa2024,
  title={Hierarchical Joint-Embedding Predictive Architecture for Self-Supervised Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the I-JEPA paper by Meta AI Research
- Vision Transformer implementation from `timm` library
- Inspired by self-supervised learning research community

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions and feedback:
- Open an issue on GitHub
- Email: your.email@example.com

## Roadmap

- [ ] Add support for video H-JEPA
- [ ] Implement additional backbone architectures
- [ ] Add transfer learning evaluation protocols
- [ ] Create pre-trained model zoo
- [ ] Add multi-node distributed training support
- [ ] Implement advanced masking strategies

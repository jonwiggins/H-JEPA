# Contributing to H-JEPA

Thank you for your interest in contributing to H-JEPA! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs actual behavior
- Your environment (OS, Python version, PyTorch version)
- Any relevant code snippets or error messages

### Suggesting Enhancements

We welcome enhancement suggestions! Please open an issue with:
- A clear description of the enhancement
- Why this enhancement would be useful
- Any relevant examples or mockups

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, readable code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=src

   # Check code formatting
   black --check src/ scripts/ tests/
   isort --check src/ scripts/ tests/
   ```

4. **Commit your changes**
   - Use clear, descriptive commit messages
   - Reference any related issues
   ```bash
   git commit -m "Add feature X that does Y (fixes #123)"
   ```

5. **Push to your fork and submit a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Wait for review**
   - Respond to any feedback
   - Make requested changes
   - Keep your branch up to date

## Development Setup

1. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/H-JEPA.git
   cd H-JEPA
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks (optional)**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

### Python Style Guide

- Follow PEP 8 guidelines
- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Use type hints where appropriate
- Write docstrings for all public functions/classes

### Example

```python
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ExampleModel(nn.Module):
    """
    Example model class following project conventions.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer
        output_dim: Dimension of output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Test both success cases and error cases

### Example Test

```python
import pytest
import torch
from src.models.example import ExampleModel


class TestExampleModel:
    def test_initialization(self):
        """Test model can be initialized"""
        model = ExampleModel(10, 20, 5)
        assert model is not None

    def test_forward_shape(self):
        """Test forward pass produces correct output shape"""
        model = ExampleModel(10, 20, 5)
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 5)

    def test_invalid_input_shape(self):
        """Test model handles invalid input shapes"""
        model = ExampleModel(10, 20, 5)
        x = torch.randn(4, 15)  # Wrong input dimension
        with pytest.raises(RuntimeError):
            model(x)
```

## Documentation

### Code Documentation

- Write docstrings for all public functions and classes
- Use Google-style docstrings
- Include type information
- Provide examples where helpful

### README Updates

- Update README.md if you add new features
- Keep installation instructions current
- Update usage examples as needed

## Project Structure

When adding new code, place it in the appropriate directory:

```
src/
├── models/      # Model architectures
├── data/        # Data loading and preprocessing
├── masks/       # Masking strategies
├── losses/      # Loss functions
├── trainers/    # Training loops
└── utils/       # Utility functions
```

## Commit Messages

Good commit messages help maintain a clear project history:

- Use the imperative mood ("Add feature" not "Added feature")
- First line: brief summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issues and PRs

Example:
```
Add hierarchical predictor module

Implements the hierarchical prediction component with:
- Multi-scale feature processing
- Configurable number of hierarchy levels
- Efficient memory usage through gradient checkpointing

Fixes #42
```

## Questions?

If you have questions about contributing:
- Open an issue with the "question" label
- Check existing issues and discussions
- Reach out to maintainers

## Recognition

Contributors will be recognized in:
- The project README
- Release notes for their contributions
- The project's contributors page

Thank you for contributing to H-JEPA!

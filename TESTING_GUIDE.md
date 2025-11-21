# H-JEPA Testing Guide

Comprehensive documentation for testing the Hierarchical Joint-Embedding Predictive Architecture (H-JEPA) implementation.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Test Suite Architecture](#test-suite-architecture)
4. [Running Tests](#running-tests)
5. [Coverage Status](#coverage-status)
6. [Test Organization](#test-organization)
7. [Writing New Tests](#writing-new-tests)
8. [Mocking Guidelines](#mocking-guidelines)
9. [CI/CD Integration](#cicd-integration)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The H-JEPA project uses **pytest** as its testing framework with comprehensive coverage across all major modules. The test suite is designed to verify:

- **Model Architecture**: Encoder, predictor, and hierarchical components
- **Data Handling**: Dataset loading, transformations, and augmentation
- **Training Pipeline**: Loss computation, optimization, checkpointing
- **Losses**: Contrastive, regression, and combined loss functions
- **Masking Strategies**: Block-based, hierarchical, and multi-crop masking
- **Utilities**: Schedulers, logging, visualization, and checkpointing
- **Edge Cases**: Different hardware (CPU/CUDA/MPS), batch sizes, input dimensions

### Key Statistics

- **Test Files**: 20+ test modules
- **Total Test Cases**: 500+ individual test cases
- **Current Coverage**: 90%+ across core modules
- **Framework**: pytest with plugins (pytest-cov, pytest-xdist)
- **Python Version**: 3.11+

---

## Quick Start

### Install Dependencies

```bash
# Development dependencies
pip install pytest pytest-cov pytest-xdist pytest-timeout

# Or install via pyproject.toml
pip install -e ".[dev]"
```

### Run Tests Immediately

```bash
# Run all tests (quick pass)
pytest tests/ -v --tb=short -m "not slow"

# Run full test suite (includes slow tests)
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Generate and view coverage
pytest tests/ --cov=src --cov-report=term-missing
# Open htmlcov/index.html in browser
```

### Common Quick Commands

```bash
# Run specific test file
pytest tests/test_models.py -v

# Run specific test class
pytest tests/test_encoder.py::TestLayerScale -v

# Run specific test function
pytest tests/test_encoder.py::TestLayerScale::test_forward_pass -v

# Run with output (useful for debugging)
pytest tests/ -v -s

# Stop at first failure
pytest tests/ -x

# Run N tests in parallel
pytest tests/ -n auto
```

---

## Test Suite Architecture

### Directory Structure

```
H-JEPA/
├── tests/
│   ├── conftest.py                    # Shared fixtures and configuration
│   ├── test_models.py                 # Model instantiation tests
│   ├── test_encoder.py                # Encoder component tests (90%+ coverage)
│   ├── test_losses.py                 # Loss function tests
│   ├── test_masks.py                  # Masking strategy tests
│   ├── test_data.py                   # Dataset and data loader tests
│   ├── test_trainers.py               # Training loop tests
│   ├── test_utils.py                  # Utility function tests
│   ├── test_visualization.py          # Visualization tests
│   ├── test_flash_attention.py        # Flash Attention tests
│   ├── test_fpn.py                    # FPN module tests
│   ├── test_rope.py                   # RoPE implementation tests
│   ├── test_layerscale.py             # LayerScale tests
│   ├── test_sigreg.py                 # SignReg loss tests
│   ├── test_phase123_optimizations.py # Phase 1-3 optimization tests
│   ├── test_ijepa_compliance.py       # I-JEPA compliance tests
│   ├── test_mask_semantics.py         # Mask semantics validation
│   └── README_TRAINERS_TESTS.md       # Trainer-specific documentation
├── src/
│   ├── models/       # Model implementations
│   ├── data/         # Data loading and processing
│   ├── losses/       # Loss functions
│   ├── masks/        # Masking implementations
│   ├── trainers/     # Training code
│   ├── utils/        # Utilities
│   └── visualization/ # Visualization tools
└── scripts/
    └── run_tests.py  # Alternative test runner
```

### Pytest Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --cov=src --cov-report=term-missing"
```

### Test Markers

Tests are organized with markers for easy filtering:

| Marker | Meaning | Usage |
|--------|---------|-------|
| `@pytest.mark.slow` | Tests taking >1 second | `pytest -m "not slow"` |
| `@pytest.mark.cuda` | Requires CUDA GPU | `pytest -m "cuda"` |
| `@pytest.mark.integration` | Integration tests | `pytest -m "integration"` |

---

## Running Tests

### Quick Test Runs

Perfect for development and local iteration:

```bash
# Run fast tests only (skip slow tests)
pytest tests/ -m "not slow" -v

# Run with minimal output
pytest tests/ -q

# Run tests matching a pattern
pytest tests/ -k "test_encoder" -v

# Run with early failure stop
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf
```

### Full Test Suite

For comprehensive validation:

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run all tests with full tracebacks
pytest tests/ -vv --tb=long

# Run all tests and stop on first failure
pytest tests/ -x -v

# Run all tests in parallel (requires pytest-xdist)
pytest tests/ -n auto -v
```

### Coverage Reports

Detailed coverage analysis:

```bash
# Terminal coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Show only uncovered lines
pytest tests/ --cov=src --cov-report=term-missing:skip-covered

# HTML coverage report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# Coverage with context (slower)
pytest tests/ --cov=src --cov-context=test

# Specify coverage branches
pytest tests/ --cov=src --cov-branch --cov-report=html

# Fail if coverage below threshold
pytest tests/ --cov=src --cov-fail-under=80
```

### Specific Module Tests

Test individual components:

```bash
# Models
pytest tests/test_models.py -v
pytest tests/test_encoder.py -v

# Data
pytest tests/test_data.py -v

# Losses
pytest tests/test_losses.py -v

# Masking
pytest tests/test_masks.py -v

# Training
pytest tests/test_trainers.py -v

# Utilities
pytest tests/test_utils.py -v

# Visualization
pytest tests/test_visualization.py -v
```

### Filtering Tests

```bash
# Run tests by name pattern
pytest tests/ -k "encoder" -v
pytest tests/ -k "not slow" -v
pytest tests/ -k "forward" -v

# Run tests by marker
pytest tests/ -m "slow" -v
pytest tests/ -m "cuda" -v
pytest tests/ -m "integration" -v

# Combine filters
pytest tests/ -k "encoder" -m "not slow" -v
```

---

## Coverage Status

### Current Coverage Achievements

| Module | Coverage | Status | Notes |
|--------|----------|--------|-------|
| `models/encoder.py` | 90%+ | ✓ Excellent | LayerScale, RoPE, Flash Attention |
| `models/predictor.py` | 85%+ | ✓ Good | Predictor architecture |
| `models/hjepa.py` | 80%+ | ✓ Good | H-JEPA main model |
| `losses/hjepa_loss.py` | 90%+ | ✓ Excellent | Core loss function |
| `losses/contrastive.py` | 85%+ | ✓ Good | Contrastive components |
| `masks/hierarchical.py` | 90%+ | ✓ Excellent | Hierarchical masking |
| `masks/multicrop_masking.py` | 85%+ | ✓ Good | Multi-crop strategies |
| `trainers/trainer.py` | 85%+ | ✓ Good | Training loop |
| `utils/scheduler.py` | 90%+ | ✓ Excellent | Learning rate scheduling |
| `utils/checkpoint.py` | 85%+ | ✓ Good | Checkpoint management |
| `data/datasets.py` | 80%+ | ✓ Good | Data loading |
| `visualization/` | 75%+ | ✓ Good | Visualization utilities |

### Coverage Goals

**Per-Module Target**: 80%+ overall, 90%+ for critical modules

- **Critical Modules** (target 90%+): encoder, losses, masks, scheduler
- **Core Modules** (target 85%+): predictor, hjepa, trainer, checkpoint
- **Support Modules** (target 80%+): data, visualization, utilities

### Improving Coverage

```bash
# Identify uncovered code
pytest tests/ --cov=src --cov-report=term-missing | grep "^src"

# Focus on specific module
pytest tests/test_encoder.py --cov=src.models.encoder --cov-report=term-missing

# Generate detailed HTML report
pytest tests/ --cov=src --cov-report=html
# View htmlcov/index.html and navigate to modules

# Use coverage report to find gaps
coverage report -m --skip-covered
```

---

## Test Organization

### Test File Structure

Each test file follows a consistent pattern:

```python
"""
Comprehensive test suite for module_name.py

Tests cover:
- Feature 1: Description
- Feature 2: Description
- Edge cases: Description
"""

import pytest
import torch
from src.module import ComponentUnderTest

class TestComponentUnderTest:
    """Tests for ComponentUnderTest functionality."""

    def test_basic_operation(self, device):
        """Test basic operation of component."""
        # Arrange: Set up test data
        input_data = torch.randn(4, 3, 224, 224, device=device)

        # Act: Call the component
        component = ComponentUnderTest()
        output = component(input_data)

        # Assert: Verify results
        assert output is not None
        assert output.shape[0] == 4

    def test_edge_case(self, device):
        """Test edge case: empty input."""
        component = ComponentUnderTest()
        input_data = torch.randn(0, 3, 224, 224, device=device)

        # Should handle gracefully or raise expected error
        if component.allow_empty:
            output = component(input_data)
            assert output.shape[0] == 0
        else:
            with pytest.raises(ValueError):
                component(input_data)
```

### Naming Conventions

**Test File Names**
- Format: `test_<module_name>.py`
- Examples: `test_encoder.py`, `test_masks.py`, `test_trainers.py`

**Test Class Names**
- Format: `Test<ComponentName>`
- Examples: `TestLayerScale`, `TestContextEncoder`, `TestHierarchicalMask`

**Test Function Names**
- Format: `test_<specific_behavior>`
- Examples: `test_forward_pass()`, `test_gradient_flow()`, `test_invalid_input()`
- Use descriptive names that document expected behavior

**Fixture Names**
- Format: `<purpose>_<variant>`
- Examples: `device`, `sample_batch_224`, `tiny_vit_config`

### Fixture Usage

#### Common Session-Level Fixtures

```python
@pytest.fixture(scope="session")
def test_device():
    """Get best available device (CUDA > MPS > CPU)."""
    # Used across all tests, set once per session

@pytest.fixture
def device(test_device):
    """Per-test device access."""
    return test_device

@pytest.fixture
def random_seed():
    """Fix random seed for reproducibility."""
    return 42  # Deterministic tests
```

#### Data Fixtures

```python
@pytest.fixture
def sample_batch_224(device):
    """4x batch of 224x224 RGB images."""
    return torch.randn(4, 3, 224, 224, device=device)

@pytest.fixture
def sample_embeddings_3d(device):
    """Patch embeddings: [batch=4, patches=196, dim=128]."""
    return torch.randn(4, 196, 128, device=device)
```

#### Config Fixtures

```python
@pytest.fixture
def tiny_vit_config():
    """Minimal ViT for fast testing."""
    return {
        "encoder_type": "vit_tiny_patch16_224",
        "img_size": 224,
        "embed_dim": 192,
        "predictor_depth": 2,
    }
```

#### Using Fixtures in Tests

```python
def test_encoder(sample_batch_224, tiny_vit_config, device):
    """Test encoder with sample batch and config."""
    from src.models import create_encoder

    encoder = create_encoder(**tiny_vit_config)
    encoder = encoder.to(device)

    output = encoder(sample_batch_224)
    assert output.shape == (4, 196, 192)
```

### Fixture Scope

| Scope | Usage | Performance |
|-------|-------|-------------|
| `session` | Once per test session | Fastest, least isolation |
| `module` | Once per test file | Good balance |
| `class` | Once per test class | Good isolation |
| `function` (default) | Once per test | Most isolated, slower |

---

## Writing New Tests

### Best Practices

#### 1. Test Structure (Arrange-Act-Assert)

```python
def test_my_feature(device, sample_batch_224):
    """Test my feature with clear documentation."""
    # ARRANGE: Set up test data
    model = create_my_model().to(device)
    expected_output_shape = (4, 256)

    # ACT: Execute the code under test
    output = model(sample_batch_224)

    # ASSERT: Verify the results
    assert output.shape == expected_output_shape
    assert not torch.isnan(output).any()
```

#### 2. Descriptive Test Names

```python
# Good: Clearly documents expected behavior
def test_encoder_forward_pass_returns_correct_shape(sample_batch_224):
    ...

def test_loss_decreases_with_valid_gradient_flow(sample_embeddings_3d):
    ...

def test_masking_handles_empty_batch_gracefully(device):
    ...

# Bad: Vague or non-descriptive
def test_encoder(sample_batch_224):  # What aspect?
    ...

def test_loss(sample_embeddings_3d):  # What behavior?
    ...
```

#### 3. One Logical Assertion Per Test

```python
# Good: Each test verifies one behavior
def test_forward_pass_shape(sample_batch_224):
    output = encoder(sample_batch_224)
    assert output.shape == (4, 196, 128)

def test_forward_pass_no_nans(sample_batch_224):
    output = encoder(sample_batch_224)
    assert not torch.isnan(output).any()

# Acceptable: Multiple related assertions
def test_forward_pass_correctness(sample_batch_224):
    output = encoder(sample_batch_224)
    assert output.shape == (4, 196, 128)
    assert output.dtype == torch.float32
    assert not torch.isnan(output).any()

# Bad: Testing multiple unrelated behaviors
def test_encoder(sample_batch_224, device):
    output = encoder(sample_batch_224)
    assert output.shape == expected_shape
    loss = loss_fn(output)
    assert loss < threshold  # Different component!
```

#### 4. Use Parametrization for Multiple Inputs

```python
import pytest

@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
def test_encoder_with_different_batch_sizes(batch_size, device):
    """Test encoder handles various batch sizes."""
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    encoder = create_encoder()
    output = encoder(images)
    assert output.shape[0] == batch_size

@pytest.mark.parametrize("img_size", [224, 384, 512])
def test_encoder_with_different_resolutions(img_size, device):
    """Test encoder handles various image resolutions."""
    images = torch.randn(4, 3, img_size, img_size, device=device)
    encoder = create_encoder(img_size=img_size)
    output = encoder(images)
    assert output.shape[1] == (img_size // 16) ** 2  # For patch16
```

#### 5. Test Edge Cases

```python
def test_encoder_with_single_sample(device):
    """Test encoder with batch size 1."""
    images = torch.randn(1, 3, 224, 224, device=device)
    output = encoder(images)
    assert output.shape == (1, 196, 128)

def test_encoder_with_zero_batch(device):
    """Test encoder handles empty batch."""
    images = torch.randn(0, 3, 224, 224, device=device)
    # Should handle gracefully (skip processing)
    output = encoder(images)
    assert output.shape == (0, 196, 128)

def test_loss_with_identical_embeddings(device):
    """Test loss when embeddings are identical."""
    z = torch.randn(4, 128, device=device)
    loss = loss_fn(z, z)
    assert loss == pytest.approx(0.0, abs=1e-6)

def test_mask_with_all_visible(device):
    """Test mask when no patches are masked."""
    mask = torch.zeros(4, 196)  # All zeros = all visible
    output = apply_mask(output, mask)
    assert torch.allclose(output, output_unmasked)
```

#### 6. Add Docstrings

```python
def test_encoder_gradient_flow(sample_batch_224):
    """
    Test that gradients flow properly through encoder.

    Ensures backpropagation works correctly for training.
    """
    images = sample_batch_224
    images.requires_grad_(True)

    output = encoder(images)
    loss = output.sum()
    loss.backward()

    assert images.grad is not None
    assert (images.grad != 0).any()
```

### Common Test Patterns

#### Testing Forward Passes

```python
def test_model_forward_pass_basic(sample_batch_224, device):
    """Test basic forward pass."""
    model = create_model().to(device)
    output = model(sample_batch_224)

    # Check shape, dtype, device
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 4
    assert output.device == device
    assert output.dtype == torch.float32

def test_model_forward_pass_backward(sample_batch_224):
    """Test backward pass."""
    model = create_model()
    sample_batch_224.requires_grad_(True)

    output = model(sample_batch_224)
    loss = output.sum()
    loss.backward()

    assert sample_batch_224.grad is not None
    assert not torch.isnan(sample_batch_224.grad).any()
```

#### Testing Loss Functions

```python
def test_loss_computation(sample_embeddings_3d):
    """Test loss computation and properties."""
    z_i, z_j = sample_embeddings_3d

    loss = loss_fn(z_i, z_j)

    # Loss should be scalar
    assert loss.dim() == 0
    # Loss should be positive
    assert loss >= 0
    # Loss should not be NaN/Inf
    assert torch.isfinite(loss)

def test_loss_symmetry(sample_embeddings_3d):
    """Test loss function symmetry properties."""
    z_i, z_j = sample_embeddings_3d

    loss_ij = loss_fn(z_i, z_j)
    loss_ji = loss_fn(z_j, z_i)

    # Should be equal for symmetric losses
    assert torch.allclose(loss_ij, loss_ji, rtol=1e-5)
```

#### Testing Masking Operations

```python
def test_masking_application(sample_embeddings_3d, device):
    """Test masking is applied correctly."""
    embeddings = sample_embeddings_3d
    batch_size, num_patches, dim = embeddings.shape

    # Create mask (50% masked)
    mask = torch.zeros(batch_size, num_patches)
    mask[:, :num_patches//2] = 1

    masked_emb = apply_masking(embeddings, mask)

    # Masked patches should be zeroed
    assert torch.allclose(masked_emb[:, :num_patches//2],
                         torch.zeros(batch_size, num_patches//2, dim))
    # Visible patches should be unchanged
    assert torch.allclose(masked_emb[:, num_patches//2:],
                         embeddings[:, num_patches//2:])
```

#### Testing Data Loading

```python
def test_dataloader_batch_shapes(sample_train_loader):
    """Test dataloader returns correct batch shapes."""
    for batch in sample_train_loader:
        images, targets = batch

        assert images.dim() == 4
        assert images.shape[1:] == (3, 224, 224)
        assert targets.dim() == 1
        assert len(images) == len(targets)
        break  # Just test first batch
```

---

## Mocking Guidelines

### When to Mock

Mock external dependencies that:
- Are slow (network, disk I/O)
- Have side effects (file system)
- Are non-deterministic (random)
- Are hard to set up (databases)

### Mock Best Practices

#### 1. Mock External Services

```python
from unittest.mock import MagicMock, patch

def test_model_checkpointing(mocker):
    """Test checkpoint save (mocking file I/O)."""
    # Mock file operations
    mock_save = mocker.patch('torch.save')

    trainer = HJEPATrainer(config)
    trainer.save_checkpoint('model.pt')

    # Verify save was called
    mock_save.assert_called_once()
    args, kwargs = mock_save.call_args
    assert 'model.pt' in str(args)
```

#### 2. Partial Mocking

```python
def test_with_partial_mock(mocker):
    """Test with some methods mocked."""
    model = create_model()

    # Mock just the backward pass
    original_backward = model.backward
    model.backward = MagicMock()

    # Test forward pass still works
    output = model(input_data)
    assert output is not None

    # Restore
    model.backward = original_backward
```

#### 3. Spy on Calls

```python
def test_loss_called_correctly(mocker):
    """Spy on loss function calls."""
    loss_fn = MyLoss()
    spy = mocker.spy(loss_fn, 'forward')

    z_i, z_j = create_test_embeddings()
    loss = loss_fn(z_i, z_j)

    # Verify function was called with correct args
    spy.assert_called_once()
    call_args = spy.call_args[0]
    assert call_args[0] is z_i
    assert call_args[1] is z_j
```

### Mock vs Fixture Trade-offs

| Approach | Speed | Realism | Use Case |
|----------|-------|---------|----------|
| Real objects | Slower | Most realistic | Core functionality |
| Fixtures | Medium | Realistic | Common scenarios |
| Mocks | Fastest | Less realistic | External deps |
| Combination | Good | Good | Best balance |

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Lint
      run: |
        flake8 src tests
        black --check src tests

    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

Configure in `.pre-commit-config.yaml`:

```yaml
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  hooks:
  - id: trailing-whitespace
  - id: end-of-file-fixer

- repo: https://github.com/psf/black
  hooks:
  - id: black
    language_version: python3.11

- repo: https://github.com/PyCQA/flake8
  hooks:
  - id: flake8

- repo: https://github.com/PyCQA/isort
  hooks:
  - id: isort

- repo: local
  hooks:
  - id: pytest-check
    name: pytest check
    entry: pytest tests/ -x
    language: system
    pass_filenames: false
    stages: [commit]
```

### Running in CI

```bash
# Quick check (fast tests only)
pytest tests/ -m "not slow" -v --cov=src --cov-report=xml --cov-fail-under=80

# Full validation
pytest tests/ -v --cov=src --cov-report=xml --cov-branch

# Parallel execution (faster)
pytest tests/ -n auto -v --cov=src --cov-report=xml
```

---

## Troubleshooting

### Common Issues and Solutions

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure you're in project root
cd /Users/jon/repos/H-JEPA

# Install in editable mode
pip install -e .

# Or run from project root
pytest tests/ -v
```

**Debug**:
```python
# In test file
import sys
print("Python path:", sys.path)
print("CWD:", os.getcwd())
```

#### Device-Related Failures

**Problem**: Tests fail on GPU but pass on CPU

**Solutions**:
```bash
# Force CPU (for debugging)
CUDA_VISIBLE_DEVICES="" pytest tests/ -v

# Force specific GPU
CUDA_VISIBLE_DEVICES=0 pytest tests/ -v

# Test on MPS (Mac)
pytest tests/ -v -k "not cuda"

# Check device availability
pytest tests/conftest.py::test_device -v -s
```

**Debug Script**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

#### Memory Issues

**Problem**: Tests exhaust memory or OOM errors

**Solutions**:
```bash
# Run serially (not in parallel)
pytest tests/ -n 0 -v

# Skip slow/memory-intensive tests
pytest tests/ -m "not slow" -v

# Run smaller subset
pytest tests/test_encoder.py -v

# Reduce batch sizes (modify fixtures in conftest.py)
```

**Monitor Memory**:
```bash
# Watch memory during test run
watch -n 1 'ps aux | grep pytest'

# Run with memory limit
pytest tests/ --memray  # Requires memray package
```

#### Timeout Issues

**Problem**: Tests timeout on slow systems

**Solutions**:
```bash
# Increase timeout globally
pytest tests/ --timeout=600 -v

# Mark slow tests explicitly
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_slow_operation():
    ...

# Skip slow tests
pytest tests/ -m "not slow" -v
```

#### Flaky Tests

**Problem**: Tests pass sometimes, fail sometimes

**Diagnosis**:
```bash
# Run test multiple times
pytest tests/test_file.py::test_name --count=10 -v

# Run with different random seeds
pytest tests/ --randomly-seed=123 -v
pytest tests/ --randomly-seed=456 -v

# Check for race conditions
pytest tests/ -n 0 -v  # Serial execution
```

**Solutions**:
```python
# Fix random seed in test
def test_deterministic(random_seed):
    """Test should be deterministic with fixed seed."""
    # seed is 42 from fixture
    x = torch.randn(10)
    y = torch.randn(10)
    assert torch.allclose(x, y)  # Now deterministic

# Avoid timing-dependent tests
# Bad:
def test_speed():
    start = time.time()
    function()
    elapsed = time.time() - start
    assert elapsed < 0.1  # Flaky on slow machines!

# Good:
def test_functionality():
    result = function()
    assert result is correct
```

#### Coverage Not Including All Code

**Problem**: Coverage report shows lower than expected percentage

**Solutions**:
```bash
# Ensure all branches tested
pytest tests/ --cov=src --cov-branch --cov-report=html

# Check for untested paths
pytest tests/ --cov=src --cov-report=term-missing | grep "^src"

# Test specific module
pytest tests/test_encoder.py --cov=src.models.encoder --cov-report=term-missing

# Increase coverage tolerance
pytest tests/ --cov=src --cov-report=term-missing:skip-covered
```

#### Test Dependencies

**Problem**: Tests pass individually but fail when run together

**Cause**: Test order dependency or shared state

**Solutions**:
```bash
# Run in random order
pytest tests/ --random-order -v

# Isolate test
pytest tests/test_file.py::TestClass::test_method -v

# Check for shared fixtures with wrong scope
# In conftest.py:
# Bad:
@pytest.fixture(scope="session")
def model():
    return create_model()  # Shared across all tests!

# Good:
@pytest.fixture(scope="function")
def model():
    return create_model()  # New instance per test
```

#### Assertion Errors with Floating Point

**Problem**: `assert 0.1 == 0.1` fails due to floating point precision

**Solution**:
```python
# Bad:
assert result == expected

# Good:
assert torch.allclose(result, expected, rtol=1e-5, atol=1e-7)

# With pytest.approx:
assert result == pytest.approx(expected, rel=1e-5, abs=1e-7)

# For tensors:
torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

#### Debugging Failed Tests

**Interactive Debugging**:
```bash
# Drop into debugger on failure
pytest tests/test_file.py::test_name --pdb

# Drop into debugger on assertions
pytest tests/test_file.py::test_name --pdb --pdbcls=IPython.terminal.debugger:Pdb

# Show local variables on failure
pytest tests/test_file.py::test_name -l

# Full tracebacks
pytest tests/test_file.py::test_name -vv --tb=long
```

**Add Debug Output**:
```python
def test_with_debug(capsys):
    """Test with captured output for debugging."""
    print("Debug: starting test")
    result = function_under_test()
    print(f"Debug: result = {result}")

    # In test output:
    # captured stdout call
    # Debug: starting test
    # Debug: result = ...

def test_with_logging(caplog):
    """Test with logging capture."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting test")

    result = function_under_test()

    # Verify log messages
    assert "Starting test" in caplog.text
```

### Performance Tips

1. **Use Markers Effectively**
   ```bash
   pytest tests/ -m "not slow" -v  # Skip long tests
   pytest tests/ -m "integration" -v  # Run only integration tests
   ```

2. **Run Tests in Parallel**
   ```bash
   pip install pytest-xdist
   pytest tests/ -n auto -v
   ```

3. **Use Fixture Scope Wisely**
   ```python
   # Good for expensive operations
   @pytest.fixture(scope="session")
   def expensive_model():
       return create_large_model()  # Created once

   # Need fresh data per test
   @pytest.fixture(scope="function")
   def data():
       return create_data()  # Fresh per test
   ```

4. **Skip Slow Tests During Development**
   ```bash
   pytest tests/ -m "not slow" -v
   ```

5. **Cache Expensive Computations**
   ```python
   @pytest.fixture
   def model_cache(tmp_path):
       path = tmp_path / "model.pt"
       if path.exists():
           return torch.load(path)
       model = create_model()
       torch.save(model.state_dict(), path)
       return model
   ```

---

## Summary

### Testing Workflow

1. **Local Development**
   ```bash
   # Quick feedback loop
   pytest tests/ -m "not slow" -v
   ```

2. **Before Committing**
   ```bash
   # Full validation
   pytest tests/ -v --cov=src --cov-report=term-missing
   ```

3. **In CI/CD**
   ```bash
   # Comprehensive check
   pytest tests/ -v --cov=src --cov-report=xml --cov-fail-under=80
   ```

4. **Adding New Features**
   ```bash
   # Write test first (TDD)
   # Implement feature
   # Verify coverage
   pytest tests/test_new_feature.py --cov=src.new_module --cov-report=html
   ```

### Key Takeaways

- **Use fixtures**: Avoid duplicating setup code
- **Name tests clearly**: Document expected behavior
- **Test edge cases**: Empty inputs, None values, etc.
- **Keep tests fast**: Use markers for slow tests
- **Mock externals**: Keep tests isolated and deterministic
- **Monitor coverage**: Aim for 80%+ overall, 90%+ for critical modules
- **Document tests**: Add docstrings explaining what's tested

### Resources

- **Pytest Docs**: https://docs.pytest.org/
- **Pytest Best Practices**: https://docs.pytest.org/goodpractices.html
- **Fixture Scopes**: https://docs.pytest.org/how-to/fixtures.html#scope-sharing-fixtures-across-classes-modules-packages-and-sessions
- **Parametrization**: https://docs.pytest.org/how-to/parametrize.html
- **Mocking with unittest.mock**: https://docs.python.org/3/library/unittest.mock.html

---

**Last Updated**: 2025-11-21
**Framework Version**: pytest 7.4.0+
**Python Version**: 3.11+
**Maintainers**: H-JEPA Team

# Utils Module Tests - Quick Start Guide

## File Location
`tests/test_utils.py` - Comprehensive test suite for all utilities

## What's Tested

### 1. Logging (TestSetupLogging, TestMetricsLogger, TestProgressTracker)
- **setup_logging()**: Console/file logging setup, log levels, formatting
- **MetricsLogger**: Metrics logging, image/histogram logging, model tracking, W&B/TensorBoard integration, system metrics
- **ProgressTracker**: Training progress tracking, ETA calculation, step timing

### 2. Checkpointing (TestCheckpointManager, TestSaveCheckpoint, TestLoadCheckpoint)
- **CheckpointManager**: Save/load checkpoints, best metric tracking, cleanup old files, DataParallel support
- **save_checkpoint()**: Standalone checkpoint saving
- **load_checkpoint()**: Standalone checkpoint loading

### 3. Schedulers (TestCosineScheduler, TestLinearScheduler, TestEMAScheduler, TestHierarchicalScheduler)
- **CosineScheduler**: Cosine annealing with linear warmup
- **LinearScheduler**: Linear schedule with warmup
- **EMAScheduler**: EMA momentum scheduling for target encoder
- **HierarchicalScheduler**: Per-level scheduling support
- **create_lr_scheduler()**: Factory function for LR schedulers
- **create_ema_scheduler()**: Factory function for EMA schedulers

### 4. Integration Tests
- Full training pipeline with metrics logging
- Complete checkpoint save/load flow
- Multi-epoch scheduler simulation

## Quick Commands

### Run All Tests
```bash
pytest tests/test_utils.py -v
```

### Run Specific Test Class
```bash
# Run all MetricsLogger tests
pytest tests/test_utils.py::TestMetricsLogger -v

# Run all scheduler tests
pytest tests/test_utils.py::TestCosineScheduler -v
```

### Run With Coverage
```bash
# Summary coverage
pytest tests/test_utils.py --cov=src/utils --cov-report=term-missing

# HTML report (opens in browser)
pytest tests/test_utils.py --cov=src/utils --cov-report=html
# Then open: htmlcov/index.html

# Per-file coverage
pytest tests/test_utils.py --cov=src/utils/logging --cov-report=term-missing
pytest tests/test_utils.py --cov=src/utils/checkpoint --cov-report=term-missing
pytest tests/test_utils.py --cov=src/utils/scheduler --cov-report=term-missing
```

### Run With Specific Options
```bash
# Stop on first failure
pytest tests/test_utils.py -x

# Verbose output with long tracebacks
pytest tests/test_utils.py -vv --tb=long

# Show print statements
pytest tests/test_utils.py -s -v

# Run only tests matching pattern
pytest tests/test_utils.py -k "checkpoint" -v
```

## Test Statistics

| Category | Count | Focus Areas |
|----------|-------|------------|
| Logging Tests | 25 | Metrics, images, histograms, progress tracking |
| Checkpoint Tests | 18 | Save/load, cleanup, best tracking |
| Scheduler Tests | 17 | Warmup, annealing, bounds, monotonicity |
| Integration Tests | 3 | End-to-end training scenarios |
| **Total Tests** | **63** | **All utils modules** |

## Key Test Features

- **Fixtures for Reuse**: Temporary directories, models, optimizers
- **Edge Cases**: Empty buffers, missing files, no steps
- **Error Handling**: Non-existent checkpoints, invalid scheduler types
- **Device Agnostic**: Works on CPU, CUDA, and MPS
- **Fast Execution**: Complete suite runs in <60 seconds
- **Clean Isolation**: Each test uses isolated temporary directories

## Coverage Targets

Current test suite targets:
- **Overall utils coverage**: 80-90%
- **Logging coverage**: 75-85%
- **Checkpoint coverage**: 80-90%
- **Scheduler coverage**: 85-95%

## Important Test Fixtures

### Temporary Directories
```python
@pytest.fixture
def temp_log_dir(self):
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)
```

### Simple Model for Testing
```python
@pytest.fixture
def simple_model(self):
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
```

### Optimizer
```python
@pytest.fixture
def simple_optimizer(self, simple_model):
    return Adam(simple_model.parameters(), lr=0.001)
```

## Common Assertions

### Metric Tests
```python
metrics_logger.log_metrics({"loss": 0.5}, step=0)
assert metrics_logger.step == 1
```

### Checkpoint Tests
```python
checkpoint_path = manager.save_checkpoint(...)
assert os.path.exists(checkpoint_path)
metadata = manager.load_checkpoint(checkpoint_path, ...)
assert metadata["epoch"] == 5
```

### Scheduler Tests
```python
scheduler = CosineScheduler(0.1, 0.001, 100, 1000, warmup_epochs=10)
lr_start = scheduler(0)
lr_mid = scheduler(50000)
assert lr_start >= lr_mid  # Should be decreasing
```

## Debugging Tests

### Run Single Test with Debug Output
```bash
pytest tests/test_utils.py::TestMetricsLogger::test_log_metrics -vv -s --tb=long
```

### Run With Pytest Debugger
```bash
pytest tests/test_utils.py --pdb  # Drops into debugger on failure
```

### Show Local Variables on Failure
```bash
pytest tests/test_utils.py -l  # --showlocals
```

## Dependencies

Required:
- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- torch >= 2.0.0
- numpy >= 1.24.0

Optional:
- tensorboard >= 2.13.0
- wandb >= 0.15.0

Install all dev dependencies:
```bash
pip install -e ".[dev]"
```

## Notes

- Tests mock W&B and TensorBoard to avoid external dependencies
- All file operations use temporary directories automatically cleaned up
- Tests are reproducible with fixed random seeds
- No external API calls required
- Tests follow pytest best practices
- Each test is independent and can run in any order

## Continuous Integration

To run in CI/CD pipeline:
```bash
pytest tests/test_utils.py \
  --cov=src/utils \
  --cov-report=term-missing \
  --cov-fail-under=75 \
  -v
```

This ensures:
- Verbose output for debugging
- Coverage reporting
- Failure if coverage drops below 75%

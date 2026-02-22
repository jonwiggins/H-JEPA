# H-JEPA Utils Testing - Best Practices & Advanced Usage

## Overview

This guide provides best practices for working with the utils test suite and extending it.

## Test Design Philosophy

### 1. Isolation
Each test is completely independent:
- Uses its own fixtures (temp directories, models, optimizers)
- No shared state between tests
- Can run in any order
- Can run in parallel

```python
# Good: Each test manages its own fixtures
def test_save_checkpoint_1(self, checkpoint_manager, simple_model):
    # Uses fresh checkpoint_manager fixture
    checkpoint_manager.save_checkpoint(...)

def test_save_checkpoint_2(self, checkpoint_manager, simple_model):
    # Uses fresh checkpoint_manager fixture
    checkpoint_manager.save_checkpoint(...)
```

### 2. Clarity
Test names describe exactly what is being tested:
- `test_<function>_<scenario>`
- `test_<function>_<condition>`

```python
# Good: Clear and descriptive
def test_log_metrics_with_prefix(self)
def test_is_better_metric_min_mode(self)
def test_get_eta_no_steps(self)

# Bad: Unclear
def test_metrics(self)
def test_metric(self)
def test_eta(self)
```

### 3. Coverage
Tests cover three categories:
- **Happy path**: Normal usage
- **Edge cases**: Empty buffers, no steps, missing values
- **Error cases**: Invalid inputs, missing files

```python
# Happy path
def test_log_metrics(self, metrics_logger):
    metrics_logger.log_metrics({"loss": 0.5})
    assert metrics_logger.step == 1

# Edge case
def test_log_accumulated_metrics_empty_buffer(self, metrics_logger):
    metrics_logger.log_accumulated_metrics()
    # Should not raise

# Error case
def test_load_checkpoint_nonexistent(self, checkpoint_manager, model):
    with pytest.raises(FileNotFoundError):
        checkpoint_manager.load_checkpoint("/nonexistent/path.pth", model)
```

## Writing New Tests

### Template for Logging Tests
```python
class TestNewLoggingFeature:
    """Test suite for new logging feature."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def logger(self, temp_log_dir):
        """Create a configured logger."""
        return MetricsLogger(
            experiment_name="test",
            log_dir=temp_log_dir,
            use_wandb=False,
            use_tensorboard=False,
        )

    def test_feature_basic(self, logger):
        """Test basic feature functionality."""
        # Setup
        input_data = {"key": "value"}

        # Execute
        result = logger.some_method(input_data)

        # Assert
        assert result is not None
        assert logger.step == 1
```

### Template for Checkpoint Tests
```python
class TestNewCheckpointFeature:
    """Test suite for new checkpoint feature."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def manager(self, temp_checkpoint_dir):
        """Create checkpoint manager."""
        return CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            keep_best_n=3,
        )

    @pytest.fixture
    def simple_model(self):
        """Simple test model."""
        return nn.Linear(10, 5)

    def test_feature_basic(self, manager, simple_model):
        """Test basic feature."""
        # Setup
        # Execute
        # Assert
        pass
```

### Template for Scheduler Tests
```python
class TestNewScheduler:
    """Test suite for new scheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = NewScheduler(base_value=0.1, final_value=0.001, epochs=100, steps_per_epoch=1000)
        assert scheduler.base_value == 0.1
        assert scheduler.final_value == 0.001

    def test_schedule_values(self):
        """Test schedule produces correct values."""
        scheduler = NewScheduler(0.1, 0.001, 100, 1000, warmup_epochs=10)

        # Test at different points
        val_0 = scheduler(0)          # Start
        val_mid = scheduler(50000)    # Middle
        val_end = scheduler(99999)    # End

        # Values should make sense
        assert 0.001 <= val_0 <= 0.1
        assert 0.001 <= val_mid <= 0.1
        assert 0.001 <= val_end <= 0.1

    def test_monotonicity(self):
        """Test schedule is monotonic."""
        scheduler = NewScheduler(0.1, 0.001, 100, 1000, warmup_epochs=10)

        prev_val = 0.1
        for step in range(10000, 100000, 10000):
            val = scheduler(step)
            assert val <= prev_val  # Should decrease
            prev_val = val
```

## Best Practices

### 1. Use Descriptive Variable Names
```python
# Good
batch_size = 4
num_epochs = 100
learning_rate = 0.1

# Bad
bs = 4
ne = 100
lr = 0.1
```

### 2. Use Type Hints in Tests
```python
# Good
def test_log_metrics(self, metrics_logger: MetricsLogger) -> None:
    metrics: Dict[str, float] = {"loss": 0.5}
    metrics_logger.log_metrics(metrics)

# Less clear
def test_log_metrics(self, metrics_logger):
    metrics = {"loss": 0.5}
    metrics_logger.log_metrics(metrics)
```

### 3. Test One Thing Per Test
```python
# Good: Single responsibility
def test_log_metrics(self, metrics_logger):
    """Test logging basic metrics."""
    metrics_logger.log_metrics({"loss": 0.5})
    assert metrics_logger.step == 1

def test_log_metrics_with_prefix(self, metrics_logger):
    """Test logging with prefix."""
    metrics_logger.log_metrics({"loss": 0.5}, prefix="train/")
    assert metrics_logger.step == 1

# Bad: Multiple concerns
def test_log_metrics_everything(self, metrics_logger):
    """Test many things at once."""
    metrics_logger.log_metrics({"loss": 0.5})
    assert metrics_logger.step == 1
    metrics_logger.log_metrics({"acc": 0.9}, prefix="train/")
    assert metrics_logger.step == 2
    # ... more assertions
```

### 4. Arrange-Act-Assert (AAA) Pattern
```python
def test_checkpoint_save_load(self, manager, model, optimizer):
    """Test saving and loading checkpoint."""
    # Arrange: Setup initial state
    checkpoint_path = os.path.join(tempfile.gettempdir(), "test.pth")

    # Act: Execute the operation
    manager.save_checkpoint(
        epoch=5,
        model=model,
        optimizer=optimizer,
        scheduler=None,
    )

    # Assert: Verify the result
    assert os.path.exists(checkpoint_path)
```

### 5. Use Fixtures for Reusable Components
```python
# Good: Reusable fixture
@pytest.fixture
def simple_model(self):
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

# Bad: Creating in each test
def test_save_checkpoint_1(self):
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    # ...

def test_save_checkpoint_2(self):
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    # ...
```

### 6. Use Context Managers for Resource Management
```python
# Good: Automatic cleanup
def test_logging(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(
            experiment_name="test",
            log_dir=tmpdir,
            use_wandb=False,
            use_tensorboard=False,
        )
        # Work with logger
    # tmpdir automatically cleaned up

# Less good: Manual cleanup
def test_logging(self):
    tmpdir = tempfile.mkdtemp()
    logger = MetricsLogger(
        experiment_name="test",
        log_dir=tmpdir,
        use_wandb=False,
        use_tensorboard=False,
    )
    # Work with logger
    shutil.rmtree(tmpdir)  # Easy to forget!
```

### 7. Test Edge Cases and Error Conditions
```python
# Good: Multiple scenarios
def test_get_eta(self):
    tracker = ProgressTracker(total_epochs=10, steps_per_epoch=100)

    # Edge case 1: No steps recorded
    eta_no_steps = tracker.get_eta(0, 0)
    assert eta_no_steps == "N/A"

    # Normal case: With recorded steps
    tracker.start_epoch()
    for _ in range(5):
        tracker.step()
    eta_with_steps = tracker.get_eta(0, 5)
    assert eta_with_steps != "N/A"
```

### 8. Use Assertions Effectively
```python
# Good: Specific assertions
assert logger.step == 1
assert os.path.exists(checkpoint_path)
assert isinstance(scheduler, CosineScheduler)
assert 0.001 <= lr <= 0.1

# Bad: Vague assertions
assert logger is not None
assert checkpoint_path is not None
assert scheduler is not None
assert lr > 0
```

## Advanced Testing Techniques

### 1. Parametrized Tests
```python
@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR])
def test_setup_logging_levels(self, level):
    """Test logging at different levels."""
    logger = setup_logging(level=level)
    assert logger.level == level
```

### 2. Testing Multiple Devices
```python
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_on_device(self, device):
    """Test checkpoint saving/loading on different devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test checkpoint operations
    pass
```

### 3. Performance Testing
```python
def test_scheduler_performance(self):
    """Test scheduler doesn't have performance issues."""
    scheduler = CosineScheduler(0.1, 0.001, 100, 1000)

    import time
    start = time.time()
    for step in range(100000):
        _ = scheduler(step)
    elapsed = time.time() - start

    # Should be fast
    assert elapsed < 1.0  # 100k steps in < 1 second
```

### 4. Testing with Temporary Files
```python
def test_checkpoint_cleanup(self, checkpoint_manager, model, optimizer):
    """Test that old checkpoints are cleaned up."""
    import glob

    # Save multiple checkpoints
    for epoch in range(10):
        checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )

    # Count files
    pattern = str(checkpoint_manager.checkpoint_dir / "checkpoint_epoch_*.pth")
    initial_count = len(glob.glob(pattern))

    # Cleanup
    checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)

    # Verify
    final_count = len(glob.glob(pattern))
    assert final_count <= 3
```

## Debugging Failed Tests

### 1. Run Single Test with Verbose Output
```bash
pytest tests/test_utils.py::TestMetricsLogger::test_log_metrics -vv -s --tb=long
```

### 2. Use pytest Debugger
```bash
pytest tests/test_utils.py --pdb  # Drops into debugger on failure
```

### 3. Print Debugging
```python
def test_example(self, logger):
    result = logger.some_method()
    print(f"Result: {result}")  # Shows up with -s flag
    print(f"Logger state: {logger.step}")
    assert result is not None
```

### 4. Check Local Variables on Failure
```bash
pytest tests/test_utils.py -l  # --showlocals
```

### 5. Stop on First Failure
```bash
pytest tests/test_utils.py -x  # Useful for quick debugging
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Utils Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: pip install -e ".[dev]"

    - name: Run tests
      run: pytest tests/test_utils.py --cov=src/utils --cov-fail-under=75
```

## Performance Optimization Tips

### 1. Use Fast Fixtures
```python
# Fast: Reused across tests
@pytest.fixture(scope="module")
def slow_model():
    """Created once per module."""
    return load_large_model()

# Use smaller models for most tests
@pytest.fixture
def simple_model():
    """Created per test, but fast."""
    return nn.Linear(10, 5)
```

### 2. Reduce Test Data Size
```python
# Good: Minimal but sufficient data
batch_size = 2
num_steps = 10

# Bad: Unnecessary large data
batch_size = 128
num_steps = 1000
```

### 3. Skip Slow Tests Conditionally
```python
@pytest.mark.slow
def test_large_checkpoint(self):
    """Skip with: pytest -m "not slow" """
    # Large checkpoint test
    pass
```

## Common Pitfalls to Avoid

### 1. Forgetting Fixture Cleanup
```python
# Bad: May leave temporary files
@pytest.fixture
def temp_dir(self):
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Forgot to clean up!

# Good: Explicit cleanup
@pytest.fixture
def temp_dir(self):
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)
```

### 2. Hardcoding Paths
```python
# Bad: May fail on different systems
checkpoint_path = "/tmp/checkpoint.pth"

# Good: Use temp directories
checkpoint_path = os.path.join(tempfile.gettempdir(), "checkpoint.pth")
```

### 3. Testing Implementation Details
```python
# Bad: Tests internal structure
def test_metrics_logger(self, logger):
    assert len(logger.metrics_buffer) == 0

# Good: Tests behavior
def test_metrics_logger(self, logger):
    logger.accumulate_metrics({"loss": 0.5})
    logger.log_accumulated_metrics()
    # Buffer should be cleared after logging
```

### 4. Weak Assertions
```python
# Bad: Too permissive
assert lr > 0

# Good: Specific bounds
assert 0.001 <= lr <= 0.1
```

## Extending the Test Suite

### Adding Tests for New Features
1. Create new test class with same pattern
2. Use existing fixtures where possible
3. Follow naming conventions
4. Include docstrings for each test
5. Test happy path, edge cases, and errors
6. Verify coverage with: `pytest --cov=src/utils`

### Adding New Fixtures
```python
@pytest.fixture
def my_new_fixture(self):
    """Description of what this fixture provides."""
    # Setup
    resource = create_resource()
    yield resource
    # Teardown
    cleanup_resource(resource)
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://realpython.com/pytest-python-testing/)
- [PyTorch Testing Guide](https://pytorch.org/docs/stable/testing.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

## Summary

Key takeaways for writing good tests:
1. **Isolation**: Each test stands alone
2. **Clarity**: Test names tell you what they test
3. **Coverage**: Happy path, edge cases, errors
4. **Efficiency**: Fast execution, minimal dependencies
5. **Maintainability**: Clear structure, good practices
6. **Debugging**: Easy to understand failures

Follow these practices and your tests will be robust, maintainable, and effective!

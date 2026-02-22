# H-JEPA Utils Module Test Coverage

## Overview

This document describes the comprehensive test suite for the H-JEPA utils module (`src/utils/`). The test suite in `tests/test_utils.py` provides extensive coverage for all utility functions and classes, targeting >60% code coverage.

## Test File Structure

The test suite is organized into the following sections:

### 1. Logging Tests (TestSetupLogging, TestMetricsLogger, TestProgressTracker)

#### TestSetupLogging
Tests for the `setup_logging()` function which configures Python logging.

**Tests:**
- `test_setup_logging_console_only`: Verifies console-only logging setup
- `test_setup_logging_with_file`: Tests file handler creation and message logging
- `test_setup_logging_different_levels`: Validates different log levels (DEBUG, INFO, WARNING, ERROR)
- `test_logging_formatter`: Confirms log format includes timestamp and level

**Coverage:**
- Console handler setup
- File handler setup
- Logger level configuration
- Formatter setup

#### TestMetricsLogger
Comprehensive tests for the `MetricsLogger` class which handles W&B and TensorBoard logging.

**Tests:**
- `test_initialization`: Basic initialization without W&B/TensorBoard
- `test_log_metrics`: Logging simple metric dictionaries
- `test_log_metrics_with_prefix`: Adding prefixes to metric names (e.g., "train/")
- `test_log_metrics_auto_increment_step`: Auto-incrementing global step counter
- `test_log_metrics_manual_step`: Manually setting step values
- `test_accumulate_metrics`: Accumulating metrics over an epoch
- `test_log_accumulated_metrics`: Logging and resetting accumulated metrics
- `test_log_accumulated_metrics_no_reset`: Logging without resetting buffer
- `test_log_accumulated_metrics_empty_buffer`: Handling empty metric buffer
- `test_log_image_tensor`: Logging images from PyTorch tensors
- `test_log_image_numpy`: Logging images from NumPy arrays
- `test_log_image_with_caption`: Adding captions to images
- `test_log_images_batch`: Logging multiple images
- `test_log_images_with_captions`: Logging images with individual captions
- `test_log_histogram_tensor`: Histogram from tensor data
- `test_log_histogram_numpy`: Histogram from NumPy array
- `test_log_model_weights`: Logging weight histograms
- `test_log_model_gradients`: Logging gradient histograms
- `test_log_hierarchical_losses`: Logging hierarchical loss per level
- `test_log_hierarchical_losses_custom_prefix`: Custom prefix for hierarchy losses
- `test_log_prediction_comparison`: Logging prediction vs target comparisons
- `test_log_embeddings`: Logging embeddings with labels
- `test_log_embeddings_no_labels`: Logging embeddings without labels
- `test_log_system_metrics`: GPU memory and utilization metrics
- `test_context_manager`: Using MetricsLogger as context manager
- `test_watch_model`: Model watching for gradient tracking

**Coverage:**
- Initialization with W&B and TensorBoard flags
- Metrics logging and step management
- Image logging (single and batch)
- Histogram creation
- Model weight and gradient tracking
- Hierarchical loss logging
- Prediction visualization
- Embedding visualization
- System metrics collection
- Context manager protocol

#### TestProgressTracker
Tests for the `ProgressTracker` class which tracks training progress and ETA.

**Tests:**
- `test_initialization`: ProgressTracker setup
- `test_start_epoch`: Epoch timing initialization
- `test_step_tracking`: Recording step times
- `test_step_times_buffer_limit`: Buffer size limiting to 100 steps
- `test_get_eta_no_steps`: ETA with no recorded steps
- `test_get_eta_with_steps`: ETA calculation with step history
- `test_get_elapsed_time`: Total elapsed time calculation
- `test_format_time`: Time formatting to HH:MM:SS

**Coverage:**
- Epoch and step timing
- ETA calculation
- Moving average of step times
- Time formatting
- Elapsed time tracking

### 2. Checkpoint Management Tests

#### TestCheckpointManager
Tests for the `CheckpointManager` class which manages training checkpoints.

**Tests:**
- `test_initialization`: Basic CheckpointManager setup
- `test_initialization_max_mode`: Initialization in max metric mode
- `test_should_save`: Frequency-based checkpoint saving
- `test_is_better_metric_min_mode`: Metric comparison (minimization)
- `test_is_better_metric_max_mode`: Metric comparison (maximization)
- `test_update_best_metric`: Tracking best metric values
- `test_save_checkpoint`: Basic checkpoint saving
- `test_save_checkpoint_creates_latest`: Latest checkpoint creation
- `test_save_checkpoint_best_flag`: Best checkpoint marking
- `test_save_checkpoint_with_scaler`: Saving with GradScaler state
- `test_save_checkpoint_with_extra_state`: Extra metadata storage
- `test_load_checkpoint`: Loading checkpoint and restoring state
- `test_load_checkpoint_nonexistent`: Error handling for missing files
- `test_get_latest_checkpoint`: Retrieving latest checkpoint
- `test_get_latest_checkpoint_none`: Handling when no checkpoints exist
- `test_get_best_checkpoint`: Retrieving best checkpoint
- `test_get_best_checkpoint_none`: Handling when no best exists
- `test_cleanup_old_checkpoints`: Cleaning up old periodic checkpoints
- `test_dataparallel_model_save_load`: DataParallel model support

**Coverage:**
- CheckpointManager initialization
- Checkpoint saving (regular, latest, best)
- Checkpoint loading
- Metric tracking
- File management and cleanup
- Support for DataParallel models
- Extra state handling

#### TestSaveCheckpoint
Tests for standalone `save_checkpoint()` function.

**Tests:**
- `test_save_checkpoint_basic`: Basic checkpoint saving
- `test_save_checkpoint_with_kwargs`: Saving extra metadata

**Coverage:**
- Standalone checkpoint saving
- Metadata inclusion

#### TestLoadCheckpoint
Tests for standalone `load_checkpoint()` function.

**Tests:**
- `test_load_checkpoint_basic`: Basic checkpoint loading
- `test_load_checkpoint_no_optimizer`: Loading without optimizer

**Coverage:**
- Standalone checkpoint loading
- Optional optimizer loading

### 3. Scheduler Tests

#### TestCosineScheduler
Tests for the `CosineScheduler` class implementing cosine annealing.

**Tests:**
- `test_initialization`: Scheduler setup
- `test_warmup_phase`: Linear warmup phase learning rate
- `test_cosine_annealing_phase`: Cosine annealing schedule
- `test_cosine_scheduler_no_warmup`: Schedule without warmup
- `test_get_epoch_value`: Getting LR for specific epoch
- `test_monotonic_during_warmup`: Warmup monotonicity
- `test_monotonic_after_warmup`: Post-warmup monotonic decrease

**Coverage:**
- Warmup phase
- Cosine annealing phase
- Step-wise and epoch-wise access
- Schedule monotonicity

#### TestLinearScheduler
Tests for the `LinearScheduler` class implementing linear scheduling.

**Tests:**
- `test_initialization`: Scheduler setup
- `test_warmup_phase`: Linear warmup phase
- `test_linear_decay_phase`: Linear decay after warmup
- `test_linear_scheduler_no_warmup`: Schedule without warmup
- `test_monotonic_decay`: Decay monotonicity

**Coverage:**
- Warmup phase
- Linear decay phase
- Schedule monotonicity

#### TestEMAScheduler
Tests for the `EMAScheduler` class for exponential moving average momentum.

**Tests:**
- `test_initialization`: Scheduler setup
- `test_warmup_phase`: Constant momentum during warmup
- `test_linear_schedule_after_warmup`: Linear schedule post-warmup
- `test_ema_scheduler_call_and_step_methods`: Both access methods
- `test_ema_bounds`: Momentum bounds (0.996-1.0)

**Coverage:**
- Warmup phase
- Linear momentum scheduling
- Bound enforcement
- Method equivalence

#### TestHierarchicalScheduler
Tests for the `HierarchicalScheduler` managing multiple schedulers.

**Tests:**
- `test_initialization`: Multi-scheduler setup
- `test_call_returns_list`: List of values per level
- `test_get_level_value`: Per-level value access
- `test_mixed_scheduler_types`: Mixed scheduler types

**Coverage:**
- Multiple scheduler management
- Per-level scheduling
- Mixed scheduler support

### 4. Factory Function Tests

#### TestCreateLRScheduler
Tests for the `create_lr_scheduler()` factory function.

**Tests:**
- `test_create_cosine_scheduler`: Create cosine scheduler
- `test_create_linear_scheduler`: Create linear scheduler
- `test_create_scheduler_with_warmup`: Scheduler with warmup
- `test_create_scheduler_invalid_type`: Error handling

**Coverage:**
- Factory function routing
- Parameter passing
- Error handling

#### TestCreateEMAScheduler
Tests for the `create_ema_scheduler()` factory function.

**Tests:**
- `test_create_ema_scheduler`: Create EMA scheduler
- `test_create_ema_scheduler_with_warmup`: EMA with warmup

**Coverage:**
- EMA scheduler creation
- Parameter handling

### 5. Integration Tests

#### TestUtilsIntegration
End-to-end integration tests combining multiple utilities.

**Tests:**
- `test_full_training_pipeline_metrics`: Complete metrics logging pipeline
- `test_full_checkpoint_pipeline`: Complete checkpoint save/load flow
- `test_scheduler_pipeline`: Multi-epoch scheduler simulation

**Coverage:**
- Real-world usage patterns
- Cross-module integration
- Multi-epoch training simulation

## Running the Tests

### Basic Test Execution
```bash
# Run all utils tests
pytest tests/test_utils.py -v

# Run specific test class
pytest tests/test_utils.py::TestCosineScheduler -v

# Run specific test
pytest tests/test_utils.py::TestCosineScheduler::test_warmup_phase -v
```

### Coverage Analysis
```bash
# Run with coverage report
pytest tests/test_utils.py --cov=src/utils --cov-report=term-missing

# Generate HTML coverage report
pytest tests/test_utils.py --cov=src/utils --cov-report=html

# Coverage for specific module
pytest tests/test_utils.py --cov=src/utils/logging --cov-report=term-missing
```

### Markers and Filtering
```bash
# Run only fast tests
pytest tests/test_utils.py -m "not slow"

# Run with detailed output
pytest tests/test_utils.py -vv --tb=long

# Stop on first failure
pytest tests/test_utils.py -x
```

## Coverage Summary

The test suite provides comprehensive coverage of:

### Logging Module (`src/utils/logging.py`)
- `setup_logging()` function: Console and file handlers, log levels, formatting
- `MetricsLogger` class: Initialization, metrics logging, image logging, histogram logging, model tracking, system metrics, context manager protocol
- `ProgressTracker` class: Epoch tracking, step timing, ETA calculation, time formatting

### Checkpoint Module (`src/utils/checkpoint.py`)
- `CheckpointManager` class: Initialization, checkpoint saving/loading, best metric tracking, cleanup, DataParallel support
- `save_checkpoint()` function: Basic checkpoint saving with metadata
- `load_checkpoint()` function: Checkpoint restoration

### Scheduler Module (`src/utils/scheduler.py`)
- `CosineScheduler` class: Warmup, cosine annealing, monotonicity
- `LinearScheduler` class: Warmup, linear decay, monotonicity
- `EMAScheduler` class: Momentum scheduling, bounds, warmup
- `HierarchicalScheduler` class: Multi-level scheduling
- `create_lr_scheduler()` function: Factory routing and error handling
- `create_ema_scheduler()` function: EMA creation

### Integration Tests
- Complete training pipelines
- Multi-module interactions
- Real-world usage patterns

## Expected Coverage Targets

Based on the test suite structure:

| Module | Expected Coverage |
|--------|------------------|
| `src/utils/logging.py` | 75-85% |
| `src/utils/checkpoint.py` | 80-90% |
| `src/utils/scheduler.py` | 85-95% |
| **Overall `src/utils/`** | **80-90%** |

## Dependencies

The test suite requires:
- `pytest>=7.4.0`
- `pytest-cov>=4.1.0`
- `torch>=2.0.0`
- `numpy>=1.24.0`
- `tensorboard>=2.13.0` (optional for TensorBoard tests)
- `wandb>=0.15.0` (optional for W&B tests)

## Key Features

1. **Comprehensive Coverage**: Tests cover normal operations, edge cases, and error conditions
2. **Fixture-Based**: Uses pytest fixtures for reusable test components
3. **Temporary Directories**: Uses `tempfile` for isolated file operations
4. **Integration Tests**: Includes end-to-end training scenarios
5. **Device Agnostic**: Tests work on CPU and CUDA devices
6. **Type Safety**: Tests validate correct types and shapes
7. **Boundary Testing**: Tests edge cases like empty buffers and missing files
8. **Error Handling**: Tests exception cases and error conditions

## Future Enhancements

Potential additions for even higher coverage:

1. W&B integration tests (currently mocked as unavailable)
2. TensorBoard integration tests (currently mocked as unavailable)
3. GPU-specific tests (CUDA memory tracking, multi-GPU)
4. Performance benchmarks for scheduler calculations
5. Stress tests with very large checkpoint files
6. Distributed training checkpoint tests
7. Custom scheduler edge cases
8. Network failure scenarios for W&B/TensorBoard

## Notes

- Tests mock external services (W&B, TensorBoard) to avoid dependencies
- Temporary directories are properly cleaned up after tests
- Tests are reproducible with fixed random seeds where applicable
- All tests are designed to run quickly (<1 minute total)
- Tests follow pytest best practices and naming conventions

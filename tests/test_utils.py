"""
Comprehensive test suite for utility modules in H-JEPA.

Tests coverage for:
- Logging utilities (setup_logging, MetricsLogger, ProgressTracker)
- Checkpoint management (CheckpointManager, save_checkpoint, load_checkpoint)
- Learning rate and EMA schedulers (CosineScheduler, LinearScheduler, EMAScheduler, HierarchicalScheduler)
"""

import logging
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from src.utils import (
    CheckpointManager,
    CosineScheduler,
    EMAScheduler,
    HierarchicalScheduler,
    LinearScheduler,
    MetricsLogger,
    ProgressTracker,
    create_ema_scheduler,
    create_lr_scheduler,
    load_checkpoint,
    save_checkpoint,
    setup_logging,
)

# ============================================================================
# LOGGING TESTS
# ============================================================================


class TestSetupLogging:
    """Test suite for setup_logging function."""

    def test_setup_logging_console_only(self):
        """Test logging setup with console output only."""
        logger = setup_logging(log_file=None, level=logging.INFO)

        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = setup_logging(log_file=log_file, level=logging.DEBUG)

            assert logger is not None
            assert logger.level == logging.DEBUG

            # Test that we can log
            logger.info("Test message")

            # Check file was created and contains message
            assert os.path.exists(log_file)
            with open(log_file, "r") as f:
                content = f.read()
                assert "Test message" in content

    def test_setup_logging_different_levels(self):
        """Test logging setup with different log levels."""
        for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]:
            logger = setup_logging(level=level)
            assert logger.level == level

    def test_logging_formatter(self):
        """Test that log formatter is correctly configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = setup_logging(log_file=log_file, level=logging.INFO)

            logger.info("Formatted log message")

            with open(log_file, "r") as f:
                content = f.read()
                # Check that formatter includes timestamp and level
                assert "Formatted log message" in content
                assert "INFO" in content


class TestMetricsLogger:
    """Test suite for MetricsLogger class."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def metrics_logger(self, temp_log_dir):
        """Create a MetricsLogger instance without W&B/TensorBoard."""
        logger = MetricsLogger(
            experiment_name="test_exp",
            log_dir=temp_log_dir,
            config={"test": "config"},
            use_wandb=False,
            use_tensorboard=False,
        )
        yield logger
        logger.finish()

    def test_initialization(self, temp_log_dir):
        """Test MetricsLogger initialization."""
        logger = MetricsLogger(
            experiment_name="test",
            log_dir=temp_log_dir,
            use_wandb=False,
            use_tensorboard=False,
        )

        assert logger.experiment_name == "test"
        assert logger.log_dir == Path(temp_log_dir)
        assert logger.step == 0
        assert logger.use_wandb is False
        assert logger.use_tensorboard is False

        logger.finish()

    def test_log_metrics(self, metrics_logger):
        """Test logging simple metrics."""
        metrics = {"loss": 0.5, "accuracy": 0.95}
        metrics_logger.log_metrics(metrics)

        assert metrics_logger.step == 1

    def test_log_metrics_with_prefix(self, metrics_logger):
        """Test logging metrics with prefix."""
        metrics = {"loss": 0.5, "accuracy": 0.95}
        metrics_logger.log_metrics(metrics, prefix="train/")

        assert metrics_logger.step == 1

    def test_log_metrics_auto_increment_step(self, metrics_logger):
        """Test that step auto-increments."""
        metrics_logger.log_metrics({"loss": 0.5})
        assert metrics_logger.step == 1

        metrics_logger.log_metrics({"loss": 0.4})
        assert metrics_logger.step == 2

    def test_log_metrics_manual_step(self, metrics_logger):
        """Test setting step manually."""
        metrics_logger.log_metrics({"loss": 0.5}, step=10)
        assert metrics_logger.step == 10

        metrics_logger.log_metrics({"loss": 0.4}, step=20)
        assert metrics_logger.step == 20

    def test_accumulate_metrics(self, metrics_logger):
        """Test metric accumulation."""
        metrics1 = {"loss": 0.5, "accuracy": 0.9}
        metrics2 = {"loss": 0.4, "accuracy": 0.92}

        metrics_logger.accumulate_metrics(metrics1)
        metrics_logger.accumulate_metrics(metrics2)

        assert len(metrics_logger.metrics_buffer["loss"]) == 2
        assert len(metrics_logger.metrics_buffer["accuracy"]) == 2

    def test_log_accumulated_metrics(self, metrics_logger):
        """Test logging accumulated metrics."""
        metrics_logger.accumulate_metrics({"loss": 0.5, "accuracy": 0.9})
        metrics_logger.accumulate_metrics({"loss": 0.4, "accuracy": 0.92})

        metrics_logger.log_accumulated_metrics(step=0)

        # Buffer should be reset
        assert len(metrics_logger.metrics_buffer) == 0

    def test_log_accumulated_metrics_no_reset(self, metrics_logger):
        """Test logging accumulated metrics without reset."""
        metrics_logger.accumulate_metrics({"loss": 0.5})
        metrics_logger.log_accumulated_metrics(reset=False)

        # Buffer should not be cleared
        assert len(metrics_logger.metrics_buffer) > 0

    def test_log_accumulated_metrics_empty_buffer(self, metrics_logger):
        """Test logging with empty buffer."""
        # Should not raise error
        metrics_logger.log_accumulated_metrics()

    def test_log_image_tensor(self, metrics_logger):
        """Test logging image from tensor."""
        image = torch.randn(3, 224, 224)
        metrics_logger.log_image("test_image", image)
        # log_image doesn't increment step - that's done by log_metrics

    def test_log_image_numpy(self, metrics_logger):
        """Test logging image from numpy array."""
        image = np.random.randn(3, 224, 224).astype(np.float32)
        metrics_logger.log_image("test_image", image)
        # log_image doesn't increment step - that's done by log_metrics

    def test_log_image_with_caption(self, metrics_logger):
        """Test logging image with caption."""
        image = torch.randn(3, 224, 224)
        metrics_logger.log_image("test_image", image, caption="Test caption")
        # log_image doesn't increment step - that's done by log_metrics

    def test_log_images_batch(self, metrics_logger):
        """Test logging multiple images."""
        images = [torch.randn(3, 224, 224) for _ in range(3)]
        metrics_logger.log_images("test_images", images)
        # log_images doesn't increment step - that's done by log_metrics

    def test_log_images_with_captions(self, metrics_logger):
        """Test logging multiple images with captions."""
        images = [torch.randn(3, 224, 224) for _ in range(3)]
        captions = ["image1", "image2", "image3"]
        metrics_logger.log_images("test_images", images, captions=captions)
        # log_images doesn't increment step - that's done by log_metrics

    def test_log_histogram_tensor(self, metrics_logger):
        """Test logging histogram from tensor."""
        values = torch.randn(1000)
        metrics_logger.log_histogram("weights", values)
        # log_histogram doesn't increment step - that's done by log_metrics

    def test_log_histogram_numpy(self, metrics_logger):
        """Test logging histogram from numpy array."""
        values = np.random.randn(1000).astype(np.float32)
        metrics_logger.log_histogram("weights", values)
        # log_histogram doesn't increment step - that's done by log_metrics

    def test_log_model_weights(self, metrics_logger):
        """Test logging model weights."""
        model = nn.Linear(10, 5)
        metrics_logger.log_model_weights(model)
        # log_model_weights doesn't increment step - that's done by log_metrics

    def test_log_model_gradients(self, metrics_logger):
        """Test logging model gradients."""
        model = nn.Linear(10, 5)
        # Create a forward pass to generate gradients
        x = torch.randn(4, 10)
        y = model(x).sum()
        y.backward()

        metrics_logger.log_model_gradients(model)
        # log_model_gradients doesn't increment step - that's done by log_metrics

    def test_log_hierarchical_losses(self, metrics_logger):
        """Test logging hierarchical losses."""
        loss_dict = {
            "loss_level_0": 0.5,
            "loss_level_1": 0.3,
            "loss_level_2": 0.2,
        }
        metrics_logger.log_hierarchical_losses(loss_dict)
        # log_hierarchical_losses calls log_metrics, which will increment step
        assert metrics_logger.step >= 1

    def test_log_hierarchical_losses_custom_prefix(self, metrics_logger):
        """Test logging hierarchical losses with custom prefix."""
        loss_dict = {
            "loss_level_0": 0.5,
            "loss_level_1": 0.3,
        }
        metrics_logger.log_hierarchical_losses(loss_dict, step=0, prefix="custom/hierarchy/")

    def test_log_prediction_comparison(self, metrics_logger):
        """Test logging prediction comparison."""
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        predictions = [torch.randn(batch_size, 196, 768) for _ in range(3)]
        targets = [torch.randn(batch_size, 196, 768) for _ in range(3)]
        masks = torch.ones(batch_size, 196)

        metrics_logger.log_prediction_comparison(images, predictions, targets, masks, step=0)

    def test_log_embeddings(self, metrics_logger):
        """Test logging embeddings."""
        embeddings = torch.randn(100, 128)
        labels = torch.randint(0, 10, (100,))
        metrics_logger.log_embeddings(embeddings, labels, step=0)

    def test_log_embeddings_no_labels(self, metrics_logger):
        """Test logging embeddings without labels."""
        embeddings = torch.randn(100, 128)
        metrics_logger.log_embeddings(embeddings, step=0)

    def test_log_system_metrics(self, metrics_logger):
        """Test logging system metrics."""
        metrics_logger.log_system_metrics()
        # log_system_metrics calls log_metrics which increments step
        assert metrics_logger.step >= 1

    def test_context_manager(self, temp_log_dir):
        """Test MetricsLogger as context manager."""
        with MetricsLogger(
            experiment_name="test",
            log_dir=temp_log_dir,
            use_wandb=False,
            use_tensorboard=False,
        ) as logger:
            logger.log_metrics({"loss": 0.5})
            assert logger.step == 1

    def test_watch_model(self, metrics_logger):
        """Test watch_model method."""
        model = nn.Linear(10, 5)
        # Should not raise error even if W&B is not available
        metrics_logger.watch_model(model)


class TestProgressTracker:
    """Test suite for ProgressTracker class."""

    def test_initialization(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker(total_epochs=10, steps_per_epoch=100)

        assert tracker.total_epochs == 10
        assert tracker.steps_per_epoch == 100
        assert tracker.total_steps == 1000
        assert tracker.step_times == []

    def test_start_epoch(self):
        """Test epoch start timing."""
        tracker = ProgressTracker(total_epochs=10, steps_per_epoch=100)
        tracker.start_epoch()

        assert tracker.epoch_start_time is not None

    def test_step_tracking(self):
        """Test step tracking."""
        tracker = ProgressTracker(total_epochs=10, steps_per_epoch=100)
        tracker.start_epoch()

        time.sleep(0.01)  # Small delay
        tracker.step()

        assert len(tracker.step_times) > 0
        assert tracker.step_times[0] > 0

    def test_step_times_buffer_limit(self):
        """Test that step times buffer is limited to 100 steps."""
        tracker = ProgressTracker(total_epochs=10, steps_per_epoch=200)
        tracker.start_epoch()

        for _ in range(150):
            tracker.step()

        # Should keep only 100 recent steps
        assert len(tracker.step_times) <= 100

    def test_get_eta_no_steps(self):
        """Test ETA calculation with no steps."""
        tracker = ProgressTracker(total_epochs=10, steps_per_epoch=100)

        eta = tracker.get_eta(current_epoch=0, current_step=0)
        assert eta == "N/A"

    def test_get_eta_with_steps(self):
        """Test ETA calculation with recorded steps."""
        tracker = ProgressTracker(total_epochs=2, steps_per_epoch=10)
        tracker.start_epoch()

        # Record some steps
        for _ in range(5):
            time.sleep(0.001)
            tracker.step()

        eta = tracker.get_eta(current_epoch=0, current_step=5)
        assert eta != "N/A"
        assert ":" in eta  # Should be HH:MM:SS format

    def test_get_elapsed_time(self):
        """Test elapsed time calculation."""
        tracker = ProgressTracker(total_epochs=10, steps_per_epoch=100)

        time.sleep(1.1)  # Sleep for > 1 second so it shows in HH:MM:SS format
        elapsed = tracker.get_elapsed_time()

        assert ":" in elapsed  # Should be HH:MM:SS format
        assert elapsed != "00:00:00"  # Should show at least 1 second

    def test_format_time(self):
        """Test time formatting."""
        # 1 hour, 2 minutes, 3 seconds = 3723 seconds
        formatted = ProgressTracker._format_time(3723)
        assert formatted == "01:02:03"

        # 10 seconds
        formatted = ProgressTracker._format_time(10)
        assert formatted == "00:00:10"

        # 3661 seconds = 1 hour, 1 minute, 1 second
        formatted = ProgressTracker._format_time(3661)
        assert formatted == "01:01:01"


# ============================================================================
# CHECKPOINT TESTS
# ============================================================================


class TestCheckpointManager:
    """Test suite for CheckpointManager class."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Create a CheckpointManager instance."""
        return CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            keep_best_n=3,
            save_frequency=5,
            metric_name="val_loss",
            mode="min",
        )

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

    @pytest.fixture
    def simple_optimizer(self, simple_model):
        """Create an optimizer for the model."""
        return Adam(simple_model.parameters(), lr=0.001)

    def test_initialization(self, temp_checkpoint_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            keep_best_n=5,
            save_frequency=10,
            metric_name="val_loss",
            mode="min",
        )

        assert manager.checkpoint_dir == Path(temp_checkpoint_dir)
        assert manager.keep_best_n == 5
        assert manager.save_frequency == 10
        assert manager.metric_name == "val_loss"
        assert manager.mode == "min"
        assert manager.best_metric == float("inf")

    def test_initialization_max_mode(self, temp_checkpoint_dir):
        """Test CheckpointManager initialization in max mode."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            mode="max",
        )

        assert manager.mode == "max"
        assert manager.best_metric == float("-inf")

    def test_should_save(self, checkpoint_manager):
        """Test should_save frequency check."""
        assert checkpoint_manager.should_save(4) is True  # (4+1) % 5 == 0
        assert checkpoint_manager.should_save(0) is False
        assert checkpoint_manager.should_save(3) is False
        assert checkpoint_manager.should_save(9) is True  # (9+1) % 5 == 0

    def test_is_better_metric_min_mode(self, checkpoint_manager):
        """Test metric comparison in min mode."""
        assert checkpoint_manager.is_better_metric(0.5) is True
        checkpoint_manager.best_metric = 0.5

        assert checkpoint_manager.is_better_metric(0.4) is True
        assert checkpoint_manager.is_better_metric(0.5) is False
        assert checkpoint_manager.is_better_metric(0.6) is False

    def test_is_better_metric_max_mode(self, temp_checkpoint_dir):
        """Test metric comparison in max mode."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            mode="max",
        )

        assert manager.is_better_metric(0.5) is True
        manager.best_metric = 0.5

        assert manager.is_better_metric(0.6) is True
        assert manager.is_better_metric(0.5) is False
        assert manager.is_better_metric(0.4) is False

    def test_update_best_metric(self, checkpoint_manager):
        """Test updating best metric."""
        assert checkpoint_manager.update_best_metric(0.5) is True
        assert checkpoint_manager.best_metric == 0.5

        assert checkpoint_manager.update_best_metric(0.4) is True
        assert checkpoint_manager.best_metric == 0.4

        assert checkpoint_manager.update_best_metric(0.5) is False
        assert checkpoint_manager.best_metric == 0.4

    def test_save_checkpoint(self, checkpoint_manager, simple_model, simple_optimizer):
        """Test saving a checkpoint."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            epoch=0,
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
            metrics={"val_loss": 0.5},
        )

        assert os.path.exists(checkpoint_path)
        assert checkpoint_path.endswith(".pth")

    def test_save_checkpoint_creates_latest(
        self, checkpoint_manager, simple_model, simple_optimizer
    ):
        """Test that save_checkpoint creates latest checkpoint."""
        checkpoint_manager.save_checkpoint(
            epoch=0,
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
        )

        latest_path = checkpoint_manager.checkpoint_dir / "checkpoint_latest.pth"
        assert latest_path.exists()

    def test_save_checkpoint_best_flag(self, checkpoint_manager, simple_model, simple_optimizer):
        """Test saving best checkpoint."""
        checkpoint_manager.save_checkpoint(
            epoch=0,
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
            metrics={"val_loss": 0.5},
            is_best=True,
        )

        best_path = checkpoint_manager.checkpoint_dir / "checkpoint_best.pth"
        assert best_path.exists()

    def test_save_checkpoint_with_scaler(self, checkpoint_manager, simple_model, simple_optimizer):
        """Test saving checkpoint with GradScaler."""
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        checkpoint_path = checkpoint_manager.save_checkpoint(
            epoch=0,
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
            scaler=scaler,
        )

        assert os.path.exists(checkpoint_path)

    def test_save_checkpoint_with_extra_state(
        self, checkpoint_manager, simple_model, simple_optimizer
    ):
        """Test saving checkpoint with extra state."""
        extra_state = {"custom_value": 42, "custom_list": [1, 2, 3]}

        checkpoint_path = checkpoint_manager.save_checkpoint(
            epoch=0,
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
            extra_state=extra_state,
        )

        checkpoint = torch.load(checkpoint_path)
        assert checkpoint["custom_value"] == 42
        assert checkpoint["custom_list"] == [1, 2, 3]

    def test_load_checkpoint(self, checkpoint_manager, simple_model, simple_optimizer):
        """Test loading a checkpoint."""
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            epoch=5,
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
            metrics={"val_loss": 0.3},
        )

        # Create new model and optimizer
        new_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        new_optimizer = Adam(new_model.parameters(), lr=0.001)

        # Load checkpoint
        metadata = checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            device="cpu",
        )

        assert metadata["epoch"] == 5
        assert metadata["metrics"]["val_loss"] == 0.3

    def test_load_checkpoint_nonexistent(self, checkpoint_manager, simple_model):
        """Test loading non-existent checkpoint."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(
                checkpoint_path="/nonexistent/path.pth",
                model=simple_model,
                device="cpu",
            )

    def test_get_latest_checkpoint(self, checkpoint_manager, simple_model, simple_optimizer):
        """Test getting latest checkpoint."""
        # Save multiple checkpoints
        checkpoint_manager.save_checkpoint(
            epoch=0,
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
        )

        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        assert os.path.exists(latest)

    def test_get_latest_checkpoint_none(self, temp_checkpoint_dir):
        """Test getting latest checkpoint when none exist."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        latest = manager.get_latest_checkpoint()
        assert latest is None

    def test_get_best_checkpoint(self, checkpoint_manager, simple_model, simple_optimizer):
        """Test getting best checkpoint."""
        checkpoint_manager.save_checkpoint(
            epoch=0,
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=None,
            is_best=True,
        )

        best = checkpoint_manager.get_best_checkpoint()
        assert best is not None
        assert os.path.exists(best)

    def test_get_best_checkpoint_none(self, temp_checkpoint_dir):
        """Test getting best checkpoint when none exist."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        best = manager.get_best_checkpoint()
        assert best is None

    def test_cleanup_old_checkpoints(self, checkpoint_manager, simple_model, simple_optimizer):
        """Test cleaning up old checkpoints."""
        # Save multiple checkpoints
        for epoch in range(10):
            checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=simple_model,
                optimizer=simple_optimizer,
                scheduler=None,
            )

        # Get initial count
        checkpoint_pattern = str(checkpoint_manager.checkpoint_dir / "checkpoint_epoch_*.pth")
        import glob

        initial_count = len(glob.glob(checkpoint_pattern))

        # Cleanup
        checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)

        # Should have reduced the number of checkpoints
        final_count = len(glob.glob(checkpoint_pattern))
        assert final_count <= 3

    def test_dataparallel_model_save_load(self, checkpoint_manager, simple_model, simple_optimizer):
        """Test saving and loading DataParallel models."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = nn.DataParallel(simple_model.to(device))
            optimizer = Adam(model.parameters(), lr=0.001)

            checkpoint_path = checkpoint_manager.save_checkpoint(
                epoch=0,
                model=model,
                optimizer=optimizer,
                scheduler=None,
            )

            assert os.path.exists(checkpoint_path)


class TestSaveCheckpoint:
    """Test standalone save_checkpoint function."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    @pytest.fixture
    def simple_optimizer(self, simple_model):
        """Create an optimizer for the model."""
        return SGD(simple_model.parameters(), lr=0.01)

    def test_save_checkpoint_basic(self, temp_checkpoint_dir, simple_model, simple_optimizer):
        """Test basic checkpoint saving."""
        filepath = os.path.join(temp_checkpoint_dir, "checkpoint.pth")

        save_checkpoint(
            filepath=filepath,
            model=simple_model,
            optimizer=simple_optimizer,
            epoch=0,
        )

        assert os.path.exists(filepath)

    def test_save_checkpoint_with_kwargs(self, temp_checkpoint_dir, simple_model, simple_optimizer):
        """Test checkpoint saving with extra kwargs."""
        filepath = os.path.join(temp_checkpoint_dir, "checkpoint.pth")

        save_checkpoint(
            filepath=filepath,
            model=simple_model,
            optimizer=simple_optimizer,
            epoch=5,
            custom_metric=0.95,
            custom_list=[1, 2, 3],
        )

        checkpoint = torch.load(filepath)
        assert checkpoint["custom_metric"] == 0.95
        assert checkpoint["custom_list"] == [1, 2, 3]


class TestLoadCheckpoint:
    """Test standalone load_checkpoint function."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    @pytest.fixture
    def simple_optimizer(self, simple_model):
        """Create an optimizer for the model."""
        return SGD(simple_model.parameters(), lr=0.01)

    def test_load_checkpoint_basic(self, temp_checkpoint_dir, simple_model, simple_optimizer):
        """Test basic checkpoint loading."""
        filepath = os.path.join(temp_checkpoint_dir, "checkpoint.pth")

        # Save checkpoint
        save_checkpoint(filepath, simple_model, simple_optimizer, epoch=5)

        # Create new model and load
        new_model = nn.Linear(10, 5)
        new_optimizer = SGD(new_model.parameters(), lr=0.01)

        checkpoint = load_checkpoint(filepath, new_model, new_optimizer, device="cpu")

        assert checkpoint["epoch"] == 5

    def test_load_checkpoint_no_optimizer(
        self, temp_checkpoint_dir, simple_model, simple_optimizer
    ):
        """Test loading checkpoint without optimizer."""
        filepath = os.path.join(temp_checkpoint_dir, "checkpoint.pth")

        # Save checkpoint
        save_checkpoint(filepath, simple_model, simple_optimizer, epoch=5)

        # Load without optimizer
        new_model = nn.Linear(10, 5)
        checkpoint = load_checkpoint(filepath, new_model, device="cpu")

        assert checkpoint["epoch"] == 5


# ============================================================================
# SCHEDULER TESTS
# ============================================================================


class TestCosineScheduler:
    """Test suite for CosineScheduler."""

    def test_initialization(self):
        """Test CosineScheduler initialization."""
        scheduler = CosineScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        assert scheduler.base_value == 0.1
        assert scheduler.final_value == 0.001
        assert scheduler.epochs == 100
        assert scheduler.steps_per_epoch == 1000
        assert scheduler.warmup_epochs == 10
        assert scheduler.warmup_steps == 10000
        assert scheduler.total_steps == 100000

    def test_warmup_phase(self):
        """Test learning rate during warmup phase."""
        scheduler = CosineScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
            start_warmup_value=0.0,
        )

        # At step 0 (start of warmup)
        lr_step_0 = scheduler(0)
        assert lr_step_0 == 0.0

        # At step 5000 (midway through warmup)
        lr_mid = scheduler(5000)
        assert 0.0 < lr_mid < 0.1

        # At step 10000 (end of warmup)
        lr_step_warmup = scheduler(10000)
        assert lr_step_warmup == 0.1

    def test_cosine_annealing_phase(self):
        """Test learning rate during cosine annealing phase."""
        scheduler = CosineScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        # Just after warmup
        lr_after_warmup = scheduler(10000)
        assert abs(lr_after_warmup - 0.1) < 0.01

        # At end of training
        lr_final = scheduler(99999)
        assert abs(lr_final - 0.001) < 0.01

    def test_cosine_scheduler_no_warmup(self):
        """Test CosineScheduler without warmup."""
        scheduler = CosineScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=0,
        )

        lr_start = scheduler(0)
        assert abs(lr_start - 0.1) < 0.01

    def test_get_epoch_value(self):
        """Test getting learning rate for epoch."""
        scheduler = CosineScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        # Should be same as calling with epoch * steps_per_epoch
        lr_epoch_0 = scheduler.get_epoch_value(0)
        lr_direct = scheduler(0)
        assert abs(lr_epoch_0 - lr_direct) < 1e-6

        lr_epoch_10 = scheduler.get_epoch_value(10)
        lr_direct_10 = scheduler(10000)
        assert abs(lr_epoch_10 - lr_direct_10) < 1e-6

    def test_monotonic_during_warmup(self):
        """Test that LR is monotonically increasing during warmup."""
        scheduler = CosineScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
            start_warmup_value=0.0,
        )

        prev_lr = 0.0
        for step in range(0, 10000, 1000):
            lr = scheduler(step)
            assert lr >= prev_lr
            prev_lr = lr

    def test_monotonic_after_warmup(self):
        """Test that LR is monotonically decreasing after warmup."""
        scheduler = CosineScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        prev_lr = 0.1
        for step in range(10000, 100000, 10000):
            lr = scheduler(step)
            assert lr <= prev_lr
            prev_lr = lr


class TestLinearScheduler:
    """Test suite for LinearScheduler."""

    def test_initialization(self):
        """Test LinearScheduler initialization."""
        scheduler = LinearScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        assert scheduler.base_value == 0.1
        assert scheduler.final_value == 0.001
        assert scheduler.warmup_steps == 10000

    def test_warmup_phase(self):
        """Test learning rate during warmup phase."""
        scheduler = LinearScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
            start_warmup_value=0.0,
        )

        # At step 0
        lr_step_0 = scheduler(0)
        assert lr_step_0 == 0.0

        # At step 5000 (midway)
        lr_mid = scheduler(5000)
        assert abs(lr_mid - 0.05) < 0.01

        # At step 10000 (end of warmup)
        lr_warmup = scheduler(10000)
        assert abs(lr_warmup - 0.1) < 0.001

    def test_linear_decay_phase(self):
        """Test learning rate during linear decay phase."""
        scheduler = LinearScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        # After warmup, should decay linearly
        lr_after_warmup = scheduler(10000)
        assert abs(lr_after_warmup - 0.1) < 0.001

        # At end of training
        lr_final = scheduler(99999)
        assert abs(lr_final - 0.001) < 0.01

    def test_linear_scheduler_no_warmup(self):
        """Test LinearScheduler without warmup."""
        scheduler = LinearScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=0,
        )

        lr_start = scheduler(0)
        assert abs(lr_start - 0.1) < 0.001

    def test_monotonic_decay(self):
        """Test monotonic decay after warmup."""
        scheduler = LinearScheduler(
            base_value=0.1,
            final_value=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        prev_lr = 0.1
        for step in range(10000, 100000, 10000):
            lr = scheduler(step)
            assert lr <= prev_lr
            prev_lr = lr


class TestEMAScheduler:
    """Test suite for EMAScheduler."""

    def test_initialization(self):
        """Test EMAScheduler initialization."""
        scheduler = EMAScheduler(
            base_value=0.996,
            final_value=1.0,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        assert scheduler.base_value == 0.996
        assert scheduler.final_value == 1.0
        assert scheduler.warmup_steps == 10000

    def test_warmup_phase(self):
        """Test EMA momentum during warmup."""
        scheduler = EMAScheduler(
            base_value=0.996,
            final_value=1.0,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        # During warmup, should stay at base value
        for step in range(0, 10000, 1000):
            momentum = scheduler(step)
            assert abs(momentum - 0.996) < 1e-6

    def test_linear_schedule_after_warmup(self):
        """Test linear schedule after warmup."""
        scheduler = EMAScheduler(
            base_value=0.996,
            final_value=1.0,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        # Just after warmup
        momentum_after = scheduler(10000)
        assert abs(momentum_after - 0.996) < 1e-4

        # At end of training
        momentum_final = scheduler(99999)
        assert momentum_final <= 1.0
        assert momentum_final > 0.996

    def test_ema_scheduler_call_and_step_methods(self):
        """Test both __call__ and step methods."""
        scheduler = EMAScheduler(
            base_value=0.996,
            final_value=1.0,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        momentum_call = scheduler(50000)
        momentum_step = scheduler.step(50000)

        assert abs(momentum_call - momentum_step) < 1e-10

    def test_ema_bounds(self):
        """Test that EMA momentum stays within bounds."""
        scheduler = EMAScheduler(
            base_value=0.996,
            final_value=1.0,
            epochs=100,
            steps_per_epoch=1000,
        )

        for step in range(0, 100000, 10000):
            momentum = scheduler(step)
            assert 0.996 <= momentum <= 1.0


class TestHierarchicalScheduler:
    """Test suite for HierarchicalScheduler."""

    def test_initialization(self):
        """Test HierarchicalScheduler initialization."""
        schedulers = [
            CosineScheduler(0.1, 0.001, 100, 1000, warmup_epochs=10),
            CosineScheduler(0.05, 0.0005, 100, 1000, warmup_epochs=10),
            CosineScheduler(0.02, 0.0002, 100, 1000, warmup_epochs=10),
        ]

        hier_scheduler = HierarchicalScheduler(schedulers)

        assert hier_scheduler.num_levels == 3
        assert len(hier_scheduler.schedulers) == 3

    def test_call_returns_list(self):
        """Test that __call__ returns list of values."""
        schedulers = [
            CosineScheduler(0.1, 0.001, 100, 1000),
            CosineScheduler(0.05, 0.0005, 100, 1000),
        ]

        hier_scheduler = HierarchicalScheduler(schedulers)

        values = hier_scheduler(0)

        assert isinstance(values, list)
        assert len(values) == 2
        assert all(isinstance(v, float) for v in values)

    def test_get_level_value(self):
        """Test getting value for specific level."""
        schedulers = [
            CosineScheduler(0.1, 0.001, 100, 1000),
            CosineScheduler(0.05, 0.0005, 100, 1000),
            CosineScheduler(0.02, 0.0002, 100, 1000),
        ]

        hier_scheduler = HierarchicalScheduler(schedulers)

        # Get values for each level
        level_0 = hier_scheduler.get_level_value(0, 0)
        level_1 = hier_scheduler.get_level_value(0, 1)
        level_2 = hier_scheduler.get_level_value(0, 2)

        assert level_0 == 0.1
        assert level_1 == 0.05
        assert level_2 == 0.02

    def test_mixed_scheduler_types(self):
        """Test HierarchicalScheduler with mixed scheduler types."""
        schedulers = [
            CosineScheduler(0.1, 0.001, 100, 1000),
            LinearScheduler(0.05, 0.0005, 100, 1000),
            EMAScheduler(0.996, 1.0, 100, 1000),
        ]

        hier_scheduler = HierarchicalScheduler(schedulers)

        values = hier_scheduler(50000)

        assert len(values) == 3
        assert all(isinstance(v, float) for v in values)


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================


class TestCreateLRScheduler:
    """Test suite for create_lr_scheduler factory function."""

    def test_create_cosine_scheduler(self):
        """Test creating cosine scheduler."""
        scheduler = create_lr_scheduler(
            optimizer_type="adam",
            base_lr=0.1,
            min_lr=0.001,
            epochs=100,
            steps_per_epoch=1000,
            schedule_type="cosine",
        )

        assert isinstance(scheduler, CosineScheduler)
        assert scheduler.base_value == 0.1
        assert scheduler.final_value == 0.001

    def test_create_linear_scheduler(self):
        """Test creating linear scheduler."""
        scheduler = create_lr_scheduler(
            optimizer_type="sgd",
            base_lr=0.1,
            min_lr=0.001,
            epochs=100,
            steps_per_epoch=1000,
            schedule_type="linear",
        )

        assert isinstance(scheduler, LinearScheduler)
        assert scheduler.base_value == 0.1
        assert scheduler.final_value == 0.001

    def test_create_scheduler_with_warmup(self):
        """Test creating scheduler with warmup."""
        scheduler = create_lr_scheduler(
            optimizer_type="adam",
            base_lr=0.1,
            min_lr=0.001,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
            schedule_type="cosine",
        )

        assert scheduler.warmup_epochs == 10

    def test_create_scheduler_invalid_type(self):
        """Test creating scheduler with invalid type."""
        with pytest.raises(ValueError):
            create_lr_scheduler(
                optimizer_type="adam",
                base_lr=0.1,
                min_lr=0.001,
                epochs=100,
                steps_per_epoch=1000,
                schedule_type="invalid",
            )


class TestCreateEMAScheduler:
    """Test suite for create_ema_scheduler factory function."""

    def test_create_ema_scheduler(self):
        """Test creating EMA scheduler."""
        scheduler = create_ema_scheduler(
            base_momentum=0.996,
            final_momentum=1.0,
            epochs=100,
            steps_per_epoch=1000,
        )

        assert isinstance(scheduler, EMAScheduler)
        assert scheduler.base_value == 0.996
        assert scheduler.final_value == 1.0

    def test_create_ema_scheduler_with_warmup(self):
        """Test creating EMA scheduler with warmup."""
        scheduler = create_ema_scheduler(
            base_momentum=0.996,
            final_momentum=1.0,
            epochs=100,
            steps_per_epoch=1000,
            warmup_epochs=10,
        )

        assert scheduler.warmup_epochs == 10


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestUtilsIntegration:
    """Integration tests for utils module."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_full_training_pipeline_metrics(self, temp_dir):
        """Test full training pipeline with metrics logging."""
        # Setup logging
        logger = setup_logging(log_file=os.path.join(temp_dir, "training.log"))

        # Create metrics logger
        metrics_logger = MetricsLogger(
            experiment_name="test_exp",
            log_dir=temp_dir,
            use_wandb=False,
            use_tensorboard=False,
        )

        # Create progress tracker
        progress_tracker = ProgressTracker(total_epochs=2, steps_per_epoch=10)

        # Simulate training
        for epoch in range(2):
            progress_tracker.start_epoch()
            metrics_logger.log_metrics({"epoch": epoch}, prefix="train/")

            for step in range(10):
                metrics_logger.accumulate_metrics({"loss": 0.5 - step * 0.01})
                progress_tracker.step()

            metrics_logger.log_accumulated_metrics(prefix="train/")

        metrics_logger.finish()

    def test_full_checkpoint_pipeline(self, temp_dir):
        """Test full checkpoint saving and loading pipeline."""
        # Create checkpoint manager
        ckpt_manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            keep_best_n=2,
        )

        # Create model and optimizer
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        optimizer = Adam(model.parameters())

        # Save multiple checkpoints
        for epoch in range(5):
            ckpt_manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                metrics={"val_loss": 0.5 - epoch * 0.05},
                is_best=(epoch > 2),
            )

        # Load latest
        latest = ckpt_manager.get_latest_checkpoint()
        assert latest is not None

        # Load best
        best = ckpt_manager.get_best_checkpoint()
        assert best is not None

    def test_scheduler_pipeline(self):
        """Test scheduler pipeline with multiple epochs."""
        # Create schedulers
        lr_scheduler = create_lr_scheduler(
            optimizer_type="adam",
            base_lr=0.1,
            min_lr=0.001,
            epochs=10,
            steps_per_epoch=100,
            warmup_epochs=1,
            schedule_type="cosine",
        )

        ema_scheduler = create_ema_scheduler(
            base_momentum=0.996,
            final_momentum=1.0,
            epochs=10,
            steps_per_epoch=100,
            warmup_epochs=1,
        )

        # Simulate training steps
        global_step = 0
        for epoch in range(10):
            for step in range(100):
                lr = lr_scheduler(global_step)
                momentum = ema_scheduler(global_step)

                # LR can be 0 at the very start of warmup (step 0)
                assert lr >= 0
                assert 0.996 <= momentum <= 1.0

                global_step += 1

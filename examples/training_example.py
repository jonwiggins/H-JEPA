"""
Example usage of H-JEPA training infrastructure.

This demonstrates how to:
1. Set up the model, data, and training components
2. Create a trainer instance
3. Start training with all features enabled
4. Resume from checkpoint
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.trainers import HJEPATrainer, create_optimizer
from src.utils import setup_logging

# ============================================================================
# Example 1: Basic Training Setup
# ============================================================================


def basic_training_example():
    """
    Basic example showing minimal training setup.
    """
    # Setup logging
    setup_logging()

    # Configuration (normally loaded from YAML)
    config = {
        "model": {
            "encoder_type": "vit_base_patch16_224",
            "embed_dim": 768,
            "num_hierarchies": 3,
            "ema": {
                "momentum": 0.996,
                "momentum_end": 1.0,
                "momentum_warmup_epochs": 30,
            },
        },
        "training": {
            "epochs": 100,
            "warmup_epochs": 10,
            "lr": 1.5e-4,
            "min_lr": 1e-6,
            "weight_decay": 0.05,
            "optimizer": "adamw",
            "betas": [0.9, 0.95],
            "lr_schedule": "cosine",
            "clip_grad": 3.0,
            "use_amp": True,
            "accumulation_steps": 1,
        },
        "checkpoint": {
            "checkpoint_dir": "results/checkpoints",
            "save_frequency": 10,
            "keep_best_n": 3,
        },
        "logging": {
            "experiment_name": "hjepa_example",
            "log_dir": "results/logs",
            "log_frequency": 100,
            "wandb": {
                "enabled": False,  # Set to True to use W&B
                "project": "h-jepa",
                "entity": None,
                "tags": ["example"],
            },
            "tensorboard": {
                "enabled": True,
            },
        },
    }

    # Create dummy model (replace with actual H-JEPA model)
    class DummyHJEPAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_encoder = nn.Linear(768, 768)
            self.target_encoder = nn.Linear(768, 768)
            self.predictor = nn.Linear(768, 768)

        def encode_context(self, images, masks):
            batch_size = images.size(0)
            return torch.randn(batch_size, 196, 768)

        def encode_target(self, images, masks):
            batch_size = images.size(0)
            return torch.randn(batch_size, 4, 768)

        def predict(self, context_emb, target_masks, context_masks):
            batch_size = context_emb.size(0)
            return torch.randn(batch_size, 4, 768)

    model = DummyHJEPAModel()

    # Create dummy data (replace with actual data loaders)
    dummy_images = torch.randn(1000, 3, 224, 224)
    train_dataset = TensorDataset(dummy_images)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    dummy_val_images = torch.randn(200, 3, 224, 224)
    val_dataset = TensorDataset(dummy_val_images)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Create loss function (replace with actual H-JEPA loss)
    class DummyLoss(nn.Module):
        def forward(self, predictions, targets):
            loss = F.smooth_l1_loss(predictions, targets)
            return loss, {"loss": loss.item(), "mse": loss.item()}

    loss_fn = DummyLoss()

    # Create masking function (replace with actual masking)
    def dummy_masking_fn(batch_size, device):
        context_masks = [torch.ones(batch_size, 196, dtype=torch.bool, device=device)]
        target_masks = [torch.ones(batch_size, 4, dtype=torch.bool, device=device)]
        return context_masks, target_masks

    # Create trainer
    trainer = HJEPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        masking_fn=dummy_masking_fn,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Start training
    trainer.train()

    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved in: {config['checkpoint']['checkpoint_dir']}")


# ============================================================================
# Example 2: Resume from Checkpoint
# ============================================================================


def resume_training_example():
    """
    Example showing how to resume training from a checkpoint.
    """
    # Similar setup as basic example...
    config = {
        # ... (same config as above)
    }

    # Create model, data, optimizer, etc. (same as above)
    # ...

    # Create trainer with resume checkpoint
    trainer = HJEPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        masking_fn=masking_fn,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        resume_checkpoint="results/checkpoints/checkpoint_latest.pth",  # Resume from here
    )

    # Training will automatically resume from the checkpoint
    trainer.train()


# ============================================================================
# Example 3: Advanced Training with Custom Schedulers
# ============================================================================


def advanced_training_example():
    """
    Example with hierarchical learning rates and custom EMA schedule.
    """
    from src.utils import CosineScheduler, HierarchicalScheduler

    # Create separate schedulers for different hierarchy levels
    schedulers = [
        CosineScheduler(
            base_value=1e-3,
            final_value=1e-6,
            epochs=100,
            steps_per_epoch=100,
            warmup_epochs=10,
        ),
        CosineScheduler(
            base_value=5e-4,
            final_value=5e-7,
            epochs=100,
            steps_per_epoch=100,
            warmup_epochs=10,
        ),
        CosineScheduler(
            base_value=2e-4,
            final_value=2e-7,
            epochs=100,
            steps_per_epoch=100,
            warmup_epochs=10,
        ),
    ]

    hierarchical_lr_scheduler = HierarchicalScheduler(schedulers)

    # Use in training by setting different learning rates for different
    # parameter groups in the optimizer
    print("Hierarchical schedulers created!")
    print(f"LRs at step 0: {hierarchical_lr_scheduler(0)}")
    print(f"LRs at step 1000: {hierarchical_lr_scheduler(1000)}")


# ============================================================================
# Example 4: Monitoring and Logging
# ============================================================================


def monitoring_example():
    """
    Example showing monitoring features.
    """
    from src.utils import MetricsLogger

    # Create metrics logger
    logger = MetricsLogger(
        experiment_name="monitoring_example",
        log_dir="results/logs",
        config={"key": "value"},
        use_wandb=False,
        use_tensorboard=True,
    )

    # Log metrics
    for step in range(100):
        metrics = {
            "loss": 0.5 * (1.0 - step / 100),
            "accuracy": 0.5 + 0.5 * (step / 100),
        }
        logger.log_metrics(metrics, step=step, prefix="train/")

    # Log images
    dummy_image = torch.randn(3, 224, 224)
    logger.log_image("example_image", dummy_image, step=0)

    # Log histograms
    dummy_gradients = torch.randn(1000)
    logger.log_histogram("gradients/layer1", dummy_gradients, step=0)

    # System metrics
    logger.log_system_metrics(step=0)

    # Cleanup
    logger.finish()

    print("Monitoring example completed!")


# ============================================================================
# Example 5: Checkpoint Management
# ============================================================================


def checkpoint_management_example():
    """
    Example showing checkpoint management features.
    """
    import torch.nn as nn

    from src.utils import CheckpointManager

    # Create checkpoint manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir="results/checkpoints_example",
        keep_best_n=3,
        save_frequency=5,
        metric_name="val_loss",
        mode="min",
    )

    # Create dummy model and optimizer
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simulate training and saving checkpoints
    for epoch in range(20):
        val_loss = 1.0 - 0.03 * epoch  # Decreasing loss

        # Check if we should save
        if ckpt_manager.should_save(epoch):
            is_best = ckpt_manager.update_best_metric(val_loss)

            ckpt_manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                metrics={"val_loss": val_loss},
                is_best=is_best,
            )

    # Get best checkpoint
    best_ckpt = ckpt_manager.get_best_checkpoint()
    print(f"Best checkpoint: {best_ckpt}")

    # Load checkpoint
    ckpt_manager.load_checkpoint(
        checkpoint_path=best_ckpt,
        model=model,
        optimizer=optimizer,
    )

    print("Checkpoint management example completed!")


if __name__ == "__main__":
    print("H-JEPA Training Infrastructure Examples")
    print("=" * 80)
    print()

    # Uncomment to run examples:

    # print("Running basic training example...")
    # basic_training_example()

    # print("\nRunning advanced training example...")
    # advanced_training_example()

    print("\nRunning monitoring example...")
    monitoring_example()

    print("\nRunning checkpoint management example...")
    checkpoint_management_example()

    print("\n" + "=" * 80)
    print("Examples completed!")

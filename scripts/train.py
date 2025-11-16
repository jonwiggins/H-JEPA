#!/usr/bin/env python3
"""
Training script for H-JEPA

Usage:
    Basic training:
        python scripts/train.py --config configs/default.yaml

    Resume from checkpoint:
        python scripts/train.py --config configs/default.yaml --resume results/checkpoints/checkpoint.pth

    Override config parameters:
        python scripts/train.py --config configs/default.yaml --batch_size 64 --lr 1e-4 --epochs 100

    Multi-GPU training:
        python scripts/train.py --config configs/default.yaml --distributed

    Custom device:
        python scripts/train.py --config configs/default.yaml --device cuda:1
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import numpy as np

# Import H-JEPA components
from src.models import create_hjepa_from_config
from src.losses import create_loss_from_config
from src.masks import MultiBlockMaskGenerator, HierarchicalMaskGenerator
from src.data import build_dataset, build_dataloader
from src.trainers import HJEPATrainer, create_optimizer
from src.utils import (
    setup_logging,
    CheckpointManager,
    MetricsLogger,
    ProgressTracker,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments with config override support."""
    parser = argparse.ArgumentParser(
        description="Train H-JEPA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python scripts/train.py --config configs/default.yaml

  # Resume training
  python scripts/train.py --config configs/default.yaml --resume results/checkpoints/latest.pth

  # Override parameters
  python scripts/train.py --config configs/default.yaml --batch_size 64 --lr 1e-4

  # Multi-GPU training
  python scripts/train.py --config configs/default.yaml --distributed
        """
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )

    # Optional arguments
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu'). Overrides config."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs. Overrides config."
    )

    # Training overrides
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, help="Warmup epochs")

    # Data overrides
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument("--num_workers", type=int, help="Number of data loading workers")

    # Distributed training
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed data parallel training"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training (set by torch.distributed.launch)"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of processes for distributed training"
    )

    # Logging
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file {config_path}: {e}")


def apply_config_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Apply command-line argument overrides to configuration.

    Args:
        config: Base configuration dictionary
        args: Parsed command-line arguments

    Returns:
        Updated configuration dictionary
    """
    # Training overrides
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
        logger.info(f"Override: batch_size = {args.batch_size}")

    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        logger.info(f"Override: epochs = {args.epochs}")

    if args.lr is not None:
        config['training']['lr'] = args.lr
        logger.info(f"Override: lr = {args.lr}")

    if args.weight_decay is not None:
        config['training']['weight_decay'] = args.weight_decay
        logger.info(f"Override: weight_decay = {args.weight_decay}")

    if args.warmup_epochs is not None:
        config['training']['warmup_epochs'] = args.warmup_epochs
        logger.info(f"Override: warmup_epochs = {args.warmup_epochs}")

    # Data overrides
    if args.data_path is not None:
        config['data']['data_path'] = args.data_path
        logger.info(f"Override: data_path = {args.data_path}")

    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
        logger.info(f"Override: num_workers = {args.num_workers}")

    # Output directory
    if args.output_dir is not None:
        config['checkpoint']['checkpoint_dir'] = os.path.join(args.output_dir, 'checkpoints')
        config['logging']['log_dir'] = os.path.join(args.output_dir, 'logs')
        logger.info(f"Override: output_dir = {args.output_dir}")

    # Resume checkpoint
    if args.resume is not None:
        config['checkpoint']['resume'] = args.resume
        logger.info(f"Override: resume = {args.resume}")

    # Logging overrides
    if args.no_wandb:
        config['logging']['wandb']['enabled'] = False
        logger.info("Override: Disabled W&B logging")

    # Distributed settings
    if args.distributed:
        config['distributed']['enabled'] = True
        config['distributed']['world_size'] = args.world_size
        logger.info(f"Override: Enabled distributed training (world_size={args.world_size})")

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['model', 'data', 'training', 'masking', 'loss', 'checkpoint', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate model config
    if config['model']['num_hierarchies'] < 2 or config['model']['num_hierarchies'] > 4:
        raise ValueError("num_hierarchies must be between 2 and 4")

    # Validate training config
    if config['training']['epochs'] <= 0:
        raise ValueError("epochs must be positive")
    if config['training']['lr'] <= 0:
        raise ValueError("lr must be positive")

    # Validate loss hierarchy weights
    if len(config['loss']['hierarchy_weights']) != config['model']['num_hierarchies']:
        raise ValueError(
            f"Loss hierarchy_weights length ({len(config['loss']['hierarchy_weights'])}) "
            f"must match num_hierarchies ({config['model']['num_hierarchies']})"
        )

    # Validate data path exists if not using downloadable datasets
    data_path = Path(config['data']['data_path'])
    use_multi_dataset = config['data'].get('use_multi_dataset', False)
    downloadable = ['cifar10', 'cifar100', 'stl10']

    if use_multi_dataset:
        # Multi-dataset mode: check if all datasets are downloadable or paths exist
        for ds_config in config['data']['datasets']:
            dataset_name = ds_config['name'].lower()
            if dataset_name not in downloadable and not data_path.exists():
                logger.warning(
                    f"Data path does not exist: {data_path}\n"
                    f"For {dataset_name}, please ensure the data is available before training."
                )
    else:
        # Single dataset mode
        dataset_name = config['data']['dataset'].lower()
        if dataset_name not in downloadable and not data_path.exists():
            logger.warning(
                f"Data path does not exist: {data_path}\n"
                f"For {dataset_name}, please ensure the data is available before training."
            )

    logger.info("Configuration validation passed")


def setup_device(config: Dict[str, Any], args: argparse.Namespace) -> torch.device:
    """
    Setup computation device.

    Args:
        config: Configuration dictionary
        args: Command-line arguments

    Returns:
        torch.device object
    """
    # Priority: args.device > config.device > auto-detect
    if args.device:
        device = torch.device(args.device)
    elif 'device' in config:
        device = torch.device(config['device'])
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    return device


def setup_distributed(args: argparse.Namespace) -> bool:
    """
    Setup distributed training environment.

    Args:
        args: Command-line arguments

    Returns:
        True if distributed training is enabled, False otherwise
    """
    if not args.distributed:
        return False

    if not torch.cuda.is_available():
        logger.warning("Distributed training requested but CUDA not available. Using single GPU.")
        return False

    # Initialize process group
    try:
        dist.init_process_group(backend='nccl')
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        logger.info(
            f"Initialized distributed training: "
            f"rank {dist.get_rank()}/{dist.get_world_size()}, "
            f"local_rank {local_rank}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        return False


def set_seed(seed: int, distributed: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
        distributed: Whether distributed training is enabled
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For reproducibility in distributed training
        if distributed:
            # Use different seed per rank to avoid identical augmentations
            rank_seed = seed + dist.get_rank()
            torch.manual_seed(rank_seed)
            np.random.seed(rank_seed)

    # Additional reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed: {seed}")


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories for outputs.

    Args:
        config: Configuration dictionary
    """
    directories = [
        config['checkpoint']['checkpoint_dir'],
        config['logging']['log_dir'],
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


def print_training_summary(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Print training configuration summary.

    Args:
        config: Configuration dictionary
        args: Command-line arguments
    """
    print("\n" + "=" * 80)
    print("H-JEPA Training Configuration".center(80))
    print("=" * 80)

    print("\nExperiment:")
    print(f"  Name: {config['logging']['experiment_name']}")
    print(f"  Seed: {config.get('seed', 42)}")
    print(f"  Output: {config['checkpoint']['checkpoint_dir']}")

    print("\nModel:")
    print(f"  Encoder: {config['model']['encoder_type']}")
    print(f"  Embedding dim: {config['model']['embed_dim']}")
    print(f"  Hierarchies: {config['model']['num_hierarchies']}")
    print(f"  Predictor depth: {config['model']['predictor']['depth']}")

    print("\nData:")
    if config['data'].get('use_multi_dataset', False):
        datasets_info = ', '.join([f"{ds['name']} ({ds['weight']:.0%})" for ds in config['data']['datasets']])
        print(f"  Datasets: {datasets_info}")
        print(f"  Sampling: {config['data']['sampling_strategy']}")
    else:
        print(f"  Dataset: {config['data']['dataset']}")
    print(f"  Data path: {config['data']['data_path']}")
    print(f"  Image size: {config['data']['image_size']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Workers: {config['data']['num_workers']}")

    print("\nTraining:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Learning rate: {config['training']['lr']}")
    print(f"  Weight decay: {config['training']['weight_decay']}")
    print(f"  Warmup epochs: {config['training']['warmup_epochs']}")
    print(f"  Optimizer: {config['training']['optimizer']}")
    print(f"  LR schedule: {config['training']['lr_schedule']}")
    print(f"  Mixed precision: {config['training']['use_amp']}")

    print("\nMasking:")
    num_masks = config['masking'].get('num_target_masks', config['masking'].get('num_masks', 4))
    print(f"  Num target masks: {num_masks}")
    print(f"  Mask scale: {config['masking']['mask_scale']}")
    print(f"  Num context masks: {config['masking']['num_context_masks']}")

    print("\nLoss:")
    print(f"  Type: {config['loss']['type']}")
    print(f"  Hierarchy weights: {config['loss']['hierarchy_weights']}")

    print("\nLogging:")
    # Handle both nested and flat wandb/tensorboard config structures
    wandb_enabled = config['logging'].get('use_wandb', config['logging'].get('wandb', {}).get('enabled', False))
    tb_enabled = config['logging'].get('use_tensorboard', config['logging'].get('tensorboard', {}).get('enabled', False))
    print(f"  W&B: {'Enabled' if wandb_enabled else 'Disabled'}")
    print(f"  TensorBoard: {'Enabled' if tb_enabled else 'Disabled'}")
    print(f"  Log frequency: {config['logging']['log_frequency']} steps")

    if args.distributed:
        print("\nDistributed:")
        print(f"  Enabled: True")
        print(f"  World size: {config['distributed']['world_size']}")
        print(f"  Backend: {config['distributed']['backend']}")

    if args.resume:
        print(f"\nResuming from: {args.resume}")

    print("=" * 80 + "\n")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    try:
        # Load and validate configuration
        config = load_config(args.config)
        config = apply_config_overrides(config, args)
        validate_config(config)

        # Setup distributed training if requested
        is_distributed = setup_distributed(args)
        is_main_process = not is_distributed or dist.get_rank() == 0

        # Setup device
        device = setup_device(config, args)

        # Set random seed
        set_seed(config.get('seed', 42), distributed=is_distributed)

        # Create output directories
        if is_main_process:
            create_directories(config)
            print_training_summary(config, args)

        # ====================================================================
        # Build Dataset and DataLoader
        # ====================================================================
        logger.info("Building datasets...")

        # Check if using multi-dataset (foundation model)
        use_multi_dataset = config['data'].get('use_multi_dataset', False)

        if use_multi_dataset:
            # Foundation model: Multiple datasets
            from src.data import build_multi_dataset

            logger.info("Using multi-dataset configuration for foundation model training")

            train_dataset = build_multi_dataset(
                dataset_configs=config['data']['datasets'],
                data_path=config['data']['data_path'],
                split='train',
                sampling_strategy=config['data'].get('sampling_strategy', 'weighted'),
                image_size=config['data']['image_size'],
                color_jitter=config['data'].get('augmentation', {}).get('color_jitter', 0.4),
            )

            # For multi-dataset, validation is tricky - use first dataset for now
            val_dataset = None
            if config.get('evaluation', {}).get('eval_frequency', 0) > 0:
                try:
                    # Use first dataset for validation
                    first_dataset = config['data']['datasets'][0]['name']
                    logger.info(f"Using {first_dataset} for validation")

                    val_dataset = build_dataset(
                        dataset_name=first_dataset,
                        data_path=config['data']['data_path'],
                        split='val',
                        image_size=config['data']['image_size'],
                        color_jitter=None,
                    )
                    logger.info(f"Validation dataset size: {len(val_dataset)}")
                except Exception as e:
                    logger.warning(f"Could not load validation dataset: {e}")

        else:
            # Single dataset (original behavior)
            train_dataset = build_dataset(
                dataset_name=config['data']['dataset'],
                data_path=config['data']['data_path'],
                split='train',
                image_size=config['data']['image_size'],
                color_jitter=config['data'].get('augmentation', {}).get('color_jitter', 0.4),
            )

            # Build validation dataset if eval is enabled
            val_dataset = None
            if config.get('evaluation', {}).get('eval_frequency', 0) > 0:
                try:
                    val_dataset = build_dataset(
                        dataset_name=config['data']['dataset'],
                        data_path=config['data']['data_path'],
                        split='val',
                        image_size=config['data']['image_size'],
                        color_jitter=None,  # No augmentation for validation
                    )
                    logger.info(f"Validation dataset size: {len(val_dataset)}")
                except Exception as e:
                    logger.warning(f"Could not load validation dataset: {e}")

        logger.info(f"Training dataset size: {len(train_dataset)}")

        # Build dataloaders
        train_loader = build_dataloader(
            dataset=train_dataset,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            pin_memory=config['data'].get('pin_memory', True),
            shuffle=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = build_dataloader(
                dataset=val_dataset,
                batch_size=config['data']['batch_size'],
                num_workers=config['data']['num_workers'],
                pin_memory=config['data'].get('pin_memory', True),
                shuffle=False,
            )

        logger.info(f"Training batches per epoch: {len(train_loader)}")

        # ====================================================================
        # Build Model
        # ====================================================================
        logger.info("Building H-JEPA model...")

        model = create_hjepa_from_config(config)
        model = model.to(device)

        # Wrap model with DDP if distributed
        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False,
            )
            logger.info("Model wrapped with DistributedDataParallel")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # ====================================================================
        # Build Masking Strategy
        # ====================================================================
        logger.info("Building masking generator...")

        # Determine image dimensions in patches
        patch_size = 16  # Most ViT models use 16x16 patches
        img_size = config['data']['image_size']
        num_patches = (img_size // patch_size) ** 2

        # Use multi-block masking strategy (I-JEPA specification)
        # Note: Hierarchy is created via pooling in the model, not different spatial masks
        masking_generator = MultiBlockMaskGenerator(
            input_size=(img_size, img_size),
            patch_size=patch_size,
            num_target_masks=config['masking'].get('num_masks', 4),
            target_scale=tuple(config['masking'].get('mask_scale', [0.15, 0.2])),
            context_scale=tuple(config['masking'].get('context_scale', [0.85, 1.0])),
            aspect_ratio_range=tuple(config['masking'].get('aspect_ratio', [0.75, 1.5])),
        )

        logger.info(
            f"Masking: {num_patches} patches, {config['masking'].get('num_masks', 4)} target blocks, "
            f"hierarchy created via pooling ({config['model']['num_hierarchies']} levels)"
        )

        # ====================================================================
        # Build Loss Function
        # ====================================================================
        logger.info("Building loss function...")

        loss_fn = create_loss_from_config(config)
        loss_fn = loss_fn.to(device)

        logger.info(f"Loss: {config['loss']['type']} with hierarchy weights {config['loss']['hierarchy_weights']}")

        # ====================================================================
        # Build Optimizer
        # ====================================================================
        logger.info("Building optimizer...")

        optimizer = create_optimizer(
            model=model,
            config=config,
        )

        logger.info(f"Optimizer: {config['training']['optimizer']} (lr={config['training']['lr']})")

        # ====================================================================
        # Build Trainer
        # ====================================================================
        logger.info("Building trainer...")

        trainer = HJEPATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            masking_fn=masking_generator,
            config=config,
            device=device,
            resume_checkpoint=args.resume,
        )

        logger.info("Trainer initialized successfully")

        # ====================================================================
        # Start Training
        # ====================================================================
        if is_main_process:
            print("\n" + "=" * 80)
            print("Starting Training".center(80))
            print("=" * 80 + "\n")

        trainer.train()

        # ====================================================================
        # Training Complete
        # ====================================================================
        if is_main_process:
            print("\n" + "=" * 80)
            print("Training Complete!".center(80))
            print("=" * 80)
            print(f"\nCheckpoints saved to: {config['checkpoint']['checkpoint_dir']}")
            print(f"Logs saved to: {config['logging']['log_dir']}")
            print("\nNext steps:")
            print("  1. Evaluate the trained model on downstream tasks")
            print("  2. Fine-tune on specific datasets")
            print("  3. Extract features for linear probing")
            print("=" * 80 + "\n")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        if is_distributed:
            dist.destroy_process_group()
        sys.exit(0)

    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        if is_distributed:
            dist.destroy_process_group()
        sys.exit(1)

    finally:
        # Cleanup
        if is_distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

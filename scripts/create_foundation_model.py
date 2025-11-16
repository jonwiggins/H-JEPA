#!/usr/bin/env python3
"""
Create a foundation model training configuration.

This script makes it easy to create multi-dataset training configs
for building foundation models with H-JEPA.

Usage:
    # Use pre-configured scales
    python3.11 scripts/create_foundation_model.py --scale mini

    # Custom dataset mixture
    python3.11 scripts/create_foundation_model.py \
        --datasets imagenet100 stl10 cifar100 \
        --weights 0.6 0.3 0.1 \
        --output-config configs/my_foundation_model.yaml
"""

import argparse
import yaml
from pathlib import Path


# Pre-configured foundation model scales
SCALES = {
    'mini': {
        'description': 'Mini foundation model - good for M1 Max (~18-24 hours, 65-75% accuracy)',
        'datasets': [
            {'name': 'imagenet100', 'weight': 0.6},
            {'name': 'stl10', 'weight': 0.3},
            {'name': 'cifar100', 'weight': 0.1},
        ],
        'total_images': 276000,
        'expected_time_hours': 20,
    },
    'medium': {
        'description': 'Medium foundation model - full ImageNet (~7-10 days, 70-78% accuracy)',
        'datasets': [
            {'name': 'imagenet', 'weight': 0.9},
            {'name': 'stl10', 'weight': 0.1},
        ],
        'total_images': 1380000,
        'expected_time_hours': 200,
    },
}


# Dataset metadata
DATASET_INFO = {
    'cifar10': {'images': 50000, 'classes': 10, 'size_gb': 0.16},
    'cifar100': {'images': 50000, 'classes': 100, 'size_gb': 0.16},
    'stl10': {'images': 100000, 'classes': 10, 'size_gb': 3},
    'imagenet100': {'images': 126689, 'classes': 100, 'size_gb': 15},
    'imagenet': {'images': 1281167, 'classes': 1000, 'size_gb': 150},
}


def create_config(
    datasets,
    weights=None,
    name='custom_foundation',
    epochs=100,
    batch_size=32,
    output_path='configs/foundation_model_custom.yaml',
):
    """Create foundation model training config."""

    # Normalize weights
    if weights is None:
        weights = [1.0 / len(datasets)] * len(datasets)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    # Build dataset configs
    dataset_configs = []
    total_images = 0
    total_size_gb = 0

    for i, dataset_name in enumerate(datasets):
        if dataset_name not in DATASET_INFO:
            print(f"Warning: Unknown dataset '{dataset_name}', skipping...")
            continue

        info = DATASET_INFO[dataset_name]
        dataset_configs.append({
            'name': dataset_name,
            'weight': weights[i],
        })

        total_images += info['images']
        total_size_gb += info['size_gb']

    if not dataset_configs:
        raise ValueError("No valid datasets specified!")

    # Create config
    config = {
        'experiment': {
            'name': name,
            'seed': 42,
            'output_dir': 'results/foundation_model',
            'save_frequency': 10,
            'eval_frequency': 10,
        },
        'model': {
            'encoder_type': 'vit_small_patch16_224',
            'embed_dim': 384,
            'num_hierarchies': 3,
            'target_encoder': {
                'ema_decay': 0.996,
                'ema_end_decay': 1.0,
                'ema_anneal_end_step': 300000,
            },
            'predictor': {
                'depth': 6,
                'num_heads': 6,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
                'dropout': 0.0,
            },
        },
        'data': {
            'use_multi_dataset': True,
            'datasets': dataset_configs,
            'sampling_strategy': 'weighted',
            'data_path': './data',
            'image_size': 224,
            'batch_size': batch_size,
            'num_workers': 6,
            'transforms': {
                'crop_scale': [0.8, 1.0],
                'horizontal_flip': True,
                'color_jitter': 0.1,
            },
        },
        'training': {
            'epochs': epochs,
            'warmup_epochs': max(10, epochs // 10),
            'optimizer': 'adamw',
            'lr': 0.0001,
            'weight_decay': 0.04,
            'betas': [0.9, 0.95],
            'lr_schedule': 'cosine',
            'min_lr_ratio': 0.01,
            'warmup_lr_ratio': 0.001,
            'use_amp': True,
        },
        'masking': {
            'num_target_masks': 4,
            'mask_scale': [0.15, 0.25],
            'aspect_ratio': [0.75, 1.5],
            'num_context_masks': 1,
        },
        'loss': {
            'type': 'smoothl1',
            'hierarchy_weights': [1.0, 0.7, 0.5],
            'use_vicreg': True,
            'vicreg': {
                'sim_coeff': 25.0,
                'std_coeff': 25.0,
                'cov_coeff': 1.0,
            },
        },
        'logging': {
            'use_wandb': False,
            'use_tensorboard': True,
            'log_frequency': 50,
            'log_images': True,
            'log_attention': True,
            'log_gradients': False,
            'log_dataset_distribution': True,
        },
        'device': 'mps',
        'checkpoint': {
            'save_best': True,
            'metric': 'val_loss',
            'mode': 'min',
            'keep_last_k': 3,
        },
    }

    # Save config
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Foundation Model Configuration Created")
    print(f"{'='*70}\n")

    print(f"Name: {name}")
    print(f"Config: {output_path}\n")

    print(f"Datasets:")
    for i, (ds_name, weight) in enumerate(zip(datasets, weights)):
        if ds_name in DATASET_INFO:
            info = DATASET_INFO[ds_name]
            print(f"  {i+1}. {ds_name}")
            print(f"     - Images: {info['images']:,}")
            print(f"     - Classes: {info['classes']}")
            print(f"     - Sampling weight: {weight*100:.1f}%")
            print(f"     - Download size: ~{info['size_gb']}GB")

    print(f"\nTotal:")
    print(f"  - Images: ~{total_images:,}")
    print(f"  - Download size: ~{total_size_gb:.1f}GB")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")

    # Estimate training time
    steps_per_epoch = total_images // batch_size
    total_steps = steps_per_epoch * epochs
    # M1 Max does ~3.2 it/s for ViT-Small
    hours = total_steps / (3.2 * 3600)
    days = hours / 24

    print(f"\nEstimated training time (M1 Max):")
    if days < 1:
        print(f"  ~{hours:.1f} hours")
    else:
        print(f"  ~{days:.1f} days ({hours:.0f} hours)")

    print(f"\nExpected results:")
    if total_images < 100000:
        print(f"  Linear probe: 50-65%")
    elif total_images < 300000:
        print(f"  Linear probe: 65-75%")
    elif total_images < 1000000:
        print(f"  Linear probe: 70-78%")
    else:
        print(f"  Linear probe: 70-80%")

    print(f"\n{'='*70}")
    print(f"Next steps:")
    print(f"{'='*70}\n")

    print(f"1. Download datasets:")
    for ds_name in datasets:
        if ds_name == 'imagenet100':
            print(f"   ./scripts/download_imagenet100.sh")
        else:
            print(f"   python3.11 scripts/download_data.py --dataset {ds_name}")

    print(f"\n2. Start training:")
    print(f"   python3.11 scripts/train.py --config {output_path}")

    print(f"\n3. Monitor progress:")
    print(f"   tensorboard --logdir results/logs/tensorboard")

    print(f"\n{'='*70}\n")

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Create foundation model training configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use pre-configured mini scale
  python3.11 scripts/create_foundation_model.py --scale mini

  # Custom dataset mixture
  python3.11 scripts/create_foundation_model.py \\
      --datasets imagenet100 stl10 cifar100 \\
      --weights 0.6 0.3 0.1

  # Custom with specific settings
  python3.11 scripts/create_foundation_model.py \\
      --datasets imagenet stl10 \\
      --weights 0.9 0.1 \\
      --epochs 50 \\
      --batch-size 48 \\
      --name my_foundation_v1
        """
    )

    parser.add_argument(
        '--scale',
        choices=['mini', 'medium'],
        help='Use pre-configured scale (overrides --datasets and --weights)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['cifar10', 'cifar100', 'stl10', 'imagenet100', 'imagenet'],
        help='Datasets to include in the mixture'
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        help='Sampling weights for each dataset (must match number of datasets)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='custom_foundation',
        help='Experiment name'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--output-config',
        type=str,
        default='configs/foundation_model_custom.yaml',
        help='Output configuration file path'
    )

    args = parser.parse_args()

    # Use scale or custom datasets
    if args.scale:
        scale_config = SCALES[args.scale]
        datasets = [ds['name'] for ds in scale_config['datasets']]
        weights = [ds['weight'] for ds in scale_config['datasets']]
        name = f"foundation_{args.scale}"
        output_path = f"configs/foundation_model_{args.scale}.yaml"

        print(f"\nUsing pre-configured scale: {args.scale}")
        print(f"Description: {scale_config['description']}\n")

    elif args.datasets:
        datasets = args.datasets
        weights = args.weights
        name = args.name
        output_path = args.output_config

        # Validate weights
        if weights and len(weights) != len(datasets):
            parser.error(f"Number of weights ({len(weights)}) must match number of datasets ({len(datasets)})")

    else:
        parser.error("Must specify either --scale or --datasets")

    # Create config
    create_config(
        datasets=datasets,
        weights=weights,
        name=name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=output_path,
    )


if __name__ == '__main__':
    main()

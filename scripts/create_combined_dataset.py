#!/usr/bin/env python3
"""
Create a combined dataset from multiple sources for H-JEPA training.

This allows combining ImageNet-100, STL-10, and other datasets for
maximum training data diversity.

Usage:
    python3.11 scripts/create_combined_dataset.py \
        --datasets imagenet100 stl10 \
        --output-config configs/combined_dataset.yaml
"""

import argparse
from pathlib import Path

import yaml


def create_combined_config(datasets, output_path, base_config=None):
    """Create a training config that uses multiple datasets."""

    # Dataset information
    dataset_info = {
        "cifar10": {"images": 50000, "resolution": 32, "classes": 10},
        "cifar100": {"images": 50000, "resolution": 32, "classes": 100},
        "stl10": {"images": 100000, "resolution": 96, "classes": 10},
        "imagenet100": {"images": 126689, "resolution": 224, "classes": 100},
        "imagenet": {"images": 1281167, "resolution": 224, "classes": 1000},
    }

    # Calculate total images
    total_images = sum(dataset_info[d]["images"] for d in datasets if d in dataset_info)

    print("Creating combined dataset configuration:")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Total images: {total_images:,}")

    for dataset in datasets:
        if dataset in dataset_info:
            info = dataset_info[dataset]
            print(
                f"    - {dataset}: {info['images']:,} images @ {info['resolution']}×{info['resolution']}"
            )

    # Load base config or create new one
    if base_config and Path(base_config).exists():
        with open(base_config) as f:
            config = yaml.safe_load(f)
    else:
        # Create default config
        config = {
            "experiment": {
                "name": f"combined_{'_'.join(datasets)}",
                "seed": 42,
                "output_dir": "results/checkpoints",
                "save_frequency": 10,
                "eval_frequency": 10,
            },
            "model": {
                "encoder_type": "vit_small_patch16_224",
                "embed_dim": 384,
                "num_hierarchies": 3,
                "target_encoder": {
                    "ema_decay": 0.996,
                    "ema_end_decay": 1.0,
                    "ema_anneal_end_step": 300000,
                },
                "predictor": {
                    "depth": 6,
                    "num_heads": 6,
                    "mlp_ratio": 4.0,
                    "qkv_bias": True,
                    "dropout": 0.0,
                },
            },
            "data": {
                "datasets": datasets,  # Multiple datasets!
                "data_path": "./data",
                "image_size": 224,
                "batch_size": 32,
                "num_workers": 6,
                "transforms": {
                    "crop_scale": [0.8, 1.0],
                    "horizontal_flip": True,
                    "color_jitter": 0.1,
                },
            },
            "training": {
                "epochs": 100,
                "warmup_epochs": 10,
                "optimizer": "adamw",
                "lr": 0.0001,
                "weight_decay": 0.04,
                "betas": [0.9, 0.95],
                "lr_schedule": "cosine",
                "min_lr_ratio": 0.01,
                "warmup_lr_ratio": 0.001,
                "use_amp": True,
            },
            "masking": {
                "num_target_masks": 4,
                "mask_scale": [0.15, 0.25],
                "aspect_ratio": [0.75, 1.5],
                "num_context_masks": 1,
            },
            "loss": {
                "type": "smoothl1",
                "hierarchy_weights": [1.0, 0.7, 0.5],
                "use_vicreg": True,
                "vicreg": {
                    "sim_coeff": 25.0,
                    "std_coeff": 25.0,
                    "cov_coeff": 1.0,
                },
            },
            "logging": {
                "use_wandb": False,
                "use_tensorboard": True,
                "log_frequency": 50,
                "log_images": True,
                "log_attention": True,
                "log_gradients": False,
            },
            "device": "mps",
            "checkpoint": {
                "save_best": True,
                "metric": "val_loss",
                "mode": "min",
                "keep_last_k": 3,
            },
        }

    # Update data section with combined datasets
    config["data"]["datasets"] = datasets
    config["experiment"]["name"] = f"combined_{'_'.join(datasets)}"

    # Adjust steps per epoch based on total images
    steps_per_epoch = total_images // config["data"]["batch_size"]
    config["training"]["steps_per_epoch"] = steps_per_epoch

    # Save config
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Created combined dataset config: {output_path}")
    print("\nEstimated training time (M1 Max, 100 epochs):")

    # Estimate time (rough calculation)
    # M1 Max does ~3.2 it/s for ViT-Small
    hours = (steps_per_epoch * 100) / (3.2 * 3600)
    print(f"  ~{hours:.1f} hours ({hours/24:.1f} days)")

    print("\nTo start training:")
    print(f"  python3.11 scripts/train.py --config {output_path}")

    return config


def main():
    parser = argparse.ArgumentParser(description="Create combined dataset configuration")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=["cifar10", "cifar100", "stl10", "imagenet100", "imagenet"],
        help="Datasets to combine",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default="configs/combined_dataset.yaml",
        help="Output configuration file path",
    )
    parser.add_argument(
        "--base-config", type=str, help="Base configuration file to extend (optional)"
    )

    args = parser.parse_args()

    create_combined_config(
        datasets=args.datasets,
        output_path=args.output_config,
        base_config=args.base_config,
    )


if __name__ == "__main__":
    main()

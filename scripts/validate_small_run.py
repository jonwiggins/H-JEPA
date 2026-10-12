"""Bounded real CIFAR-10 diagnostics; invoke through bounded_validation.py.

Public uoft-cs/cifar10 Parquet files are a download fallback, not synthetic data.
Requires pyarrow in the checkout venv. Uses the project's JEPA transforms.
"""

import argparse
import copy
import io
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import timm
import torch
import torchvision
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.datasets import JEPAEvalTransform, JEPATransform
from src.losses.combined import create_loss_from_config
from src.masks.hierarchical import HierarchicalMaskGenerator
from src.models.hjepa import create_hjepa_from_config
from src.trainers.trainer import HJEPATrainer, create_optimizer


class PublicCIFARSubset(Dataset):
    def __init__(self, split, count):
        rows = next(
            pq.ParquetFile(f"data/cifar10-{split}.parquet").iter_batches(batch_size=count)
        ).to_pylist()
        self.images = [Image.open(io.BytesIO(r["img"]["bytes"])).convert("RGB") for r in rows]
        self.labels = [r["label"] for r in rows]
        self.transform = JEPATransform(64) if split == "train" else JEPAEvalTransform(64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def prepare_cli_folder(config):
    """Export the same real CIFAR rows to the existing ImageFolder loader.

    `dataset: imagenet` selects that loader; the images are still CIFAR-10.
    This path validates scripts/train.py without the slow Toronto tar download.
    """
    root = Path("data/cifar10-folder")
    for source, split, count in [("train", "train", 32), ("test", "val", 16)]:
        dataset = PublicCIFARSubset(source, count)
        for i, (image, label) in enumerate(zip(dataset.images, dataset.labels)):
            directory = root / split / str(label)
            directory.mkdir(parents=True, exist_ok=True)
            image.save(directory / f"{i:05d}.png")
    config["data"].update(dataset="imagenet", data_path=str(root))
    config["experiment"]["output_dir"] = "results/validation/cli"
    config["checkpoint"]["checkpoint_dir"] = "results/validation/cli/checkpoints"
    config["logging"]["log_dir"] = "results/validation/cli/logs"
    Path("results/validation/cli.yaml").write_text(yaml.safe_dump(config))
    print("Prepared real CIFAR-10 PNGs and results/validation/cli.yaml")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build(config, device, resume=None):
    model = create_hjepa_from_config(config)
    loss = create_loss_from_config(config).to(device)
    train = DataLoader(PublicCIFARSubset("train", 32), batch_size=4, shuffle=True, num_workers=0)
    val = DataLoader(PublicCIFARSubset("test", 16), batch_size=4, num_workers=0)
    masks = HierarchicalMaskGenerator(
        input_size=(64, 64),
        patch_size=16,
        num_hierarchies=2,
        num_target_masks=2,
        base_scale=(0.25, 0.4),
    )
    trainer = HJEPATrainer(
        model,
        train,
        val,
        create_optimizer(model, config),
        loss,
        masks,
        config,
        device,
        resume_checkpoint=resume,
    )
    return trainer


def epoch(trainer, index, records):
    original = trainer._train_step

    def diagnostic_step(*args):
        start = time.monotonic()
        loss, values = original(*args)
        timings.append(time.monotonic() - start)
        assert torch.isfinite(loss)
        records.append(
            {
                k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else v
                for k, v in values.items()
            }
        )
        return loss, values

    trainer._train_step = diagnostic_step
    grad_norms = []
    raw_norms = []
    timings = []
    ema_errors = []
    original_clip = torch.nn.utils.clip_grad_norm_
    original_ema = trainer._update_target_encoder

    def clip(*args, **kwargs):
        norm = original_clip(*args, **kwargs)
        assert torch.isfinite(norm)
        raw_norms.append(float(norm))
        return norm

    def ema(momentum):
        if not trainer.model.use_target_encoder:
            return original_ema(momentum)
        target = next(trainer.model.target_encoder.parameters())
        student = next(trainer.model.context_encoder.parameters())
        expected = target.detach().clone().mul_(momentum).add_(student.detach(), alpha=1 - momentum)
        original_ema(momentum)
        error = float((target - expected).abs().max())
        ema_errors.append(error)
        assert error < 1e-7

    torch.nn.utils.clip_grad_norm_ = clip
    trainer._update_target_encoder = ema

    def before_step(opt, args, kwargs):
        grads = [
            p.grad.detach().float().cpu() for p in trainer.model.parameters() if p.grad is not None
        ]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        grad_norms.append(float(torch.stack([g.square().sum() for g in grads]).sum().sqrt()))

    handle = trainer.optimizer.register_step_pre_hook(before_step)
    try:
        trainer._train_epoch(index)
    finally:
        handle.remove()
        trainer._train_step = original
        trainer._update_target_encoder = original_ema
        torch.nn.utils.clip_grad_norm_ = original_clip
    return dict(
        clipped_norms=grad_norms,
        raw_norms=raw_norms,
        ema_errors=ema_errors,
        forward_seconds=timings,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume-check", action="store_true")
    parser.add_argument("--prepare-cli", action="store_true")
    parser.add_argument("--regime", choices=["classic", "sigreg", "end_to_end"], default="classic")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed_all(42)
    config = yaml.safe_load(Path("configs/validation_bounded.yaml").read_text())
    if args.prepare_cli:
        prepare_cli_folder(config)
        return
    config["checkpoint"]["checkpoint_dir"] = f"results/validation/{args.name}/checkpoints"
    config["logging"]["log_dir"] = f"results/validation/{args.name}/logs"
    if args.regime != "classic":
        config["loss"].update(
            type="hjepa_sigreg",
            sigreg_num_slices=16,
            sigreg_test_method="char_function",
            sigreg_vectorized=True,
            detach_targets=args.regime != "end_to_end",
        )
        config["model"]["use_target_encoder"] = args.regime != "end_to_end"
        config["model"]["projection_norm"] = "batchnorm"
    device = (
        ("mps" if torch.backends.mps.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    trainer = build(config, device)
    before = next(trainer.model.context_encoder.parameters()).detach().cpu().clone()
    records = []
    norms = epoch(trainer, 0, records)
    val = trainer._validate_epoch(0)
    path = trainer._save_checkpoint(0, val["loss"], False)
    result = dict(
        name=args.name,
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        working_tree_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        device=device,
        platform=platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        timm=timm.__version__,
        config=config,
        data="uoft-cs/cifar10 Parquet: first 32 train / first 16 test rows",
        losses=records,
        gradient_norms=norms,
        validation=val,
        global_step=trainer.global_step,
        context_parameter_max_update=float(
            (before - next(trainer.model.context_encoder.parameters()).detach().cpu()).abs().max()
        ),
    )
    Path(f"results/validation/{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.resume_check:
        continuation = []
        result["continuation_diagnostics"] = epoch(trainer, 1, continuation)
        expected = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
        expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
        resumed = build(config, device, resume=path)
        replay = []
        result["replay_diagnostics"] = epoch(resumed, 1, replay)
        max_diff = max(
            float((v.detach().cpu() - expected[k]).abs().max())
            for k, v in resumed.model.state_dict().items()
        )
        result.update(
            resume_max_parameter_difference=max_diff,
            continuation=continuation,
            replay=replay,
            resumed_global_step=resumed.global_step,
            expected_global_step=trainer.global_step,
            resumed_lr=resumed.optimizer.param_groups[0]["lr"],
            expected_lr=trainer.optimizer.param_groups[0]["lr"],
        )
        result["resume_optimizer_max_difference"] = max(
            float(
                (value.detach().cpu() - expected_optimizer["state"][pid][key].detach().cpu())
                .abs()
                .max()
            )
            for pid, state in resumed.optimizer.state_dict()["state"].items()
            for key, value in state.items()
            if isinstance(value, torch.Tensor)
        )
        result["resume_loss_max_difference"] = max(
            abs(a["loss"] - b["loss"]) for a, b in zip(continuation, replay)
        )
        resumed.metrics_logger.finish()
    trainer.model.eval()
    with torch.no_grad():
        x, labels = next(iter(trainer.val_loader))
        features = trainer.model.encode_context(x.to(device))[:, 1:].mean(1).cpu()
        centered = features - features.mean(0)
        s = torch.linalg.svdvals(centered)
        prob = s / s.sum()
        result["representation"] = dict(
            batch_feature_std=float(features.std(0).mean()),
            centered_rank=int(torch.linalg.matrix_rank(centered)),
            effective_rank=float((-(prob * (prob + 1e-12).log()).sum()).exp()),
        )
        masks = trainer.masking_fn(batch_size=4, device=trainer.device)["level_0"]["targets"].any(1)
        output = trainer.model(x.to(device), masks)
        result["hierarchy_shapes"] = [list(p.shape) for p in output["predictions"]]
        result["valid_tokens"] = [m.sum(1).tolist() for m in output["masks_valid"]]
        result["teacher_gradients_absent"] = (
            all(
                p.grad is None and not p.requires_grad
                for p in trainer.model.target_encoder.parameters()
            )
            if trainer.model.use_target_encoder
            else None
        )
    trainer.online_eval_k = 3
    trainer.online_eval_max_samples = 32
    result["evaluation"] = trainer._run_online_eval()
    assert "knn_accuracy" in result["evaluation"]
    trainer.metrics_logger.finish()
    out = Path(f"results/validation/{args.name}.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("config", "losses", "continuation", "replay")
            }
        )
    )
    if args.resume_check:
        assert result["resumed_global_step"] == result["expected_global_step"]
        assert result["resume_max_parameter_difference"] < 1e-6
        assert result["resume_optimizer_max_difference"] < 1e-6
        assert result["resume_loss_max_difference"] < 1e-6
    assert result["context_parameter_max_update"] > 0


if __name__ == "__main__":
    main()

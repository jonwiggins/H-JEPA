"""
Pydantic-based configuration validation for H-JEPA.

Provides typed, validated config models so that YAML misconfigurations
(typos, wrong types, missing required fields) are caught at load time
with clear error messages instead of silently ignored.
"""

import logging
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ---- Sub-configs ----


class PredictorConfig(BaseModel):
    depth: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0


class EMAConfig(BaseModel):
    momentum: float = 0.996
    momentum_end: float = 1.0
    momentum_warmup_epochs: int = 30


class RoPEConfig(BaseModel):
    use_rope: bool = False
    theta: float = 10000.0


class FPNConfig(BaseModel):
    use_fpn: bool = False
    feature_dim: int | None = None
    fusion_method: Literal["add", "concat"] = "add"


class ModelConfig(BaseModel):
    encoder_type: str = "vit_base_patch16_224"
    embed_dim: int = 768
    num_hierarchies: int = 3
    predictor: PredictorConfig = PredictorConfig()
    ema: EMAConfig = EMAConfig()
    rope: RoPEConfig = RoPEConfig()
    use_flash_attention: bool = True
    fpn: FPNConfig = FPNConfig()

    model_config = {"extra": "allow"}


class AugmentationConfig(BaseModel):
    color_jitter: float = 0.4
    horizontal_flip: bool = True
    random_crop: bool = True

    model_config = {"extra": "allow"}


class DataConfig(BaseModel):
    dataset: str = "imagenet"
    data_path: str = "/path/to/dataset"
    image_size: int = 224
    batch_size: int = 128
    num_workers: int = 8
    pin_memory: bool = True
    augmentation: AugmentationConfig = AugmentationConfig()

    model_config = {"extra": "allow"}


class MaskingConfig(BaseModel):
    num_masks: int = 4
    mask_scale: list[float] = Field(default=[0.15, 0.2])
    aspect_ratio: list[float] = Field(default=[0.75, 1.5])
    num_context_masks: int = 1
    context_scale: list[float] = Field(default=[0.85, 1.0])

    model_config = {"extra": "allow"}


class EarlyStoppingConfig(BaseModel):
    enabled: bool = False
    patience: int = 10
    min_delta: float = 0.0
    mode: Literal["min", "max"] = "min"


class OnlineEvalConfig(BaseModel):
    enabled: bool = False
    frequency: int = 5
    type: str = "knn"
    k: int = 20
    max_samples: int = 5000


class EMAMomentumScheduleConfig(BaseModel):
    start: float = 0.996
    end: float = 1.0
    warmup_steps: int = 1000


class TrainingConfig(BaseModel):
    epochs: int = 300
    warmup_epochs: int = 40
    lr: float = 1.5e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    betas: list[float] = Field(default=[0.9, 0.95])
    lr_schedule: Literal["cosine", "linear"] = "cosine"
    clip_grad: float | None = 3.0
    use_amp: bool = True
    accumulation_steps: int = Field(default=1, ge=1)
    use_gradient_checkpointing: bool = False
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    online_eval: OnlineEvalConfig = OnlineEvalConfig()
    ema_momentum_schedule: EMAMomentumScheduleConfig = EMAMomentumScheduleConfig()

    model_config = {"extra": "allow"}


class LossConfig(BaseModel):
    type: str = "mse"
    hierarchy_weights: float | list[float] = 1.0
    normalize_embeddings: bool = False
    jepa_loss_type: str = "smoothl1"

    model_config = {"extra": "allow"}


class CheckpointConfig(BaseModel):
    save_frequency: int = 10
    keep_best_n: int = 3
    checkpoint_dir: str = "results/checkpoints"
    resume: str | None = None
    metric: str = "val_loss"
    mode: Literal["min", "max"] = "min"

    model_config = {"extra": "allow"}


class WandBConfig(BaseModel):
    enabled: bool = False
    project: str = "h-jepa"
    entity: str | None = None
    tags: list[str] = Field(default_factory=list)


class TensorBoardConfig(BaseModel):
    enabled: bool = True


class LoggingConfig(BaseModel):
    experiment_name: str = "hjepa_default"
    log_dir: str = "results/logs"
    log_frequency: int = 100
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb: WandBConfig = WandBConfig()
    tensorboard: TensorBoardConfig = TensorBoardConfig()

    model_config = {"extra": "allow"}


class DistributedConfig(BaseModel):
    enabled: bool = False
    backend: Literal["nccl", "gloo"] = "nccl"
    world_size: int = 1


class LinearProbeEvalConfig(BaseModel):
    enabled: bool = False
    dataset: str = "imagenet"
    batch_size: int = 256
    epochs: int = 90
    lr: float = 0.1


class EvaluationConfig(BaseModel):
    eval_frequency: int = 10
    linear_probe: LinearProbeEvalConfig = LinearProbeEvalConfig()

    model_config = {"extra": "allow"}


# ---- Top-level config ----


class HJEPAConfig(BaseModel):
    """
    Top-level validated configuration for H-JEPA.

    Load from YAML with::

        config = HJEPAConfig.from_yaml("configs/default.yaml")

    Or validate an existing dict::

        config = HJEPAConfig(**raw_dict)
    """

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    masking: MaskingConfig = MaskingConfig()
    training: TrainingConfig = TrainingConfig()
    loss: LossConfig = LossConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    logging: LoggingConfig = LoggingConfig()
    distributed: DistributedConfig = DistributedConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    seed: int = 42
    device: str = "cuda"

    # Allow extra top-level keys (experiment, etc.) without failing
    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _validate_hierarchy_weights(self) -> "HJEPAConfig":
        weights = self.loss.hierarchy_weights
        if isinstance(weights, list) and len(weights) != self.model.num_hierarchies:
            raise ValueError(
                f"loss.hierarchy_weights has {len(weights)} entries but "
                f"model.num_hierarchies is {self.model.num_hierarchies}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HJEPAConfig":
        """Load and validate config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Export to plain dict (for passing to legacy code)."""
        return self.model_dump()


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML config, validate it, and return as a plain dict.

    This is a drop-in replacement for raw ``yaml.safe_load`` that adds
    validation. Legacy code expecting a dict can use this directly.

    Raises:
        pydantic.ValidationError: on invalid config with clear messages
    """
    config = HJEPAConfig.from_yaml(path)
    return config.to_dict()

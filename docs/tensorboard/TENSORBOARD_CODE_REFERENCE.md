# H-JEPA TensorBoard Code Reference

Detailed code snippets and method signatures for the TensorBoard logging implementation.

---

## 1. MetricsLogger Class

**File:** `/Users/jon/repos/H-JEPA/src/utils/logging.py`

### 1.1 Class Initialization

**Lines:** 62-105

```python
def __init__(
    self,
    experiment_name: str,
    log_dir: str,
    config: Optional[Dict[str, Any]] = None,
    use_wandb: bool = True,
    use_tensorboard: bool = True,
    wandb_project: str = "h-jepa",
    wandb_entity: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
):
    self.experiment_name = experiment_name
    self.log_dir = Path(log_dir)
    self.log_dir.mkdir(parents=True, exist_ok=True)

    self.use_wandb = use_wandb and WANDB_AVAILABLE
    self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE

    # Initialize TensorBoard
    if self.use_tensorboard:
        try:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"TensorBoard initialized: {tb_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.use_tensorboard = False

    # Metrics aggregation
    self.metrics_buffer = defaultdict(list)
    self.step = 0
```

**Key Points:**
- TensorBoard initialized in `<log_dir>/tensorboard/` subdirectory
- Graceful fallback if initialization fails
- `SummaryWriter` is stored as `self.tb_writer`
- Internal step counter for tracking global progress

---

### 1.2 Core Logging Methods

#### log_metrics()

**Lines:** 111-149

**Signature:**
```python
def log_metrics(
    self,
    metrics: Dict[str, Union[float, int]],
    step: Optional[int] = None,
    prefix: str = "",
    commit: bool = True,
) -> None
```

**TensorBoard Call:**
```python
for name, value in metrics.items():
    self.tb_writer.add_scalar(name, value, self.step)
```

**Usage Example from Trainer:**
```python
# Log training metrics
self.metrics_logger.log_metrics(
    train_metrics,
    step=self.global_step,
    prefix="train/",
)

# Log validation metrics
self.metrics_logger.log_metrics(
    val_metrics,
    step=self.global_step,
    prefix="val/",
)
```

---

#### log_image()

**Lines:** 151-186

**Signature:**
```python
def log_image(
    self,
    name: str,
    image: Union[np.ndarray, torch.Tensor],
    step: Optional[int] = None,
    caption: Optional[str] = None,
) -> None
```

**TensorBoard Call:**
```python
if isinstance(image, torch.Tensor):
    image = image.detach().cpu().numpy()

if self.use_tensorboard:
    try:
        self.tb_writer.add_image(name, image, step)
    except Exception as e:
        logger.warning(f"Failed to log image to TensorBoard: {e}")
```

**Expected Image Format:**
- Torch Tensor: `(C, H, W)` with values in [0, 1] or [0, 255]
- NumPy Array: Same as above

**Usage Example (Not Currently Called):**
```python
# Log masked image
masked_img = images * mask.unsqueeze(-1)
self.metrics_logger.log_image(
    "training/masked_image",
    masked_img,
    step=self.global_step
)
```

---

#### log_images()

**Lines:** 188-237

**Signature:**
```python
def log_images(
    self,
    name: str,
    images: List[Union[np.ndarray, torch.Tensor]],
    step: Optional[int] = None,
    captions: Optional[List[str]] = None,
) -> None
```

**TensorBoard Implementation:**
```python
images_tensor = torch.stack([
    torch.from_numpy(img) if isinstance(img, np.ndarray) else img
    for img in images_np
])
from torchvision.utils import make_grid
grid = make_grid(images_tensor, nrow=4)
self.tb_writer.add_image(name, grid, step)
```

**Behavior:**
- Stacks multiple images into a grid
- Default: 4 images per row
- Creates a single TensorBoard image entry

**Usage Example (Not Currently Called):**
```python
# Log predictions vs targets
pred_images = [pred[i] for i in range(8)]
target_images = [target[i] for i in range(8)]

self.metrics_logger.log_images(
    "predictions/vs_targets",
    pred_images + target_images,
    step=self.global_step,
    captions=["pred"] * 8 + ["target"] * 8
)
```

---

#### log_histogram()

**Lines:** 239-272

**Signature:**
```python
def log_histogram(
    self,
    name: str,
    values: Union[np.ndarray, torch.Tensor],
    step: Optional[int] = None,
) -> None
```

**TensorBoard Call:**
```python
if isinstance(values, torch.Tensor):
    values = values.detach().cpu().numpy()

if self.use_tensorboard:
    try:
        self.tb_writer.add_histogram(name, values, step)
    except Exception as e:
        logger.warning(f"Failed to log histogram to TensorBoard: {e}")
```

**Usage Example (Not Currently Called):**
```python
# Log embedding distributions
embeddings = model.encode_context(images)
self.metrics_logger.log_histogram(
    "features/context_embeddings",
    embeddings,
    step=self.global_step
)
```

---

#### log_model_gradients()

**Lines:** 274-295

**Signature:**
```python
def log_model_gradients(
    self,
    model: nn.Module,
    step: Optional[int] = None,
) -> None
```

**Implementation:**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        self.log_histogram(
            f"gradients/{name}",
            param.grad,
            step=step,
        )
```

**Creates Histograms For:**
- All learnable parameters with gradients
- Prefix: `gradients/{param_name}`

**Usage Example (Not Currently Called):**
```python
# Log after backward pass, before optimizer step
self.metrics_logger.log_model_gradients(
    self.model,
    step=self.global_step
)
```

---

#### log_model_weights()

**Lines:** 297-317

**Signature:**
```python
def log_model_weights(
    self,
    model: nn.Module,
    step: Optional[int] = None,
) -> None
```

**Implementation:**
```python
for name, param in model.named_parameters():
    self.log_histogram(
        f"weights/{name}",
        param.data,
        step=step,
    )
```

**Creates Histograms For:**
- All learnable parameters
- Prefix: `weights/{param_name}`

**Usage Example (Not Currently Called):**
```python
# Periodically log weight distributions
if step % (log_frequency * 100) == 0:
    self.metrics_logger.log_model_weights(
        self.model,
        step=self.global_step
    )
```

---

### 1.3 Metrics Aggregation Methods

#### accumulate_metrics()

**Lines:** 319-332

**Signature:**
```python
def accumulate_metrics(
    self,
    metrics: Dict[str, float],
) -> None
```

**Purpose:** Buffer metrics for later averaging (e.g., per-epoch statistics)

**Usage in Trainer (Lines 230, 282):**
```python
# Start of epoch
self.metrics_logger.accumulate_metrics({})  # Reset

# During epoch
self.metrics_logger.accumulate_metrics(loss_dict)

# End of epoch
self.metrics_logger.log_accumulated_metrics(
    step=self.global_step,
    prefix="train_epoch/",
    reset=True,
)
```

---

#### log_accumulated_metrics()

**Lines:** 334-367

**Signature:**
```python
def log_accumulated_metrics(
    self,
    step: Optional[int] = None,
    prefix: str = "",
    reset: bool = True,
) -> None
```

**Behavior:**
1. Computes mean of accumulated metrics
2. Logs averaged values using `log_metrics()`
3. Optionally resets buffer for next epoch

**Implementation Details (Lines 354-360):**
```python
# Convert to CPU if tensors, then to numpy array
cpu_values = [
    v.cpu().item() if isinstance(v, torch.Tensor) else v
    for v in values
]
averaged_metrics[name] = np.mean(cpu_values)
```

---

### 1.4 System Monitoring

#### log_system_metrics()

**Lines:** 369-406

**Signature:**
```python
def log_system_metrics(
    self,
    step: Optional[int] = None,
) -> None
```

**Metrics Collected:**

1. **GPU Memory (Always)**
   ```python
   metrics[f"system/gpu{i}_memory_allocated_gb"] = torch.cuda.memory_allocated(i) / 1e9
   metrics[f"system/gpu{i}_memory_reserved_gb"] = torch.cuda.memory_reserved(i) / 1e9
   ```

2. **GPU Utilization (Optional - if pynvml available)**
   ```python
   import pynvml
   pynvml.nvmlInit()
   handle = pynvml.nvmlDeviceGetHandleByIndex(i)
   util = pynvml.nvmlDeviceGetUtilizationRates(handle)
   metrics[f"system/gpu{i}_utilization"] = util.gpu
   metrics[f"system/gpu{i}_memory_utilization"] = util.memory
   ```

**Usage in Trainer (Lines 204-206):**
```python
if epoch % 10 == 0:
    self.metrics_logger.log_system_metrics(step=self.global_step)
```

---

### 1.5 Cleanup

#### finish()

**Lines:** 427-441

**Signature:**
```python
def finish(self) -> None
```

**Implementation:**
```python
if self.use_tensorboard:
    try:
        self.tb_writer.close()
        logger.info("TensorBoard writer closed")
    except Exception as e:
        logger.warning(f"Error closing TensorBoard: {e}")
```

**When Called:** At end of training (see context manager support below)

---

### 1.6 Context Manager Support

**Lines:** 443-449

```python
def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.finish()
```

**Usage:**
```python
with MetricsLogger(...) as logger:
    # Training loop
    logger.log_metrics(metrics)
    # Automatic cleanup on exit
```

---

## 2. Trainer Integration

**File:** `/Users/jon/repos/H-JEPA/src/trainers/trainer.py`

### 2.1 Trainer Initialization

**Lines:** 122-135

```python
self.metrics_logger = MetricsLogger(
    experiment_name=config['logging']['experiment_name'],
    log_dir=config['logging']['log_dir'],
    config=config,
    use_wandb=wandb_config.get('enabled', False),
    use_tensorboard=tensorboard_config.get('enabled', True),
    wandb_project=wandb_config.get('project', 'h-jepa'),
    wandb_entity=wandb_config.get('entity', None),
    wandb_tags=wandb_config.get('tags', []),
)
```

---

### 2.2 Per-Step Logging

**Lines:** 295-306

```python
if batch_idx % (self.log_frequency * 5) == 0:
    log_dict = {
        'lr': current_lr,
        'ema_momentum': self.ema_scheduler(self.global_step),
    }
    log_dict.update(loss_dict)
    self.metrics_logger.log_metrics(
        log_dict,
        step=self.global_step,
        prefix="train/",
    )
```

**Frequency:** Every `log_frequency * 5` steps (default: 500 steps with log_frequency=100)

**Logged Metrics:**
- `train/loss` - Main training loss
- `train/lr` - Current learning rate
- `train/ema_momentum` - Current EMA momentum value
- `train/{other}` - Any additional metrics from loss_dict

---

### 2.3 Per-Epoch Logging

**Lines:** 177-191**

```python
# Train one epoch
train_metrics = self._train_epoch(epoch)

# Log epoch metrics
self.metrics_logger.log_metrics(
    train_metrics,
    step=self.global_step,
    prefix="train/",
)

# Validation
if self.val_loader is not None:
    val_metrics = self._validate_epoch(epoch)
    self.metrics_logger.log_metrics(
        val_metrics,
        step=self.global_step,
        prefix="val/",
    )
```

**Logged Metrics:**
- `train/loss` - Epoch average training loss
- `val/loss` - Epoch validation loss

---

### 2.4 Accumulated Metrics Logging

**Lines:** 309-313 (end of _train_epoch)**

```python
# Compute epoch averages
self.metrics_logger.log_accumulated_metrics(
    step=self.global_step,
    prefix="train_epoch/",
    reset=True,
)
```

**Purpose:** Average all metrics accumulated during epoch

---

### 2.5 Collapse Monitoring

**Lines:** 385-390**

```python
# Monitor representation collapse
if step % (self.log_frequency * 10) == 0:
    collapse_metrics = self._compute_collapse_metrics(
        outputs['context_features'],
        outputs['target_features'],
    )
    loss_dict.update(collapse_metrics)
```

**Metrics Computed (Lines 495-542):**

```python
# Standard deviation (should be > 0)
metrics['context_std'] = context_flat.std().item()
metrics['target_std'] = target_flat.std().item()

# Mean L2 norm
metrics['context_norm'] = context_flat.norm(dim=1).mean().item()
metrics['target_norm'] = target_flat.norm(dim=1).mean().item()

# Effective rank (using SVD)
context_sv = torch.svd(context_sample)[1]
target_sv = torch.svd(target_sample)[1]

# Entropy-based effective rank
context_sv_norm = context_sv / context_sv.sum()
target_sv_norm = target_sv / target_sv.sum()

context_entropy = -(context_sv_norm * torch.log(context_sv_norm + 1e-8)).sum()
target_entropy = -(target_sv_norm * torch.log(target_sv_norm + 1e-8)).sum()

metrics['context_rank'] = torch.exp(context_entropy).item()
metrics['target_rank'] = torch.exp(target_entropy).item()
```

**Frequency:** Every `log_frequency * 10` steps (default: 1000 steps)

**Logged Metrics:**
- `train/context_std` - Context encoder output std dev
- `train/target_std` - Target encoder output std dev
- `train/context_norm` - Context encoder output L2 norm
- `train/target_norm` - Target encoder output L2 norm
- `train/context_rank` - Context encoder effective rank
- `train/target_rank` - Target encoder effective rank

---

### 2.6 System Metrics Logging

**Lines:** 204-206**

```python
if epoch % 10 == 0:
    self.metrics_logger.log_system_metrics(step=self.global_step)
```

**Frequency:** Every 10 epochs

**Logged Metrics:**
- `system/gpu{i}_memory_allocated_gb` - Allocated GPU memory
- `system/gpu{i}_memory_reserved_gb` - Reserved GPU memory
- `system/gpu{i}_utilization` - GPU utilization % (if pynvml available)
- `system/gpu{i}_memory_utilization` - GPU memory utilization % (if pynvml available)

---

## 3. Configuration

**File:** `/Users/jon/repos/H-JEPA/configs/default.yaml`

### 3.1 Logging Configuration

**Lines:** 164-184

```yaml
logging:
  # Experiment name
  experiment_name: "hjepa_default"

  # Logging directory
  log_dir: "results/logs"

  # Log every N steps
  log_frequency: 100

  # Weights & Biases settings
  wandb:
    enabled: true
    project: "h-jepa"
    entity: null  # Your W&B username/team
    tags: ["baseline", "vit-base"]

  # TensorBoard settings
  tensorboard:
    enabled: true
```

### 3.2 Config Handling in Trainer

**Lines:** 123-124**

```python
wandb_config = config['logging'].get('wandb', {})
tensorboard_config = config['logging'].get('tensorboard', {})
```

**Safe defaults ensure TensorBoard works even with minimal config**

---

## 4. TensorBoard Scalar Naming Convention

All scalars follow a hierarchical naming convention for organization in TensorBoard:

### 4.1 Training Metrics
- `train/loss` - Main training loss
- `train/lr` - Learning rate
- `train/ema_momentum` - EMA momentum coefficient
- `train/collapse_*` - Collapse detection metrics
- `train_epoch/loss` - Per-epoch average loss

### 4.2 Validation Metrics
- `val/loss` - Validation loss

### 4.3 System Metrics
- `system/gpu{i}_memory_allocated_gb` - GPU memory allocated
- `system/gpu{i}_memory_reserved_gb` - GPU memory reserved
- `system/gpu{i}_utilization` - GPU utilization percentage
- `system/gpu{i}_memory_utilization` - GPU memory utilization percentage

### 4.4 Optional (if implemented)
- `gradients/{param_name}` - Gradient histograms
- `weights/{param_name}` - Weight histograms

---

## 5. Example Usage Patterns

### 5.1 Logging during Training (Currently Used)

```python
# In trainer loop
metrics_dict = {'loss': loss.item(), 'grad_norm': grad_norm}
self.metrics_logger.log_metrics(
    metrics_dict,
    step=self.global_step,
    prefix="train/"
)
```

### 5.2 Logging Images (Ready but Not Used)

```python
# Log masked images
masked_img = image * mask.unsqueeze(-1)
self.metrics_logger.log_image(
    "training/masked_samples/batch_0",
    masked_img,
    step=self.global_step
)

# Log multiple images as grid
predictions = [pred[i] for i in range(8)]
self.metrics_logger.log_images(
    "training/predictions_grid",
    predictions,
    step=self.global_step
)
```

### 5.3 Logging Histograms (Ready but Not Used)

```python
# Log feature distributions
embeddings = model.encode_context(images)
self.metrics_logger.log_histogram(
    "features/context_embeddings",
    embeddings,
    step=self.global_step
)

# Log gradients
self.metrics_logger.log_model_gradients(
    self.model,
    step=self.global_step
)

# Log weights
self.metrics_logger.log_model_weights(
    self.model,
    step=self.global_step
)
```

### 5.4 Logging Accumulated Metrics

```python
# During epoch
for batch in dataloader:
    metrics = compute_metrics(batch)
    self.metrics_logger.accumulate_metrics(metrics)

# At end of epoch
self.metrics_logger.log_accumulated_metrics(
    step=epoch,
    prefix="epoch/",
    reset=True
)
```

---

## 6. Error Handling

All TensorBoard operations are wrapped in try-except blocks to ensure training continues even if logging fails:

**Pattern (used throughout logging.py):**
```python
if self.use_tensorboard:
    try:
        self.tb_writer.add_scalar(name, value, self.step)
    except Exception as e:
        logger.warning(f"Failed to log to TensorBoard: {e}")
```

**Benefits:**
- Logging failures don't crash training
- Warnings logged for debugging
- Graceful degradation

---

## 7. File Organization

```
/Users/jon/repos/H-JEPA/
├── src/
│   ├── utils/
│   │   ├── logging.py          # MetricsLogger class
│   │   ├── checkpoint.py
│   │   ├── scheduler.py
│   │   └── __init__.py
│   ├── trainers/
│   │   ├── trainer.py          # HJEPATrainer with logging integration
│   │   └── __init__.py
│   └── visualization/
│       ├── training_viz.py     # Separate visualization utilities
│       ├── masking_viz.py
│       ├── attention_viz.py
│       └── __init__.py
├── configs/
│   └── default.yaml            # TensorBoard configuration
├── scripts/
│   └── train.py                # Main training entry point
└── launch_tensorboard.sh        # TensorBoard launch script
```

---

**End of Code Reference**

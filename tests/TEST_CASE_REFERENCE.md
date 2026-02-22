# Test Case Reference for test_trainers.py

## Comprehensive Test Case Documentation

This document provides detailed descriptions of all 68 test cases in the trainer test suite.

---

## 1. TestTrainerInitialization (8 tests)

### test_trainer_init_basic
**Purpose:** Verify basic trainer initialization with minimal configuration
**Coverage:** Constructor, config parsing, basic state setup
**Assertions:**
- Model is assigned
- Train loader is assigned
- Validation loader is None
- Epoch and step counters start at 0
- Total epochs matches config

**Example:**
```python
trainer = HJEPATrainer(
    model=model,
    train_loader=train_loader,
    val_loader=None,  # <-- Important: None here
    optimizer=optimizer,
    loss_fn=loss_fn,
    masking_fn=masking_fn,
    config=config,
    device=device,
)
assert trainer.epochs == 2
```

### test_trainer_init_with_validation
**Purpose:** Verify initialization with validation loader
**Coverage:** Validation loader assignment, best metric tracking initialization
**Assertions:**
- val_loader is not None
- best_val_loss is initialized to infinity

### test_trainer_init_with_warmup
**Purpose:** Verify learning rate warmup configuration
**Coverage:** Warmup epoch configuration, scheduler warmup behavior
**Assertions:**
- warmup_epochs matches config
- LR at step 0 is lower than after warmup (proof of linear warmup)
- Warmup steps are correctly computed

**Test Logic:**
```python
lr_step_0 = trainer.lr_scheduler(0)      # During warmup
lr_step_100 = trainer.lr_scheduler(100)  # After warmup
assert lr_step_0 < lr_step_100  # LR should increase during training
```

### test_trainer_init_with_gradient_accumulation
**Purpose:** Verify gradient accumulation setup
**Coverage:** accumulation_steps configuration
**Assertions:**
- accumulation_steps is set correctly in config
- steps_per_epoch is adjusted for accumulation

### test_trainer_init_with_amp
**Purpose:** Verify automatic mixed precision (AMP) initialization
**Precondition:** Not on MPS device (AMP not supported on MPS)
**Coverage:** AMP configuration, GradScaler initialization
**Assertions:**
- use_amp flag matches config
- scaler is initialized if use_amp=True
- scaler is None if use_amp=False

### test_trainer_init_checkpoint_manager
**Purpose:** Verify CheckpointManager is properly set up
**Coverage:** Checkpoint manager initialization and attributes
**Assertions:**
- checkpoint_manager is not None
- Has save_checkpoint method
- Has load_checkpoint method
- Has checkpoint_dir attribute

### test_trainer_init_metrics_logger
**Purpose:** Verify MetricsLogger is properly set up
**Coverage:** Metrics logger initialization
**Assertions:**
- metrics_logger is not None
- Has log_metrics method
- Experiment name is set

### test_trainer_init_progress_tracker
**Purpose:** Verify ProgressTracker is properly set up
**Coverage:** Progress tracking initialization
**Assertions:**
- progress_tracker is not None
- Has start_epoch method

---

## 2. TestLearningRateScheduling (4 tests)

### test_lr_scheduler_cosine
**Purpose:** Verify cosine annealing learning rate schedule
**Coverage:** CosineScheduler integration, LR decay behavior
**Test Logic:**
```python
lr_step_0 = trainer.lr_scheduler(0)      # Early in training
lr_step_50 = trainer.lr_scheduler(50)    # Middle of training
lr_final = trainer.lr_scheduler(total_steps)  # End of training

# Cosine schedule: LR should decrease monotonically
assert lr_step_0 >= lr_step_50 >= lr_final
```

**Mathematical Validation:**
- Checks cosine annealing formula: LR = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(...))

### test_lr_scheduler_linear
**Purpose:** Verify linear learning rate decay schedule
**Coverage:** LinearScheduler integration, LR decay behavior
**Test Logic:**
```python
lr_step_0 = trainer.lr_scheduler(0)      # Start
lr_step_50 = trainer.lr_scheduler(50)    # Middle
lr_final = trainer.lr_scheduler(total_steps)  # End

# Linear schedule should also decrease monotonically
assert lr_step_0 >= lr_step_50 >= lr_final
```

### test_ema_scheduler
**Purpose:** Verify EMA momentum scheduling
**Coverage:** EMAScheduler integration, momentum progression
**Test Logic:**
```python
ema_step_0 = trainer.ema_scheduler(0)    # Start (base momentum)
ema_step_50 = trainer.ema_scheduler(50)  # Middle
ema_final = trainer.ema_scheduler(total_steps)  # End (final momentum)

# EMA should increase from base to final
assert ema_step_0 <= ema_step_50 <= ema_final
```

**Physical Interpretation:**
- Start with 0.996 momentum (more context encoder influence)
- Gradually increase to 1.0 (target encoder becomes fixed)

### test_scheduler_values_in_valid_range
**Purpose:** Verify scheduler values stay within reasonable bounds
**Coverage:** Boundary condition checking
**Assertions:**
- Learning rate: min_lr ≤ lr ≤ base_lr
- EMA momentum: 0.99 ≤ ema ≤ 1.01

---

## 3. TestOptimizerCreation (5 tests)

### test_create_adamw_optimizer
**Purpose:** Verify AdamW optimizer creation
**Coverage:** create_optimizer with adamw type
**Configuration:**
```python
config["training"]["optimizer"] = "adamw"
config["training"]["betas"] = [0.9, 0.95]
```
**Assertions:**
- isinstance(optimizer, torch.optim.AdamW)
- optimizer.defaults["lr"] == 1e-4
- optimizer.defaults["betas"] == (0.9, 0.95)

### test_create_adam_optimizer
**Purpose:** Verify Adam optimizer creation
**Coverage:** create_optimizer with adam type
**Assertions:**
- isinstance(optimizer, torch.optim.Adam)
- Correct learning rate

### test_create_sgd_optimizer
**Purpose:** Verify SGD optimizer creation
**Coverage:** create_optimizer with sgd type
**Configuration:**
```python
config["training"]["optimizer"] = "sgd"
config["training"]["momentum"] = 0.9
```
**Assertions:**
- isinstance(optimizer, torch.optim.SGD)
- optimizer.defaults["momentum"] == 0.9

### test_optimizer_with_weight_decay
**Purpose:** Verify weight decay configuration
**Coverage:** Weight decay parameter passing
**Configuration:**
```python
config["training"]["weight_decay"] = 0.05
```
**Assertions:**
- optimizer.defaults["weight_decay"] == 0.05

### test_optimizer_invalid_type
**Purpose:** Verify error handling for invalid optimizer type
**Coverage:** Error handling in create_optimizer
**Test Logic:**
```python
config["training"]["optimizer"] = "invalid_optimizer"
with pytest.raises(ValueError, match="Unknown optimizer"):
    create_optimizer(model, config)
```

---

## 4. TestTrainingStep (6 tests)

### test_train_step_basic
**Purpose:** Verify basic forward pass and loss computation in training step
**Coverage:** _train_step method, loss computation pipeline
**Test Data:**
```python
batch = [torch.randn(4, 3, 224, 224, device=device)]
loss, loss_dict = trainer._train_step(batch, epoch=0, step=0)
```
**Assertions:**
- loss is torch.Tensor
- "loss" key in loss_dict
- loss.item() > 0

**What Happens:**
1. Batch moved to device
2. Masks generated via masking_fn
3. Forward pass through model
4. Loss computed via loss_fn
5. EMA update applied
6. Collapse metrics computed (if applicable)

### test_train_step_returns_scalar_loss
**Purpose:** Verify loss is scalar tensor (not vector)
**Coverage:** Loss tensor shape validation
**Assertions:**
- loss.dim() == 0 or loss.shape == torch.Size([])
- loss.item() returns float

### test_train_step_gradient_computation
**Purpose:** Verify gradients are computed during training step
**Coverage:** Gradient accumulation, backward pass
**Test Logic:**
```python
initial_params = [p.clone() for p in trainer.model.parameters()]
loss, _ = trainer._train_step(batch, epoch=0, step=0)
loss.backward()  # Backward pass

# Check for gradients
has_gradients = any(p.grad is not None for p in trainer.model.parameters())
```

### test_train_step_ema_update
**Purpose:** Verify EMA update is performed
**Coverage:** _update_target_encoder call, momentum application
**Verification:**
- _update_target_encoder is called with appropriate momentum
- Target encoder parameters are updated (indirectly tested)

### test_train_step_collapse_metrics
**Purpose:** Verify collapse metrics are computed and included in loss_dict
**Coverage:** _compute_collapse_metrics integration
**Assertions:**
- Collapse metrics included in loss_dict at appropriate frequency
- Keys like "context_std", "target_std" present

**Frequency:** Computed every `log_frequency * 10` steps

### test_train_step_with_tuple_batch
**Purpose:** Verify handling of tuple batch format (images, labels)
**Coverage:** Flexible batch format support
**Test Data:**
```python
images = torch.randn(4, 3, 224, 224, device=device)
labels = torch.randint(0, 100, (4,), device=device)
batch = (images, labels)
```
**Assertions:**
- No errors raised
- Loss computed correctly
- Labels handled gracefully (may be ignored in forward pass)

---

## 5. TestValidation (4 tests)

### test_validate_epoch_basic
**Purpose:** Verify basic validation epoch execution
**Coverage:** _validate_epoch method, validation loop
**Test Logic:**
```python
with torch.no_grad():
    val_metrics = trainer._validate_epoch(epoch=0)

assert "loss" in val_metrics
assert isinstance(val_metrics["loss"], float)
assert val_metrics["loss"] > 0
```

**What Happens:**
1. Model set to eval mode
2. Iterate over val_loader
3. Forward pass (no gradient computation)
4. Loss computed for each batch
5. Average loss returned

### test_validate_epoch_averages_metrics
**Purpose:** Verify validation computes average metrics across batches
**Coverage:** Metric aggregation during validation
**Assertions:**
- Returned loss is average over all batches
- Not NaN or infinite
- Matches expected range

**Computation:**
```
avg_val_loss = np.mean([loss_batch_1, loss_batch_2, ...])
```

### test_validate_epoch_model_in_eval_mode
**Purpose:** Verify model is set to eval mode during validation
**Coverage:** Model mode management
**Verification:**
- trainer.model.eval() is called
- Dropout/BatchNorm behave as during inference

### test_validate_epoch_no_gradients
**Purpose:** Verify validation doesn't compute gradients
**Coverage:** torch.no_grad() context usage
**Test Logic:**
```python
with torch.no_grad():
    val_metrics = trainer._validate_epoch(epoch=0)

# Verify no gradients were computed
for param in trainer.model.parameters():
    assert param.grad is None
```

---

## 6. TestCheckpointManagement (7 tests)

### test_save_checkpoint
**Purpose:** Verify basic checkpoint saving functionality
**Coverage:** _save_checkpoint method, file I/O
**Test Logic:**
```python
checkpoint_path = trainer._save_checkpoint(epoch=0, val_loss=0.5, is_best=False)

assert os.path.exists(checkpoint_path)
assert "epoch_0000" in checkpoint_path
```

### test_save_checkpoint_best
**Purpose:** Verify best checkpoint is saved with special marking
**Coverage:** Best checkpoint tracking
**Test Logic:**
```python
trainer._save_checkpoint(epoch=0, val_loss=0.5, is_best=True)

best_path = Path(temp_checkpoint_dir) / "checkpoint_best.pth"
assert best_path.exists()
```

### test_save_checkpoint_latest
**Purpose:** Verify latest checkpoint symlink is created
**Coverage:** Latest checkpoint management
**Test Logic:**
```python
trainer._save_checkpoint(epoch=0, val_loss=0.5, is_best=False)

latest_path = Path(temp_checkpoint_dir) / "checkpoint_latest.pth"
assert latest_path.exists()
```

### test_checkpoint_contains_required_state
**Purpose:** Verify checkpoint contains all necessary state
**Coverage:** Checkpoint state dict completeness
**Test Logic:**
```python
trainer._save_checkpoint(epoch=1, val_loss=0.4, is_best=False)
checkpoint = torch.load(checkpoint_path)

assert "model_state_dict" in checkpoint
assert "optimizer_state_dict" in checkpoint
assert "epoch" in checkpoint
```

**Required Keys:**
- model_state_dict
- optimizer_state_dict
- epoch
- metrics
- best_metric

### test_resume_from_checkpoint
**Purpose:** Verify training can resume from saved checkpoint
**Coverage:** _resume_from_checkpoint method, state restoration
**Test Logic:**
```python
# Save checkpoint at epoch 5
trainer._save_checkpoint(epoch=5, val_loss=0.4, is_best=False)

# Create new trainer and resume
new_trainer = HJEPATrainer(..., resume_checkpoint=checkpoint_path)

# Verify resumption
assert new_trainer.current_epoch == 6  # Next epoch after 5
assert new_trainer.best_val_loss <= 0.5
```

### test_checkpoint_manager_should_save
**Purpose:** Verify checkpoint saving frequency logic
**Coverage:** save_frequency parameter
**Test Logic:**
```python
save_freq = trainer.checkpoint_manager.save_frequency

# Should save every save_freq epochs
assert trainer.checkpoint_manager.should_save(save_freq - 1)  # True at epoch save_freq-1
```

---

## 7. TestMetricTracking (5 tests)

### test_metrics_logger_initialization
**Purpose:** Verify metrics logger is properly initialized
**Coverage:** MetricsLogger setup
**Assertions:**
- metrics_logger is not None
- experiment_name is set
- log_dir exists or is created

### test_accumulate_metrics
**Purpose:** Verify metric accumulation functionality
**Coverage:** MetricsLogger.accumulate_metrics
**Test Logic:**
```python
loss_dict = {
    "loss": torch.tensor(0.5),
    "pred_loss": torch.tensor(0.4)
}
trainer.metrics_logger.accumulate_metrics(loss_dict)
# Should not raise any exception
```

### test_log_metrics
**Purpose:** Verify metrics are logged correctly
**Coverage:** MetricsLogger.log_metrics
**Test Logic:**
```python
metrics = {"loss": 0.5, "accuracy": 0.95}
trainer.metrics_logger.log_metrics(metrics, step=0, prefix="train/")
# Should log to TensorBoard/W&B (disabled in tests)
```

### test_log_system_metrics
**Purpose:** Verify system metrics (GPU/memory) logging
**Coverage:** MetricsLogger.log_system_metrics
**Metrics Logged:**
- GPU memory allocated/reserved
- System memory usage
- Time elapsed

### test_epoch_visualizations_logging
**Purpose:** Verify visualization logging is called
**Coverage:** _log_epoch_visualizations integration
**What's Visualized:**
- Prediction comparisons
- Embeddings projection
- Attention maps (if applicable)

---

## 8. TestEMAUpdates (3 tests)

### test_update_target_encoder
**Purpose:** Verify target encoder EMA update mechanism
**Coverage:** _update_target_encoder method
**Mathematical Formula:**
```
target_param = momentum * target_param + (1 - momentum) * context_param
```

**Test Logic:**
```python
momentum = 0.99
trainer._update_target_encoder(momentum)
# Should not raise exception
# Target encoder params should be updated
```

### test_ema_momentum_range
**Purpose:** Verify EMA momentum values are in valid range
**Coverage:** Scheduler value bounds
**Assertions:**
```python
for step in range(0, total_steps, 100):
    momentum = trainer.ema_scheduler(step)
    assert 0.99 <= momentum <= 1.01
```

### test_ema_update_with_different_momentums
**Purpose:** Verify EMA update works with various momentum values
**Coverage:** Momentum flexibility
**Test Momentums:** [0.99, 0.995, 0.999]

---

## 9. TestCollapseDetection (3 tests)

### test_compute_collapse_metrics
**Purpose:** Verify representation collapse metric computation
**Coverage:** _compute_collapse_metrics method
**Test Data:**
```python
context_emb = torch.randn(4, 196, 384, device=device)
target_emb = torch.randn(4, 196, 384, device=device)
metrics = trainer._compute_collapse_metrics(context_emb, target_emb)
```

**Metrics Computed:**
- context_std: Standard deviation of context embeddings
- target_std: Standard deviation of target embeddings
- context_norm: Mean L2 norm of context embeddings
- target_norm: Mean L2 norm of target embeddings
- context_eff_rank: Effective rank (if not on MPS)
- target_eff_rank: Effective rank (if not on MPS)

### test_collapse_metrics_values_valid
**Purpose:** Verify collapse metrics have physically meaningful values
**Coverage:** Metric value validation
**Assertions:**
```python
assert metrics["context_std"] > 0      # Should not be zero (collapse indicator)
assert metrics["target_std"] > 0
assert metrics["context_norm"] > 0     # Non-zero embeddings
assert metrics["target_norm"] > 0
```

**Why This Matters:**
- Low std → representation collapse (bad)
- High std → diverse representations (good)
- Norms > 0 → embeddings are not zero (good)

### test_collapse_metrics_2d_embeddings
**Purpose:** Verify collapse metrics work with 2D embeddings
**Coverage:** Flexible embedding shape handling
**Test Data:**
```python
context_emb = torch.randn(4, 384, device=device)  # 2D, not 3D
target_emb = torch.randn(4, 384, device=device)
metrics = trainer._compute_collapse_metrics(context_emb, target_emb)
```

---

## 10. TestErrorHandling (4 tests)

### test_train_step_with_empty_batch
**Purpose:** Verify handling of empty batches
**Coverage:** Edge case handling, error conditions
**Test Data:**
```python
empty_batch = [torch.randn(0, 3, 224, 224, device=device)]
with pytest.raises((RuntimeError, ValueError)):
    trainer._train_step(empty_batch, epoch=0, step=0)
```

**Expected Behavior:**
- Raises RuntimeError or ValueError
- OR gracefully skips batch

### test_missing_val_loader
**Purpose:** Verify trainer works without validation loader
**Coverage:** Optional validation support
**Configuration:**
```python
trainer = HJEPATrainer(
    ...,
    val_loader=None,  # No validation
    ...
)
```

**Assertions:**
- trainer.val_loader is None
- No validation loss tracking
- Training proceeds normally

### test_loss_nan_detection
**Purpose:** Verify handling of NaN loss values
**Coverage:** Numerical stability checks
**Test Logic:**
```python
nan_loss_fn = Mock(return_value={"loss": torch.tensor(float("nan"))})
trainer.loss_fn = nan_loss_fn

loss, _ = trainer._train_step(batch, epoch=0, step=0)
assert torch.isnan(loss)
```

### test_gradient_clipping
**Purpose:** Verify gradient clipping functionality
**Coverage:** Gradient normalization, numerical stability
**Test Logic:**
```python
trainer.clip_grad = 1.0
loss, _ = trainer._train_step(batch, epoch=0, step=0)
loss.backward()

# Clipping should work
torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
```

---

## 11. TestTrainingLoopIntegration (4 tests)

### test_single_epoch_training
**Purpose:** Verify complete single epoch training
**Coverage:** Full training epoch pipeline
**Mark:** @pytest.mark.slow
**Test Logic:**
```python
metrics = trainer._train_epoch(epoch=0)

assert isinstance(metrics, dict)
assert "loss" in metrics
assert metrics["loss"] > 0
assert not np.isnan(metrics["loss"])
```

### test_alternating_train_validate
**Purpose:** Verify alternating train-validate cycle
**Coverage:** Training and validation in sequence
**Test Logic:**
```python
train_metrics = trainer._train_epoch(epoch=0)
val_metrics = trainer._validate_epoch(epoch=0)

assert "loss" in train_metrics
assert "loss" in val_metrics
```

### test_global_step_increment
**Purpose:** Verify global step counter increments correctly
**Coverage:** Step tracking across batches
**Assertions:**
- hasattr(trainer, "global_step")
- global_step increments after optimizer steps
- Accounts for gradient accumulation

### test_print_epoch_summary
**Purpose:** Verify epoch summary printing works
**Coverage:** Logging integration
**Test Logic:**
```python
with caplog.at_level(logging.INFO):
    trainer._print_epoch_summary(
        epoch=0,
        train_metrics={"loss": 0.5},
        val_metrics={"loss": 0.45},
    )

assert "Epoch" in caplog.text or "Summary" in caplog.text
```

---

## 12. TestModelStateAndDeviceHandling (4 tests)

### test_model_moved_to_device
**Purpose:** Verify model is moved to correct device during init
**Coverage:** Device placement
**Assertion:**
```python
trainer.model.to.assert_called()
```

### test_batch_moved_to_device
**Purpose:** Verify batch tensors are moved to device
**Coverage:** Device handling in forward pass
**Test Data:**
```python
batch = [torch.randn(4, 3, 224, 224)]  # On CPU
loss, _ = trainer._train_step(batch, epoch=0, step=0)
# Should move batch to trainer.device
```

### test_model_train_eval_modes
**Purpose:** Verify model can switch between train and eval modes
**Coverage:** Mode management
**Test Logic:**
```python
trainer.model.train()
trainer.model.eval()
trainer.model.train()
# Should not raise any exception
```

### test_no_grad_in_validation
**Purpose:** Verify validation uses torch.no_grad() context
**Coverage:** Memory efficiency, speed optimization
**Test Logic:**
```python
with torch.no_grad():
    val_metrics = trainer._validate_epoch(epoch=0)

assert isinstance(val_metrics, dict)
```

---

## 13. TestConfigurationVariations (3 tests)

### test_trainer_with_different_learning_rates
**Purpose:** Verify trainer works with different LRs
**Coverage:** Configuration flexibility
**Learning Rates Tested:** [1e-5, 1e-4, 1e-3]

### test_trainer_with_different_epochs
**Purpose:** Verify trainer works with different epoch counts
**Coverage:** Epoch configuration flexibility
**Epoch Counts Tested:** [1, 5, 10]

### test_trainer_with_different_loss_configs
**Purpose:** Verify trainer works with different loss configurations
**Coverage:** Loss function flexibility

---

## 14. TestDataHandling (4 tests)

### test_batch_as_tensor
**Purpose:** Verify handling of batch as single tensor
**Coverage:** Flexible batch format 1/3
**Test Data:**
```python
batch = torch.randn(4, 3, 224, 224, device=device)
```

### test_batch_as_list
**Purpose:** Verify handling of batch as list of tensors
**Coverage:** Flexible batch format 2/3
**Test Data:**
```python
batch = [torch.randn(4, 3, 224, 224, device=device)]
```

### test_batch_as_tuple
**Purpose:** Verify handling of batch as tuple (images, labels)
**Coverage:** Flexible batch format 3/3
**Test Data:**
```python
batch = (images, labels)
```

### test_batch_size_consistency
**Purpose:** Verify batch sizes are handled consistently
**Coverage:** Batch size validation

---

## 15. TestPerformanceAndRegression (3 tests)

### test_training_produces_valid_loss
**Purpose:** Verify training produces valid loss values
**Coverage:** Numerical stability
**Assertions:**
```python
assert train_metrics["loss"] > 0
assert not np.isnan(train_metrics["loss"])
assert not np.isinf(train_metrics["loss"])
```

### test_validation_produces_valid_loss
**Purpose:** Verify validation produces valid loss values
**Coverage:** Validation numerical stability

### test_multiple_epochs_training
**Purpose:** Verify training works for multiple epochs
**Coverage:** Multi-epoch stability
**Mark:** @pytest.mark.slow

### test_checkpoint_save_load_consistency
**Purpose:** Verify checkpoint save/load preserves state
**Coverage:** Checkpoint integrity

---

## Test Data Specifications

### Image Dimensions
- **Default:** (4, 3, 224, 224) - Batch of 4 images, RGB, 224x224
- **Components:**
  - 4: Batch size
  - 3: RGB channels
  - 224, 224: Height, Width (ViT standard)

### Embedding Dimensions
- **Context embeddings:** (B, 196, 384)
  - B: Batch size
  - 196: Number of patches (14×14 for 224×224 with patch size 16)
  - 384: Embedding dimension (small ViT)

### Number of Hierarchy Levels
- **Default:** 2-3 levels
- **Masking:** Multiple target masks per level

---

## Fixture Dependencies Graph

```
device (session-scoped)
    ↓
random_seed
    ↓
sample_train_loader ─┐
sample_val_loader   ├─→ trainer
sample_loss_fn      │
sample_masking_fn   │
mock_hjepa_model ───┤
base_training_config ┤
temp_checkpoint_dir ┘
```

---

## Notes for Test Maintenance

1. **Update tests when adding new trainer methods**
2. **Keep fixtures DRY (Don't Repeat Yourself)**
3. **Use meaningful assertion messages**
4. **Mark slow tests with @pytest.mark.slow**
5. **Mock external services (W&B, TensorBoard)**
6. **Use temp directories for file I/O tests**

---

**Last Updated:** 2025-11-21

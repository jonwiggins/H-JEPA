# Training Script Architecture

## Component Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Script                           │
│                     (scripts/train.py)                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     1. Initialization Phase                      │
├─────────────────────────────────────────────────────────────────┤
│ • Parse command-line arguments                                  │
│ • Setup logging system                                          │
│ • Load YAML configuration                                       │
│ • Apply CLI overrides                                           │
│ • Validate configuration                                        │
│ • Setup distributed training (optional)                         │
│ • Configure device (GPU/CPU)                                    │
│ • Set random seeds                                              │
│ • Create output directories                                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     2. Data Loading Phase                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐      ┌────────────────┐                   │
│  │ build_dataset  │──────▶│  JEPATransform │                   │
│  │  (training)    │      │  (augmentation)│                   │
│  └────────────────┘      └────────────────┘                   │
│          │                                                      │
│          ▼                                                      │
│  ┌────────────────┐      ┌────────────────┐                   │
│  │ build_dataset  │──────▶│JEPAEvalTransform│                  │
│  │ (validation)   │      │  (no aug)      │                   │
│  └────────────────┘      └────────────────┘                   │
│          │                                                      │
│          ▼                                                      │
│  ┌────────────────┐      ┌────────────────┐                   │
│  │build_dataloader│──────▶│DistributedSampler                 │
│  │ (both sets)    │      │  (optional)    │                   │
│  └────────────────┘      └────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     3. Model Creation Phase                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  create_hjepa_from_config                                       │
│          │                                                      │
│          ├──────▶ Context Encoder (ViT)                        │
│          │       • Processes visible patches                    │
│          │       • No EMA                                       │
│          │                                                      │
│          ├──────▶ Target Encoder (ViT)                         │
│          │       • Processes full image                         │
│          │       • With EMA                                     │
│          │                                                      │
│          ├──────▶ Predictor (Transformer)                      │
│          │       • Predicts target from context                │
│          │       • Hierarchical predictions                     │
│          │                                                      │
│          └──────▶ Hierarchy Projections                        │
│                  • Per-level projection heads                   │
│                  • 3 levels by default                          │
│                                                                  │
│  If distributed: wrap with DistributedDataParallel             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   4. Training Setup Phase                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────┐                   │
│  │ HierarchicalMaskGenerator               │                   │
│  │ • Creates context and target masks      │                   │
│  │ • Multi-level masking                   │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                  │
│  ┌─────────────────────────────────────────┐                   │
│  │ create_loss_from_config                 │                   │
│  │ • CombinedLoss (JEPA + VICReg)         │                   │
│  │ • Hierarchical weights                  │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                  │
│  ┌─────────────────────────────────────────┐                   │
│  │ create_optimizer                        │                   │
│  │ • AdamW optimizer                       │                   │
│  │ • Layer-wise weight decay               │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                  │
│  ┌─────────────────────────────────────────┐                   │
│  │ Learning Rate Schedulers                │                   │
│  │ • Cosine schedule with warmup           │                   │
│  │ • EMA momentum scheduler                │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     5. Training Loop Phase                       │
│                      (HJEPATrainer.train())                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each epoch:                                                │
│    ┌──────────────────────────────────────────────┐            │
│    │ Training Step                                │            │
│    │  1. Load batch                               │            │
│    │  2. Generate masks (context + targets)       │            │
│    │  3. Forward pass:                            │            │
│    │     • Context encoder on visible patches     │            │
│    │     • Target encoder on full image (no grad) │            │
│    │     • Predictor on context                   │            │
│    │  4. Compute hierarchical loss                │            │
│    │  5. Backward pass                            │            │
│    │  6. Gradient clipping                        │            │
│    │  7. Optimizer step                           │            │
│    │  8. Update EMA for target encoder            │            │
│    │  9. Update learning rate                     │            │
│    │ 10. Log metrics                              │            │
│    └──────────────────────────────────────────────┘            │
│                                                                  │
│    Every eval_frequency epochs:                                │
│    ┌──────────────────────────────────────────────┐            │
│    │ Validation Step                              │            │
│    │  • Run on validation set                     │            │
│    │  • Compute validation loss                   │            │
│    │  • Check for representation collapse         │            │
│    │  • Log validation metrics                    │            │
│    └──────────────────────────────────────────────┘            │
│                                                                  │
│    Every save_frequency epochs:                                │
│    ┌──────────────────────────────────────────────┐            │
│    │ Checkpoint Saving                            │            │
│    │  • Save model state                          │            │
│    │  • Save optimizer state                      │            │
│    │  • Save scheduler state                      │            │
│    │  • Save training metadata                    │            │
│    │  • Save configuration                        │            │
│    └──────────────────────────────────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      6. Logging & Monitoring                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌──────────────┐ │
│  │  TensorBoard   │    │  Weights&Biases│    │Console Logging│ │
│  ├────────────────┤    ├────────────────┤    ├──────────────┤ │
│  │• Loss curves   │    │• Experiment    │    │• Progress    │ │
│  │• LR schedule   │    │  tracking      │    │  bars        │ │
│  │• Gradients     │    │• Hyperparams   │    │• Step info   │ │
│  │• Collapse      │    │• Comparisons   │    │• Errors      │ │
│  │  metrics       │    │• Model saving  │    │• Warnings    │ │
│  └────────────────┘    └────────────────┘    └──────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       7. Completion Phase                        │
├─────────────────────────────────────────────────────────────────┤
│ • Save final checkpoint                                         │
│ • Log final metrics                                             │
│ • Clean up distributed processes                                │
│ • Print training summary                                        │
│ • Provide next steps guidance                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Input Image (224x224)
        │
        ▼
┌─────────────────┐
│  Augmentation   │
│  • RandomCrop   │
│  • ColorJitter  │
│  • Flip         │
└─────────────────┘
        │
        ├────────────────────────────┐
        ▼                            ▼
┌───────────────┐          ┌────────────────┐
│Context Encoder│          │Target Encoder  │
│ (trainable)   │          │ (EMA, no grad) │
└───────────────┘          └────────────────┘
        │                            │
        │ Visible patches            │ All patches
        │ (context)                  │ (target)
        ▼                            ▼
┌───────────────┐          ┌────────────────┐
│ [B, N_ctx, D] │          │  [B, N, D]     │
└───────────────┘          └────────────────┘
        │                            │
        ▼                            │
┌───────────────┐                   │
│   Predictor   │                   │
│ + Mask Tokens │                   │
└───────────────┘                   │
        │                            │
        │ Predictions                │ Targets
        │ [B, N_tgt, D]              │ [B, N_tgt, D]
        │                            │
        ├────────────────────────────┤
        ▼                            ▼
┌─────────────────────────────────────┐
│    Hierarchical Projections         │
│    Level 0, 1, 2, ...               │
└─────────────────────────────────────┘
        │                            │
        ├────────────────────────────┤
        ▼                            ▼
┌─────────────────────────────────────┐
│         Loss Computation            │
│  • Per-level losses                 │
│  • Weighted combination             │
│  • Optional VICReg                  │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────┐
│   Backprop      │
│ • Clip gradients│
│ • Update weights│
│ • Update EMA    │
└─────────────────┘
```

## Key Components

### 1. Configuration System
- YAML-based configuration
- Command-line overrides
- Validation and error checking
- Flexible and extensible

### 2. Data Pipeline
- Multiple dataset support
- Automatic augmentation
- Distributed sampling
- Efficient loading

### 3. Model Architecture
- Context encoder (ViT)
- Target encoder (ViT + EMA)
- Predictor (Transformer)
- Hierarchical projections

### 4. Training Loop
- Forward/backward pass
- Gradient management
- EMA updates
- Learning rate scheduling

### 5. Loss Computation
- Hierarchical JEPA loss
- Optional VICReg regularization
- Weighted combination
- Collapse prevention

### 6. Checkpointing
- Periodic saving
- Best model tracking
- Resume capability
- Full state preservation

### 7. Logging
- Multi-backend support
- Real-time metrics
- Progress tracking
- Error handling

## Error Handling

```
┌─────────────────────────────────────┐
│    Input Validation                 │
│    • Config file exists             │
│    • Required sections present      │
│    • Parameter ranges valid         │
│    • Cross-param consistency        │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│    Resource Checking                │
│    • CUDA availability              │
│    • GPU memory                     │
│    • Disk space                     │
│    • Dataset accessibility          │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│    Runtime Monitoring               │
│    • OOM detection                  │
│    • NaN/Inf checking               │
│    • Gradient explosion             │
│    • Representation collapse        │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│    Graceful Handling                │
│    • Try-except blocks              │
│    • Informative messages           │
│    • Cleanup on failure             │
│    • State preservation             │
└─────────────────────────────────────┘
```

## Multi-GPU Support

```
Single GPU:
    Model ──▶ Device 0 ──▶ Training

Multi-GPU (DDP):

    Process 0:                Process 1:
    Model Replica 0           Model Replica 1
    │                         │
    ├──▶ GPU 0               ├──▶ GPU 1
    │   • Batch 0-31         │   • Batch 32-63
    │   • Forward            │   • Forward
    │   • Backward           │   • Backward
    │   • Compute grads      │   • Compute grads
    │                         │
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  All-Reduce     │
    │  (sync grads)   │
    └─────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
    Update Model 0    Update Model 1
    (synchronized)    (synchronized)
```

## File I/O

```
Input:
    configs/default.yaml ──▶ Configuration
    /path/to/dataset ──▶ Training data
    checkpoint.pth ──▶ Resume state (optional)

Output:
    results/
    ├── checkpoints/
    │   ├── checkpoint_epoch_*.pth
    │   ├── checkpoint_latest.pth
    │   └── checkpoint_best.pth
    └── logs/
        ├── tensorboard/
        │   └── events.out.tfevents.*
        └── train.log
```

## Integration with Components

```
scripts/train.py
    │
    ├──▶ src/models/
    │    ├── encoder.py (ViT encoders)
    │    ├── predictor.py (Transformer)
    │    └── hjepa.py (Full model)
    │
    ├──▶ src/losses/
    │    ├── hjepa_loss.py (Prediction loss)
    │    ├── vicreg.py (Regularization)
    │    └── combined.py (Full loss)
    │
    ├──▶ src/masks/
    │    ├── multi_block.py (Block masking)
    │    └── hierarchical.py (Multi-level)
    │
    ├──▶ src/data/
    │    ├── datasets.py (Dataset classes)
    │    └── download.py (Auto-download)
    │
    ├──▶ src/trainers/
    │    └── trainer.py (Training loop)
    │
    └──▶ src/utils/
         ├── scheduler.py (LR, EMA)
         ├── checkpoint.py (Save/load)
         └── logging.py (Metrics)
```

This architecture ensures:
- Modularity and reusability
- Clear separation of concerns
- Easy debugging and testing
- Extensibility for new features
- Production-ready reliability

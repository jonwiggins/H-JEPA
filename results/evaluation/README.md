# H-JEPA Evaluation Results

This directory contains evaluation results, visualizations, and analysis for H-JEPA models.

## Directory Structure

```
results/evaluation/
├── README.md                          # This file
├── mock_results/                      # Example/template results
│   ├── evaluation_results_example.json    # Example JSON output
│   └── evaluation_summary_example.md      # Example summary report
│
├── <experiment_name>/                 # Actual evaluation results (created during eval)
│   ├── evaluation_results.json        # Complete evaluation metrics
│   ├── evaluation_summary.txt         # Quick text summary
│   ├── evaluation_report.md           # Detailed markdown report
│   └── visualizations/                # Generated plots
│       ├── evaluation_dashboard.png       # Main summary dashboard
│       ├── hierarchy_comparison.png       # Cross-level comparison
│       ├── knn_k_sweep.png               # k-NN hyperparameter sweep
│       ├── per_class_accuracy.png        # Per-class performance
│       ├── feature_quality_summary.png   # Feature quality metrics
│       ├── confusion_matrix_level0.png   # Confusion matrices
│       ├── tsne_level0.png               # t-SNE visualizations
│       └── singular_values_level0.png    # Rank analysis
│
└── comparison/                        # Cross-experiment comparisons (optional)
    ├── experiments_comparison.json
    └── comparison_plots.png
```

## Quick Start

### 1. Run Evaluation

#### Quick k-NN Evaluation (Fastest - 15 min)
```bash
python scripts/quick_eval.py \
    --checkpoint results/checkpoints/best_model.pth \
    --method knn
```

#### Standard Evaluation (2 hours)
```bash
python scripts/evaluate.py \
    --checkpoint results/checkpoints/best_model.pth \
    --dataset cifar10 \
    --eval-type linear_probe knn \
    --hierarchy-levels 0 1 \
    --output-dir results/evaluation/standard
```

#### Comprehensive Evaluation (4 hours)
```bash
python scripts/evaluate.py \
    --checkpoint results/checkpoints/best_model.pth \
    --dataset cifar10 \
    --eval-type all \
    --hierarchy-levels 0 1 \
    --save-visualizations \
    --output-dir results/evaluation/comprehensive
```

### 2. Generate Visualizations

```bash
python scripts/generate_eval_visualizations.py \
    --results results/evaluation/comprehensive/evaluation_results.json \
    --output-dir results/evaluation/comprehensive/visualizations
```

### 3. View Results

**JSON Results:**
```bash
cat results/evaluation/comprehensive/evaluation_results.json | jq '.summary'
```

**Markdown Report:**
```bash
less results/evaluation/comprehensive/evaluation_report.md
```

**Visualizations:**
Open `results/evaluation/comprehensive/visualizations/evaluation_dashboard.png` in image viewer.

## Evaluation Protocols

### 1. Linear Probe
- **Purpose:** Standard SSL evaluation metric
- **Method:** Train linear classifier on frozen features
- **Time:** ~45 min per hierarchy level
- **Target:** >70% on CIFAR-10

### 2. k-NN Classification
- **Purpose:** Training-free evaluation
- **Method:** k-nearest neighbors on frozen features
- **Time:** ~15 min (all levels)
- **Target:** >68% on CIFAR-10

### 3. Feature Quality
- **Purpose:** Assess representation health
- **Metrics:** Effective rank, isotropy, variance
- **Time:** ~10 min
- **Red Flags:** Rank <10%, uniformity >-1.0

### 4. Fine-tuning
- **Purpose:** Transfer learning capability
- **Method:** Fine-tune encoder or train head only
- **Time:** ~1.5 hours
- **Target:** Significant improvement over linear probe

### 5. Few-shot Learning
- **Purpose:** Low-data regime performance
- **Method:** N-way K-shot episode evaluation
- **Time:** ~30 min
- **Expected:** ~40% (1-shot), ~60% (5-shot), ~70% (10-shot)

## Understanding Results

### Performance Tiers

| Tier | Linear Probe | Interpretation | Action |
|------|--------------|----------------|--------|
| **Excellent** | >80% | Strong features | Ready for deployment |
| **Good** | 70-80% | Solid features | Good for fine-tuning |
| **Moderate** | 60-70% | Basic features | Needs improvement |
| **Poor** | <60% | Weak features | Debug training |

### Collapse Indicators

**Healthy representations:**
- ✅ Effective rank >40% of dimensions
- ✅ Uniformity <-2.0
- ✅ Mean variance >0.3

**Warning signs:**
- ⚠️ Effective rank 20-40% (mild collapse)
- ⚠️ Uniformity -2.0 to -1.0 (moderate similarity)
- ⚠️ Mean variance 0.1-0.3 (low variance)

**Critical issues:**
- ❌ Effective rank <20% (severe collapse)
- ❌ Uniformity >-1.0 (mode collapse)
- ❌ Mean variance <0.1 (variance collapse)

### Hierarchy Level Selection

**Level 0 (Finest):**
- Captures: Textures, local patterns, edges
- Best for: Fine-grained classification, texture recognition
- Typically: Highest accuracy

**Level 1 (Mid):**
- Captures: Object parts, mid-level structures
- Best for: Part-based recognition, structural tasks
- Typically: -2% to -5% vs Level 0

**Level 2 (Coarsest):**
- Captures: Global semantics, context
- Best for: Scene understanding, high-level semantics
- Typically: -5% to -10% vs Level 0

## Expected Performance

### CIFAR-10 Benchmarks

**H-JEPA (ViT-Tiny, 20 epochs):**
- Linear Probe: 70-78% (target: 75%)
- k-NN: 68-76% (target: 73%)
- Effective Rank: 85-100 / 192 dims

**Published SSL Methods (larger models, more epochs):**
- I-JEPA (ViT-Base, 300ep): ~89%
- SimCLR (ResNet-50, 1000ep): ~91%
- MoCo v2 (ResNet-50, 800ep): ~91%

**Baselines:**
- Random features: ~30%
- Supervised (ViT-Tiny): ~95%

## Troubleshooting

### Low Performance (<60%)

**Check:**
1. Training loss converged?
2. EMA momentum schedule completed?
3. Data augmentation appropriate?
4. Any NaN/Inf in training?

**Solutions:**
- Review training logs
- Check for collapse indicators
- Verify data pipeline
- Increase training epochs

### Collapse Detected

**Symptoms:**
- Very low effective rank (<20)
- High uniformity (>-1.0)
- Low variance (<0.1)

**Solutions:**
1. Check loss function weights
2. Verify EMA is working
3. Review augmentation strength
4. Check for gradient issues
5. Try different learning rate

### Slow Evaluation

**Speed up:**
- Use `quick_eval.py` for fast k-NN only
- Reduce linear probe epochs: `--linear-probe-epochs 50`
- Evaluate single level: `--hierarchy-levels 0`
- Increase batch size: `--batch-size 512`
- Skip fine-tuning and few-shot for quick check

## File Formats

### evaluation_results.json

Complete evaluation results in JSON format. Contains:
- `metadata`: Checkpoint info, dataset, config
- `level_<N>`: Results for each hierarchy level
  - `linear_probe`: Accuracy, per-class metrics, confusion matrix
  - `knn`: Accuracy, k-sweep, distance statistics
  - `feature_quality`: Rank, isotropy, variance, collapse
  - `fine_tune`: Frozen and full fine-tuning results
  - `few_shot`: N-way K-shot results
- `hierarchy_comparison`: Cross-level comparison
- `summary`: Overall status, recommendations, next steps

### evaluation_report.md

Human-readable markdown report with:
- Executive summary
- Detailed metrics
- Per-class analysis
- Feature quality assessment
- Comparison to baselines
- Recommendations
- Next steps

## Citation

If you use these evaluation protocols in your research:

```bibtex
@misc{hjepa_eval,
  title={H-JEPA Evaluation Framework},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/H-JEPA}}
}
```

## Contributing

To add new evaluation protocols:

1. Add evaluation function to `src/evaluation/`
2. Update `scripts/evaluate.py` to include new protocol
3. Add visualization to `scripts/generate_eval_visualizations.py`
4. Update this README with usage instructions

## License

See LICENSE file in repository root.

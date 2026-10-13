# Bounded H-JEPA correctness validation — 2026-10-12

This investigation reproduced and fixed core correctness failures that the
existing tests missed. Small real CIFAR-10 runs now have finite losses and
gradients, nonzero optimizer updates, correct sampled-context isolation, exact
EMA arithmetic, and reproducible epoch-boundary continuation. These observations
are **not evidence of convergence, competitive accuracy, or universal correctness**.

## Revisions and environment

- Baseline: `eb995a02678dfa5bcc808e782af035121962a5d4`, also the fetched `origin/main`.
- Original validated implementation: `1bc7e40425856bfc92725d9e1ec73c19666c28db`.
  CI follow-through implementation: `0d1e058eaa4b4cf81be0927ab22273cb5381a46c`.
  See the follow-through section for its repeat diagnostics and dependency versions.
- Task branch: `optio/task-8e636231-a839-4b71-b129-f99c8cf411d7`.
- Apple M1 Max, 64 GiB unified memory, macOS 27.2 arm64, Python 3.12.12.
  PyTorch 2.14.1, torchvision 0.29.1, timm 1.0.30, NumPy 2.5.4.
  MPS detected and exercised successfully, float32, AMP disabled; 2 CPU threads.
  Full package versions used for diagnostics are in `validation/environment.json`.
- Dependencies installed only into this checkout's `.venv` using `uv pip install
  -e '.[dev]' scikit-learn scipy isort pyarrow`; Black 24.8.0 matches CI.
  No cloud compute, external experiment logging, or changes to other checkouts.

## Intended regimes and reference fidelity

| Regime | Implemented objective and target behavior | Validation scope |
|---|---|---|
| Classic image SSL | Hierarchy-weighted prediction against a frozen EMA backbone; shared FPN/projectors; variance/covariance regularization on context features | Real CIFAR-10 MPS training, regressions, CLI, resume, frozen-feature k-NN output |
| EMA + SIGReg | Prediction plus configured SIGReg weight; frozen EMA backbone | Real CIFAR-10, characteristic-function test, BatchNorm projectors, resume |
| End-to-end image SSL | Shared encoder, `detach_targets: false`, prediction plus SIGReg, no EMA teacher | Real CIFAR-10, both-branch gradient tests, resume |
| Action-conditioned world model | Frame encoder/projector, causal AdaLN-Zero predictor, `predictions[:, :-1]` against `embeddings[:, 1:]`, SIGReg and CEM | Code review and synthetic/analytic unit tests only; **no real-trajectory training** |

[I-JEPA's reference implementation](https://github.com/facebookresearch/ijepa/blob/main/src/train.py)
initializes its teacher from the student, predicts hidden target features, and
updates the teacher after optimizer updates. H-JEPA adds hierarchy/FPN and
regularization. It uses dense, zeroed hidden patch embeddings rather than token
removal, pools the **ordered selected-token sequence in 1D** with `2**level`
kernels, and unions target blocks. It is not a faithful spatial-pyramid or
multi-block reproduction of [I-JEPA](https://arxiv.org/abs/2301.08243).
Only the backbone has an EMA copy; shared projection/FPN heads remain extensions.

[VICReg](https://arxiv.org/abs/2105.04906) matches corresponding views and controls
variance and off-diagonal covariance. Pairing independent halves of an image batch
is not a valid invariance pair. The image trainer now uses only variance/covariance
on its single context view. Covariance is normalized by feature dimension, and
single-observation statistics remain finite.

[LeJEPA](https://arxiv.org/abs/2511.08544) combines prediction with an isotropic
Gaussian regularizer; it does not imply removing prediction loss. Both local
Gaussian tests previously standardized away the mean and scale being tested.
They now retain those signals. The default finite-quantile kernel test is a
**project approximation**, quadratic in sample count; the characteristic-function
quadrature is the closer reference-inspired variant used in real runs. Its finite
quadrature/weighting and overall statistic scaling are not asserted to reproduce
paper hyperparameters or results. [LeWorldModel](https://arxiv.org/abs/2603.19312)
inspires the action-conditioned stack; `scripts/train.py` remains an image SSL
trainer and does not train the world-model config.

## Initial hypotheses and reproduced failures

The priority was invariants over a falling loss: identical initial teacher/student,
frozen and deterministic target backbone, no dependence on hidden pixels, correct
target indices, batch-padding independence, valid-token-only loss, anti-collapse
location/scale sensitivity, optimizer-step-based EMA/LR, and equivalent checkpoint
continuation. Architecture/config wiring and temporal prediction were reviewed too.

The baseline relevant suite passed **184 tests, 5 skipped**, despite seven failures
in eight new behavioral cases. Several existing “compliance” tests print findings
or return booleans without asserting them; their passing status alone is weak evidence.

| Iteration | Finding and fix | Observed result |
|---|---|---|
| Baseline tests | New tests exposed reinitialized teacher, learned FPN masks, padded-query interference, masked-loss dimension scaling, both SIGReg variants losing scale/location, and unrelated-image VICReg matching | 7 failed, 1 passed; masked-pixel isolation already passed |
| Baseline real run | 32 real train / 16 test images; MPS; 8 microbatches per epoch | Training losses finite (first epoch 3.8415 to 3.3318); checkpoint resume crashed on a NumPy scalar under PyTorch weights-only loading |
| First fixes | Preserve encoder initialization; teacher eval/freeze; predictor padding mask; unpadded per-sample pyramids and deterministic validity masks; correct element mean; repair regularizers | Original 8 regressions passed |
| Trainer/state fixes | Sample-weighted accumulation, flush remainder, LR before update, EMA once after successful optimizer update, restore steps/RNG/loss state and keep checkpoint scalar metadata safe | First real resume matched model, optimizer, losses, LR and step count exactly |
| Second review | Honor sampled context and flat FPN/RoPE/LayerScale flags; encoder checkpointing; SIGReg configured weight/cache state; scheduler endpoint; centered collapse diagnostics | Focused behavioral tests added; real-data variants retested |
| Broader review | Removed Matplotlib `cm.get_cmap` API broke six mask visualization tests; CEM one-elite unbiased standard deviation produced NaNs | Compatible colormap lookup; population standard deviation; analytic rollout and finite-action test |
| Retest | Update mock loss to honor `nn.Module` contract; correct an invalid tiny-encoder dimension fixture; expect explicit empty-mask validation | 438 passed / 6 skipped in broad focused suite; then 207 passed / 5 skipped in core suite |
| Final revision | Two consecutive independent fresh classic diagnostics, each including checkpoint continuation; both SIGReg regimes and CLI | Measurements below; model/optimizer/loss resume deltas zero |

Focused regressions include hidden-pixel perturbation, target-index gather oracle,
independence from another sample's padding, all-valid FPN masks, one-target-token
hierarchies, all-true masked/unmasked loss equivalence, wrong-Gaussian-scale and
location penalties, batch-order invariance, exact analytic accumulation including
short batches, LR timing, post-update EMA, RNG continuation, and a constant-vector
collapse counterexample. Temporal causality is tested **after opening AdaLN gates**,
so zero initialization cannot make that test vacuous. CEM rollout is compared with
known additive dynamics. Synthetic fixtures are labeled in the test module.

## Bounded data and reproducible commands

Seed 42 initializes Python, NumPy, CPU/MPS RNGs. The run uses ViT-Tiny/16 (192
channels, its standard 12 backbone blocks), a one-block/three-head predictor,
32-channel FPN, two hierarchy levels, 64x64 images (16 patches), batch 4,
accumulation 2, AdamW LR `1e-4`, weight decay `.04`, clip norm 1, two epochs and
`max_steps: 8` optimizer updates. Each resume diagnostic runs epoch 1 once and
epoch 2 twice, once uninterrupted and once restored, to compare complete states.
Sixteen random SIGReg directions and 32 quadrature points keep that path bounded.
The two classic runs are separate new processes; neither consumes the other's
checkpoint. Config: `configs/validation_bounded.yaml`.

The Toronto torchvision archive download was approximately 55 kB/s and was
stopped. A public CIFAR-10 mirror succeeded. Only the first 32 train and first 16
test rows enter training/evaluation, not synthetic images. Full Parquet archives
are downloaded for reproducibility, but only the requested first batch of rows is
materialized. Source revision: `uoft-cs/cifar10@0b2714987fa478483af9968de7c934580d0bb9a2`.
SHA-256:

- Train: `8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883`
- Test: `841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de`

Run from the dedicated checkout; retain the budget ledger between attempts:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]' scikit-learn scipy pyarrow isort black==24.8.0
mkdir -p data results/validation
curl -L --fail --max-time 90 -o data/cifar10-train.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text/train-00000-of-00001.parquet
curl -L --fail --max-time 90 -o data/cifar10-test.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text/test-00000-of-00001.parquet
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name classic_a -- .venv/bin/python scripts/validate_small_run.py --name classic_a --resume-check
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name classic_b -- .venv/bin/python scripts/validate_small_run.py --name classic_b --resume-check
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name sigreg -- .venv/bin/python scripts/validate_small_run.py --name sigreg --regime sigreg --resume-check
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name e2e -- .venv/bin/python scripts/validate_small_run.py --name e2e --regime end_to_end --resume-check
```

The normal CLI was also exercised on the same real images exported losslessly to
PNG. `dataset: imagenet` in the generated config selects the existing ImageFolder
storage loader; **the data are CIFAR-10, not ImageNet**. This checks CLI configuration,
subset handling, training, validation and checkpoints, without waiting for the
Toronto archive. It does not validate successful archive download/checksumming.

```sh
.venv/bin/python scripts/validate_small_run.py --name prepare --prepare-cli
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name cli -- .venv/bin/python scripts/train.py --config results/validation/cli.yaml --device mps
cp results/validation/cli/checkpoints/checkpoint_epoch_0001.pth results/validation/cli/checkpoints/expected_epoch_0001.pth
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name cli_resume -- .venv/bin/python scripts/train.py --config results/validation/cli.yaml --device mps --resume results/validation/cli/checkpoints/checkpoint_epoch_0000.pth
```

## Measured final outcomes

All rows below use implementation `1bc7e40`. Loss values are individual
microbatch objectives, not claims of a monotonic trajectory. Validation is the
mean of four batches after epoch 1. Gradient ranges include both uninterrupted
epochs, before clipping. Clipped norms are approximately 1.0 (float32 rounding).

| Run | First → last epoch-1 loss | Last epoch-2 loss | Validation loss | Raw gradient norm range | Pooled-image std / effective rank |
|---|---:|---:|---:|---:|---:|
| Classic A | 3.284759 → 1.445586 | 1.442349 | 1.797784 | 2.6853–118.5948 | .344399 / 2.662456 |
| Classic B, fresh process | 3.284759 → 1.445586 | 1.442349 | 1.797784 | 2.6853–118.5948 | .344399 / 2.662456 |
| EMA + SIGReg | 4.198111 → 3.245729 | 4.285867 | 3.920182 | 37.5747–152.9819 | .340924 / 2.606551 |
| End-to-end + SIGReg | 4.198111 → 3.245465 | 4.285448 | 3.919696 | 37.5720–152.9823 | .340958 / 2.606532 |

All four runs have finite losses and checked finite gradients at every optimizer
update. The first context parameter changed by about `3.5e-4` after epoch 1.
Teacher parameters had no gradients and remained frozen; observed post-update
EMA arithmetic error on the monitored parameter was **0** at every update.
All classic first-epoch diagnostics match between the two fresh processes.

Every diagnostic resume reported **0 maximum difference** in model state,
optimizer tensor state and continuation losses. Both paths finish at optimizer
step 8 and LR `4.767963140691306e-6`. This includes Python/NumPy/torch/MPS RNG
continuation; the optimizer is AdamW. MPS uses no GradScaler, so CUDA AMP/scaler
replay is not empirically established. Config and optimizer-step count reproduce
the pure LR/EMA schedule; changing schedule settings on resume is not an
exact-continuation experiment.

The full CLI replay separately matched model/optimizer tensors exactly, retained
step 8 and identical scheduler values, and reproduced final validation loss
`1.4506632685661316`. See `validation/cli_comparison.json`.

Hierarchy outputs on a sampled batch were `[4,11,192]` and `[4,6,192]`; valid-token
counts were `[8,8,11,8]` and `[4,4,6,4]`. The extra entries are excluded from loss,
attention keys and hierarchy pooling. Shape/count checks supplement the direct
index oracle and padding-invariance tests.

Representation statistics use the mean patch embedding for **four held-out real
images**, then center across images. Centered matrix rank was 3, the maximum for
four samples. This detects trivial identity-of-all-images collapse in that batch;
it does not establish learned semantics. Token-only variance could conceal image
collapse, hence the separate pooled-image diagnostic. The frozen-feature k-NN
path returned 25% for classic/EMA+SIGReg and 12.5% for end-to-end on only 16 test
images using 32 training images and k=3. These are output sanity checks, **not
benchmarks**; sampling noise and random initialization dominate.

The cold first baseline batch took 6.72 s. The warmed final classic-A forward
median was .0355 s, excluding backward/optimizer work. The complete diagnostic
process (startup, 12 total optimizer updates including replay, validation, loading,
feature diagnostics and k-NN) took 13.74 s. Explicit step limits ended the
three-minute-capped experiments early.

## Coverage and remaining limits

- Real data: both the Parquet diagnostic adapter with repository transforms and
  the CLI's ImageFolder loader worked. The standard torchvision CIFAR archive
  download did not finish; that download/integrity path is not claimed validated.
- CPU invariants and MPS image training were exercised. CUDA/AMP, multi-GPU/DDP,
  alternative timm backbones and large-memory configurations remain unvalidated.
  DDP is still described as future support by the trainer; it was not promoted to
  supported status in this change.
- The world model has no production sequential trainer in this repository.
  Temporal shifts, opened-gate latent causality and analytic CEM rollout were
  checked; real dynamics learning and control success were not. BatchNorm over
  frames means latent-input causality tests do not prove strictly causal encoder
  statistics during training.
- Multi-crop/contrastive and legacy per-level loss APIs have existing unit tests,
  but are not integrated real-data training validations here. The three image
  regimes above are the scope of the clean training passes.
- Replay guarantees tested here are **epoch-boundary, same config, workers=0**.
  Stateful workers, distributed samplers and changed data/schedules may differ.
  Mid-epoch max-step checkpoints reject resume explicitly rather than silently
  skipping unsaved sampler position. They remain loadable for inference.
- Accumulation averages microbatch objectives by sample count. BatchNorm and
  batch-statistical regularizers still operate per microbatch; accumulation does
  not turn their statistics into those of one larger batch.
- FPN pooling is 1D over selected tokens, not spatial 2D hierarchy. Shared heads,
  normalization, masking and loss scaling differ from cited papers. The fixes
  change objectives; historical hyperparameters/results must be reevaluated.
- No long-run anti-collapse guarantee, downstream quality claim or benchmark
  comparison follows from 32 training images and eight optimizer updates.


## Test/check outcomes and hard time budget

Final full suite excluding the network-downloading `tests/test_data.py`: **1426 passed, 13 skipped** (216.99 s pytest time). The three network-free transform classes from that file added **36 passed** (1.34 s). Thus 1462 tests passed across those two commands; the remaining archive-download integration tests were not run to completion.

```sh
.venv/bin/python scripts/bounded_validation.py --seconds 600 --name final_suite -- .venv/bin/python -m pytest -q -o addopts='' tests --ignore=tests/test_data.py
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name transforms -- .venv/bin/python -m pytest -q -o addopts='' tests/test_data.py::TestJEPATransforms tests/test_data.py::TestAugmentations tests/test_data.py::TestMultiCropTransforms
.venv/bin/black --check src scripts tests
.venv/bin/ruff check src tests scripts/train.py scripts/bounded_validation.py scripts/validate_small_run.py
.venv/bin/isort --check-only --profile black --line-length 100 src scripts tests
.venv/bin/mypy src
```

Black, Ruff, isort and `git diff --check` passed. The original local mypy result recorded 14 baseline errors and 11 errors in unchanged evaluation/visualization files after fixes, using NumPy 2.5.4. This was incomplete evidence: a later nonincremental check also found the trainer float/`detach` error. CI, using the requirements-file NumPy 1.26.4 constraint, found three trainer errors. The follow-through below reproduces and fixes those errors; mypy now passes all 50 files under CI dependencies. The original source comparison used a `git archive eb995a0` copy inside this checkout; it did not establish CI compatibility.

Every training/evaluation attempt ran under an isolated process group and a monotonic deadline, with a persistent locked 3300-second ledger. First real experiment cap: 180 seconds; all later caps at most 600 seconds. Tests, startup, checkpoint I/O and interrupted work are charged conservatively as well. A one-second timeout exercise killed both its sleeping parent and child; both PIDs were absent afterward. No training jobs were left running.

**Original investigation charge: 1066.41 seconds (17 min 46.41 s).** The cumulative follow-through total appears below. Measured real-data experiment process time, including startup/evaluation/replay, totals 141.04 seconds. Unit-test training/evaluation is additionally covered by the conservative all-process total. Downloads and static code review are not training.

One early supervisor cleanup hit macOS `PermissionError` while signaling a reaped process group. Its tests had finished, but the supervisor end timestamp was not recorded. We retained its entire 180-second reservation rather than invent elapsed time. The committed ledger records the failure and last test output time; the wrapper now preserves accounting on this exception. Other attempts have start/end timestamps and elapsed times.

| Attempt | Charged seconds | Outcome |
|---|---:|---|
| baseline_core | 60.71 | completed; exit 0 |
| baseline_regressions | 6.64 | completed; exit 1 |
| baseline_real | 43.59 | completed; exit 1 |
| fixed_regressions_1 | 5.15 | completed; exit 0 |
| fixed_core_1 | 180.00 | cleanup_error_full_reservation; exit 1 |
| fixed_real_1 | 18.45 | completed; exit 0 |
| fixed_core_2 | 35.45 | completed; exit 1 |
| final_suite_candidate | 99.85 | completed; exit -15 |
| suite_offline | 220.82 | completed; exit 1 |
| regressions_review | 71.48 | completed; exit 0 |
| final_core | 21.56 | completed; exit 0 |
| final_classic_resume | 13.74 | completed; exit 0 |
| final_classic_fresh | 14.08 | completed; exit 0 |
| final_sigreg | 18.57 | completed; exit 0 |
| final_end_to_end | 13.60 | completed; exit 0 |
| final_cli | 10.49 | completed; exit 0 |
| final_cli_resume | 8.52 | completed; exit 0 |
| final_suite | 219.67 | completed; exit 0 |
| final_data_transforms | 3.03 | completed; exit 0 |
| timeout_cleanup_check | 1.01 | timeout; exit 124 |

Full UTC start/end times, commands and per-attempt caps are in `validation/budget.json`. `validation/metrics.json` contains loss components, raw/clipped gradients, EMA errors, hierarchy validity, representation diagnostics and checkpoint comparison measurements. Large logs, datasets, source snapshots, checkpoint files and credentials are not committed.

To independently compare the CLI checkpoints after the commands above:

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
import torch
root = Path('results/validation/cli/checkpoints')
a = torch.load(root / 'expected_epoch_0001.pth', map_location='cpu')
b = torch.load(root / 'checkpoint_epoch_0001.pth', map_location='cpu')
for key, value in a['model_state_dict'].items():
    torch.testing.assert_close(value, b['model_state_dict'][key], rtol=0, atol=0)
for pid, state in a['optimizer_state_dict']['state'].items():
    for key, value in state.items():
        torch.testing.assert_close(value, b['optimizer_state_dict']['state'][pid][key], rtol=0, atol=0)
assert a['global_step'] == b['global_step'] == 8
assert a['scheduler_state_dict'] == b['scheduler_state_dict']
assert a['metrics'] == b['metrics']
print('Exact CLI checkpoint continuation verified')
PY
```

## CI follow-through (PR #6)

The first PR head, `b9da295f7fc8b268d44859957711d5228a6d4ca9`, failed
[Tests run 38301325713](https://github.com/jonwiggins/H-JEPA/actions/runs/38301325713)
and [Docker run 38301325630](https://github.com/jonwiggins/H-JEPA/actions/runs/38301325630).
The earlier local mypy paragraph above did **not** establish CI compatibility:
installing `requirements.txt` gives NumPy 1.26.4, not the original editable-only
installation's NumPy 2.5.4. A fresh, nonincremental check reproduced exactly CI's
three trainer errors with mypy 2.4.0, NumPy 1.26.4 and PyTorch 2.14.1.

- `_train_step` promised float metrics but returned tensors, and progress logging
  called `detach()` on a declared float. Metrics now become detached Python
  scalars at the training-step boundary, while the separate loss tensor retains
  its gradient. This also prevents epoch metric buffers retaining autograd graphs.
  A regression checks both the buffer values and nonzero backpropagation.
- NumPy 1.26's `get_state()` default overload resolves to a dictionary; its explicit
  legacy overload permits either tuple or dictionary. Capture now requests
  `legacy=True` and checks the tuple format before weights-only-safe serialization.
  This preserves the existing checkpoint schema and explicitly rejects an
  unsupported nonlegacy generator instead of mis-indexing its dictionary.
- Docker metadata's `{{branch}}-` SHA prefix becomes `-` on PR and tag refs, creating
  invalid Docker references. All three images now use the action's `sha-` default.
  Existing PR login/publish guards remain intact. The actual metadata-action v4
  bundle was exercised for PR #6, main push, v1.2.3 tag push and workflow_dispatch:
  all 12 cases generated valid tags; PRs had `pr-6` and `sha-1d97f43`, no `main` or
  `latest`. Push/manual events retain their publication gates. This is a metadata
  test, not evidence of a successful registry push or GPU execution.

Reproduction commands (all installation is inside the existing checkout venv):

```sh
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/mypy --no-incremental src/
git clone --depth 1 --branch v4 https://github.com/docker/metadata-action.git results/validation/metadata-action
.venv/bin/python scripts/validate_docker_metadata.py
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name ci_trainer_regressions_final -- .venv/bin/python -m pytest -q -o addopts='' tests/test_correctness_regressions.py tests/test_trainers.py
```

Mypy: **no issues in 50 source files**. Black 24.8.0 and Ruff pass.
The original budget ledger remains intact, including its 1066.4106376233976-second
charge; all follow-through test/training attempts append to that same ledger.
Measured smoke/resume results follow; remote CI is checked on each pushed head.

Follow-through regressions: **113 passed, 5 skipped**. Two consecutive fresh real
CIFAR-10 MPS processes at clean revision `0d1e058` each completed eight optimizer
updates plus checkpoint replay. Both had zero weight, optimizer-tensor and loss
replay differences; EMA error was zero and teacher gradients absent. First/last
first-epoch losses were 3.284759/1.445586; epoch-two final loss 1.442349; validation
loss 1.797784; raw gradient norm range 2.685261–118.594795; clipped norms ≤1.000001.
Context parameter update was 0.000348842, pooled feature std 0.344399, centered
rank 3/effective rank 2.662456 on four images; k-NN output 25% on the tiny split.
These match the original classic diagnostics, with the same limitations.

CI dependency matching also downgraded SciPy to 1.17.1 and contourpy to 1.3.3.
The optional Parquet harness needed PyArrow 19.0.1 because the formerly installed
26.0.0 required NumPy 2; that import-only failed attempt remains charged. No
project dependency constraint was relaxed. Metrics/environment and all 12 metadata
cases are in `validation/ci-followthrough.json` and `validation/docker-metadata.json`.
The metadata action source revision was `818d4b7b91585d195f67373fd9cb0332e31a7175`.

```sh
uv pip install --python .venv/bin/python pyarrow==19.0.1
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name ci_classic_resume_retry -- .venv/bin/python scripts/validate_small_run.py --name ci_classic_resume_retry --resume-check
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name ci_classic_fresh -- .venv/bin/python scripts/validate_small_run.py --name ci_classic_fresh --resume-check
```

| Follow-through attempt | Charged seconds | Outcome |
|---|---:|---|
| ci_trainer_regressions | 7.11 | exit 4 |
| ci_trainer_regressions_retry | 11.07 | exit 1 |
| ci_trainer_regressions_fixed | 11.04 | exit 1 |
| ci_trainer_regressions_final | 11.25 | exit 0 |
| ci_classic_resume | 0.24 | exit 1 |
| ci_classic_resume_retry | 43.82 | exit 0 |
| ci_classic_fresh | 12.39 | exit 0 |
| ci_dependency_suite | 218.59 | exit 0; 1427 passed, 13 skipped |
| container_cpu_resume | 64.28 | exit 0; candidate dependency environment |
| container_cpu_final_resume | 8.42 | exit 0; clean a81666a |
| container_cpu_final_fresh | 8.43 | exit 0; clean a81666a |

**Cumulative charge: 1463.04 seconds (24m 23.04s) / 3300 seconds.**
The two earlier regression failures exposed obsolete tensor-metric assertions;
the first invocation used a mistyped test filename. All attempts remain charged.
No training processes remain. Exact-head remote checks are pending this push.

The complete local network-free suite was repeated under CI dependencies:
**1,427 passed, 13 skipped**, 215.86 seconds pytest / 218.59 seconds process time.
The additional test is the metric-buffer/backpropagation regression. The separate
archive-download tests still run in the unchanged CI test command; they were not
rerun locally because the real-data path already has bounded mirror/CLI evidence.

```sh
.venv/bin/python scripts/bounded_validation.py --seconds 600 --name ci_dependency_suite -- .venv/bin/python -m pytest -q -o addopts='' tests --ignore=tests/test_data.py
```

### Container dependency findings after tag repair

The CPU build and its original model-import smoke check succeeded, but its build
log exposed `fastapi 0.104.1 requires anyio<4` alongside installed AnyIO 4.15.1.
W&B 0.30.0 requires Pydantic >=2.6, conflicting with the image's 2.5 pin; its
HTTP dependencies also require AnyIO 4. The final editable install silently
replaced packages and left that inconsistent environment. FastAPI is now pinned
to 0.143.0, with Pydantic >=2.9,<3. Final installs run `pip check` and import both
the serving app and trainer, making dependency/import failures fatal to the build.

The Dockerfiles also pinned torch 2.1.0, whose
[AMP exports](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/amp/__init__.py)
lack the `GradScaler` already imported by the project's trainer/checkpoint code.
The [2.4.1 exports](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/amp/__init__.py)
include it. Declared minimums are raised to torch 2.4.1 / torchvision 0.19.1;
Docker pins torch 2.7.1 / torchvision 0.22.1, a CPU/CUDA-11.8 pair listed in the
[official installation matrix](https://pytorch.org/get-started/previous-versions/).
Training builds now execute `scripts/train.py --help`, which imports the actual
entrypoint without starting a training experiment. Pip commands use python3.11
explicitly, and `.venv/` is excluded from Docker build context.

A separate checkout-local `results/validation/container-venv` resolved these
versions together with requirements, serving dependencies and the editable
project. `uv pip check`, serving/trainer imports and training `--help` passed.
The first CPU real-CIFAR smoke/replay passed with exact weights/optimizer/loss
continuation. This is macOS CPU dependency evidence, **not Linux-container or CUDA
execution evidence**. Linux builds/import checks are validated by subsequent CI.
The earlier CI smoke only imported the model, so it could miss the trainer import
failure; no existing checks have been weakened or removed.

Two final fresh CPU processes on clean revision `a81666a23530c5a59e2407d0b9ed09938a154152` reproduced
exact checkpoint continuation (zero model/optimizer/loss differences, step 8,
matching scheduler LR). Both had finite losses and gradients, zero EMA error and
no teacher gradients. Loss was 3.284728 to 1.446319
in epoch one; epoch-two last loss 1.443557; validation
1.800071; raw gradient norms 2.706037–118.613701.
Pooled feature std 0.344457, effective rank
2.662146, centered rank 3. CPU/PyTorch 2.7.1
numbers differ slightly from MPS/PyTorch 2.14.1; exact equality is established
within each environment, not across them. No CUDA computation or trained serving
request was exercised.

```sh
uv venv --python .venv/bin/python results/validation/container-venv
uv pip install --python results/validation/container-venv/bin/python torch==2.7.1 torchvision==0.22.1 -r requirements.txt -e . fastapi==0.143.0 'uvicorn[standard]==0.24.0' python-multipart==0.0.6 aiofiles==23.2.1 prometheus-client==0.19.0 pyarrow==19.0.1
uv pip check --python results/validation/container-venv/bin/python
results/validation/container-venv/bin/python -c 'from src.serving.model_server import app; from src.trainers import HJEPATrainer'
results/validation/container-venv/bin/python scripts/train.py --help
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name container_cpu_final_resume -- results/validation/container-venv/bin/python scripts/validate_small_run.py --name container_cpu_final_resume --device cpu --resume-check
.venv/bin/python scripts/bounded_validation.py --seconds 180 --name container_cpu_final_fresh -- results/validation/container-venv/bin/python scripts/validate_small_run.py --name container_cpu_final_fresh --device cpu --resume-check
```


### Post-merge CI timeout investigation (2026-10-13 UTC)

GitHub records PR #6 as externally squash-merged at 2026-10-12 21:35:02 UTC,
before its final checks completed. Task head
`7bd532c50d2365daca0ad96b3fa3d3cf0b61b8a1` and merge commit
`4696ce60ff47f5c37d20c31b74ef5a2cc4ae14ef` have identical Git trees. The agent
issued no merge command. Both Docker workflows (38311530050 and 38311536007)
passed, including all three images, dependency/import/CLI checks, container smoke
checks and security checks. CUDA tests were skipped because no CUDA runner was
available; successful GPU image construction is not CUDA execution evidence.

Both Python workflows (38311530081 and 38311536204) passed formatting, lint and
mypy, then all six matrix jobs exceeded GitHub's six-hour execution limit.
Each logged 183 passing tests with no assertion failure; its last completed test
was `TestVerification::test_verify_cifar10`. The download tests earlier in the
same logs passed. For example, Ubuntu/Python 3.11 first loaded CIFAR-10 at 22:02
and CIFAR-100 at 22:26, then downloaded them again for subsequent classes until
the job was killed at 03:35. This confirms repeated slow archive downloads,
not a trainer/type-check failure. Compact evidence is in
`docs/validation/ci-timeout-summary.json`; full logs remain untracked.

Seven class-scoped temporary directories caused repeated full CIFAR downloads;
the multicrop tests additionally used two different roots. The follow-up on the
same task branch shares a pytest-owned module-scoped directory and one root per
dataset. It retains all 77 test cases, all assertions, real downloads, torchvision
integrity checks, and separate dataset objects/transforms. No test is skipped,
mocked, or weakened. A structural comparison confirmed unchanged test names and
assertions; collection, Black and Ruff passed. A manual execution of the unchanged
Tests workflow on the follow-up branch is required to validate the entire suite.

The fix changes test data reuse only; model/training behavior and the cumulative
experiment charge (1463.04 seconds) are unchanged. Further commits cannot be
merged through the already-merged PR #6. The user's instruction forbids another
PR or a direct main push, so landing this follow-up requires a scope exception;
no replacement PR or direct main push has been made. Full green CI for the merged
revision has **not** been established.

Reproduction on an empty pytest data directory:

```sh
python -m pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
# Inspect the recorded failures without rerunning a six-hour job:
gh run view 38311530081 --log
gh run view 38311536204 --log
```


The follow-up fix is committed as
`9cafbafa70b73ca78088c30d5136750bebddca2a` and pushed to the original task branch.
Manual Tests workflow [38342722329](https://github.com/jonwiggins/H-JEPA/actions/runs/38342722329)
checks that exact commit with the unchanged full test command. As of
2026-10-13 05:38 UTC, Ubuntu/Python 3.11 passed **1,494 tests, 23 skipped**
(pytest 4552.82 seconds, including cold downloads), and macOS/Python 3.12 passed
**1,504 tests, 13 skipped** (5596.13 seconds). Both completed every data test.
Ubuntu/Python 3.12 remains in pytest, so the complete workflow is not yet green.
The separate GPU availability job skipped CUDA tests. No further local training
or evaluation experiment was launched; the original budget ledger is unchanged.
These are ordinary CI runs, separately authorized in the follow-through request.

The Ubuntu/Python 3.11 log shows CIFAR-100's first load completed at 04:52:45;
subsequent builder and verification tests reused the data and completed without
another archive download (verification at 04:55:15). This directly validates the
fixture change while retaining the real loader/integrity behavior. Compact check
status is in `docs/validation/ci-data-reuse-status.json`. Later report-only commits
do not change the tested Python code or workflow. The follow-up remains unmerged;
landing it requires permission for a follow-up PR because PR #6 is already merged
and both a replacement PR and a direct main push are explicitly forbidden.

```sh
gh workflow run test.yml --ref optio/task-8e636231-a839-4b71-b129-f99c8cf411d7
gh run view 38342722329
```


Final follow-up CI outcome: workflow **38342722329 succeeded** on exact code
revision `9cafbafa70b73ca78088c30d5136750bebddca2a`. Ubuntu/Python 3.12 also completed:
**1,494 passed, 23 skipped**, 7462.74 seconds including cold archive downloads.
All three platform/Python combinations passed formatting, Ruff, mypy, the full
pytest command, and coverage upload. Every real-data test remained enabled.
The three full suites respectively completed in 1h15m52s, 2h04m22s, and 1h33m16s
(Ubuntu 3.11, Ubuntu 3.12, macOS 3.12). The difference is dominated by slow cold
archive transfers; the fixture fix eliminates subsequent copies within each job.
These are CI durations, not new training experiments. No checks were disabled or
suppressed to obtain success. The GPU availability job still did not execute CUDA.

This green result applies to the **unmerged follow-up**, not retrospectively to
the six-hour timeout workflows on merged commit `4696ce6`. The only remaining
blocker is permission to land the follow-up through another PR: the original PR
was already merged before these failures became visible. All follow-up changes,
validation evidence and this report are committed/pushed on the same task branch;
report-only commits after `9cafbaf` leave tested source, tests and workflows
unchanged. No replacement PR has been created and main has not been pushed.

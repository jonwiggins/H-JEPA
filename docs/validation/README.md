# Bounded validation artifacts

The report `../VALIDATION_2026-10-12.md` describes scope, commands and limitations.
JSON files here are compact measurements; datasets, checkpoint files and full logs
are deliberately excluded. Training and evaluation commands run under
`scripts/bounded_validation.py`, which records a persistent, cumulative 3300-second
budget in `results/validation/budget.json`, caps each command at 600 seconds, and
kills its isolated process group on timeout or supervisor interruption. Test runs
are conservatively charged too. Run only one supervisor at a time; the ledger is
locked. Do not delete or reset the ledger during a task.

CIFAR-10 mirror: `uoft-cs/cifar10`, revision
`0b2714987fa478483af9968de7c934580d0bb9a2`, `plain_text` Parquet files.
The diagnostic harness reads only the first 32 training and first 16 test rows,
uses the repository's JEPA transforms at 64x64, and never substitutes synthetic
images. PyArrow is an optional dependency of this harness only.

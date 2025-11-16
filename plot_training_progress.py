#!/usr/bin/env python3
"""Plot H-JEPA training progress from log file."""

import re

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend

# Read log file
with open("foundation_model_training.log", "r") as f:
    log_content = f.read()

# Extract loss values from progress bars
# Pattern: loss=0.0058
loss_pattern = r"loss=(\d+\.\d+)"
losses = re.findall(loss_pattern, log_content)
losses = [float(loss) for loss in losses]

# Extract step numbers from progress bars
# Pattern: Epoch 1/100:  21%|██        | 326/1562
step_pattern = r"Epoch \d+/\d+:.*?\|\s*(\d+)/\d+"
steps = re.findall(step_pattern, log_content)
steps = [int(step) for step in steps]

# Ensure we have matching data
min_len = min(len(losses), len(steps))
losses = losses[:min_len]
steps = steps[:min_len]

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(steps, losses, linewidth=2, color="#2E86AB", alpha=0.8)
plt.xlabel("Training Step", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("H-JEPA Foundation Model Training Progress (Epoch 1)", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)

# Add some statistics
if losses:
    plt.axhline(
        y=losses[0], color="red", linestyle="--", alpha=0.5, label=f"Initial: {losses[0]:.4f}"
    )
    plt.axhline(
        y=losses[-1], color="green", linestyle="--", alpha=0.5, label=f"Current: {losses[-1]:.4f}"
    )
    improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
    plt.text(
        0.02,
        0.98,
        f"Improvement: {improvement:.1f}%",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

plt.legend()
plt.tight_layout()
plt.savefig("training_progress.png", dpi=150, bbox_inches="tight")
print(f"Plot saved to training_progress.png")
print(f"Data points: {len(losses)}")
print(f"Steps: {steps[0]} to {steps[-1]}")
print(f"Loss range: {min(losses):.4f} to {max(losses):.4f}")

#!/usr/bin/env python3
"""Real-time training monitoring dashboard for H-JEPA."""

import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def parse_log_file(log_path):
    """Parse the training log file and extract metrics."""
    if not log_path.exists():
        return None

    with open(log_path) as f:
        content = f.read()

    # Extract progress information from the most recent line
    # Pattern: Epoch 1/100:  20%|█▉        | 312/1562 [06:33<24:39,  1.18s/it, loss=0.0035, lr=1.92e-06]
    progress_pattern = r"Epoch (\d+)/(\d+):\s+(\d+)%.*?\|\s*(\d+)/(\d+)\s+\[([\d:]+)<([\d:]+),\s+([\d.]+).*?loss=([\d.]+).*?lr=([\d.e+-]+)"

    matches = list(re.finditer(progress_pattern, content))
    if not matches:
        return None

    # Get the most recent match
    match = matches[-1]

    current_epoch = int(match.group(1))
    total_epochs = int(match.group(2))
    progress_pct = int(match.group(3))
    current_step = int(match.group(4))
    total_steps = int(match.group(5))
    elapsed = match.group(6)
    remaining = match.group(7)
    iter_time = float(match.group(8))
    loss = float(match.group(9))
    lr = float(match.group(10))

    # Extract all losses for trend calculation
    loss_pattern = r"loss=([\d.]+)"
    all_losses = [float(l) for l in re.findall(loss_pattern, content)]

    # Calculate statistics
    initial_loss = all_losses[0] if all_losses else loss
    min_loss = min(all_losses) if all_losses else loss
    recent_losses = all_losses[-10:] if len(all_losses) >= 10 else all_losses
    avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else loss

    # Calculate improvement
    improvement = ((initial_loss - loss) / initial_loss * 100) if initial_loss > 0 else 0

    return {
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "progress_pct": progress_pct,
        "current_step": current_step,
        "total_steps": total_steps,
        "elapsed": elapsed,
        "remaining": remaining,
        "iter_time": iter_time,
        "loss": loss,
        "lr": lr,
        "initial_loss": initial_loss,
        "min_loss": min_loss,
        "avg_recent_loss": avg_recent_loss,
        "improvement": improvement,
        "total_data_points": len(all_losses),
    }


def format_time(time_str):
    """Format time string to be more readable."""
    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return f"{hours}h {minutes}m {seconds}s"
    return time_str


def create_progress_bar(current, total, width=40):
    """Create a text-based progress bar."""
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * current / total
    return f"[{bar}] {pct:.1f}%"


def display_dashboard(metrics):
    """Display the training dashboard."""
    clear_screen()

    # Header
    print("=" * 80)
    print(" " * 20 + "H-JEPA FOUNDATION MODEL TRAINING DASHBOARD")
    print("=" * 80)
    print()

    # Current status
    print(f"🕐 Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Epoch progress
    print("📊 EPOCH PROGRESS")
    print(f"   Epoch: {metrics['current_epoch']}/{metrics['total_epochs']}")
    epoch_bar = create_progress_bar(metrics["current_epoch"], metrics["total_epochs"], 50)
    print(f"   {epoch_bar}")
    print()

    # Step progress (within current epoch)
    print(f"📈 STEP PROGRESS (Epoch {metrics['current_epoch']})")
    print(f"   Step: {metrics['current_step']:,}/{metrics['total_steps']:,}")
    step_bar = create_progress_bar(metrics["current_step"], metrics["total_steps"], 50)
    print(f"   {step_bar}")
    print()

    # Training metrics
    print("🎯 TRAINING METRICS")
    print(f"   Current Loss:      {metrics['loss']:.6f}")
    print(f"   Initial Loss:      {metrics['initial_loss']:.6f}")
    print(f"   Minimum Loss:      {metrics['min_loss']:.6f}")
    print(f"   Avg Recent (10):   {metrics['avg_recent_loss']:.6f}")
    print(f"   Improvement:       {metrics['improvement']:.1f}%")
    print(f"   Learning Rate:     {metrics['lr']:.2e}")
    print()

    # Performance
    print("⚡ PERFORMANCE")
    print(f"   Iteration Time:    {metrics['iter_time']:.2f}s/it")
    steps_per_min = 60 / metrics["iter_time"] if metrics["iter_time"] > 0 else 0
    print(f"   Steps per Minute:  {steps_per_min:.1f}")
    print()

    # Time estimates
    print("⏱️  TIME ESTIMATES")
    print(f"   Elapsed:           {format_time(metrics['elapsed'])}")
    print(f"   Remaining (Epoch): {format_time(metrics['remaining'])}")

    # Calculate total remaining time
    steps_remaining_this_epoch = metrics["total_steps"] - metrics["current_step"]
    steps_remaining_total = (
        steps_remaining_this_epoch
        + (metrics["total_epochs"] - metrics["current_epoch"]) * metrics["total_steps"]
    )
    total_remaining_seconds = steps_remaining_total * metrics["iter_time"]
    total_remaining = str(timedelta(seconds=int(total_remaining_seconds)))
    print(f"   Remaining (Total): {total_remaining}")

    # Calculate ETA
    eta = datetime.now() + timedelta(seconds=total_remaining_seconds)
    print(f"   ETA:               {eta.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Statistics
    print("📊 STATISTICS")
    print(f"   Total Data Points: {metrics['total_data_points']:,}")
    print(
        f"   Steps Completed:   {metrics['current_step']:,} / {metrics['total_steps'] * metrics['total_epochs']:,} (total)"
    )
    total_progress = (
        ((metrics["current_epoch"] - 1) * metrics["total_steps"] + metrics["current_step"])
        / (metrics["total_epochs"] * metrics["total_steps"])
        * 100
    )
    print(f"   Overall Progress:  {total_progress:.2f}%")
    print()

    # Footer
    print("=" * 80)
    print("Press Ctrl+C to stop monitoring")
    print("=" * 80)


def main():
    """Main monitoring loop."""
    log_path = Path("foundation_model_training.log")

    print("Starting H-JEPA Training Monitor...")
    print(f"Watching: {log_path}")
    print()

    # Wait for log file to exist
    while not log_path.exists():
        print("Waiting for training log file...")
        time.sleep(2)

    try:
        while True:
            metrics = parse_log_file(log_path)

            if metrics:
                display_dashboard(metrics)
            else:
                clear_screen()
                print("=" * 80)
                print(" " * 25 + "Waiting for training data...")
                print("=" * 80)
                print()
                print(f"Log file: {log_path}")
                print(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Refresh every 5 seconds
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()

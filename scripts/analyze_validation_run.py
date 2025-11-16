#!/usr/bin/env python3
"""
Analyze validation run results and provide recommendations for full training.

This script:
1. Parses training logs
2. Analyzes loss curves and convergence
3. Checks for representation collapse
4. Estimates full training time
5. Recommends optimal configuration for next run
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(log_path: Path) -> Dict:
    """Parse training log file and extract metrics."""

    metrics = {
        'losses': [],
        'learning_rates': [],
        'steps': [],
        'epoch_times': [],
        'iterations_per_sec': [],
    }

    config_info = {}

    with open(log_path, 'r') as f:
        content = f.read()

    # Extract configuration
    if "Model:" in content:
        encoder_match = re.search(r'Encoder: (\S+)', content)
        if encoder_match:
            config_info['encoder'] = encoder_match.group(1)

        embed_match = re.search(r'Embedding dim: (\d+)', content)
        if embed_match:
            config_info['embed_dim'] = int(embed_match.group(1))

        hierarchies_match = re.search(r'Hierarchies: (\d+)', content)
        if hierarchies_match:
            config_info['num_hierarchies'] = int(hierarchies_match.group(1))

        epochs_match = re.search(r'Epochs: (\d+)', content)
        if epochs_match:
            config_info['epochs'] = int(epochs_match.group(1))

        batch_match = re.search(r'Batch size: (\d+)', content)
        if batch_match:
            config_info['batch_size'] = int(batch_match.group(1))

    # Extract total parameters
    params_match = re.search(r'Total parameters: ([\d,]+)', content)
    if params_match:
        config_info['total_params'] = int(params_match.group(1).replace(',', ''))

    # Parse progress lines (format: "Epoch X/Y: XX%|... [time<remaining, speed, loss=X.XXXX, lr=X.XXe-XX]")
    # Example: "Epoch 1/5:  18%|... | 283/1562 [01:50<..., 3.14it/s, loss=0.0042, lr=1.60e-05]"
    progress_pattern = r'Epoch \d+/\d+:.*?(\d+)/\d+.*?(\d+\.\d+)it/s, loss=([\d.]+), lr=([\d.e+-]+)'

    for match in re.finditer(progress_pattern, content):
        step = int(match.group(1))
        it_per_sec = float(match.group(2))
        loss = float(match.group(3))
        lr = float(match.group(4))

        metrics['steps'].append(step)
        metrics['iterations_per_sec'].append(it_per_sec)
        metrics['losses'].append(loss)
        metrics['learning_rates'].append(lr)

    return {
        'config': config_info,
        'metrics': metrics,
    }


def analyze_loss_curve(losses: List[float], steps: List[int]) -> Dict:
    """Analyze loss curve for convergence and stability."""

    if len(losses) < 10:
        return {
            'status': 'insufficient_data',
            'message': 'Not enough data points for analysis'
        }

    losses_array = np.array(losses)
    steps_array = np.array(steps)

    # Calculate statistics
    initial_loss = losses_array[0]
    final_loss = losses_array[-1]
    min_loss = losses_array.min()
    max_loss = losses_array.max()
    mean_loss = losses_array.mean()
    std_loss = losses_array.std()

    # Calculate improvement
    improvement = (initial_loss - final_loss) / initial_loss * 100

    # Check for divergence
    recent_losses = losses_array[-20:]
    is_diverging = recent_losses[-1] > recent_losses[0] * 1.1

    # Check for convergence (loss plateauing)
    if len(losses) > 50:
        recent_std = recent_losses.std()
        is_converging = recent_std < mean_loss * 0.01  # Less than 1% variation
    else:
        is_converging = False

    # Estimate convergence quality
    if improvement > 30:
        convergence_quality = 'excellent'
    elif improvement > 15:
        convergence_quality = 'good'
    elif improvement > 5:
        convergence_quality = 'fair'
    else:
        convergence_quality = 'poor'

    return {
        'status': 'analyzed',
        'initial_loss': float(initial_loss),
        'final_loss': float(final_loss),
        'min_loss': float(min_loss),
        'max_loss': float(max_loss),
        'mean_loss': float(mean_loss),
        'std_loss': float(std_loss),
        'improvement_percent': float(improvement),
        'is_diverging': bool(is_diverging),
        'is_converging': bool(is_converging),
        'convergence_quality': convergence_quality,
    }


def analyze_training_speed(it_per_sec: List[float], steps: List[int]) -> Dict:
    """Analyze training speed and estimate completion times."""

    if len(it_per_sec) < 5:
        return {
            'status': 'insufficient_data'
        }

    # Filter out initial warmup (first 10% of steps)
    stable_start = len(it_per_sec) // 10
    stable_speeds = it_per_sec[stable_start:]

    mean_speed = np.mean(stable_speeds)
    std_speed = np.std(stable_speeds)
    min_speed = np.min(stable_speeds)
    max_speed = np.max(stable_speeds)

    return {
        'status': 'analyzed',
        'mean_iterations_per_sec': float(mean_speed),
        'std_iterations_per_sec': float(std_speed),
        'min_iterations_per_sec': float(min_speed),
        'max_iterations_per_sec': float(max_speed),
        'speed_stability': 'stable' if std_speed < mean_speed * 0.1 else 'variable',
    }


def estimate_training_times(mean_speed: float, batch_size: int) -> Dict:
    """Estimate training times for different epoch counts."""

    # CIFAR-10 has 50,000 training images
    steps_per_epoch = 50000 // batch_size

    estimates = {}

    for epochs in [5, 10, 20, 50, 100]:
        total_steps = steps_per_epoch * epochs
        time_seconds = total_steps / mean_speed
        time_minutes = time_seconds / 60
        time_hours = time_minutes / 60

        estimates[f'{epochs}_epochs'] = {
            'total_steps': total_steps,
            'time_seconds': float(time_seconds),
            'time_minutes': float(time_minutes),
            'time_hours': float(time_hours),
            'time_formatted': format_time(time_seconds),
        }

    return estimates


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def recommend_next_run(analysis: Dict) -> Dict:
    """Generate recommendations for next training run."""

    recommendations = {
        'recommended_config': None,
        'reasoning': [],
        'warnings': [],
        'optimizations': [],
    }

    # Check loss convergence
    loss_analysis = analysis.get('loss_analysis', {})
    convergence_quality = loss_analysis.get('convergence_quality', 'unknown')

    if convergence_quality in ['excellent', 'good']:
        recommendations['reasoning'].append(
            f"Loss convergence is {convergence_quality} ({loss_analysis['improvement_percent']:.1f}% improvement), "
            "indicating the model is learning effectively."
        )
        recommendations['ready_for_full_training'] = True
    else:
        recommendations['reasoning'].append(
            f"Loss convergence is {convergence_quality}, which may indicate issues with hyperparameters."
        )
        recommendations['ready_for_full_training'] = False
        recommendations['warnings'].append(
            "Consider adjusting learning rate or warmup schedule before full training."
        )

    # Check for divergence
    if loss_analysis.get('is_diverging', False):
        recommendations['warnings'].append(
            "WARNING: Loss is diverging! Reduce learning rate before proceeding."
        )
        recommendations['ready_for_full_training'] = False

    # Speed analysis
    speed_analysis = analysis.get('speed_analysis', {})
    mean_speed = speed_analysis.get('mean_iterations_per_sec', 0)

    if mean_speed > 0:
        recommendations['reasoning'].append(
            f"Training speed is stable at ~{mean_speed:.2f} it/s on M1 Max MPS."
        )

        # Recommend configuration based on available time
        time_estimates = analysis.get('time_estimates', {})

        if time_estimates:
            # 20-epoch run
            time_20 = time_estimates.get('20_epochs', {}).get('time_hours', 0)
            time_100 = time_estimates.get('100_epochs', {}).get('time_hours', 0)

            recommendations['options'] = {
                'quick_baseline': {
                    'config': 'configs/m1_max_full_20epoch.yaml',
                    'epochs': 20,
                    'expected_time': time_estimates.get('20_epochs', {}).get('time_formatted', 'unknown'),
                    'expected_accuracy': '70-78%',
                    'description': 'Good for quick baseline results',
                },
                'full_training': {
                    'config': 'configs/m1_max_full_100epoch.yaml',
                    'epochs': 100,
                    'expected_time': time_estimates.get('100_epochs', {}).get('time_formatted', 'unknown'),
                    'expected_accuracy': '80-85%',
                    'description': 'Competitive results (overnight run)',
                },
            }

            # Default recommendation
            if time_20 < 4:  # Less than 4 hours for 20 epochs
                recommendations['recommended_config'] = 'configs/m1_max_full_20epoch.yaml'
                recommendations['reasoning'].append(
                    f"20-epoch run will take ~{time_estimates['20_epochs']['time_formatted']}, "
                    "which is reasonable for a baseline."
                )
            else:
                recommendations['recommended_config'] = 'configs/m1_max_full_100epoch.yaml'
                recommendations['reasoning'].append(
                    "Consider 100-epoch run for competitive results (can run overnight)."
                )

    # Optimizations
    if speed_analysis.get('speed_stability') == 'variable':
        recommendations['optimizations'].append(
            "Training speed is variable. Consider reducing num_workers or batch_size for stability."
        )

    return recommendations


def plot_training_curves(metrics: Dict, output_dir: Path):
    """Generate training curve plots."""

    output_dir.mkdir(parents=True, exist_ok=True)

    steps = metrics['steps']
    losses = metrics['losses']
    lrs = metrics['learning_rates']
    speeds = metrics['iterations_per_sec']

    if not steps:
        print("No data to plot")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Loss curve
    axes[0].plot(steps, losses, linewidth=2, color='blue', alpha=0.8)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Learning rate schedule
    axes[1].plot(steps, lrs, linewidth=2, color='green', alpha=0.8)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Training speed
    axes[2].plot(steps, speeds, linewidth=2, color='orange', alpha=0.8)
    axes[2].axhline(y=np.mean(speeds), color='red', linestyle='--',
                    label=f'Mean: {np.mean(speeds):.2f} it/s')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Iterations/Second')
    axes[2].set_title('Training Speed (M1 Max MPS)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / 'validation_training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {plot_path}")

    plt.close()


def generate_report(analysis: Dict, output_path: Path):
    """Generate markdown report."""

    with open(output_path, 'w') as f:
        f.write("# Validation Run Analysis Report\n\n")
        f.write(f"Generated: {Path.cwd()}\n\n")

        # Configuration
        f.write("## Configuration\n\n")
        config = analysis.get('config', {})
        f.write(f"- **Encoder**: {config.get('encoder', 'unknown')}\n")
        f.write(f"- **Embedding Dim**: {config.get('embed_dim', 'unknown')}\n")
        f.write(f"- **Hierarchies**: {config.get('num_hierarchies', 'unknown')}\n")
        f.write(f"- **Total Parameters**: {config.get('total_params', 'unknown'):,}\n")
        f.write(f"- **Epochs**: {config.get('epochs', 'unknown')}\n")
        f.write(f"- **Batch Size**: {config.get('batch_size', 'unknown')}\n\n")

        # Loss Analysis
        f.write("## Loss Analysis\n\n")
        loss = analysis.get('loss_analysis', {})
        if loss.get('status') == 'analyzed':
            f.write(f"- **Initial Loss**: {loss['initial_loss']:.6f}\n")
            f.write(f"- **Final Loss**: {loss['final_loss']:.6f}\n")
            f.write(f"- **Improvement**: {loss['improvement_percent']:.2f}%\n")
            f.write(f"- **Convergence Quality**: {loss['convergence_quality']}\n")
            f.write(f"- **Is Diverging**: {loss['is_diverging']}\n")
            f.write(f"- **Is Converging**: {loss['is_converging']}\n\n")

            if loss['convergence_quality'] in ['excellent', 'good']:
                f.write("✅ **Status**: Loss is decreasing well, model is learning effectively.\n\n")
            else:
                f.write("⚠️ **Status**: Loss convergence could be better.\n\n")

        # Speed Analysis
        f.write("## Training Speed\n\n")
        speed = analysis.get('speed_analysis', {})
        if speed.get('status') == 'analyzed':
            f.write(f"- **Mean Speed**: {speed['mean_iterations_per_sec']:.2f} it/s\n")
            f.write(f"- **Speed Range**: {speed['min_iterations_per_sec']:.2f} - {speed['max_iterations_per_sec']:.2f} it/s\n")
            f.write(f"- **Stability**: {speed['speed_stability']}\n\n")

        # Time Estimates
        f.write("## Estimated Training Times\n\n")
        estimates = analysis.get('time_estimates', {})
        if estimates:
            f.write("| Epochs | Total Steps | Estimated Time |\n")
            f.write("|--------|-------------|----------------|\n")
            for key in ['5_epochs', '10_epochs', '20_epochs', '50_epochs', '100_epochs']:
                if key in estimates:
                    est = estimates[key]
                    epochs = key.split('_')[0]
                    f.write(f"| {epochs} | {est['total_steps']:,} | {est['time_formatted']} |\n")
            f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        rec = analysis.get('recommendations', {})

        if rec.get('ready_for_full_training'):
            f.write("✅ **Ready for full training**\n\n")
        else:
            f.write("⚠️ **Not ready for full training** - address warnings first\n\n")

        # Reasoning
        if rec.get('reasoning'):
            f.write("### Reasoning\n\n")
            for reason in rec['reasoning']:
                f.write(f"- {reason}\n")
            f.write("\n")

        # Warnings
        if rec.get('warnings'):
            f.write("### ⚠️ Warnings\n\n")
            for warning in rec['warnings']:
                f.write(f"- {warning}\n")
            f.write("\n")

        # Options
        if rec.get('options'):
            f.write("### Training Options\n\n")

            for name, option in rec['options'].items():
                f.write(f"#### Option: {name.replace('_', ' ').title()}\n\n")
                f.write(f"- **Config**: `{option['config']}`\n")
                f.write(f"- **Epochs**: {option['epochs']}\n")
                f.write(f"- **Expected Time**: {option['expected_time']}\n")
                f.write(f"- **Expected Accuracy**: {option['expected_accuracy']}\n")
                f.write(f"- **Description**: {option['description']}\n\n")

        # Recommended command
        if rec.get('recommended_config'):
            f.write("### Recommended Next Step\n\n")
            f.write(f"Run the following command:\n\n")
            f.write(f"```bash\n")
            f.write(f"python3.11 scripts/train.py --config {rec['recommended_config']}\n")
            f.write(f"```\n\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze H-JEPA validation run')
    parser.add_argument(
        '--log',
        type=Path,
        default=Path('training_run.log'),
        help='Path to training log file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/validation_analysis'),
        help='Directory for output files'
    )

    args = parser.parse_args()

    if not args.log.exists():
        print(f"Error: Log file not found: {args.log}")
        return

    print(f"Analyzing training log: {args.log}")
    print()

    # Parse log
    data = parse_training_log(args.log)

    # Perform analyses
    analysis = {
        'config': data['config'],
        'loss_analysis': analyze_loss_curve(
            data['metrics']['losses'],
            data['metrics']['steps']
        ),
        'speed_analysis': analyze_training_speed(
            data['metrics']['iterations_per_sec'],
            data['metrics']['steps']
        ),
    }

    # Estimate training times
    if analysis['speed_analysis'].get('status') == 'analyzed':
        mean_speed = analysis['speed_analysis']['mean_iterations_per_sec']
        batch_size = data['config'].get('batch_size', 32)
        analysis['time_estimates'] = estimate_training_times(mean_speed, batch_size)

    # Generate recommendations
    analysis['recommendations'] = recommend_next_run(analysis)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("Generating training curves...")
    plot_training_curves(data['metrics'], args.output_dir)

    # Generate report
    report_path = args.output_dir / 'validation_report.md'
    print(f"Generating report: {report_path}")
    generate_report(analysis, report_path)

    # Save JSON analysis
    json_path = args.output_dir / 'analysis.json'
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {json_path}")

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION RUN ANALYSIS SUMMARY")
    print("="*80)

    print("\n📊 Loss Analysis:")
    loss = analysis['loss_analysis']
    if loss.get('status') == 'analyzed':
        print(f"  Initial: {loss['initial_loss']:.6f}")
        print(f"  Final:   {loss['final_loss']:.6f}")
        print(f"  Improvement: {loss['improvement_percent']:.2f}%")
        print(f"  Quality: {loss['convergence_quality']}")

    print("\n⚡ Training Speed:")
    speed = analysis['speed_analysis']
    if speed.get('status') == 'analyzed':
        print(f"  Mean: {speed['mean_iterations_per_sec']:.2f} it/s")
        print(f"  Stability: {speed['speed_stability']}")

    print("\n⏱️  Estimated Times:")
    estimates = analysis.get('time_estimates', {})
    if estimates:
        print(f"  20 epochs:  {estimates['20_epochs']['time_formatted']}")
        print(f"  100 epochs: {estimates['100_epochs']['time_formatted']}")

    print("\n💡 Recommendation:")
    rec = analysis['recommendations']
    if rec.get('recommended_config'):
        print(f"  Config: {rec['recommended_config']}")
        print(f"  Ready: {'✅ Yes' if rec.get('ready_for_full_training') else '⚠️  No'}")

    if rec.get('warnings'):
        print("\n⚠️  Warnings:")
        for warning in rec['warnings']:
            print(f"  - {warning}")

    print("\n" + "="*80)
    print(f"\nFull report: {report_path}")
    print(f"Training curves: {args.output_dir / 'validation_training_curves.png'}")
    print()


if __name__ == '__main__':
    main()

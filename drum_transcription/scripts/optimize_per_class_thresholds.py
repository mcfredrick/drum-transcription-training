"""
Per-Class Threshold Optimization for Rhythm Game

Finds optimal threshold for each drum class independently to maximize performance
for rhythm game use case (prioritizing recall while maintaining acceptable precision).
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.crnn import DrumTranscriptionCRNN
from src.data.data_module import EGMDDataModule
from src.utils.config import load_config


class PerClassThresholdOptimizer:
    """Finds optimal threshold for each class independently."""

    def __init__(self, model, dataloader, class_names, priority_classes=None):
        """
        Initialize optimizer.

        Args:
            model: Trained model
            dataloader: Validation or test dataloader
            class_names: List of drum class names
            priority_classes: List of class names to prioritize (Tier 1)
        """
        self.model = model
        self.dataloader = dataloader
        self.class_names = class_names
        self.n_classes = len(class_names)

        # Priority classes for rhythm game
        if priority_classes is None:
            priority_classes = ['kick', 'snare_head', 'snare_rim',
                              'hihat_closed', 'ride', 'floor_tom']
        self.priority_classes = priority_classes
        self.priority_indices = [i for i, name in enumerate(class_names)
                                if name in priority_classes]

    def collect_predictions(self):
        """Run model on all data and collect predictions and labels."""
        print("Collecting predictions from model...")

        all_predictions = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Processing batches"):
                specs, labels, lengths = batch

                # Move to device
                specs = specs.to(self.model.device)

                # Forward pass
                predictions = self.model(specs)  # (batch, time, n_classes)

                # Apply sigmoid to get probabilities
                predictions_prob = torch.sigmoid(predictions)

                # Adjust lengths for pooling
                adjusted_lengths = self.model._adjust_lengths_for_pooling(lengths)

                # Downsample labels
                labels_downsampled = self.model._downsample_labels(labels, predictions.size(1))

                # Extract valid frames
                for i in range(predictions.size(0)):
                    length = adjusted_lengths[i]
                    all_predictions.append(predictions_prob[i, :length].cpu().numpy())
                    all_labels.append(labels_downsampled[i, :length].cpu().numpy())

        # Concatenate all frames
        self.predictions = np.concatenate(all_predictions, axis=0)  # (total_frames, n_classes)
        self.labels = np.concatenate(all_labels, axis=0)  # (total_frames, n_classes)

        print(f"Collected {self.predictions.shape[0]} frames across {self.n_classes} classes")

    def find_optimal_threshold_for_class(self, class_idx, strategy='max_f1',
                                         min_precision=0.60, min_recall=0.70):
        """
        Find optimal threshold for a single class.

        Args:
            class_idx: Index of the class
            strategy: 'max_f1', 'max_recall', 'balanced', 'rhythm_game'
            min_precision: Minimum acceptable precision (for rhythm_game strategy)
            min_recall: Minimum acceptable recall (for rhythm_game strategy)

        Returns:
            dict with threshold, precision, recall, f1, support
        """
        y_true = self.labels[:, class_idx]
        y_scores = self.predictions[:, class_idx]

        support = int(y_true.sum())

        if support == 0:
            return {
                'threshold': 0.5,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'support': 0
            }

        # Compute precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

        # Compute F1 for each threshold
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

        if strategy == 'max_f1':
            # Simple max F1
            best_idx = np.argmax(f1_scores)

        elif strategy == 'max_recall':
            # Maximum recall while maintaining some minimum precision
            valid_indices = np.where(precisions[:-1] >= min_precision)[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmax(recalls[:-1][valid_indices])]
            else:
                best_idx = np.argmax(f1_scores)

        elif strategy == 'rhythm_game':
            # Prioritize recall but maintain minimum precision
            # For rhythm game: missing notes is worse than false positives
            valid_indices = np.where(precisions[:-1] >= min_precision)[0]
            if len(valid_indices) > 0:
                # Among valid precision, maximize recall
                best_idx = valid_indices[np.argmax(recalls[:-1][valid_indices])]
            else:
                # If can't meet min precision, just maximize F1
                best_idx = np.argmax(f1_scores)

        elif strategy == 'balanced':
            # Balance precision and recall (closest to equal)
            balance = np.abs(precisions[:-1] - recalls[:-1])
            best_idx = np.argmin(balance)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        optimal_threshold = float(thresholds[best_idx])
        optimal_precision = float(precisions[best_idx])
        optimal_recall = float(recalls[best_idx])
        optimal_f1 = float(f1_scores[best_idx])

        return {
            'threshold': optimal_threshold,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1': optimal_f1,
            'support': support
        }

    def optimize_all_classes(self, strategy='rhythm_game', min_precision=0.60):
        """
        Find optimal thresholds for all classes.

        Args:
            strategy: Optimization strategy
            min_precision: Minimum acceptable precision (rhythm game needs this)

        Returns:
            dict mapping class_idx -> threshold info
        """
        print(f"\nOptimizing thresholds using strategy: {strategy}")
        print(f"Minimum precision requirement: {min_precision:.1%}\n")

        results = {}

        for class_idx in range(self.n_classes):
            class_name = self.class_names[class_idx]

            # Priority classes get more aggressive recall optimization
            if class_idx in self.priority_indices:
                result = self.find_optimal_threshold_for_class(
                    class_idx,
                    strategy=strategy,
                    min_precision=min_precision,
                    min_recall=0.85  # Target >85% recall for priority classes
                )
                is_priority = True
            else:
                result = self.find_optimal_threshold_for_class(
                    class_idx,
                    strategy=strategy,
                    min_precision=max(0.50, min_precision - 0.1),  # Lower bar for non-priority
                    min_recall=0.70
                )
                is_priority = False

            results[class_idx] = {
                'name': class_name,
                'is_priority': is_priority,
                **result
            }

        return results

    def evaluate_with_per_class_thresholds(self, thresholds):
        """
        Evaluate model using per-class thresholds.

        Args:
            thresholds: dict mapping class_idx -> threshold

        Returns:
            dict with overall and per-class metrics
        """
        all_y_true = []
        all_y_pred = []

        for class_idx in range(self.n_classes):
            threshold = thresholds.get(class_idx, 0.5)
            y_true = self.labels[:, class_idx]
            y_pred = (self.predictions[:, class_idx] >= threshold).astype(int)

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

        all_y_true = np.stack(all_y_true, axis=1)
        all_y_pred = np.stack(all_y_pred, axis=1)

        # Compute overall metrics (micro-average)
        total_tp = ((all_y_true == 1) & (all_y_pred == 1)).sum()
        total_fp = ((all_y_true == 0) & (all_y_pred == 1)).sum()
        total_fn = ((all_y_true == 1) & (all_y_pred == 0)).sum()

        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # Per-class metrics
        per_class = {}
        for class_idx in range(self.n_classes):
            y_true = all_y_true[:, class_idx]
            y_pred = all_y_pred[:, class_idx]

            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()

            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)
            f = 2 * p * r / (p + r + 1e-10)

            per_class[class_idx] = {
                'precision': float(p),
                'recall': float(r),
                'f1': float(f),
                'support': int(y_true.sum())
            }

        return {
            'overall': {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            },
            'per_class': per_class
        }

    def compare_baseline_vs_optimized(self, optimized_results, baseline_threshold=0.5):
        """Compare baseline (single threshold) vs optimized (per-class thresholds)."""
        print("\n" + "="*80)
        print("BASELINE vs OPTIMIZED COMPARISON")
        print("="*80)

        # Baseline: single threshold for all classes
        baseline_thresholds = {i: baseline_threshold for i in range(self.n_classes)}
        baseline_metrics = self.evaluate_with_per_class_thresholds(baseline_thresholds)

        # Optimized: per-class thresholds
        optimized_thresholds = {i: res['threshold'] for i, res in optimized_results.items()}
        optimized_metrics = self.evaluate_with_per_class_thresholds(optimized_thresholds)

        # Overall comparison
        print(f"\nOVERALL METRICS:")
        print(f"{'Metric':<15} {'Baseline':>12} {'Optimized':>12} {'Change':>12}")
        print("-" * 55)

        for metric in ['precision', 'recall', 'f1']:
            baseline_val = baseline_metrics['overall'][metric]
            optimized_val = optimized_metrics['overall'][metric]
            change = optimized_val - baseline_val
            change_pct = (change / baseline_val * 100) if baseline_val > 0 else 0

            print(f"{metric.capitalize():<15} {baseline_val:>11.1%} {optimized_val:>11.1%} "
                  f"{change:>+6.1%} ({change_pct:+.1f}%)")

        # Priority classes comparison
        print(f"\n{'PRIORITY CLASSES (Tier 1)':^55}")
        print(f"{'Class':<18} {'Metric':<10} {'Baseline':>10} {'Optimized':>10} {'Change':>10}")
        print("-" * 65)

        for class_idx in self.priority_indices:
            class_name = self.class_names[class_idx]

            for metric in ['recall', 'precision', 'f1']:
                baseline_val = baseline_metrics['per_class'][class_idx][metric]
                optimized_val = optimized_metrics['per_class'][class_idx][metric]
                change = optimized_val - baseline_val

                metric_display = metric if metric == 'recall' else ''
                print(f"{class_name:<18} {metric_display:<10} {baseline_val:>9.1%} "
                      f"{optimized_val:>9.1%} {change:>+9.1%}")

        return baseline_metrics, optimized_metrics

    def generate_report(self, optimized_results, output_dir):
        """Generate detailed report with recommendations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "per_class_optimization_report.txt"

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PER-CLASS THRESHOLD OPTIMIZATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total frames analyzed: {self.predictions.shape[0]:,}\n")
            f.write(f"Number of classes: {self.n_classes}\n")
            f.write(f"Priority classes: {', '.join(self.priority_classes)}\n\n")

            # Priority classes
            f.write("-"*80 + "\n")
            f.write("PRIORITY CLASSES (TIER 1 - Critical for Rhythm Game)\n")
            f.write("-"*80 + "\n\n")

            f.write(f"{'Class':<18} {'Threshold':>10} {'Precision':>11} {'Recall':>11} "
                   f"{'F1':>11} {'Support':>12}\n")
            f.write("-"*80 + "\n")

            for class_idx in self.priority_indices:
                res = optimized_results[class_idx]
                f.write(f"{res['name']:<18} {res['threshold']:>10.2f} "
                       f"{res['precision']:>11.1%} {res['recall']:>11.1%} "
                       f"{res['f1']:>11.1%} {res['support']:>12,}\n")

            # Other classes
            f.write("\n" + "-"*80 + "\n")
            f.write("OTHER CLASSES (TIER 2/3)\n")
            f.write("-"*80 + "\n\n")

            f.write(f"{'Class':<18} {'Threshold':>10} {'Precision':>11} {'Recall':>11} "
                   f"{'F1':>11} {'Support':>12}\n")
            f.write("-"*80 + "\n")

            for class_idx in range(self.n_classes):
                if class_idx not in self.priority_indices:
                    res = optimized_results[class_idx]
                    f.write(f"{res['name']:<18} {res['threshold']:>10.2f} "
                           f"{res['precision']:>11.1%} {res['recall']:>11.1%} "
                           f"{res['f1']:>11.1%} {res['support']:>12,}\n")

            # Recommendations
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")

            # Check if targets are met
            targets_met = []
            targets_failed = []

            for class_idx in self.priority_indices:
                res = optimized_results[class_idx]
                if res['recall'] >= 0.85 and res['precision'] >= 0.70:
                    targets_met.append(res['name'])
                else:
                    targets_failed.append((res['name'], res['recall'], res['precision']))

            if targets_met:
                f.write("✅ TARGETS MET:\n")
                for name in targets_met:
                    f.write(f"  - {name}: Ready for rhythm game!\n")
                f.write("\n")

            if targets_failed:
                f.write("❌ TARGETS NOT MET (need further action):\n")
                for name, recall, precision in targets_failed:
                    f.write(f"  - {name}: recall={recall:.1%}, precision={precision:.1%}\n")
                    if recall < 0.85:
                        f.write(f"    → Recall too low - consider model retraining or class simplification\n")
                    if precision < 0.70:
                        f.write(f"    → Precision too low - may need higher threshold or better features\n")
                f.write("\n")

            # Implementation guide
            f.write("-"*80 + "\n")
            f.write("IMPLEMENTATION GUIDE\n")
            f.write("-"*80 + "\n\n")

            f.write("To use these per-class thresholds in your inference:\n\n")
            f.write("1. Update scripts/transcribe.py:\n\n")
            f.write("```python\n")
            f.write("# Per-class optimized thresholds\n")
            f.write("class_thresholds = {\n")
            for class_idx, res in optimized_results.items():
                f.write(f"    {class_idx}: {res['threshold']:.2f},  # {res['name']}\n")
            f.write("}\n\n")
            f.write("# Apply per-class thresholds\n")
            f.write("predictions_binary = torch.zeros_like(predictions)\n")
            f.write("for class_idx, threshold in class_thresholds.items():\n")
            f.write("    predictions_binary[:, :, class_idx] = (predictions[:, :, class_idx] >= threshold)\n")
            f.write("```\n\n")

            f.write("2. Or create a config file (per_class_thresholds.yaml):\n\n")
            f.write("```yaml\n")
            f.write("class_thresholds:\n")
            for class_idx, res in optimized_results.items():
                f.write(f"  {res['name']}: {res['threshold']:.2f}\n")
            f.write("```\n\n")

        print(f"\nReport saved to: {report_path}")

        # Save thresholds as YAML config
        config_path = output_dir / "optimized_thresholds.yaml"
        with open(config_path, 'w') as f:
            f.write("# Per-class optimized thresholds for rhythm game\n")
            f.write("# Generated by optimize_per_class_thresholds.py\n\n")
            f.write("class_thresholds:\n")
            for class_idx, res in optimized_results.items():
                f.write(f"  {res['name']}: {res['threshold']:.3f}  # recall={res['recall']:.1%}, precision={res['precision']:.1%}\n")

        print(f"Config saved to: {config_path}")

        return report_path

    def plot_optimization_results(self, optimized_results, output_dir):
        """Create visualization of optimized thresholds."""
        output_dir = Path(output_dir)

        # Prepare data
        names = [optimized_results[i]['name'] for i in range(self.n_classes)]
        thresholds = [optimized_results[i]['threshold'] for i in range(self.n_classes)]
        recalls = [optimized_results[i]['recall'] for i in range(self.n_classes)]
        precisions = [optimized_results[i]['precision'] for i in range(self.n_classes)]
        is_priority = [optimized_results[i]['is_priority'] for i in range(self.n_classes)]

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Thresholds
        ax = axes[0]
        colors = ['#2ecc71' if p else '#95a5a6' for p in is_priority]
        bars = ax.barh(names, thresholds, color=colors, alpha=0.7)
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Baseline (0.5)')
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_title('Optimized Per-Class Thresholds', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        # Add threshold values on bars
        for i, (bar, val) in enumerate(zip(bars, thresholds)):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=9)

        # Plot 2: Recall vs Precision
        ax = axes[1]
        for i, (name, recall, precision, priority) in enumerate(zip(names, recalls, precisions, is_priority)):
            marker = 'o' if priority else 's'
            color = '#2ecc71' if priority else '#95a5a6'
            size = 150 if priority else 100
            ax.scatter(recall, precision, marker=marker, s=size, color=color, alpha=0.7,
                      label=name if i < 5 else '')  # Only label first few to avoid clutter

        # Target box for priority classes
        ax.axvline(x=0.85, color='green', linestyle='--', alpha=0.5, label='Target Recall (85%)')
        ax.axhline(y=0.70, color='blue', linestyle='--', alpha=0.5, label='Target Precision (70%)')
        ax.fill_between([0.85, 1.0], 0.70, 1.0, alpha=0.1, color='green', label='Target Zone')

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Per-Class Performance (Priority=circles, Other=squares)',
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        plot_path = output_dir / "per_class_optimization.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Plots saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Optimize per-class thresholds for rhythm game')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint (default: latest in checkpoints/)')
    parser.add_argument('--config', type=str, default='configs/full_training_config.yaml',
                       help='Path to config file')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--output-dir', type=str, default='per_class_threshold_results',
                       help='Output directory for results')
    parser.add_argument('--strategy', type=str, default='rhythm_game',
                       choices=['max_f1', 'max_recall', 'balanced', 'rhythm_game'],
                       help='Optimization strategy')
    parser.add_argument('--min-precision', type=float, default=0.60,
                       help='Minimum acceptable precision (default: 0.60)')

    args = parser.parse_args()

    # Load config
    print("Loading configuration...")
    config = load_config(args.config)

    # Find checkpoint if not specified
    if args.checkpoint is None:
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        checkpoints = sorted(checkpoint_dir.glob('*.ckpt'))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        args.checkpoint = str(checkpoints[-1])
        print(f"Using latest checkpoint: {args.checkpoint}")

    # Load model
    print("Loading model...")
    model = DrumTranscriptionCRNN.load_from_checkpoint(
        args.checkpoint,
        config=config
    )
    model.eval()

    # Setup data
    print("Setting up data...")
    data_module = EGMDDataModule(config)
    data_module.setup('fit')

    if args.split == 'val':
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()

    class_names = config['data']['class_names']

    # Initialize optimizer
    optimizer = PerClassThresholdOptimizer(
        model=model,
        dataloader=dataloader,
        class_names=class_names
    )

    # Collect predictions
    optimizer.collect_predictions()

    # Optimize thresholds
    optimized_results = optimizer.optimize_all_classes(
        strategy=args.strategy,
        min_precision=args.min_precision
    )

    # Compare baseline vs optimized
    baseline_metrics, optimized_metrics = optimizer.compare_baseline_vs_optimized(
        optimized_results,
        baseline_threshold=0.5
    )

    # Generate report
    output_dir = Path(args.output_dir)
    optimizer.generate_report(optimized_results, output_dir)

    # Create plots
    optimizer.plot_optimization_results(optimized_results, output_dir)

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review the report: cat per_class_threshold_results/per_class_optimization_report.txt")
    print("2. Check the plots: open per_class_threshold_results/per_class_optimization.png")
    print("3. Use optimized_thresholds.yaml in your inference pipeline")
    print("\nIf priority classes don't meet targets (>85% recall, >70% precision),")
    print("you may need to consider model retraining or class simplification.")


if __name__ == '__main__':
    main()

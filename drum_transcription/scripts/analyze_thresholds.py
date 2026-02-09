"""
Threshold Analysis Script for Drum Transcription Model

Tests model performance across multiple thresholds to find optimal operating points.
Helps determine if poor recall is due to suboptimal threshold vs. model quality.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.crnn import DrumTranscriptionCRNN
from src.data.data_module import EGMDDataModule
from src.utils.config import load_config


class ThresholdAnalyzer:
    """Analyzes model performance across different thresholds."""

    def __init__(self, model, dataloader, class_names):
        """
        Initialize analyzer.

        Args:
            model: Trained model
            dataloader: Validation or test dataloader
            class_names: List of drum class names
        """
        self.model = model
        self.dataloader = dataloader
        self.class_names = class_names
        self.n_classes = len(class_names)

        # Core classes for rhythm game (kick, snare, hihat/ride)
        self.core_class_indices = self._identify_core_classes()

    def _identify_core_classes(self):
        """Identify core drum classes for rhythm game."""
        core_keywords = ['kick', 'snare', 'hihat', 'ride']
        core_indices = []

        for idx, name in enumerate(self.class_names):
            if any(keyword in name.lower() for keyword in core_keywords):
                core_indices.append(idx)

        return core_indices

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
                predictions = self.model(specs)  # (batch, time, n_classes) - logits

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

    def compute_metrics_at_threshold(self, threshold, class_indices=None):
        """
        Compute metrics at a specific threshold.

        Args:
            threshold: Threshold value
            class_indices: If provided, only compute metrics for these classes

        Returns:
            Dictionary of metrics
        """
        if class_indices is None:
            class_indices = range(self.n_classes)

        # Binarize predictions
        pred_binary = (self.predictions[:, class_indices] > threshold).astype(float)
        true_binary = self.labels[:, class_indices]

        # Compute metrics
        tp = ((pred_binary == 1) & (true_binary == 1)).sum()
        fp = ((pred_binary == 1) & (true_binary == 0)).sum()
        fn = ((pred_binary == 0) & (true_binary == 1)).sum()
        tn = ((pred_binary == 0) & (true_binary == 0)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }

    def compute_per_class_metrics(self, threshold):
        """Compute metrics for each class at a threshold."""
        per_class_metrics = []

        for class_idx in range(self.n_classes):
            y_true = self.labels[:, class_idx]
            y_pred = (self.predictions[:, class_idx] > threshold).astype(float)

            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            per_class_metrics.append({
                'class': self.class_names[class_idx],
                'class_idx': class_idx,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(y_true.sum())
            })

        return per_class_metrics

    def analyze_threshold_range(self, thresholds):
        """
        Analyze performance across a range of thresholds.

        Args:
            thresholds: List of threshold values to test

        Returns:
            DataFrame with results
        """
        print(f"\nAnalyzing {len(thresholds)} thresholds...")

        results = []

        for threshold in tqdm(thresholds, desc="Testing thresholds"):
            # Overall metrics
            overall_metrics = self.compute_metrics_at_threshold(threshold)
            overall_metrics['subset'] = 'all_classes'
            results.append(overall_metrics)

            # Core classes only
            if self.core_class_indices:
                core_metrics = self.compute_metrics_at_threshold(threshold, self.core_class_indices)
                core_metrics['subset'] = 'core_classes'
                results.append(core_metrics)

        return pd.DataFrame(results)

    def find_optimal_thresholds(self, results_df):
        """Find optimal thresholds for different objectives."""
        all_class_results = results_df[results_df['subset'] == 'all_classes']

        optimal = {
            'max_f1': {
                'threshold': all_class_results.loc[all_class_results['f1'].idxmax(), 'threshold'],
                'f1': all_class_results['f1'].max(),
                'precision': all_class_results.loc[all_class_results['f1'].idxmax(), 'precision'],
                'recall': all_class_results.loc[all_class_results['f1'].idxmax(), 'recall']
            },
            'max_recall': {
                'threshold': all_class_results.loc[all_class_results['recall'].idxmax(), 'threshold'],
                'recall': all_class_results['recall'].max(),
                'precision': all_class_results.loc[all_class_results['recall'].idxmax(), 'precision'],
                'f1': all_class_results.loc[all_class_results['recall'].idxmax(), 'f1']
            },
            'balanced': {  # F1 closest to average of precision and recall
                'threshold': all_class_results.loc[
                    (all_class_results['precision'] - all_class_results['recall']).abs().idxmin(),
                    'threshold'
                ],
            }
        }

        # Add metrics for balanced threshold
        balanced_row = all_class_results[
            all_class_results['threshold'] == optimal['balanced']['threshold']
        ].iloc[0]
        optimal['balanced']['precision'] = balanced_row['precision']
        optimal['balanced']['recall'] = balanced_row['recall']
        optimal['balanced']['f1'] = balanced_row['f1']

        return optimal

    def plot_threshold_curves(self, results_df, save_path=None):
        """Plot precision/recall/F1 curves across thresholds."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: All classes
        all_class_results = results_df[results_df['subset'] == 'all_classes']
        ax = axes[0]
        ax.plot(all_class_results['threshold'], all_class_results['precision'],
                label='Precision', marker='o', markersize=4)
        ax.plot(all_class_results['threshold'], all_class_results['recall'],
                label='Recall', marker='s', markersize=4)
        ax.plot(all_class_results['threshold'], all_class_results['f1'],
                label='F1', marker='^', markersize=4, linewidth=2)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('All Classes Performance vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Current (0.5)')

        # Plot 2: Core classes vs All classes F1
        ax = axes[1]
        ax.plot(all_class_results['threshold'], all_class_results['f1'],
                label='All Classes', marker='o', markersize=4, linewidth=2)

        if 'core_classes' in results_df['subset'].values:
            core_class_results = results_df[results_df['subset'] == 'core_classes']
            ax.plot(core_class_results['threshold'], core_class_results['f1'],
                    label='Core Classes (Kick/Snare/HH/Ride)', marker='s', markersize=4, linewidth=2)

        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score: Core vs All Classes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Current (0.5)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig

    def generate_report(self, results_df, optimal_thresholds, output_path=None):
        """Generate a comprehensive text report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("THRESHOLD ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Dataset info
        report_lines.append(f"Dataset: {self.predictions.shape[0]:,} frames, {self.n_classes} classes")
        report_lines.append(f"Core classes (indices): {self.core_class_indices}")
        report_lines.append(f"Core classes (names): {[self.class_names[i] for i in self.core_class_indices]}")
        report_lines.append("")

        # Current performance (threshold=0.5)
        report_lines.append("-" * 80)
        report_lines.append("CURRENT PERFORMANCE (threshold=0.5)")
        report_lines.append("-" * 80)
        current = results_df[(results_df['threshold'] == 0.5) & (results_df['subset'] == 'all_classes')].iloc[0]
        report_lines.append(f"Precision: {current['precision']:.4f}")
        report_lines.append(f"Recall:    {current['recall']:.4f}")
        report_lines.append(f"F1 Score:  {current['f1']:.4f}")
        report_lines.append("")

        # Optimal thresholds
        report_lines.append("-" * 80)
        report_lines.append("OPTIMAL THRESHOLDS")
        report_lines.append("-" * 80)

        for objective, metrics in optimal_thresholds.items():
            report_lines.append(f"\n{objective.upper().replace('_', ' ')}:")
            report_lines.append(f"  Threshold:  {metrics['threshold']:.2f}")
            report_lines.append(f"  Precision:  {metrics['precision']:.4f}")
            report_lines.append(f"  Recall:     {metrics['recall']:.4f}")
            report_lines.append(f"  F1 Score:   {metrics['f1']:.4f}")

        # Per-class analysis at current threshold
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("PER-CLASS PERFORMANCE (threshold=0.5)")
        report_lines.append("-" * 80)
        per_class = self.compute_per_class_metrics(0.5)

        report_lines.append(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        report_lines.append("-" * 65)
        for metrics in per_class:
            is_core = "(*)" if metrics['class_idx'] in self.core_class_indices else "   "
            report_lines.append(
                f"{metrics['class']:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                f"{metrics['f1']:>10.4f} {metrics['support']:>10} {is_core}"
            )
        report_lines.append("")
        report_lines.append("(*) = Core class for rhythm game")

        # Recommendations
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 80)

        best_f1_threshold = optimal_thresholds['max_f1']['threshold']
        current_f1 = current['f1']
        best_f1 = optimal_thresholds['max_f1']['f1']
        improvement = ((best_f1 - current_f1) / current_f1) * 100

        if best_f1_threshold != 0.5:
            report_lines.append(f"\n1. THRESHOLD ADJUSTMENT:")
            report_lines.append(f"   Changing threshold from 0.5 to {best_f1_threshold:.2f} would improve:")
            report_lines.append(f"   - F1 Score: {current_f1:.4f} → {best_f1:.4f} (+{improvement:.1f}%)")
            report_lines.append(f"   - Recall: {current['recall']:.4f} → {optimal_thresholds['max_f1']['recall']:.4f}")
        else:
            report_lines.append(f"\n1. THRESHOLD OPTIMIZATION:")
            report_lines.append(f"   Current threshold (0.5) is already optimal for F1 score.")

        # Check if core classes would benefit from different threshold
        if self.core_class_indices:
            core_results = results_df[results_df['subset'] == 'core_classes']
            best_core_f1 = core_results['f1'].max()
            best_core_threshold = core_results.loc[core_results['f1'].idxmax(), 'threshold']
            current_core_f1 = core_results[core_results['threshold'] == 0.5].iloc[0]['f1']

            report_lines.append(f"\n2. CORE CLASSES OPTIMIZATION:")
            report_lines.append(f"   Core classes would perform best at threshold {best_core_threshold:.2f}:")
            report_lines.append(f"   - Core F1: {current_core_f1:.4f} → {best_core_f1:.4f}")

        report_lines.append("")
        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Saved report to {output_path}")

        return report


def main():
    parser = argparse.ArgumentParser(description='Analyze model performance across thresholds')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='threshold_analysis',
        help='Output directory for results'
    )
    parser.add_argument(
        '--thresholds',
        type=str,
        default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
        help='Comma-separated list of thresholds to test'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Which dataset split to analyze'
    )

    args = parser.parse_args()

    # Parse thresholds
    thresholds = [float(t) for t in args.thresholds.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Thresholds: {thresholds}")
    print(f"Split: {args.split}")
    print(f"Output: {output_dir}")
    print()

    # Load config
    config = load_config(args.config)

    # Initialize DataModule
    print("Loading dataset...")
    data_module = EGMDDataModule(
        processed_root=config.data.processed_root,
        splits_dir=config.data.splits_dir,
        batch_size=config.training.batch_size,
        num_workers=config.hardware.num_workers,
        use_hdf5=True
    )
    data_module.setup('fit')

    # Get dataloader
    if args.split == 'train':
        dataloader = data_module.train_dataloader()
    elif args.split == 'val':
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = DrumTranscriptionCRNN.load_from_checkpoint(args.checkpoint)
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # Get class names
    class_names = config.drum_midi.drum_names

    # Initialize analyzer
    analyzer = ThresholdAnalyzer(model, dataloader, class_names)

    # Collect predictions
    analyzer.collect_predictions()

    # Analyze thresholds
    results_df = analyzer.analyze_threshold_range(thresholds)

    # Save raw results
    results_csv = output_dir / 'threshold_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved results to {results_csv}")

    # Find optimal thresholds
    optimal_thresholds = analyzer.find_optimal_thresholds(results_df)

    # Generate plots
    plot_path = output_dir / 'threshold_curves.png'
    analyzer.plot_threshold_curves(results_df, save_path=plot_path)

    # Generate report
    report_path = output_dir / 'analysis_report.txt'
    report = analyzer.generate_report(results_df, optimal_thresholds, output_path=report_path)

    # Print report to console
    print("\n")
    print(report)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

"""Evaluation script for detailed model assessment."""

import argparse
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import json

import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.crnn import DrumTranscriptionCRNN
from src.data.data_module import EGMDDataModule
from src.data.midi_processing import get_drum_names
from src.utils.config import load_config


def evaluate_model(
    checkpoint_path: str,
    config_path: str = "configs/default_config.yaml",
    split: str = "test",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    threshold: float = 0.5
):
    """
    Evaluate model on test set with detailed per-class metrics.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        split: Which split to evaluate ('val' or 'test')
        device: Device to run on
        threshold: Threshold for binary predictions
    """
    # Load config
    config = load_config(config_path)
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = DrumTranscriptionCRNN.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    
    # Load data
    print(f"Loading {split} data...")
    data_module = EGMDDataModule(
        processed_root=config.data.processed_root,
        splits_dir=config.data.splits_dir,
        batch_size=config.training.batch_size,
        num_workers=config.hardware.num_workers,
        use_hdf5=True,
        augmentation_config=None  # No augmentation for evaluation
    )
    data_module.setup('test' if split == 'test' else 'fit')
    
    if split == 'test':
        dataloader = data_module.test_dataloader()
    else:
        dataloader = data_module.val_dataloader()
    
    # Initialize metrics
    drum_names = get_drum_names()
    n_classes = len(drum_names)
    
    per_class_tp = np.zeros(n_classes)
    per_class_fp = np.zeros(n_classes)
    per_class_fn = np.zeros(n_classes)
    
    total_loss = 0
    n_batches = 0
    
    # Evaluate
    print(f"Evaluating on {split} set...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            specs, labels, lengths = batch
            specs = specs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            predictions = model(specs)
            
            # Binarize predictions
            pred_binary = (predictions > threshold).float()
            
            # Compute metrics for each sample
            batch_size = specs.size(0)
            for i in range(batch_size):
                length = lengths[i]
                pred = pred_binary[i, :length].cpu().numpy()
                label = labels[i, :length].cpu().numpy()
                
                # Per-class metrics
                for c in range(n_classes):
                    tp = ((pred[:, c] == 1) & (label[:, c] == 1)).sum()
                    fp = ((pred[:, c] == 1) & (label[:, c] == 0)).sum()
                    fn = ((pred[:, c] == 0) & (label[:, c] == 1)).sum()
                    
                    per_class_tp[c] += tp
                    per_class_fp[c] += fp
                    per_class_fn[c] += fn
            
            # Loss
            loss = model._compute_masked_loss(predictions, labels, lengths)
            total_loss += loss.item()
            n_batches += 1
    
    # Compute overall metrics
    avg_loss = total_loss / n_batches
    
    # Compute per-class precision, recall, F1
    print(f"\n{'='*70}")
    print(f"Evaluation Results ({split} set)")
    print(f"{'='*70}")
    print(f"\nAverage Loss: {avg_loss:.4f}")
    print(f"\nPer-Class Metrics (threshold={threshold}):")
    print(f"{'='*70}")
    print(f"{'Drum':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"{'-'*70}")
    
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    
    results = {}
    
    for i, drum_name in enumerate(drum_names):
        tp = per_class_tp[i]
        fp = per_class_fp[i]
        fn = per_class_fn[i]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        
        print(f"{drum_name:<12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {int(support):>10d}")
        
        results[drum_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(support)
        }
        
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
    
    # Overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"{'-'*70}")
    print(f"{'Overall':<12} {overall_precision:>10.4f} {overall_recall:>10.4f} {overall_f1:>10.4f} {int(overall_tp + overall_fn):>10d}")
    print(f"{'='*70}")
    
    results['overall'] = {
        'precision': float(overall_precision),
        'recall': float(overall_recall),
        'f1': float(overall_f1),
        'loss': float(avg_loss)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate drum transcription model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Which split to evaluate'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary predictions'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results JSON'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    
    args = parser.parse_args()
    
    # Evaluate
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        device=args.device,
        threshold=args.threshold
    )
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

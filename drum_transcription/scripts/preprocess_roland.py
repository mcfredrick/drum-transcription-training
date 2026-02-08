#!/usr/bin/env python3
"""
Preprocess E-GMD dataset with Roland TD-17 mapping.
This script creates a new processed dataset using the 26-class Roland mapping.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import h5py
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import random
from typing import Optional, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.audio_processing import extract_log_mel_spectrogram
from src.data.midi_processing import midi_to_frame_labels
from src.utils.config import load_config


def process_file(
    file_path: str,
    egmd_root: Path,
    output_root: Path,
    config,
    use_hdf5: bool = True,
    force: bool = False
):
    """
    Process a single audio/MIDI file pair using Roland TD-17 mapping.
    
    Args:
        file_path: Path to audio file (relative to egmd_root)
        egmd_root: Root directory of E-GMD dataset
        output_root: Root directory for processed output
        config: Configuration object
        use_hdf5: Save as HDF5 (faster) or .npy files
        force: Force reprocessing even if output exists
    """
    try:
        # CHECK IF FILE ALREADY EXISTS (skip if not forcing)
        if use_hdf5:
            output_file_path = output_root / file_path
            output_h5 = output_file_path.with_suffix('.h5')
            if output_h5.exists() and not force:
                return True  # Skip already processed files
        
        # Full paths
        audio_path = egmd_root / file_path
        midi_path = audio_path.with_suffix('.midi')
        # Try .mid if .midi doesn't exist
        if not midi_path.exists():
            midi_path = audio_path.with_suffix('.mid')
        
        # Check if files exist
        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            return False
        if not midi_path.exists():
            print(f"Warning: MIDI file not found: {midi_path}")
            return False
        
        # Extract spectrogram
        spec = extract_log_mel_spectrogram(
            str(audio_path),
            sr=config.audio.sample_rate,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            n_mels=config.audio.n_mels,
            fmin=config.audio.fmin,
            fmax=config.audio.fmax
        )
        
        # Extract labels from MIDI using Roland mapping
        num_frames = spec.shape[1]
        drum_mapping = config.roland_midi.midi_to_class.to_dict()  # Use Roland mapping from config
        labels = midi_to_frame_labels(
            str(midi_path),
            num_frames=num_frames,
            drum_mapping=drum_mapping,
            hop_length=config.audio.hop_length,
            sr=config.audio.sample_rate
        )
        
        # Create output directory
        output_file_path = output_root / file_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        if use_hdf5:
            # Save as HDF5 (more efficient)
            output_h5 = output_file_path.with_suffix('.h5')
            with h5py.File(output_h5, 'w') as f:
                f.create_dataset('spectrogram', data=spec, compression='gzip')
                f.create_dataset('labels', data=labels, compression='gzip')
        else:
            # Save as separate .npy files
            spec_output = output_file_path.with_suffix('.spec.npy')
            labels_output = output_file_path.with_suffix('.labels.npy')
            np.save(spec_output, spec)
            np.save(labels_output, labels)
        
        return True
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def create_dataset_splits(
    egmd_root: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_files: Optional[int] = None
):
    """
    Create train/val/test splits and save to text files.
    
    Args:
        egmd_root: Root directory of E-GMD dataset
        output_dir: Directory to save split files
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility
        max_files: Maximum number of files to use (for testing)
    """
    # Find all audio files
    audio_files = list(egmd_root.rglob("*.wav"))
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    # Convert to relative paths
    audio_files = [f.relative_to(egmd_root) for f in audio_files]
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(audio_files)
    
    # Calculate split sizes
    n_files = len(audio_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    n_test = n_files - n_train - n_val
    
    # Split files
    train_files = audio_files[:n_train]
    val_files = audio_files[n_train:n_train + n_val]
    test_files = audio_files[n_train + n_val:]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save split files
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_file = output_dir / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for file_path in files:
                f.write(str(file_path) + '\n')
    
    print(f"Created splits:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val: {len(val_files)} files")
    print(f"  Test: {len(test_files)} files")
    
    return train_files, val_files, test_files


def main():
    """Main function to preprocess with Roland mapping."""
    parser = argparse.ArgumentParser(description='Preprocess E-GMD dataset with Roland TD-17 mapping')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/roland_config.yaml',
        help='Path to Roland config file'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--use-hdf5',
        action='store_true',
        default=True,
        help='Save as HDF5 format (faster loading)'
    )
    parser.add_argument(
        '--create-splits-only',
        action='store_true',
        help='Only create splits, do not preprocess'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing of already processed files'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("E-GMD Dataset Preprocessing - Roland TD-17 Mapping (26 classes)")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Workers: {args.num_workers}")
    print(f"HDF5: {args.use_hdf5}")
    print(f"Force: {args.force}")
    print("="*80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup paths
    egmd_root = Path(config.data.egmd_root)
    output_root = Path(config.data.processed_root)
    splits_dir = Path(config.data.splits_dir)
    
    print(f"E-GMD root: {egmd_root}")
    print(f"Output root: {output_root}")
    print(f"Splits dir: {splits_dir}")
    print(f"Number of classes: {config.model.n_classes}")
    print("="*80)
    
    # Create splits
    if args.create_splits_only or not (splits_dir / "train.txt").exists():
        print("Creating dataset splits...")
        train_files, val_files, test_files = create_dataset_splits(
            egmd_root=egmd_root,
            output_dir=splits_dir,
            train_ratio=config.data.train_split,
            val_ratio=config.data.val_split,
            test_ratio=config.data.test_split,
            seed=config.data.random_seed
        )
    else:
        print("Loading existing splits...")
        train_files = [Path(line.strip()) for line in open(splits_dir / "train.txt")]
        val_files = [Path(line.strip()) for line in open(splits_dir / "val.txt")]
        test_files = [Path(line.strip()) for line in open(splits_dir / "test.txt")]
    
    if args.create_splits_only:
        print("Splits created successfully!")
        return
    
    # Process all files
    all_files = train_files + val_files + test_files
    print(f"Processing {len(all_files)} files...")
    
    # Setup multiprocessing
    process_func = partial(
        process_file,
        egmd_root=egmd_root,
        output_root=output_root,
        config=config,
        use_hdf5=args.use_hdf5,
        force=args.force
    )
    
    # Process files in parallel
    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, all_files),
            total=len(all_files),
            desc="Processing files"
        ))
    
    # Count successes
    successful = sum(results)
    failed = len(results) - successful
    
    print("\n" + "="*80)
    print("Roland preprocessing completed!")
    print("="*80)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_root}")
    print(f"Number of classes: {config.model.n_classes}")
    print("Next steps:")
    print("1. Verify the processed dataset has the correct number of classes")
    print("2. Run: uv run python train_with_optuna.py")
    print("3. Monitor training with wandb")


if __name__ == "__main__":
    main()

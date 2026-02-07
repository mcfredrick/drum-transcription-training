"""Preprocess E-GMD dataset: convert audio to spectrograms and MIDI to labels."""

import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import sys
import random
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.audio_processing import extract_log_mel_spectrogram
from src.data.midi_processing import midi_to_frame_labels, get_egmd_drum_mapping
from src.utils.config import load_config


def process_single_file(
    file_path: Path,
    egmd_root: Path,
    output_root: Path,
    config,
    use_hdf5: bool = True,
    force: bool = False
):
    """
    Process a single audio/MIDI file pair.
    
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
        
        # Extract labels from MIDI
        num_frames = spec.shape[1]
        drum_mapping = get_egmd_drum_mapping()
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
    print("\nCreating dataset splits...")
    
    # Find all audio files
    audio_files = sorted(egmd_root.rglob('*.wav'))
    
    # Convert to relative paths
    relative_paths = [f.relative_to(egmd_root) for f in audio_files]
    
    print(f"Found {len(relative_paths)} audio files")
    
    # Limit files if max_files is set (for quick testing)
    if max_files is not None and max_files > 0:
        relative_paths = relative_paths[:max_files]
        print(f"Limited to {len(relative_paths)} files for quick test")
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(relative_paths)
    
    # Split
    n_total = len(relative_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = relative_paths[:n_train]
    val_files = relative_paths[n_train:n_train + n_val]
    test_files = relative_paths[n_train + n_val:]
    
    print(f"Split sizes: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'train_split.txt', 'w') as f:
        for path in train_files:
            f.write(f"{path}\n")
    
    with open(output_dir / 'val_split.txt', 'w') as f:
        for path in val_files:
            f.write(f"{path}\n")
    
    with open(output_dir / 'test_split.txt', 'w') as f:
        for path in test_files:
            f.write(f"{path}\n")
    
    print(f"Splits saved to {output_dir}")
    
    return train_files, val_files, test_files


def main():
    parser = argparse.ArgumentParser(description='Preprocess E-GMD dataset')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to config file'
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
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Paths
    egmd_root = Path(config.data.egmd_root)
    output_root = Path(config.data.processed_root)
    splits_dir = Path(config.data.splits_dir)
    
    # Check if E-GMD exists
    if not egmd_root.exists():
        print(f"Error: E-GMD root directory not found: {egmd_root}")
        print("Please download E-GMD dataset first.")
        return
    
    # Create splits
    max_files = getattr(config.data, 'max_files', None)
    train_files, val_files, test_files = create_dataset_splits(
        egmd_root=egmd_root,
        output_dir=splits_dir,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split,
        seed=config.data.random_seed,
        max_files=max_files
    )
    
    if args.create_splits_only:
        print("Splits created. Exiting (--create-splits-only flag set).")
        return
    
    # Get all files to process
    all_files = train_files + val_files + test_files
    
    print(f"\nPreprocessing {len(all_files)} files...")
    print(f"Output directory: {output_root}")
    print(f"Using HDF5: {args.use_hdf5}")
    print(f"Workers: {args.num_workers}")
    
    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Process files
    if args.num_workers > 1:
        # Parallel processing
        process_fn = partial(
            process_single_file,
            egmd_root=egmd_root,
            output_root=output_root,
            config=config,
            use_hdf5=args.use_hdf5,
            force=args.force
        )
        
        with mp.Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_fn, all_files),
                total=len(all_files),
                desc="Processing files"
            ))
    else:
        # Serial processing
        results = []
        for file_path in tqdm(all_files, desc="Processing files"):
            result = process_single_file(
                file_path,
                egmd_root,
                output_root,
                config,
                args.use_hdf5,
                args.force
            )
            results.append(result)
    
    # Summary
    n_success = sum(results)
    n_failed = len(results) - n_success
    
    print(f"\nPreprocessing complete!")
    print(f"Success: {n_success}/{len(all_files)}")
    print(f"Failed: {n_failed}/{len(all_files)}")
    
    if n_failed > 0:
        print("\nSome files failed to process. Check warnings above.")
        print("Updating split files to exclude failed files...")
        
        # Filter out failed files from each split
        successful_files = [f for f, success in zip(all_files, results) if success]
        successful_set = set(successful_files)
        
        # Update train split
        train_files_filtered = [f for f in train_files if f in successful_set]
        with open(splits_dir / 'train_split.txt', 'w') as f:
            for path in train_files_filtered:
                f.write(f"{path}\n")
        
        # Update val split
        val_files_filtered = [f for f in val_files if f in successful_set]
        with open(splits_dir / 'val_split.txt', 'w') as f:
            for path in val_files_filtered:
                f.write(f"{path}\n")
        
        # Update test split
        test_files_filtered = [f for f in test_files if f in successful_set]
        with open(splits_dir / 'test_split.txt', 'w') as f:
            for path in test_files_filtered:
                f.write(f"{path}\n")
        
        print(f"Updated splits: Train={len(train_files_filtered)}, Val={len(val_files_filtered)}, Test={len(test_files_filtered)}")
        print(f"Removed {n_failed} failed files from splits.")


if __name__ == "__main__":
    main()

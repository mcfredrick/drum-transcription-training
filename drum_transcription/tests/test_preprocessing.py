#!/usr/bin/env python3
"""Test Roland preprocessing on a single file to verify 26-class output."""

import sys
from pathlib import Path
import h5py
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data.audio_processing import extract_log_mel_spectrogram
from src.data.midi_processing import midi_to_frame_labels
from src.utils.config import load_config

def main():
    print("Testing Roland preprocessing on single file...")
    
    # Load Roland config
    config = load_config('configs/roland_config.yaml')
    print(f"Config loaded: {config.model.n_classes} classes")
    
    # Setup paths
    egmd_root = Path(config.data.egmd_root)
    
    # Find first audio file
    audio_files = list(egmd_root.rglob("*.wav"))
    if not audio_files:
        print("No audio files found!")
        return
    
    first_file = audio_files[0]
    relative_path = first_file.relative_to(egmd_root)
    print(f"Testing file: {relative_path}")
    
    # Full paths
    audio_path = egmd_root / relative_path
    midi_path = audio_path.with_suffix('.midi')
    if not midi_path.exists():
        midi_path = audio_path.with_suffix('.mid')
    
    print(f"Audio: {audio_path}")
    print(f"MIDI: {midi_path}")
    
    # Extract spectrogram
    print("\nExtracting spectrogram...")
    spec = extract_log_mel_spectrogram(
        str(audio_path),
        sr=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        n_mels=config.audio.n_mels,
        fmin=config.audio.fmin,
        fmax=config.audio.fmax
    )
    print(f"Spectrogram shape: {spec.shape}")
    
    # Extract labels using Roland mapping
    print("\nExtracting labels with Roland mapping...")
    num_frames = spec.shape[1]
    drum_mapping = config.roland_midi.midi_to_class.to_dict()
    print(f"Roland mapping type: {type(drum_mapping)}")
    print(f"Number of mapping entries: {len(drum_mapping)}")
    print(f"Sample mapping entries: {dict(list(drum_mapping.items())[:5])}")
    
    labels = midi_to_frame_labels(
        str(midi_path),
        num_frames=num_frames,
        drum_mapping=drum_mapping,
        hop_length=config.audio.hop_length,
        sr=config.audio.sample_rate
    )
    
    print(f"Labels shape: {labels.shape}")
    print(f"Number of classes in labels: {labels.shape[1]}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Non-zero frames: {np.any(labels > 0, axis=1).sum()}")
    
    # Show sample of labels
    print(f"\nSample labels (first 5 frames):")
    print(labels[:5])
    
    # Verify class count
    expected_classes = config.model.n_classes
    actual_classes = labels.shape[1]
    
    if actual_classes == expected_classes:
        print(f"\n✅ SUCCESS: Labels have {actual_classes} classes as expected!")
    else:
        print(f"\n❌ ERROR: Expected {expected_classes} classes, got {actual_classes}")
    
    # Test saving to HDF5
    print(f"\nTesting HDF5 save...")
    output_path = Path("/tmp/test_roland.h5")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('spectrogram', data=spec, compression='gzip')
        f.create_dataset('labels', data=labels, compression='gzip')
    
    # Verify saved file
    with h5py.File(output_path, 'r') as f:
        saved_spec = f['spectrogram'][:]
        saved_labels = f['labels'][:]
        print(f"Saved spectrogram shape: {saved_spec.shape}")
        print(f"Saved labels shape: {saved_labels.shape}")
        print(f"Saved labels classes: {saved_labels.shape[1]}")
    
    print(f"\n✅ Test completed successfully!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()

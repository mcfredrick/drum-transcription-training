"""MIDI processing utilities for extracting drum labels."""

import numpy as np
import pretty_midi
from typing import Dict, List, Tuple


def midi_to_frame_labels(
    midi_path: str,
    num_frames: int,
    drum_mapping: Dict[int, int],
    hop_length: int = 512,
    sr: int = 22050,
    onset_tolerance_frames: int = 2
) -> np.ndarray:
    """
    Convert MIDI file to frame-level labels for drum transcription.
    
    Args:
        midi_path: Path to MIDI file
        num_frames: Number of frames in corresponding spectrogram
        drum_mapping: Dictionary mapping MIDI note numbers to drum class indices
        hop_length: Hop length used in spectrogram
        sr: Sample rate
        onset_tolerance_frames: Number of frames before/after onset to mark as positive
        
    Returns:
        Binary label matrix of shape (num_frames, num_classes)
    """
    # Initialize labels (all zeros)
    num_classes = max(drum_mapping.values()) + 1
    labels = np.zeros((num_frames, num_classes), dtype=np.float32)
    
    # Load MIDI file
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Warning: Could not load MIDI file {midi_path}: {e}")
        return labels
    
    # Extract drum notes
    for instrument in midi.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                # Check if this note is in our mapping
                if note.pitch in drum_mapping:
                    # Convert time to frame
                    frame_idx = int(note.start * sr / hop_length)
                    
                    # Get drum class
                    drum_class = drum_mapping[note.pitch]
                    
                    # Mark frames around onset (tolerance window)
                    start_frame = max(0, frame_idx - onset_tolerance_frames)
                    end_frame = min(num_frames, frame_idx + onset_tolerance_frames + 1)
                    
                    labels[start_frame:end_frame, drum_class] = 1.0
    
    return labels


def extract_drum_onsets(
    midi_path: str,
    drum_mapping: Dict[int, int]
) -> List[Tuple[float, int, int]]:
    """
    Extract drum onsets from MIDI file.
    
    Args:
        midi_path: Path to MIDI file
        drum_mapping: Dictionary mapping MIDI note numbers to drum class indices
        
    Returns:
        List of (time, drum_class, velocity) tuples
    """
    onsets = []
    
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Warning: Could not load MIDI file {midi_path}: {e}")
        return onsets
    
    for instrument in midi.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                if note.pitch in drum_mapping:
                    drum_class = drum_mapping[note.pitch]
                    onsets.append((note.start, drum_class, note.velocity))
    
    # Sort by time
    onsets.sort(key=lambda x: x[0])
    
    return onsets


def create_midi_from_onsets(
    onsets: List[Tuple[float, str, int]],
    output_path: str,
    gm_mapping: Dict[str, int],
    tempo: int = 120,
    note_duration: float = 0.1
):
    """
    Create MIDI file from drum onsets.
    
    Args:
        onsets: List of (time, drum_name, velocity) tuples
        output_path: Path to save MIDI file
        gm_mapping: General MIDI drum mapping (drum_name -> MIDI note)
        tempo: Tempo in BPM
        note_duration: Duration of each note in seconds
    """
    # Create MIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Create drum instrument (channel 9)
    drum_program = 0  # Standard drum kit
    drums = pretty_midi.Instrument(program=drum_program, is_drum=True)
    
    # Add notes
    for time, drum_name, velocity in onsets:
        if drum_name in gm_mapping:
            note_number = gm_mapping[drum_name]
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=note_number,
                start=time,
                end=time + note_duration
            )
            drums.notes.append(note)
    
    # Add instrument to MIDI
    midi.instruments.append(drums)
    
    # Write MIDI file
    midi.write(output_path)


def get_drum_mapping() -> Dict[int, int]:
    """
    Get MIDI note to drum class mapping for standard drum kit (12-class system).
    Covers ~99.7% of E-GMD dataset with focus on standard drum kit sounds.

    Returns:
        Dictionary mapping MIDI note numbers to drum class indices (0-11)
    """
    return {
        # Core drums (0-3)
        36: 0,   # Kick
        38: 1,   # Snare head
        40: 2,   # Snare rim
        37: 3,   # Side stick

        # Hi-hats (4-6)
        44: 4,   # Pedal hi-hat
        42: 5,   # Closed hi-hat
        46: 6,   # Open hi-hat

        # Toms (7-8)
        43: 7,   # Floor tom
        48: 8,   # High-mid tom

        # Cymbals (9-11)
        51: 9,   # Ride
        53: 10,  # Ride bell
        49: 11,  # Crash cymbal
    }


def get_drum_name_mapping() -> Dict[str, int]:
    """
    Get drum name to MIDI note mapping for standard drum kit export.

    Returns:
        Dictionary mapping drum names to MIDI note numbers
    """
    return {
        'kick': 36,
        'snare_head': 38,
        'snare_rim': 40,
        'side_stick': 37,
        'hihat_pedal': 44,
        'hihat_closed': 42,
        'hihat_open': 46,
        'floor_tom': 43,
        'high_mid_tom': 48,
        'ride': 51,
        'ride_bell': 53,
        'crash': 49,
    }


def get_drum_names() -> List[str]:
    """Get list of drum class names for standard drum kit (12 classes)."""
    return [
        'kick', 'snare_head', 'snare_rim', 'side_stick',
        'hihat_pedal', 'hihat_closed', 'hihat_open',
        'floor_tom', 'high_mid_tom',
        'ride', 'ride_bell', 'crash'
    ]


def get_class_weights() -> List[float]:
    """
    Get class weights for handling class imbalance in standard drum kit system.

    Weights are inverse frequency-based, clipped to [0.5, 15.0] range:
    - Common classes (kick, snare_head): 0.5
    - Rare classes (hihat_open, ride_bell, crash): 8.0-15.0
    - Others: 0.8-3.5 based on frequency

    Returns:
        List of 12 class weights for use with BCEWithLogitsLoss pos_weight
    """
    return [
        0.5,   # 0: kick (very common)
        0.5,   # 1: snare_head (very common)
        1.5,   # 2: snare_rim (medium)
        3.5,   # 3: side_stick (less common)
        0.8,   # 4: hihat_pedal (common)
        0.9,   # 5: hihat_closed (common)
        15.0,  # 6: hihat_open (rare)
        1.5,   # 7: floor_tom (medium)
        1.6,   # 8: high_mid_tom (medium)
        1.0,   # 9: ride (medium)
        13.0,  # 10: ride_bell (rare)
        8.0,   # 11: crash (moderately rare accent)
    ]


if __name__ == "__main__":
    # Test MIDI processing with 12-class standard drum kit mapping
    from pathlib import Path

    test_midi = "data/e-gmd/drummer1/session1/1_funk_120_beat_4-4.mid"

    if Path(test_midi).exists():
        print(f"Processing MIDI file: {test_midi}")

        # Get standard drum kit mapping
        drum_mapping = get_drum_mapping()
        print(f"Standard drum kit mapping covers {len(drum_mapping)} MIDI notes (12 classes)")

        # Extract onsets
        onsets = extract_drum_onsets(test_midi, drum_mapping)
        print(f"\nFound {len(onsets)} drum onsets")

        # Show first 10 onsets
        print("\nFirst 10 onsets:")
        drum_names = get_drum_names()
        for i, (time, drum_class, velocity) in enumerate(onsets[:10]):
            print(f"  {i+1}. Time: {time:.3f}s, Drum: {drum_names[drum_class]}, Velocity: {velocity}")

        # Convert to frame labels (assuming 43 FPS)
        num_frames = 1000  # Example
        labels = midi_to_frame_labels(test_midi, num_frames, drum_mapping)
        print(f"\nLabel matrix shape: {labels.shape}")
        print(f"Non-zero frames: {np.count_nonzero(labels.sum(axis=1))}")
        print(f"Number of drum classes: {labels.shape[1]}")

    else:
        print(f"Test MIDI file not found: {test_midi}")
        print("Download E-GMD dataset to test this module.")

        # Show mapping information
        print("\nStandard Drum Kit Mapping (12 Classes):")
        drum_mapping = get_drum_mapping()
        drum_names = get_drum_names()
        for midi_note, class_idx in sorted(drum_mapping.items(), key=lambda x: x[1]):
            print(f"  MIDI {midi_note:2d} -> Class {class_idx:2d}: {drum_names[class_idx]}")

        # Show class weights
        print("\nClass Weights (for handling imbalance):")
        weights = get_class_weights()
        for i, (name, weight) in enumerate(zip(drum_names, weights)):
            print(f"  Class {i:2d} ({name:15s}): {weight:4.1f}")

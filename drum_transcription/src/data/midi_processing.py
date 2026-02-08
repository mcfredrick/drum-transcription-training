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
    Get Roland TD-17 MIDI note to drum class mapping.
    Based on Roland TD-17 default mapping used in E-GMD dataset.
    
    Returns:
        Dictionary mapping MIDI note numbers to drum class indices
        Covers 26 drum classes found in the E-GMD dataset
    """
    return {
        # Core drums (GM compatible)
        36: 0,  # Kick
        38: 1,  # Snare Head
        37: 2,  # Snare X-Stick/Side Stick
        40: 3,  # Snare Rim/Hi Floor Tom
        
        # Toms
        48: 4,  # Tom 1 Head (Hi-Mid Tom)
        50: 5,  # Tom 1 Rim (High Tom)
        45: 6,  # Tom 2 Head (Low Tom)
        47: 7,  # Tom 2 Rim (Low-Mid Tom)
        43: 8,  # Tom 3 Head (High Floor Tom)
        58: 9,  # Tom 3 Rim (Vibraslap)
        41: 10, # Tom 4 Head (Low Floor Tom)
        39: 11, # Tom 4 Rim (Hand Clap)
        
        # Hi-Hats
        42: 12, # Hi-Hat Closed (Bow)
        22: 13, # Hi-Hat Closed (Edge) - Roland specific
        46: 14, # Hi-Hat Open (Bow)
        26: 15, # Hi-Hat Open (Edge) - Roland specific
        44: 16, # Hi-Hat Pedal
        
        # Cymbals
        49: 17, # Crash 1 (Bow)
        55: 18, # Crash 1 (Edge/Splash)
        57: 19, # Crash 2 (Bow)
        52: 20, # Crash 2 (Edge/Chinese)
        51: 21, # Ride (Bow)
        59: 22, # Ride (Edge)
        53: 23, # Ride Bell
        
        # Additional percussion
        54: 24, # Tambourine
        56: 25, # Cowbell
    }


def get_drum_name_mapping() -> Dict[str, int]:
    """
    Get Roland TD-17 drum name to MIDI note mapping for export.
    
    Returns:
        Dictionary mapping drum names to MIDI note numbers
    """
    return {
        'kick': 36,
        'snare_head': 38,
        'snare_xstick': 37,
        'snare_rim': 40,
        'tom1_head': 48,
        'tom1_rim': 50,
        'tom2_head': 45,
        'tom2_rim': 47,
        'tom3_head': 43,
        'tom3_rim': 58,
        'tom4_head': 41,
        'tom4_rim': 39,
        'hihat_closed': 42,
        'hihat_closed_edge': 22,
        'hihat_open': 46,
        'hihat_open_edge': 26,
        'hihat_pedal': 44,
        'crash1_bow': 49,
        'crash1_edge': 55,
        'crash2_bow': 57,
        'crash2_edge': 52,
        'ride_bow': 51,
        'ride_edge': 59,
        'ride_bell': 53,
        'tambourine': 54,
        'cowbell': 56,
    }


def get_drum_names() -> List[str]:
    """Get list of Roland drum class names."""
    return [
        'kick', 'snare_head', 'snare_xstick', 'snare_rim',
        'tom1_head', 'tom1_rim', 'tom2_head', 'tom2_rim',
        'tom3_head', 'tom3_rim', 'tom4_head', 'tom4_rim',
        'hihat_closed', 'hihat_closed_edge', 'hihat_open', 'hihat_open_edge',
        'hihat_pedal', 'crash1_bow', 'crash1_edge', 'crash2_bow', 'crash2_edge',
        'ride_bow', 'ride_edge', 'ride_bell', 'tambourine', 'cowbell'
    ]


if __name__ == "__main__":
    # Test MIDI processing with Roland mapping
    from pathlib import Path
    
    test_midi = "data/e-gmd/drummer1/session1/1_funk_120_beat_4-4.mid"
    
    if Path(test_midi).exists():
        print(f"Processing MIDI file: {test_midi}")
        
        # Get Roland drum mapping
        drum_mapping = get_drum_mapping()
        print(f"Roland mapping covers {len(drum_mapping)} MIDI notes")
        
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
        print("\nRoland TD-17 Drum Mapping:")
        drum_mapping = get_drum_mapping()
        drum_names = get_drum_names()
        for midi_note, class_idx in drum_mapping.items():
            print(f"  MIDI {midi_note:2d} -> Class {class_idx:2d}: {drum_names[class_idx]}")

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


def get_egmd_drum_mapping() -> Dict[int, int]:
    """
    Get E-GMD MIDI note to drum class mapping.
    
    Returns:
        Dictionary mapping MIDI note numbers to drum class indices
        0: kick, 1: snare, 2: hihat, 3: hi_tom, 4: mid_tom, 
        5: low_tom, 6: crash, 7: ride
    """
    return {
        36: 0,  # Kick
        38: 1,  # Snare
        42: 2,  # Closed Hi-Hat
        44: 2,  # Pedal Hi-Hat -> Hi-Hat
        46: 2,  # Open Hi-Hat -> Hi-Hat
        50: 3,  # High Tom
        47: 4,  # Mid Tom (Low-Mid Tom)
        48: 4,  # Mid Tom (Hi-Mid Tom) -> Mid Tom
        45: 5,  # Low Tom
        41: 5,  # Floor Tom -> Low Tom
        43: 5,  # Floor Tom (high) -> Low Tom
        49: 6,  # Crash Cymbal 1
        55: 6,  # Splash Cymbal -> Crash
        57: 6,  # Crash Cymbal 2 -> Crash
        51: 7,  # Ride Cymbal 1
        53: 7,  # Ride Bell -> Ride
        59: 7,  # Ride Cymbal 2 -> Ride
    }


def get_gm_drum_mapping() -> Dict[str, int]:
    """
    Get General MIDI drum mapping for export.
    
    Returns:
        Dictionary mapping drum names to MIDI note numbers
    """
    return {
        'kick': 36,      # Bass Drum 1
        'snare': 38,     # Acoustic Snare
        'hihat': 42,     # Closed Hi-Hat
        'hi_tom': 50,    # High Tom
        'mid_tom': 47,   # Low-Mid Tom
        'low_tom': 45,   # Low Tom
        'crash': 49,     # Crash Cymbal 1
        'ride': 51,      # Ride Cymbal 1
    }


def get_drum_names() -> List[str]:
    """Get list of drum class names."""
    return ['kick', 'snare', 'hihat', 'hi_tom', 'mid_tom', 'low_tom', 'crash', 'ride']


if __name__ == "__main__":
    # Test MIDI processing
    from pathlib import Path
    
    test_midi = "data/e-gmd/drummer1/session1/1_funk_120_beat_4-4.mid"
    
    if Path(test_midi).exists():
        print(f"Processing MIDI file: {test_midi}")
        
        # Get drum mapping
        drum_mapping = get_egmd_drum_mapping()
        
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
        
    else:
        print(f"Test MIDI file not found: {test_midi}")
        print("Download E-GMD dataset to test this module.")

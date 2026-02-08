#!/usr/bin/env python3
"""
Analyze MIDI files to identify rare drum notes in the Roland drum mapping.
"""

import os
import json
from collections import defaultdict, Counter
from pathlib import Path
import mido

# Roland drum mapping (General MIDI drum notes)
ROLAND_DRUM_MAP = {
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    37: "Side Stick",
    38: "Acoustic Snare",
    39: "Hand Clap",
    40: "Electric Snare",
    41: "Low Floor Tom",
    42: "Closed Hi-Hat",
    43: "High Floor Tom",
    44: "Pedal Hi-Hat",
    45: "Low Tom",
    46: "Open Hi-Hat",
    47: "Low-Mid Tom",
    48: "High-Mid Tom",
    49: "Crash Cymbal 1",
    50: "High Tom",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    54: "Tambourine",
    55: "Splash Cymbal",
    56: "Cowbell",
    57: "Crash Cymbal 2",
    58: "Vibraslap",
    59: "Ride Cymbal 2",
    60: "Hi Bongo",
    61: "Low Bongo",
    62: "Mute Hi Conga",
    63: "Open Hi Conga",
    64: "Low Conga",
    65: "High Timbale",
    66: "Low Timbale",
    67: "High Agogo",
    68: "Low Agogo",
    69: "Cabasa",
    70: "Maracas",
    71: "Short Whistle",
    72: "Long Whistle",
    73: "Short Guiro",
    74: "Long Guiro",
    75: "Claves",
    76: "Hi Wood Block",
    77: "Low Wood Block",
    78: "Mute Cuica",
    79: "Open Cuica",
    80: "Mute Triangle",
    81: "Open Triangle",
}

def analyze_midi_file(filepath):
    """Extract drum notes from a MIDI file."""
    notes = []
    try:
        mid = mido.MidiFile(filepath)
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Check if note is in drum range
                    if 35 <= msg.note <= 81:
                        notes.append(msg.note)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return notes

def main():
    dataset_path = Path("/mnt/hdd/drum-tranxn/e-gmd-v1.0.0")
    
    # Collect all drum notes
    note_counts = Counter()
    total_files = 0
    files_with_errors = 0
    
    print(f"Analyzing MIDI files in {dataset_path}...")
    print("=" * 70)
    
    # Find all MIDI files
    midi_files = list(dataset_path.rglob("*.mid")) + list(dataset_path.rglob("*.midi"))
    print(f"Found {len(midi_files)} MIDI files\n")
    
    # Process each file
    for i, midi_file in enumerate(midi_files):
        if (i + 1) % 500 == 0:
            print(f"Processing file {i + 1}/{len(midi_files)}...")
        
        notes = analyze_midi_file(midi_file)
        if notes:
            note_counts.update(notes)
            total_files += 1
        else:
            files_with_errors += 1
    
    print(f"\nProcessed {total_files} files with drum notes")
    print(f"Files with no drum notes or errors: {files_with_errors}\n")
    
    # Calculate statistics
    total_notes = sum(note_counts.values())
    unique_notes = len(note_counts)
    
    print(f"Total drum notes found: {total_notes}")
    print(f"Unique note classes: {unique_notes}\n")
    
    # Sort by frequency
    sorted_notes = note_counts.most_common()
    
    print("=" * 70)
    print(f"{'Rank':<5} {'Note#':<7} {'Count':<10} {'%':<8} {'Instrument Name':<30}")
    print("=" * 70)
    
    for rank, (note, count) in enumerate(sorted_notes, 1):
        percentage = (count / total_notes) * 100
        instrument = ROLAND_DRUM_MAP.get(note, "Unknown")
        print(f"{rank:<5} {note:<7} {count:<10} {percentage:>6.2f}% {instrument:<30}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS FOR CLASS REDUCTION:")
    print("=" * 70)
    
    # Find percentile cutoffs
    cumsum = 0
    cumsum_95 = None
    cumsum_99 = None
    
    for note, count in sorted_notes:
        cumsum += count
        cumsum_pct = (cumsum / total_notes) * 100
        if cumsum_pct >= 95 and cumsum_95 is None:
            cumsum_95 = note
            print(f"\n95% of data comes from top classes down to note {cumsum_95}")
        if cumsum_pct >= 99 and cumsum_99 is None:
            cumsum_99 = note
            print(f"99% of data comes from top classes down to note {cumsum_99}")
    
    # Show rarest notes
    print(f"\nRarest {min(15, len(sorted_notes))} note classes:")
    for rank, (note, count) in enumerate(sorted(sorted_notes, key=lambda x: x[1])[-15:], 1):
        percentage = (count / total_notes) * 100
        instrument = ROLAND_DRUM_MAP.get(note, "Unknown")
        print(f"  {note:3d} - {instrument:30s} : {count:6d} ({percentage:6.3f}%)")
    
    # Suggestions
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    
    # Find notes that make up <0.1% of data
    rare_threshold = 0.001  # 0.1%
    rare_notes = [note for note, count in sorted_notes if (count / total_notes) < rare_threshold]
    
    print(f"\nNotes with <0.1% of data ({len(rare_notes)} classes):")
    if rare_notes:
        print(f"  Note numbers: {sorted(rare_notes)}")
        rare_count = sum(note_counts[n] for n in rare_notes)
        rare_pct = (rare_count / total_notes) * 100
        print(f"  Combined count: {rare_count} ({rare_pct:.3f}%)")
        print(f"  → Consider merging these into \"Other\" category")
    
    # Find notes that make up <1% of data
    very_rare_threshold = 0.01  # 1%
    very_rare_notes = [note for note, count in sorted_notes if (count / total_notes) < very_rare_threshold]
    
    print(f"\nNotes with <1% of data ({len(very_rare_notes)} classes):")
    if very_rare_notes:
        very_rare_count = sum(note_counts[n] for n in very_rare_notes)
        very_rare_pct = (very_rare_count / total_notes) * 100
        print(f"  Combined count: {very_rare_count} ({very_rare_pct:.3f}%)")
    
    # Output to JSON for further processing
    output_data = {
        "total_notes": total_notes,
        "unique_classes": unique_notes,
        "files_processed": total_files,
        "note_frequencies": {str(note): count for note, count in sorted_notes},
        "instrument_names": {str(note): ROLAND_DRUM_MAP.get(note, "Unknown") for note, _ in sorted_notes}
    }
    
    output_file = "/home/matt/Documents/drum-tranxn/drum_note_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()

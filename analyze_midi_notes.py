#!/usr/bin/env python3
"""
Script to analyze all MIDI files in the project and determine:
1. Complete range of MIDI notes used
2. Frequency distribution of each note
3. MIDI mapping format analysis
4. Potential for direct MIDI note prediction vs classification
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import json
import pretty_midi
import numpy as np
from typing import Dict, List, Set, Tuple

def find_all_midi_files(root_dir: Path) -> List[Path]:
    """Find all MIDI files in directory tree."""
    midi_files = []
    for ext in ['*.mid', '*.midi']:
        midi_files.extend(root_dir.rglob(ext))
    return midi_files

def analyze_midi_file(midi_path: Path) -> Dict:
    """Analyze a single MIDI file and extract note information."""
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        analysis = {
            'file_path': str(midi_path),
            'total_notes': 0,
            'drum_notes': [],
            'melodic_notes': [],
            'instruments': [],
            'duration': midi_data.get_end_time(),
            'tempo_changes': len(midi_data._get_tempo_changes()) if hasattr(midi_data, '_get_tempo_changes') else 0
        }
        
        for instrument in midi_data.instruments:
            instrument_info = {
                'program': instrument.program,
                'is_drum': instrument.is_drum,
                'name': instrument.name,
                'note_count': len(instrument.notes)
            }
            analysis['instruments'].append(instrument_info)
            
            for note in instrument.notes:
                note_info = {
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity,
                    'duration': note.end - note.start
                }
                
                if instrument.is_drum:
                    analysis['drum_notes'].append(note_info)
                else:
                    analysis['melodic_notes'].append(note_info)
                
                analysis['total_notes'] += 1
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing {midi_path}: {e}")
        return None

def get_general_midi_drum_mapping() -> Dict[int, str]:
    """Get standard General MIDI drum mapping."""
    return {
        35: 'Acoustic Bass Drum',
        36: 'Bass Drum 1',
        37: 'Side Stick',
        38: 'Acoustic Snare',
        39: 'Hand Clap',
        40: 'Hi Floor Tom',
        41: 'Low Floor Tom',
        42: 'Closed Hi Hat',
        43: 'High Floor Tom',
        44: 'Pedal Hi-Hat',
        45: 'Low Tom',
        46: 'Open Hi-Hat',
        47: 'Low-Mid Tom',
        48: 'Hi-Mid Tom',
        49: 'Crash Cymbal 1',
        50: 'High Tom',
        51: 'Ride Cymbal 1',
        52: 'Chinese Cymbal',
        53: 'Ride Bell',
        54: 'Tambourine',
        55: 'Splash Cymbal',
        56: 'Cowbell',
        57: 'Crash Cymbal 2',
        58: 'Vibraslap',
        59: 'Ride Cymbal 2',
        60: 'Hi Bongo',
        61: 'Low Bongo',
        62: 'Mute Hi Conga',
        63: 'Open Hi Conga',
        64: 'Low Conga',
        65: 'High Timbale',
        66: 'Low Timbale',
        67: 'High Agogo',
        68: 'Low Agogo',
        69: 'Cabasa',
        70: 'Maracas',
        71: 'Short Whistle',
        72: 'Long Whistle',
        73: 'Short Guiro',
        74: 'Long Guiro',
        75: 'Claves',
        76: 'Hi Wood Block',
        77: 'Low Wood Block',
        78: 'Mute Cuica',
        79: 'Open Cuica',
        80: 'Mute Triangle',
        81: 'Open Triangle'
    }

def analyze_all_midi_files(root_dir: Path) -> Dict:
    """Analyze all MIDI files and generate comprehensive report."""
    print(f"Searching for MIDI files in {root_dir}...")
    midi_files = find_all_midi_files(root_dir)
    
    if not midi_files:
        print("No MIDI files found!")
        return {}
    
    print(f"Found {len(midi_files)} MIDI files")
    
    # Initialize counters
    all_drum_notes = Counter()
    all_melodic_notes = Counter()
    drum_note_velocities = defaultdict(list)
    melodic_note_velocities = defaultdict(list)
    drum_note_durations = defaultdict(list)
    melodic_note_durations = defaultdict(list)
    
    file_analyses = []
    total_files = 0
    successful_files = 0
    
    # Analyze each file
    for midi_path in midi_files:
        total_files += 1
        analysis = analyze_midi_file(midi_path)
        
        if analysis:
            successful_files += 1
            file_analyses.append(analysis)
            
            # Count drum notes
            for note in analysis['drum_notes']:
                pitch = note['pitch']
                all_drum_notes[pitch] += 1
                drum_note_velocities[pitch].append(note['velocity'])
                drum_note_durations[pitch].append(note['duration'])
            
            # Count melodic notes
            for note in analysis['melodic_notes']:
                pitch = note['pitch']
                all_melodic_notes[pitch] += 1
                melodic_note_velocities[pitch].append(note['velocity'])
                melodic_note_durations[pitch].append(note['duration'])
    
    # Generate statistics
    gm_drum_mapping = get_general_midi_drum_mapping()
    
    drum_analysis = {}
    if all_drum_notes:
        drum_analysis = {
            'unique_notes': len(all_drum_notes),
            'total_occurrences': sum(all_drum_notes.values()),
            'note_range': [min(all_drum_notes.keys()), max(all_drum_notes.keys())],
            'frequency_distribution': dict(all_drum_notes.most_common()),
            'note_statistics': {}
        }
        
        for pitch, count in all_drum_notes.items():
            drum_analysis['note_statistics'][pitch] = {
                'count': count,
                'percentage': (count / drum_analysis['total_occurrences']) * 100,
                'gm_name': gm_drum_mapping.get(pitch, f'Unknown ({pitch})'),
                'velocity_stats': {
                    'mean': np.mean(drum_note_velocities[pitch]),
                    'std': np.std(drum_note_velocities[pitch]),
                    'min': min(drum_note_velocities[pitch]),
                    'max': max(drum_note_velocities[pitch])
                },
                'duration_stats': {
                    'mean': np.mean(drum_note_durations[pitch]),
                    'std': np.std(drum_note_durations[pitch]),
                    'min': min(drum_note_durations[pitch]),
                    'max': max(drum_note_durations[pitch])
                }
            }
    
    melodic_analysis = {}
    if all_melodic_notes:
        melodic_analysis = {
            'unique_notes': len(all_melodic_notes),
            'total_occurrences': sum(all_melodic_notes.values()),
            'note_range': [min(all_melodic_notes.keys()), max(all_melodic_notes.keys())],
            'frequency_distribution': dict(all_melodic_notes.most_common(20))  # Top 20
        }
    
    # Compare with current E-GMD mapping
    current_egmd_mapping = {
        36: 0,  # Kick
        38: 1,  # Snare
        42: 2,  # Closed Hi-Hat
        44: 2,  # Pedal Hi-Hat -> Hi-Hat
        46: 2,  # Open Hi-Hat -> Hi-Hat
        50: 3,  # High Tom
        47: 4,  # Mid Tom
        48: 4,  # Mid Tom
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
    
    mapping_analysis = {
        'current_egmd_notes': set(current_egmd_mapping.keys()),
        'found_drum_notes': set(all_drum_notes.keys()) if all_drum_notes else set(),
        'additional_notes_found': set(all_drum_notes.keys()) - set(current_egmd_mapping.keys()) if all_drum_notes else set(),
        'missing_egmd_notes': set(current_egmd_mapping.keys()) - set(all_drum_notes.keys()) if all_drum_notes else set(current_egmd_mapping.keys()),
        'coverage_percentage': len(set(all_drum_notes.keys()) & set(current_egmd_mapping.keys())) / len(current_egmd_mapping.keys()) * 100 if all_drum_notes else 0
    }
    
    return {
        'summary': {
            'total_files_found': total_files,
            'successfully_analyzed': successful_files,
            'analysis_success_rate': (successful_files / total_files) * 100 if total_files > 0 else 0
        },
        'drum_analysis': drum_analysis,
        'melodic_analysis': melodic_analysis,
        'mapping_analysis': mapping_analysis,
        'file_details': file_analyses[:5]  # First 5 files for detailed view
    }

def print_analysis_report(analysis: Dict):
    """Print a comprehensive analysis report."""
    print("\n" + "="*80)
    print("MIDI NOTE ANALYSIS REPORT")
    print("="*80)
    
    # Summary
    summary = analysis['summary']
    print(f"\nSUMMARY:")
    print(f"  Files found: {summary['total_files_found']}")
    print(f"  Successfully analyzed: {summary['successfully_analyzed']}")
    print(f"  Success rate: {summary['analysis_success_rate']:.1f}%")
    
    # Drum analysis
    drum = analysis['drum_analysis']
    if drum:
        print(f"\nDRUM ANALYSIS:")
        print(f"  Unique drum notes: {drum['unique_notes']}")
        print(f"  Total drum occurrences: {drum['total_occurrences']}")
        print(f"  Note range: {drum['note_range'][0]} - {drum['note_range'][1]}")
        
        print(f"\n  Top 10 most frequent drum notes:")
        for i, (note, count) in enumerate(list(drum['frequency_distribution'].items())[:10]):
            note_stats = drum['note_statistics'][note]
            print(f"    {i+1}. Note {note} ({note_stats['gm_name']}): {count} times ({note_stats['percentage']:.1f}%)")
    
    # Mapping analysis
    mapping = analysis['mapping_analysis']
    print(f"\nMAPPING ANALYSIS:")
    print(f"  Current E-GMD mapping covers {len(mapping['current_egmd_notes'])} notes")
    print(f"  Found {len(mapping['found_drum_notes'])} unique drum notes in data")
    print(f"  Coverage of current mapping: {mapping['coverage_percentage']:.1f}%")
    
    if mapping['additional_notes_found']:
        print(f"\n  Additional notes found beyond current mapping:")
        for note in sorted(mapping['additional_notes_found']):
            gm_mapping = get_general_midi_drum_mapping()
            print(f"    Note {note}: {gm_mapping.get(note, 'Unknown')}")
    
    if mapping['missing_egmd_notes']:
        print(f"\n  E-GMD mapping notes not found in data:")
        for note in sorted(mapping['missing_egmd_notes']):
            gm_mapping = get_general_midi_drum_mapping()
            print(f"    Note {note}: {gm_mapping.get(note, 'Unknown')}")
    
    # Melodic analysis
    melodic = analysis['melodic_analysis']
    if melodic:
        print(f"\nMELODIC ANALYSIS:")
        print(f"  Unique melodic notes: {melodic['unique_notes']}")
        print(f"  Total melodic occurrences: {melodic['total_occurrences']}")
        print(f"  Note range: {melodic['note_range'][0]} - {melodic['note_range'][1]}")

def main():
    """Main function to run MIDI analysis."""
    # Search in project directory and dataset locations
    search_paths = [
        Path('/home/matt/Documents/drum-tranxn'),
        Path('/mnt/hdd/drum-tranxn'),
        Path('/home/matt/tensorflow_datasets'),
        Path('/home/matt/Documents'),
    ]
    
    all_results = {}
    
    for search_path in search_paths:
        if search_path.exists():
            print(f"\nAnalyzing: {search_path}")
            results = analyze_all_midi_files(search_path)
            if results:
                all_results[str(search_path)] = results
                print_analysis_report(results)
    
    # Save comprehensive results
    if all_results:
        output_file = Path('/home/matt/Documents/drum-tranxn/midi_analysis_results.json')
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
    
    # Provide recommendation
    print(f"\n" + "="*80)
    print("RECOMMENDATION FOR MODEL TRAINING")
    print("="*80)
    
    if any(results['drum_analysis'] for results in all_results.values()):
        print("\n✓ The model CAN be trained to predict actual MIDI notes instead of classes")
        print("  Benefits:")
        print("    - More granular control over drum sounds")
        print("    - Ability to distinguish between similar drums (e.g., different cymbals)")
        print("    - Better compatibility with standard MIDI equipment")
        print("    - More expressive outputs")
        
        print("\n  Implementation approach:")
        print("    1. Modify output layer to predict MIDI note numbers (0-127)")
        print("    2. Use multi-label classification for simultaneous notes")
        print("    3. Consider velocity prediction as additional output")
        print("    4. May need larger model capacity for 128 possible outputs")
    else:
        print("\n⚠ Limited drum data found - may need more MIDI files for comprehensive analysis")

if __name__ == "__main__":
    main()

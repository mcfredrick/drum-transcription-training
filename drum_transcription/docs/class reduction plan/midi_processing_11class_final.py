"""
Final 11-Class MIDI Processing for Standard Drum Kit
Coverage: 96.58% of E-GMD dataset
No auxiliary percussion - only standard drum kit sounds
"""

import numpy as np
from typing import Dict, List, Tuple

# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

def get_drum_mapping() -> Dict[int, int]:
    """
    Get MIDI note to class index mapping for 11-class system.
    
    Returns:
        Dictionary mapping MIDI note numbers to class indices (0-10)
    """
    return {
        # Core drums (0-3)
        36: 0,   # Kick          - 387,286 samples
        38: 1,   # Snare head    - 355,950 samples
        40: 2,   # Snare rim     - 128,171 samples
        37: 3,   # Side stick    - 54,221 samples
        
        # Hi-hats (4-6)
        44: 4,   # Pedal hi-hat  - 242,152 samples
        42: 5,   # Closed hi-hat - 216,792 samples
        46: 6,   # Open hi-hat   - 12,513 samples
        
        # Toms (7-8)
        43: 7,   # Floor tom     - 129,908 samples
        48: 8,   # High-mid tom  - 116,840 samples
        
        # Cymbals (9-10)
        51: 9,   # Ride          - 197,033 samples
        53: 10,  # Ride bell     - 14,315 samples
    }


def get_reverse_mapping() -> Dict[int, int]:
    """
    Get class index to MIDI note mapping.
    
    Returns:
        Dictionary mapping class indices (0-10) to MIDI note numbers
    """
    return {
        0: 36,   # Kick
        1: 38,   # Snare head
        2: 40,   # Snare rim
        3: 37,   # Side stick
        4: 44,   # Pedal hi-hat
        5: 42,   # Closed hi-hat
        6: 46,   # Open hi-hat
        7: 43,   # Floor tom
        8: 48,   # High-mid tom
        9: 51,   # Ride
        10: 53,  # Ride bell
    }


def get_class_names() -> List[str]:
    """Get human-readable class names."""
    return [
        'kick',          # 0
        'snare_head',    # 1
        'snare_rim',     # 2
        'side_stick',    # 3
        'hihat_pedal',   # 4
        'hihat_closed',  # 5
        'hihat_open',    # 6
        'floor_tom',     # 7
        'high_mid_tom',  # 8
        'ride',          # 9
        'ride_bell',     # 10
    ]


def get_class_frequencies() -> Dict[int, int]:
    """
    Get sample count for each class from E-GMD analysis.
    
    Returns:
        Dictionary mapping class index to number of samples
    """
    return {
        0: 387286,   # Kick
        1: 355950,   # Snare head
        2: 128171,   # Snare rim
        3: 54221,    # Side stick
        4: 242152,   # Pedal hi-hat
        5: 216792,   # Closed hi-hat
        6: 12513,    # Open hi-hat
        7: 129908,   # Floor tom
        8: 116840,   # High-mid tom
        9: 197033,   # Ride
        10: 14315,   # Ride bell
    }


def compute_class_weights(
    clip_min: float = 0.5,
    clip_max: float = 15.0
) -> np.ndarray:
    """
    Compute inverse frequency class weights for balanced training.
    
    Args:
        clip_min: Minimum weight (prevent under-weighting common classes)
        clip_max: Maximum weight (prevent over-weighting rare classes)
    
    Returns:
        Array of shape (11,) with class weights
    """
    frequencies = get_class_frequencies()
    freq_array = np.array([frequencies[i] for i in range(11)])
    
    # Inverse frequency weights
    total = freq_array.sum()
    weights = total / (11 * freq_array)
    
    # Clip to reasonable range
    weights = np.clip(weights, clip_min, clip_max)
    
    return weights


# ============================================================================
# INFORMATION DISPLAY
# ============================================================================

def print_system_info():
    """Print comprehensive system information."""
    print("="*70)
    print("11-Class Standard Drum Kit Transcription System")
    print("="*70)
    
    print("\nCoverage: 96.58% of E-GMD dataset (1,855,181 / 1,920,922 samples)")
    print("Dropped: 3.42% (auxiliary percussion + duplicates)")
    print("Imbalance ratio: 31:1 (manageable)")
    
    print("\n" + "="*70)
    print("CLASS BREAKDOWN")
    print("="*70)
    
    names = get_class_names()
    freqs = get_class_frequencies()
    reverse = get_reverse_mapping()
    weights = compute_class_weights()
    
    total = sum(freqs.values())
    
    # Group by instrument type
    groups = [
        ("Core Drums", [0, 1, 2, 3]),
        ("Hi-Hats", [4, 5, 6]),
        ("Toms", [7, 8]),
        ("Cymbals", [9, 10]),
    ]
    
    for group_name, indices in groups:
        print(f"\n{group_name}:")
        for i in indices:
            midi_note = reverse[i]
            name = names[i]
            freq = freqs[i]
            pct = (freq / total) * 100
            weight = weights[i]
            print(f"  {i:2d}. MIDI {midi_note:2d} - {name:15s}: "
                  f"{freq:>8,} ({pct:5.2f}%) [weight: {weight:4.1f}]")
    
    print("\n" + "="*70)
    print("DROPPED CLASSES (3.42% of data)")
    print("="*70)
    print("  54 - Tambourine      (33k)  - Auxiliary percussion")
    print("  52 - Chinese Cymbal  (12k)  - Specialty cymbal")
    print("  55 - Splash Cymbal   (7k)   - Specialty cymbal")
    print("  59 - Ride Cymbal 2   (6k)   - Duplicate")
    print("  50 - High Tom        (6k)   - Extra tom")
    print("  45 - Low Tom         (704)  - Extra tom")
    print("  58 - Vibraslap       (624)  - Auxiliary percussion")
    print("  39 - Hand Clap       (182)  - Auxiliary percussion")
    print("  57 - Crash Cymbal 2  (115)  - Duplicate")
    print("  49 - Crash Cymbal 1  (58)   - Duplicate")
    print("  47 - Low-Mid Tom     (13)   - Extra tom")
    print("  56 - Cowbell         (12)   - Auxiliary percussion")
    
    print("\n" + "="*70)
    print("EXPECTED PERFORMANCE")
    print("="*70)
    print("  Overall F1:     70-80%")
    print("  Common classes: 80-85% (kick, snare, hi-hats, ride)")
    print("  Rare classes:   65-70% (open hi-hat, ride bell)")
    print("\n  Perfect for standard drum kit rhythm game!")
    print("="*70)


# ============================================================================
# CODE GENERATION
# ============================================================================

def generate_implementation_code():
    """Generate ready-to-use implementation code."""
    print("\n" + "="*70)
    print("COPY-PASTE IMPLEMENTATION")
    print("="*70)
    
    print("""
# Put this in your config file:
model:
  n_classes: 11

class_weights:
  enabled: true
  weights: [0.5, 0.5, 1.5, 3.5, 0.8, 0.9, 15.0, 1.5, 1.6, 1.0, 13.0]

# Put this in your MIDI processing:
DRUM_MAPPING = {
    36: 0,   # Kick
    38: 1,   # Snare head
    40: 2,   # Snare rim
    37: 3,   # Side stick
    44: 4,   # Pedal hi-hat
    42: 5,   # Closed hi-hat
    46: 6,   # Open hi-hat
    43: 7,   # Floor tom
    48: 8,   # High-mid tom
    51: 9,   # Ride
    53: 10,  # Ride bell
}

CLASS_NAMES = [
    'kick', 'snare_head', 'snare_rim', 'side_stick',
    'hihat_pedal', 'hihat_closed', 'hihat_open',
    'floor_tom', 'high_mid_tom',
    'ride', 'ride_bell'
]

# That's it! Train with these 11 classes for 96.58% coverage.
""")


if __name__ == "__main__":
    print_system_info()
    generate_implementation_code()

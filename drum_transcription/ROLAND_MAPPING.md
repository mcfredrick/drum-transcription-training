# Roland TD-17 Drum Mapping Reference

Complete reference for the Roland TD-17 electronic drum kit MIDI mapping used in this transcription system.

## Overview

This system uses the **Roland TD-17 drum mapping** which provides detailed articulation for professional drum transcription.

**Key features:**
- **26 drum classes** (vs. 8 in simplified General MIDI)
- **Articulation detail:** Head/rim hits, bow/edge cymbals, hi-hat variations
- **Roland TD-17 compatible:** Direct MIDI import to TD-17 module
- **Professional quality:** Matches real drum kit playing techniques

## Complete Drum Mapping

### Kicks and Snares

| Class | Drum Name | MIDI Note | Roland Pad | Playing Technique |
|-------|-----------|-----------|------------|-------------------|
| 0 | Kick | 36 | KD | Bass drum pedal |
| 1 | Snare Head | 38 | Pad 1 (center) | Center snare hit |
| 2 | Snare X-Stick | 37 | Pad 1 (rim + head) | Cross-stick/rim click |
| 3 | Snare Rim | 40 | Pad 1 (rim only) | Rim shot |

### Toms

| Class | Drum Name | MIDI Note | Roland Pad | Playing Technique |
|-------|-----------|-----------|------------|-------------------|
| 4 | Tom 1 Head | 48 | Pad 2 (center) | High tom center hit |
| 5 | Tom 1 Rim | 50 | Pad 2 (rim) | High tom rim shot |
| 6 | Tom 2 Head | 45 | Pad 3 (center) | Mid-high tom center hit |
| 7 | Tom 2 Rim | 47 | Pad 3 (rim) | Mid-high tom rim shot |
| 8 | Tom 3 Head | 43 | Pad 4 (center) | Mid-low tom center hit |
| 9 | Tom 3 Rim | 58 | Pad 4 (rim) | Mid-low tom rim shot |
| 10 | Tom 4 Head | 41 | Pad 5 (center) | Floor tom center hit |
| 11 | Tom 4 Rim | 39 | Pad 5 (rim) | Floor tom rim shot |

### Hi-Hat

| Class | Drum Name | MIDI Note | Roland Pad | Playing Technique |
|-------|-----------|-----------|------------|-------------------|
| 12 | Hi-Hat Closed | 42 | HH Pad (bow) | Closed hi-hat, bow position |
| 13 | Hi-Hat Closed Edge | 22 | HH Pad (edge) | Closed hi-hat, edge hit |
| 14 | Hi-Hat Open | 46 | HH Pad (bow) | Open hi-hat, bow position |
| 15 | Hi-Hat Open Edge | 26 | HH Pad (edge) | Open hi-hat, edge hit |
| 16 | Hi-Hat Pedal | 44 | HH Controller | Foot pedal closed |

### Crashes

| Class | Drum Name | MIDI Note | Roland Pad | Playing Technique |
|-------|-----------|-----------|------------|-------------------|
| 17 | Crash 1 Bow | 49 | CY1 (bow) | Crash cymbal 1, bow area |
| 18 | Crash 1 Edge | 55 | CY1 (edge) | Crash cymbal 1, edge area |
| 19 | Crash 2 Bow | 57 | CY2 (bow) | Crash cymbal 2, bow area |
| 20 | Crash 2 Edge | 52 | CY2 (edge) | Crash cymbal 2, edge area |

### Ride

| Class | Drum Name | MIDI Note | Roland Pad | Playing Technique |
|-------|-----------|-----------|------------|-------------------|
| 21 | Ride Bow | 51 | CY3 (bow) | Ride cymbal, bow area |
| 22 | Ride Edge | 59 | CY3 (edge) | Ride cymbal, edge area |
| 23 | Ride Bell | 53 | CY3 (bell) | Ride cymbal, bell area |

### Auxiliary

| Class | Drum Name | MIDI Note | Roland Pad | Playing Technique |
|-------|-----------|-----------|------------|-------------------|
| 24 | Tambourine | 54 | N/A | Tambourine |
| 25 | Cowbell | 56 | N/A | Cowbell |

## Class Index Reference

**Quick lookup by class number:**

```python
CLASS_TO_NAME = {
    0: "kick",
    1: "snare_head",
    2: "snare_xstick",
    3: "snare_rim",
    4: "tom1_head",
    5: "tom1_rim",
    6: "tom2_head",
    7: "tom2_rim",
    8: "tom3_head",
    9: "tom3_rim",
    10: "tom4_head",
    11: "tom4_rim",
    12: "hihat_closed",
    13: "hihat_closed_edge",
    14: "hihat_open",
    15: "hihat_open_edge",
    16: "hihat_pedal",
    17: "crash1_bow",
    18: "crash1_edge",
    19: "crash2_bow",
    20: "crash2_edge",
    21: "ride_bow",
    22: "ride_edge",
    23: "ride_bell",
    24: "tambourine",
    25: "cowbell"
}
```

## MIDI Note Reference

**Quick lookup by MIDI note:**

```python
MIDI_TO_CLASS = {
    36: 0,   # Kick
    37: 2,   # Snare X-Stick
    38: 1,   # Snare Head
    39: 11,  # Tom 4 Rim
    40: 3,   # Snare Rim
    41: 10,  # Tom 4 Head
    42: 12,  # Hi-Hat Closed
    43: 8,   # Tom 3 Head
    44: 16,  # Hi-Hat Pedal
    45: 6,   # Tom 2 Head
    46: 14,  # Hi-Hat Open
    47: 7,   # Tom 2 Rim
    48: 4,   # Tom 1 Head
    49: 17,  # Crash 1 Bow
    50: 5,   # Tom 1 Rim
    51: 21,  # Ride Bow
    52: 20,  # Crash 2 Edge
    53: 23,  # Ride Bell
    54: 24,  # Tambourine
    55: 18,  # Crash 1 Edge
    56: 25,  # Cowbell
    57: 19,  # Crash 2 Bow
    58: 9,   # Tom 3 Rim
    59: 22,  # Ride Edge
    22: 13,  # Hi-Hat Closed Edge
    26: 15   # Hi-Hat Open Edge
}
```

## Comparison to General MIDI

### Roland TD-17 (26 classes) vs. General MIDI (8 classes)

**Roland TD-17 advantages:**

1. **Articulation detail:**
   - Snare: head/rim/x-stick (3 sounds vs. 1)
   - Toms: head/rim for each tom (8 sounds vs. 3)
   - Cymbals: bow/edge/bell (7 sounds vs. 2)
   - Hi-hat: open/closed/pedal/edge (5 sounds vs. 1)

2. **Musical expression:**
   - Captures playing technique
   - Enables realistic transcription
   - Supports advanced drumming styles

3. **TD-17 compatibility:**
   - Direct import to Roland module
   - Matches kit layout
   - No remapping needed

### General MIDI Simplified Mapping

For reference, the old 8-class system used:
```
0: kick
1: snare (all types)
2: hihat (all types)
3: hi_tom
4: mid_tom
5: low_tom
6: crash (all cymbals)
7: ride
```

**Limitations:** Loss of articulation detail, no rim shots, no cymbal zones.

## Roland TD-17 Kit Configuration

### Default Kit Setup

**Pads:**
- **KD**: Kick drum (MIDI 36)
- **Pad 1**: Snare (center=38, rim=40, x-stick=37)
- **Pad 2**: High Tom (center=48, rim=50)
- **Pad 3**: Mid-High Tom (center=45, rim=47)
- **Pad 4**: Mid-Low Tom (center=43, rim=58)
- **Pad 5**: Floor Tom (center=41, rim=39)

**Cymbals:**
- **HH**: Hi-Hat (closed=42/22, open=46/26, pedal=44)
- **CY1**: Crash 1 (bow=49, edge=55)
- **CY2**: Crash 2 (bow=57, edge=52)
- **CY3**: Ride (bow=51, edge=59, bell=53)

### Trigger Sensitivity

**For best transcription results on TD-17:**
1. Calibrate pad sensitivity (TD-17 settings)
2. Adjust threshold in transcription (see [INFERENCE.md](INFERENCE.md))
3. Match pad configuration to mapping above

## Usage in Code

### Configuration File

**In `configs/*.yaml` files:**

```yaml
model:
  n_classes: 26  # Roland TD-17 mapping

roland_midi:
  drum_names: [
    "kick", "snare_head", "snare_xstick", "snare_rim",
    "tom1_head", "tom1_rim", "tom2_head", "tom2_rim",
    "tom3_head", "tom3_rim", "tom4_head", "tom4_rim",
    "hihat_closed", "hihat_closed_edge", "hihat_open", "hihat_open_edge",
    "hihat_pedal", "crash1_bow", "crash1_edge", "crash2_bow", "crash2_edge",
    "ride_bow", "ride_edge", "ride_bell", "tambourine", "cowbell"
  ]

  midi_to_class:
    36: 0   # Kick
    38: 1   # Snare Head
    # ... (full mapping)

  class_to_midi:
    0: 36   # Kick
    1: 38   # Snare Head
    # ... (full mapping)
```

### Python Code

**Loading mapping from config:**

```python
import yaml

with open('configs/roland_config.yaml') as f:
    config = yaml.safe_load(f)

drum_names = config['roland_midi']['drum_names']
midi_to_class = config['roland_midi']['midi_to_class']
class_to_midi = config['roland_midi']['class_to_midi']
```

## Preprocessing with Roland Mapping

### E-GMD to Roland Mapping

The preprocessing script maps E-GMD MIDI notes to Roland classes:

**E-GMD → Roland examples:**
- E-GMD MIDI 38 (Snare) → Class 1 (Snare Head)
- E-GMD MIDI 37 (Side Stick) → Class 2 (Snare X-Stick)
- E-GMD MIDI 42 (HH Closed) → Class 12 (Hi-Hat Closed)
- E-GMD MIDI 49 (Crash) → Class 17 (Crash 1 Bow)

**Preprocessing command:**
```bash
uv run python scripts/preprocess_roland.py \
    --config configs/roland_config.yaml
```

This creates labels with 26 classes matching Roland TD-17.

## Performance Characteristics

### Typical F1 Scores by Drum Type

Based on full training:

**Easy to transcribe (F1: 0.85-0.95):**
- Kick (0)
- Snare Head (1)
- Hi-Hat Closed (12)

**Medium difficulty (F1: 0.70-0.85):**
- Tom heads (4, 6, 8, 10)
- Crash cymbals (17, 19)
- Ride Bow (21)

**Challenging (F1: 0.60-0.75):**
- Rim shots (3, 5, 7, 9, 11)
- Hi-Hat articulations (13, 14, 15, 16)
- Ride Edge/Bell (22, 23)

**Rare (F1: varies):**
- Tambourine (24)
- Cowbell (25)

### Why Some Drums Are Harder

**Rim shots:**
- Subtler acoustic signature
- Less training data
- Overlap with head sounds

**Hi-hat articulations:**
- Complex interactions (open/closed/pedal)
- Temporal dependencies
- High variability in playing style

**Rare instruments:**
- Limited training examples
- Less robust features learned

## Customizing the Mapping

### Modifying for Other Kits

To adapt this system for different electronic drum kits:

1. **Edit config file:**
   - Change MIDI note numbers in `midi_to_class` and `class_to_midi`
   - Update `drum_names` for clarity

2. **Keep 26 classes:**
   - Model is trained for 26 outputs
   - Changing requires retraining

3. **Remap MIDI output:**
   - Edit `scripts/transcribe.py`
   - Change `class_to_midi` dictionary
   - Output will match your kit

### Example: Alesis Nitro Mesh

```yaml
# Custom mapping for Alesis Nitro Mesh
alesis_midi:
  # Update MIDI note numbers to match Alesis kit
  class_to_midi:
    0: 36   # Kick (same)
    1: 38   # Snare (same)
    2: 37   # Snare Rim (same)
    # ... adjust others as needed
```

## Additional Resources

- **Roland TD-17 Manual:** Official MIDI implementation chart
- **E-GMD Documentation:** Source dataset MIDI mapping
- **General MIDI Specification:** Standard drum mapping reference

## Related Documentation

- **[SETUP.md](SETUP.md)** - Installation and setup
- **[TRAINING.md](TRAINING.md)** - Training models
- **[INFERENCE.md](INFERENCE.md)** - Using trained models
- **[docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md)** - Tuning guide

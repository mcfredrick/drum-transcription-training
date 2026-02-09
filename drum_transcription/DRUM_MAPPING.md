# Standard Drum Kit Mapping (11 Classes)

Complete reference for the 11-class standard drum kit MIDI mapping used in this transcription system.

## Overview

This system uses a **standard drum kit mapping** focused on essential drum sounds for efficient and accurate transcription.

**Key features:**
- **11 drum classes** (focused on essential sounds)
- **96.58% dataset coverage** of E-GMD
- **Efficient training** with lower VRAM requirements
- **General MIDI compatible** for wide software support
- **Class weights** to handle imbalanced data

## Complete Drum Mapping

### 11-Class System

| Class | Drum Name | MIDI Note | Sample Count | Percentage | Class Weight | Description |
|-------|-----------|-----------|--------------|------------|--------------|-------------|
| 0 | Kick | 36 | 387,286 | 20.9% | 0.5 | Bass drum / kick drum |
| 1 | Snare head | 38 | 355,950 | 19.2% | 0.5 | Center snare hit |
| 2 | Snare rim | 40 | 128,171 | 6.9% | 1.5 | Rim shot |
| 3 | Side stick | 37 | 54,221 | 2.9% | 3.5 | Cross-stick / rim click |
| 4 | Pedal hi-hat | 44 | 242,152 | 13.1% | 0.8 | Foot pedal closed hi-hat |
| 5 | Closed hi-hat | 42 | 216,792 | 11.7% | 0.9 | Closed hi-hat bow |
| 6 | Open hi-hat | 46 | 12,513 | 0.7% | 15.0 | Open hi-hat |
| 7 | Floor tom | 43 | 129,908 | 7.0% | 1.5 | Floor tom |
| 8 | High-mid tom | 48 | 116,840 | 6.3% | 1.6 | High or mid tom |
| 9 | Ride | 51 | 197,033 | 10.6% | 1.0 | Ride cymbal bow |
| 10 | Ride bell | 53 | 14,315 | 0.8% | 13.0 | Ride cymbal bell |

**Total coverage:** 1,855,181 samples (96.58% of E-GMD dataset)

### Class Weights Explained

Class weights are used during training to handle imbalanced data:
- **Low weights (0.5-0.9):** Common drums (kick, snare, hi-hat) - reduce their dominance
- **Medium weights (1.0-1.6):** Medium frequency drums (toms, ride) - standard weighting
- **High weights (3.5-15.0):** Rare drums (side stick, open hi-hat, ride bell) - boost their importance

This ensures the model learns all drum types effectively, not just the most common ones.

## Dataset Statistics

### Coverage Analysis

**Included (96.58% of data):**
- All essential drum kit sounds
- Complete rhythmic vocabulary
- Suitable for most musical styles

**Excluded (3.42% of data):**
- Auxiliary percussion (tambourine, cowbell, shaker)
- Duplicate cymbals (crash variations, splash)
- Additional toms beyond standard kit
- Specialized articulations (cymbal chokes, rim clicks on toms)

### Sample Distribution

```
Kick          ████████████████████ 20.9%
Snare head    ███████████████████  19.2%
Pedal hi-hat  █████████████        13.1%
Closed hi-hat ███████████          11.7%
Ride          ██████████           10.6%
Snare rim     ███████              6.9%
Floor tom     ███████              7.0%
High-mid tom  ██████               6.3%
Side stick    ███                  2.9%
Open hi-hat   █                    0.7%
Ride bell     █                    0.8%
```

## MIDI Note Reference

### Quick Lookup by MIDI Note

```python
MIDI_TO_CLASS = {
    36: 0,   # Kick
    37: 3,   # Side stick
    38: 1,   # Snare head
    40: 2,   # Snare rim
    42: 5,   # Closed hi-hat
    43: 7,   # Floor tom
    44: 4,   # Pedal hi-hat
    46: 6,   # Open hi-hat
    48: 8,   # High-mid tom
    51: 9,   # Ride
    53: 10   # Ride bell
}
```

### Class Index Reference

```python
CLASS_TO_NAME = {
    0: "kick",
    1: "snare_head",
    2: "snare_rim",
    3: "side_stick",
    4: "hihat_pedal",
    5: "hihat_closed",
    6: "hihat_open",
    7: "floor_tom",
    8: "high_mid_tom",
    9: "ride",
    10: "ride_bell"
}

CLASS_TO_MIDI = {
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
    10: 53   # Ride bell
}
```

## General MIDI Compatibility

This mapping uses standard General MIDI drum notes (GM Level 1), ensuring compatibility with:

**Software:**
- All major DAWs (Ableton, Logic, FL Studio, Reaper, etc.)
- VST drum plugins (Superior Drummer, EZdrummer, Addictive Drums, etc.)
- General MIDI synthesizers
- Notation software (MuseScore, Sibelius, Guitar Pro)

**Hardware:**
- Most MIDI-compatible electronic drum kits
- Hardware synthesizers with drum sounds
- MIDI controllers

## Usage in Code

### Configuration File

**In `configs/drum_config.yaml`:**

```yaml
model:
  n_classes: 11  # Standard drum kit mapping

drum_mapping:
  num_classes: 11

  class_names: [
    "kick", "snare_head", "snare_rim", "side_stick",
    "hihat_pedal", "hihat_closed", "hihat_open",
    "floor_tom", "high_mid_tom", "ride", "ride_bell"
  ]

  midi_to_class:
    36: 0   # Kick
    38: 1   # Snare head
    40: 2   # Snare rim
    37: 3   # Side stick
    44: 4   # Pedal hi-hat
    42: 5   # Closed hi-hat
    46: 6   # Open hi-hat
    43: 7   # Floor tom
    48: 8   # High-mid tom
    51: 9   # Ride
    53: 10  # Ride bell

  class_to_midi:
    0: 36   # Kick
    1: 38   # Snare head
    2: 40   # Snare rim
    3: 37   # Side stick
    4: 44   # Pedal hi-hat
    5: 42   # Closed hi-hat
    6: 46   # Open hi-hat
    7: 43   # Floor tom
    8: 48   # High-mid tom
    9: 51   # Ride
    10: 53  # Ride bell

  class_weights: [0.5, 0.5, 1.5, 3.5, 0.8, 0.9, 15.0, 1.5, 1.6, 1.0, 13.0]
```

### Python Code

**Loading mapping from config:**

```python
import yaml

with open('configs/drum_config.yaml') as f:
    config = yaml.safe_load(f)

class_names = config['drum_mapping']['class_names']
midi_to_class = config['drum_mapping']['midi_to_class']
class_to_midi = config['drum_mapping']['class_to_midi']
class_weights = config['drum_mapping']['class_weights']
```

## Preprocessing with Standard Mapping

### E-GMD to Standard Mapping

The preprocessing script maps E-GMD MIDI notes to the 11 standard classes:

**E-GMD → Standard mapping examples:**
- E-GMD MIDI 36 (Kick) → Class 0 (Kick)
- E-GMD MIDI 38 (Snare) → Class 1 (Snare head)
- E-GMD MIDI 40 (Rim shot) → Class 2 (Snare rim)
- E-GMD MIDI 37 (Side stick) → Class 3 (Side stick)
- E-GMD MIDI 42 (HH Closed) → Class 5 (Closed hi-hat)
- E-GMD MIDI 44 (HH Pedal) → Class 4 (Pedal hi-hat)
- E-GMD MIDI 46 (HH Open) → Class 6 (Open hi-hat)
- E-GMD MIDI 43 (Floor tom) → Class 7 (Floor tom)
- E-GMD MIDI 48 (High tom) → Class 8 (High-mid tom)
- E-GMD MIDI 51 (Ride) → Class 9 (Ride)
- E-GMD MIDI 53 (Ride bell) → Class 10 (Ride bell)

**Preprocessing command:**
```bash
uv run python scripts/preprocess.py \
    --config configs/drum_config.yaml
```

This creates labels with 11 classes covering 96.58% of the dataset.

### What Was Dropped

The following E-GMD drums are excluded (3.42% of data):

**Auxiliary percussion:**
- Tambourine (MIDI 54)
- Cowbell (MIDI 56)
- Shaker instruments

**Duplicate cymbals:**
- Crash variations (multiple crash cymbals)
- Splash cymbals
- China cymbals

**Extra toms:**
- Toms beyond standard 5-piece kit
- High rack toms

**Specialized articulations:**
- Tom rim shots (consolidated into standard toms)
- Cymbal edge hits (consolidated into bow hits)
- Hi-hat edge variations

## Performance Characteristics

### Typical F1 Scores by Drum Type

Based on full training with 11-class system:

**High accuracy (F1: 0.85-0.95):**
- Kick (0) - Very consistent, strong attack
- Snare head (1) - Clear signature, high sample count
- Closed hi-hat (5) - Common, distinctive timbre

**Good accuracy (F1: 0.75-0.85):**
- Pedal hi-hat (4) - Clear foot stomp sound
- Floor tom (7) - Low frequency, distinct pitch
- High-mid tom (8) - Medium frequency range
- Ride (9) - Sustained, bright sound
- Snare rim (2) - Moderate sample count

**Challenging (F1: 0.60-0.75):**
- Side stick (3) - Less common, subtle sound
- Open hi-hat (6) - Rare in dataset (0.7%)
- Ride bell (10) - Rare, high-pitched (0.8%)

### Why Some Drums Are Harder

**Rare drums (open hi-hat, ride bell, side stick):**
- Limited training examples (< 3% of dataset)
- Higher class weights help but don't fully compensate
- Model sees fewer variations

**Similar timbres:**
- Toms can be confused with each other
- Hi-hat variations share similar frequencies
- Snare rim vs. side stick have subtle differences

**Temporal patterns:**
- Open hi-hat depends on pedal state
- Hi-hat articulations require context understanding

## Design Rationale

### Why 11 Classes?

**Advantages over larger systems (e.g., 26-class Roland):**
1. **Higher dataset coverage** (96.58% vs. ~85%)
2. **Fewer rare classes** to learn
3. **Lower VRAM requirements** for training
4. **Faster training** with smaller model
5. **Better generalization** with more samples per class
6. **Universal compatibility** with General MIDI

**Advantages over simpler systems (e.g., 8-class):**
1. **Preserves important articulations** (rim shots, side stick)
2. **Distinguishes hi-hat types** (open/closed/pedal)
3. **Includes ride bell** for cymbal expression
4. **Captures snare variations** (head vs. rim)
5. **Professional quality** transcription

### Trade-offs

**What you gain:**
- Essential drum vocabulary for most musical styles
- Efficient training on consumer hardware (6GB+ VRAM)
- High accuracy on common drums
- Broad software/hardware compatibility

**What you lose:**
- Detailed cymbal articulations (bow/edge distinctions)
- Auxiliary percussion (tambourine, cowbell)
- Multiple tom rack positions
- Multiple crash cymbals

For most applications, the 11-class system provides the optimal balance of detail and practicality.

## Comparison to Other Systems

### 11-Class vs. 26-Class Roland

| Aspect | 11-Class | 26-Class Roland |
|--------|----------|-----------------|
| **Coverage** | 96.58% | ~85% |
| **VRAM needed** | 6GB+ | 8GB+ |
| **Training time** | Faster | Slower |
| **Rare drums** | 3 classes < 3% | 10+ classes < 2% |
| **Compatibility** | General MIDI | Roland-specific |
| **Use case** | General transcription | Roland TD-17 practice |

### 11-Class vs. 8-Class Simplified

| Aspect | 11-Class | 8-Class |
|--------|----------|---------|
| **Articulations** | Snare (head/rim/stick), HH (open/closed/pedal) | Basic drums only |
| **Cymbals** | Ride + bell | Ride only |
| **Expressiveness** | Professional | Basic |
| **Model size** | Slightly larger | Smaller |

## Customizing the Mapping

### Adapting for Your Needs

To modify the mapping for different requirements:

1. **Edit config file:**
   - Add/remove drum classes in `class_names`
   - Update `midi_to_class` and `class_to_midi` dictionaries
   - Adjust `class_weights` for new class balance

2. **Retrain model:**
   - Model must be retrained with new number of classes
   - Cannot just change config on existing checkpoint

3. **Update preprocessing:**
   - Modify `src/data/midi_utils.py` if needed
   - Re-run preprocessing with new mapping

### Example: Add Crash Cymbal

```yaml
# Add crash as class 11
model:
  n_classes: 12  # Increase from 11

drum_mapping:
  class_names: [
    "kick", "snare_head", "snare_rim", "side_stick",
    "hihat_pedal", "hihat_closed", "hihat_open",
    "floor_tom", "high_mid_tom", "ride", "ride_bell",
    "crash"  # New class
  ]

  midi_to_class:
    # ... existing mappings ...
    49: 11  # Crash

  class_to_midi:
    # ... existing mappings ...
    11: 49  # Crash

  class_weights: [0.5, 0.5, 1.5, 3.5, 0.8, 0.9, 15.0, 1.5, 1.6, 1.0, 13.0, 1.2]
```

Then re-preprocess and retrain.

## Related Documentation

- **[SETUP.md](SETUP.md)** - Installation and setup
- **[TRAINING.md](TRAINING.md)** - Training models
- **[INFERENCE.md](INFERENCE.md)** - Using trained models
- **[docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md)** - Tuning guide

## Additional Resources

- **General MIDI Specification:** Standard drum note assignments
- **E-GMD Documentation:** Source dataset information
- **MIDI.org:** MIDI protocol specifications

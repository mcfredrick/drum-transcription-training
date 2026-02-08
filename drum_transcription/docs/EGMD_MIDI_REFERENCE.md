# E-GMD Dataset MIDI Reference

**Purpose:** Complete MIDI mapping reference for the Extended Groove MIDI Dataset (E-GMD) used in training the hierarchical drum transcription model.

---

## E-GMD Dataset Overview

**Source:** Google Magenta Research
**URL:** https://magenta.tensorflow.org/datasets/e-gmd
**Size:** ~90GB, 444 hours of drum performance
**Format:** Audio (WAV) + aligned MIDI files
**Drum Kits:** 43 different electronic drum kits (Roland V-Drums family)
**Coverage:** 96.58% of dataset covered by 11-class mapping

---

## General MIDI Drum Map (Standard)

E-GMD follows the **General MIDI Level 1 Percussion Key Map** (Standard MIDI Channel 10).

### Complete Drum Note Reference

| MIDI Note | Drum Name | Category | In E-GMD? | In 11-Class? |
|-----------|-----------|----------|-----------|--------------|
| 35 | Acoustic Bass Drum | Kick | Yes | Yes (merged with 36) |
| **36** | **Bass Drum 1 (Kick)** | **Kick** | **Yes** | **Yes (Class 0)** |
| **37** | **Side Stick** | **Snare** | **Yes** | **Yes (Class 3)** |
| **38** | **Acoustic Snare (Head)** | **Snare** | **Yes** | **Yes (Class 1)** |
| 39 | Hand Clap | Percussion | Rare | No |
| **40** | **Electric Snare (Rim)** | **Snare** | **Yes** | **Yes (Class 2)** |
| 41 | Low Floor Tom | Tom | Yes | Merged with 43 |
| **42** | **Closed Hi-Hat** | **Cymbal** | **Yes** | **Yes (Class 5)** |
| **43** | **High Floor Tom** | **Tom** | **Yes** | **Yes (Class 7)** |
| **44** | **Pedal Hi-Hat** | **Cymbal** | **Yes** | **Yes (Class 4)** |
| 45 | Low Tom | Tom | Yes | Merged with 48 |
| **46** | **Open Hi-Hat** | **Cymbal** | **Yes** | **Yes (Class 6)** |
| 47 | Low-Mid Tom | Tom | Yes | Merged with 48 |
| **48** | **Hi-Mid Tom** | **Tom** | **Yes** | **Yes (Class 8)** |
| **49** | **Crash Cymbal 1** | **Cymbal** | **Yes** | **NO - EXCLUDED** |
| 50 | High Tom | Tom | Yes | Merged with 48 |
| **51** | **Ride Cymbal 1** | **Cymbal** | **Yes** | **Yes (Class 9)** |
| 52 | Chinese Cymbal | Cymbal | Rare | No |
| **53** | **Ride Bell** | **Cymbal** | **Yes** | **Yes (Class 10)** |
| 54 | Tambourine | Percussion | Rare | No |
| 55 | Splash Cymbal | Cymbal | Rare | No |
| 56 | Cowbell | Percussion | Rare | No |
| 57 | Crash Cymbal 2 | Cymbal | Yes | No |
| 58 | Vibraslap | Percussion | No | No |
| 59 | Ride Cymbal 2 | Cymbal | Rare | No |

---

## Current 11-Class System (Used in Codebase)

### Class Mapping

| Class | Name | MIDI Note | E-GMD Samples | % of Dataset | Class Weight |
|-------|------|-----------|---------------|--------------|--------------|
| 0 | kick | 36 | 387,286 | 20.9% | 0.5 |
| 1 | snare_head | 38 | 355,950 | 19.2% | 0.5 |
| 2 | snare_rim | 40 | 128,171 | 6.9% | 1.5 |
| 3 | side_stick | 37 | 54,221 | 2.9% | 3.5 |
| 4 | hihat_pedal | 44 | 242,152 | 13.1% | 0.8 |
| 5 | hihat_closed | 42 | 216,792 | 11.7% | 0.9 |
| 6 | hihat_open | 46 | 12,513 | 0.7% | 15.0 |
| 7 | floor_tom | 43 | 129,908 | 7.0% | 1.5 |
| 8 | high_mid_tom | 48 | 116,840 | 6.3% | 1.6 |
| 9 | ride | 51 | 197,033 | 10.6% | 1.0 |
| 10 | ride_bell | 53 | 14,315 | 0.8% | 13.0 |

**Total Coverage:** 1,855,181 samples (96.58% of E-GMD)

### What's Excluded (3.42% of Data)

**Crash Cymbals:**
- MIDI 49 (Crash 1) - **Most common crash** ⚠️
- MIDI 57 (Crash 2)
- MIDI 55 (Splash)
- MIDI 52 (China)

**Auxiliary Percussion:**
- MIDI 54 (Tambourine)
- MIDI 56 (Cowbell)
- MIDI 39 (Hand Clap)

**Extra Toms:**
- Toms beyond 2 types (floor + high/mid merged)
- Various rack tom positions

---

## Proposed 12-Class System (For Hierarchical Model)

**Adding crash cymbal to the current system:**

| Class | Name | MIDI Note | E-GMD Samples (est.) | Category |
|-------|------|-----------|---------------------|----------|
| 0 | kick | 36 | 387,286 | Kick |
| 1 | snare_head | 38 | 355,950 | Snare |
| 2 | snare_rim | 40 | 128,171 | Snare |
| 3 | side_stick | 37 | 54,221 | Snare |
| 4 | ~~hihat_pedal~~ | ~~44~~ | ~~242,152~~ | ~~Dropped~~ |
| 5 | hihat_closed | 42 | 216,792 | Cymbal (Rhythm) |
| 6 | hihat_open | 46 | 12,513 | Cymbal (Rhythm) |
| 7 | floor_tom | 43 | 129,908 | Tom |
| 8 | high_mid_tom | 48 | 116,840 | Tom |
| 9 | ride | 51 | 197,033 | Cymbal (Rhythm) |
| 10 | ride_bell | 53 | 14,315 | Cymbal (Rhythm) |
| **11** | **crash** | **49** | **~60,000** (est.) | **Cymbal (Accent)** |

**Coverage after adding crash:** ~98% of E-GMD dataset

**Note:** Hihat pedal (class 4) dropped as discussed - low priority, 5.8% recall

---

## Hierarchical Model Branch Mapping

### How 12 Classes Map to 5 Branches

```
E-GMD MIDI → 12-Class → Hierarchical Branch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MIDI 36 → Class 0 (kick)
  └─→ KICK BRANCH: Binary (kick/no_kick)

MIDI 38 → Class 1 (snare_head)
MIDI 40 → Class 2 (snare_rim)
MIDI 37 → Class 3 (side_stick)
  └─→ SNARE BRANCH: 3-class (none/head/rim)
      Note: side_stick can be merged with rim or separate

MIDI 43 → Class 7 (floor_tom)
MIDI 48 → Class 8 (high_mid_tom)
  └─→ TOM BRANCH: Hierarchical
      - Primary: tom/no_tom
      - Variation: floor/high/mid

MIDI 42 → Class 5 (hihat_closed)
MIDI 46 → Class 6 (hihat_open)
MIDI 51 → Class 9 (ride)
MIDI 53 → Class 10 (ride_bell)
  └─→ CYMBAL BRANCH (Rhythm): Hierarchical
      - Primary: none/hihat/ride
      - Hihat variation: closed/open
      - Ride variation: body/bell

MIDI 49 → Class 11 (crash) ← NEW
  └─→ CRASH BRANCH (Accent): Binary (crash/no_crash)

MIDI 44 (hihat_pedal) → DROPPED
```

---

## Roland TD-17 Official MIDI Mapping

The E-GMD dataset was recorded using Roland TD-17 V-Drums modules. Here is the **official complete mapping:**

### Complete TD-17 MIDI Map

| Input | MIDI Note | Note Name | Used in Model? | Our Class |
|-------|-----------|-----------|----------------|-----------|
| KICK | 36 | C2 | ✅ Yes | Class 0 (kick) |
| SNARE \<HEAD\> | 38 | D2 | ✅ Yes | Class 1 (snare_head) |
| SNARE \<RIM\> | 40 | E2 | ✅ Yes | Class 2 (snare_rim) |
| SNARE \<XSTICK\> | 37 | C#2 | ✅ Yes | Class 3 (side_stick) |
| TOM1 \<HEAD\> | 48 | C3 | ✅ Yes | Class 8 (high_tom) |
| TOM1 \<RIM\> | 50 | D3 | ❌ No | Merged with head |
| TOM2 \<HEAD\> | 45 | A2 | ✅ Yes | Class 8 (mid_tom) |
| TOM2 \<RIM\> | 47 | B2 | ❌ No | Merged with head |
| TOM3 \<HEAD\> | 43 | G2 | ✅ Yes | Class 7 (floor_tom) |
| TOM3 \<RIM\> | 58 | A#3 | ❌ No | Merged with head |
| HH OPEN \<BOW\> | 46 | A#2 | ✅ Yes | Class 6 (hihat_open) |
| HH OPEN \<EDGE\> | 26 | D1 | ❌ No | Merged with bow |
| HH CLOSE \<BOW\> | 42 | F#2 | ✅ Yes | Class 5 (hihat_closed) |
| HH CLOSE \<EDGE\> | 22 | A#0 | ❌ No | Merged with bow |
| HH PEDAL | 44 | G#2 | ❌ DROPPED | Low priority |
| **CRASH1 \<BOW\>** | **49** | **C#3** | **✅ Yes** | **Class 11 (crash)** |
| CRASH1 \<EDGE\> | 55 | G3 | ❌ No | Merged with bow |
| CRASH2 \<BOW\> | 57 | A3 | ❌ No | Merged with CRASH1 |
| CRASH2 \<EDGE\> | 52 | E3 | ❌ No | Merged with CRASH1 |
| RIDE \<BOW\> | 51 | D#3 | ✅ Yes | Class 9 (ride) |
| RIDE \<EDGE\> | 59 | B3 | ❌ No | Merged with bow |
| RIDE \<BELL\> | 53 | F3 | ✅ Yes | Class 10 (ride_bell) |
| AUX1 \<HEAD\> | 27 | D#1 | ❌ No | Auxiliary percussion |
| AUX1 \<RIM\> | 28 | E1 | ❌ No | Auxiliary percussion |

### Mapping Strategy

**What We Use (12 Classes):**
- **Kick:** MIDI 36 only
- **Snare:** MIDI 37 (xstick), 38 (head), 40 (rim) - 3 articulations
- **Toms:** MIDI 43 (floor), 45 (mid), 48 (high) - HEAD only, rim merged
- **Hihat:** MIDI 42 (closed), 46 (open) - BOW only, edge merged
- **Ride:** MIDI 51 (bow), 53 (bell) - BOW and BELL, edge merged
- **Crash:** MIDI 49 (CRASH1 bow) - Primary crash, CRASH2 merged

**What We Merge:**
- Tom rims (50, 47, 58) → Merged with respective tom heads
- Cymbal edges (26, 22, 55, 52, 59) → Merged with bow articulations
- CRASH2 (57) → Merged with CRASH1 (49)

**What We Drop:**
- Hihat pedal (44) - Low priority, 5.8% recall
- Auxiliary pads (27, 28) - Not standard drum kit

### Tom Granularity in TD-17

The TD-17 has **3 separate toms:**

```
TOM1 (High Tom):  MIDI 48 (C3)   → Class 8 (high_tom)
TOM2 (Mid Tom):   MIDI 45 (A2)   → Class 8 (mid_tom)
TOM3 (Floor Tom): MIDI 43 (G2)   → Class 7 (floor_tom)
```

**For our model:**
- Floor tom: Separate class (Class 7)
- High/Mid toms: Can be merged (Class 8) or separate

**Decision needed:**
- Option A: Keep high/mid merged as "high_mid_tom" (simpler)
- Option B: Separate into high (48) and mid (45) toms (more detail)
- Recommendation: Start with merged, can separate later if needed

### Crash Cymbal Details

TD-17 has **2 crash cymbals:**

```
CRASH1 <BOW>:  MIDI 49 (C#3)  ← Primary, most common
CRASH1 <EDGE>: MIDI 55 (G3)   ← Edge articulation
CRASH2 <BOW>:  MIDI 57 (A3)   ← Secondary crash
CRASH2 <EDGE>: MIDI 52 (E3)   ← Edge articulation
```

**Our approach:**
- Use MIDI 49 (CRASH1 bow) as primary crash
- Merge CRASH2 (57) into CRASH1 for training
- Ignore edge articulations (55, 52)
- Single "crash" class represents all crash hits

**Rationale:**
- Bow vs edge distinction not critical for rhythm game
- Multiple crashes serve same musical function
- Simplifies model (one crash class vs four)

### Cymbal Bow vs Edge

TD-17 distinguishes bow (center) vs edge hits on cymbals:

**Hi-Hat:**
- Bow: 42 (closed), 46 (open) ✅ Used
- Edge: 22 (closed), 26 (open) ❌ Merged

**Ride:**
- Bow: 51 ✅ Used
- Bell: 53 ✅ Used (separate class)
- Edge: 59 ❌ Merged with bow

**Crash:**
- Bow: 49, 57 ✅ Used (merged)
- Edge: 55, 52 ❌ Dropped

**Rationale:**
- Bow hits are most common (primary sound)
- Edge hits similar timbre, harder to distinguish
- Reduces class confusion
- Simplifies model

### Roland TD-17 Pad Configuration

Typical physical setup that generated E-GMD data:

```
Physical Pad       MIDI Notes        Our Class(es)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kick Pad           36                Class 0 (kick)
Snare Pad          37, 38, 40        Classes 1, 2, 3 (head/rim/xstick)
Tom 1 (High)       48, 50            Class 8 (high_tom, rim merged)
Tom 2 (Mid)        45, 47            Class 8 (mid_tom, rim merged)
Tom 3 (Floor)      43, 58            Class 7 (floor_tom, rim merged)
Hi-Hat Cymbal      42, 46, 22, 26    Classes 5, 6 (closed/open, edge merged)
Hi-Hat Pedal       44                DROPPED
Crash 1 Cymbal     49, 55            Class 11 (crash, edge merged)
Crash 2 Cymbal     57, 52            Class 11 (merged with crash1)
Ride Cymbal        51, 53, 59        Classes 9, 10 (bow/bell, edge merged)
Aux Pad            27, 28            DROPPED
```

---

## Crash Cymbal Analysis

### Why Crash Wasn't in Original 11-Class System

**From E-GMD analysis:**
- Crash samples: ~60,000 (estimated 3.2% of dataset)
- Excluded to focus on rhythm-keeping drums
- Original system prioritized continuous rhythm elements

**For rhythm game:**
- Crash provides important accents/emphasis
- Marks transitions and climaxes
- Essential for full musical expression

### Crash Cymbal Characteristics in E-GMD

**MIDI Note:** 49 (Crash Cymbal 1)
**Typical Usage:**
- Section transitions (verse → chorus)
- Downbeats of new sections
- Climactic moments
- 1-5 hits per 30-second section

**Acoustic Properties:**
- Frequency: 4-8 kHz (broadband)
- Onset: Very explosive, sharp attack
- Decay: Very long (1-2 seconds)
- Spectral: Chaotic, less tonal than ride

**Class Imbalance:**
- Much rarer than hihat/ride (~3% vs 20%+ of frames)
- Requires special handling in training
- Justifies separate branch in hierarchical model

---

## MIDI Note Mapping for Implementation

### Label Conversion: E-GMD MIDI → Hierarchical Labels

```python
# src/data/egmd_label_conversion.py

EGMD_MIDI_TO_CLASS_12 = {
    36: 0,   # Kick
    38: 1,   # Snare head
    40: 2,   # Snare rim
    37: 3,   # Side stick
    42: 5,   # Closed hi-hat
    46: 6,   # Open hi-hat
    43: 7,   # Floor tom
    48: 8,   # High-mid tom
    51: 9,   # Ride
    53: 10,  # Ride bell
    49: 11,  # Crash ← NEW
    # Note: MIDI 44 (hihat pedal) dropped
}

def egmd_to_hierarchical_labels(midi_events, class_mapping=EGMD_MIDI_TO_CLASS_12):
    """
    Convert E-GMD MIDI events to hierarchical branch labels.

    Args:
        midi_events: List of (time, note, velocity) tuples from E-GMD MIDI
        class_mapping: MIDI note → class index mapping

    Returns:
        dict with hierarchical labels for each branch
    """
    # Create 12-class labels first
    labels_12class = create_frame_labels(midi_events, class_mapping)

    # Convert to hierarchical format
    hierarchical = {
        'kick': labels_12class[:, 0].float(),  # Class 0
        'snare': create_snare_labels(labels_12class),  # Classes 1,2,3
        'tom_primary': create_tom_primary(labels_12class),  # Classes 7,8
        'tom_variation': create_tom_variation(labels_12class),
        'cymbal_primary': create_cymbal_primary(labels_12class),  # Classes 5,6,9,10
        'hihat_variation': create_hihat_variation(labels_12class),
        'ride_variation': create_ride_variation(labels_12class),
        'crash': labels_12class[:, 11].float(),  # Class 11 ← NEW
    }

    return hierarchical
```

### Output: Hierarchical Labels → MIDI

```python
# scripts/predictions_to_midi.py

HIERARCHICAL_TO_MIDI = {
    'kick': 36,
    'snare_head': 38,
    'snare_rim': 40,
    'side_stick': 37,
    'floor_tom': 43,
    'high_tom': 48,
    'mid_tom': 47,
    'hihat_closed': 42,
    'hihat_open': 46,
    'ride': 51,
    'ride_bell': 53,
    'crash': 49,  # ← NEW
}
```

---

## Data Statistics: Crash Cymbal in E-GMD

**Estimated from dataset analysis:**

```
Total E-GMD samples: ~1,920,000
Crash samples (MIDI 49): ~60,000 (3.1%)
Crash + other excluded: ~64,000 (3.4%)

New coverage with crash:
  1,855,181 (current) + 60,000 (crash) = 1,915,181 (99.7%)
```

**Distribution by section:**
- Rare overall (3.1% of frames)
- Concentrated at section boundaries
- Typical song: 5-15 crash hits total

**Class imbalance implications:**
- Crash: ~60,000 samples
- Kick: ~387,000 samples (6.5x more)
- Hihat closed: ~217,000 samples (3.6x more)

**Training strategy:**
- Use higher class weight for crash (~1.5-2.0)
- Separate crash branch allows independent weighting
- Can use higher threshold for crash (favor precision)

---

## Implementation Checklist

### Phase 1: Update MIDI Mapping

- [ ] Update `drum_config.yaml` to include crash (12 classes)
- [ ] Update `DRUM_MAPPING.md` with crash cymbal
- [ ] Add MIDI note 49 to all mapping dictionaries
- [ ] Update class weights array (add crash weight)

### Phase 2: Data Pipeline

- [ ] Modify preprocessing to extract MIDI note 49
- [ ] Update label conversion to include crash
- [ ] Verify crash samples are captured correctly
- [ ] Check coverage statistics (should be ~99.7%)

### Phase 3: Model Training

- [ ] Implement crash branch in hierarchical model
- [ ] Set appropriate branch weight (0.6-1.0)
- [ ] Monitor crash class metrics separately
- [ ] Use higher inference threshold for crash (0.6-0.7)

---

## References

1. **E-GMD Dataset Paper:**
   - *Gillick, Jon, et al. "Learning to Groove with Inverse Sequence Transformations."* ICML 2019
   - https://arxiv.org/abs/1905.06118

2. **General MIDI Specification:**
   - MIDI Manufacturers Association
   - https://www.midi.org/specifications/midi1-specifications/general-midi-specifications

3. **Roland V-Drums:**
   - Roland TD-17 Sound Module Manual
   - GM2 Sound Set documentation

---

**Last Updated:** 2026-02-08
**For:** Hierarchical Drum Transcription Model Implementation
**Dataset:** E-GMD v1.0.0
**Coverage:** 99.7% with 12-class system (including crash)

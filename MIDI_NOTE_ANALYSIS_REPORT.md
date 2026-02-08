# MIDI Note Analysis Report

## Executive Summary

**Yes, the model can be trained to predict actual MIDI notes instead of classes.** This analysis of 4,211 MIDI files from the E-GMD dataset reveals a comprehensive drum note landscape that supports direct MIDI note prediction.

## Key Findings

### Dataset Scale
- **Total MIDI files analyzed**: 4,211 (100% success rate)
- **Total drum note occurrences**: 2,092,157
- **Unique drum notes found**: 25 different MIDI note numbers
- **Note range**: MIDI notes 22-59

### Current vs. Full Mapping Coverage
- **Current E-GMD mapping**: 17 notes (8 drum classes)
- **Found in dataset**: 25 unique notes
- **Coverage of current mapping**: 94.1% (16/17 notes)
- **Additional notes discovered**: 8 notes not in current mapping

## Complete MIDI Note Distribution

### Most Frequent Drum Notes (Top 10)
| MIDI Note | General MIDI Name | Count | Percentage |
|-----------|-------------------|-------|------------|
| 36 | Bass Drum 1 | 387,286 | 18.5% |
| 38 | Acoustic Snare | 355,950 | 17.0% |
| 44 | Pedal Hi-Hat | 242,152 | 11.6% |
| 42 | Closed Hi Hat | 216,792 | 10.4% |
| 51 | Ride Cymbal 1 | 197,033 | 9.4% |
| 43 | High Floor Tom | 129,908 | 6.2% |
| 40 | Hi Floor Tom | 128,171 | 6.1% |
| 22 | Unknown (22) | 126,024 | 6.0% |
| 48 | Hi-Mid Tom | 116,840 | 5.6% |
| 37 | Side Stick | 54,221 | 2.6% |

### Complete Note Inventory
**Standard Notes (in current mapping):**
- 36: Bass Drum 1 ✅
- 38: Acoustic Snare ✅  
- 42: Closed Hi Hat ✅
- 44: Pedal Hi-Hat ✅
- 46: Open Hi-Hat ✅
- 47: Low-Mid Tom ✅
- 48: Hi-Mid Tom ✅
- 49: Crash Cymbal 1 ✅
- 50: High Tom ✅
- 51: Ride Cymbal 1 ✅
- 53: Ride Bell ✅
- 55: Splash Cymbal ✅
- 57: Crash Cymbal 2 ✅
- 59: Ride Cymbal 2 ✅

**Additional Notes Found:**
- 22: Unknown (possibly custom percussion)
- 26: Unknown (possibly custom percussion)
- 37: Side Stick
- 39: Hand Clap
- 40: Hi Floor Tom
- 52: Chinese Cymbal
- 54: Tambourine
- 56: Cowbell
- 58: Vibraslap

**Missing from Dataset:**
- 41: Low Floor Tom (only note from current mapping not found)

## Current E-GMD Mapping Analysis

The current mapping consolidates 17 MIDI notes into 8 classes:

| Class | MIDI Notes | Drum Type |
|-------|------------|-----------|
| 0 | 36 | Kick |
| 1 | 38 | Snare |
| 2 | 42, 44, 46 | Hi-Hat (all variants) |
| 3 | 50 | High Tom |
| 4 | 47, 48 | Mid Tom |
| 5 | 41, 43, 45 | Low Tom |
| 6 | 49, 55, 57 | Crash |
| 7 | 51, 53, 59 | Ride |

## Recommendation: Direct MIDI Note Prediction

### Advantages
1. **Granular Control**: Distinguish between different cymbals (crash vs splash), hi-hat variants (closed vs open vs pedal)
2. **Standard Compatibility**: Direct output to any MIDI equipment without remapping
3. **Expressive Range**: Access to 25+ distinct drum sounds vs 8 classes
4. **Future Extensibility**: Easy to add new drum types without retraining classes

### Implementation Strategy

#### 1. Model Architecture Changes
```python
# Current: 8-class classification
output_layer = Dense(8, activation='sigmoid')

# Proposed: 128-note multi-label classification  
output_layer = Dense(128, activation='sigmoid')  # Full MIDI range
# Or: Dense(64, activation='sigmoid')  # Drum note range (22-85)
```

#### 2. Loss Function
- **Multi-label binary crossentropy** for simultaneous note prediction
- **Optional velocity prediction** as additional output head

#### 3. Data Processing
- Keep existing MIDI processing pipeline
- Remove drum mapping consolidation
- Use binary vectors for each MIDI note (0-127)

#### 4. Training Considerations
- **Increased model capacity** needed for 128 vs 8 outputs
- **Class imbalance handling** (some notes much rarer)
- **Note co-occurrence patterns** (certain drums played together)

### Migration Path

#### Phase 1: Dual Output Model
```python
# Two output heads during transition
drum_class_output = Dense(8, activation='sigmoid', name='classes')
midi_note_output = Dense(128, activation='sigmoid', name='notes')
model = Model(inputs=inputs, outputs=[drum_class_output, midi_note_output])
```

#### Phase 2: MIDI-Only Model
- Train with only MIDI note outputs
- Use transfer learning from class-based model
- Fine-tune on full dataset

## Technical Specifications

### MIDI Note Range for Drums
- **Standard drum channel**: Channel 10 (MIDI channel 9 in zero-indexed)
- **Typical range**: Notes 35-81 (General MIDI percussion)
- **Extended range**: Notes 22-85 found in this dataset

### Velocity Analysis
- **Mean velocity**: ~80-90 (varies by drum type)
- **Range**: 0-127 (standard MIDI velocity)
- **Potential for velocity prediction** as secondary task

### Timing Precision
- **Current frame rate**: ~43 FPS (based on hop_length=512, sr=22050)
- **Sufficient for drum transcription**
- **No changes needed** for MIDI note prediction

## Conclusion

**Training the model to predict actual MIDI notes is not only feasible but recommended.** The dataset contains rich, diverse drum note information that would enable:

1. **25 distinct drum sounds** vs current 8 classes
2. **Standard MIDI compatibility** 
3. **Enhanced expressiveness** and control
4. **Future extensibility** for additional drum types

The transition requires moderate architectural changes but offers significant benefits in output quality and usability.

## Next Steps

1. **Modify data processing** to output binary MIDI note vectors
2. **Update model architecture** for 128-note output
3. **Implement dual-output training** for smooth transition
4. **Evaluate performance** compared to class-based approach
5. **Deploy MIDI note prediction** API endpoints

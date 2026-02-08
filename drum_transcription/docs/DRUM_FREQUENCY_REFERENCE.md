# Drum Kit Frequency Ranges Reference

## Frequency Ranges by Drum Piece

Based on typical microphone placement and frequency response:

```
Drum Piece          Frequency Range    Primary Energy    Mic Type
------------------------------------------------------------------------
Kick drum           20-100 Hz          40-80 Hz          Large diaphragm condenser
Snare drum          200-1000 Hz        240-400 Hz        Small diaphragm condenser
Toms (high/mid)     100-400 Hz         150-300 Hz        Small/large diaphragm
Floor tom           80-200 Hz          100-150 Hz        Small/large diaphragm
Hi-hats             2-8 kHz            3-6 kHz           Small diaphragm condenser
Crash cymbal        4-8 kHz            5-7 kHz           Small diaphragm condenser
Ride cymbal         4-8 kHz            4-6 kHz           Small diaphragm condenser
```

## Frequency Grouping for Model Architecture

Based on these ranges, we can group drums for specialized branches:

### Low-Frequency Branch (20-200 Hz)
**Drums:**
- Kick drum (20-100 Hz)
- Floor tom (80-200 Hz)

**Characteristics:**
- Deep, resonant fundamental frequencies
- Sharp attack with quick decay
- Minimal harmonic content above 200 Hz

**Model implications:**
- Can use frequency masking to emphasize 20-200 Hz range
- Convolutional filters optimized for low frequencies
- Helps distinguish kick from floor tom (kick lower, tom higher in this range)

---

### Mid-Frequency Branch (100-1000 Hz)
**Drums:**
- Snare drum (200-1000 Hz) - Primary
- Toms (100-400 Hz)
- Floor tom (80-200 Hz) - Overlaps with low freq

**Characteristics:**
- Resonant shell frequencies
- Snare has additional broadband noise from wires
- Attack + sustain + resonance

**Model implications:**
- Snare has unique signature (mid-freq resonance + high-freq rattle)
- Toms are primarily in this range
- Can distinguish snare from toms by presence of high-freq noise component

---

### High-Frequency Branch (2-8 kHz)
**Drums:**
- Hi-hats (2-8 kHz)
- Crash cymbal (4-8 kHz)
- Ride cymbal (4-8 kHz)

**Characteristics:**
- Metallic, bright timbre
- Complex harmonic content
- Long decay (especially crash/ride)
- Open vs closed hihat changes frequency content and decay

**Model implications:**
- All cymbals operate in similar frequency range
- Distinction is more about:
  - Attack characteristics (hihat sharper, ride more sustained)
  - Decay time (closed hihat short, open hihat medium, ride long)
  - Spectral envelope (hihat more noise-like, ride more tonal)

---

## Overlaps and Challenges

### 1. **Floor Tom Overlap (80-200 Hz)**
- Overlaps with both kick (20-100 Hz) and toms (100-400 Hz)
- **Solution:** Use temporal patterns and attack characteristics
  - Kick: Very short, punchy attack
  - Floor tom: Slightly longer attack with more resonance

### 2. **Cymbal Overlap (4-8 kHz)**
- Hi-hats, crash, and ride all in similar range
- **Solution:** Use temporal characteristics
  - Hihat: Shortest decay, more noise-like
  - Crash: Very long decay, explosion-like onset
  - Ride: Medium decay, bell-like tone

### 3. **Snare Complexity (200-1000 Hz + broadband noise)**
- Snare has both tonal component (shell resonance) and noise component (wires)
- Extends into high frequencies (1-10 kHz for wire rattle)
- **Solution:** Look for combination of mid-freq tone + high-freq noise
  - Snare head vs rim: Rim has sharper attack, less wire rattle

---

## Implications for Hierarchical Architecture

### Branch Design Based on Frequency Ranges

```
Input Spectrogram (full frequency range)
    ↓
Shared CNN Encoder (learns features across all frequencies)
    ↓
    ├─→ KICK BRANCH
    │   ├─→ Focus: 20-100 Hz
    │   ├─→ Look for: Sharp attack, low fundamental
    │   └─→ Output: kick/no_kick
    │
    ├─→ SNARE BRANCH
    │   ├─→ Focus: 200-1000 Hz (shell) + 1-10 kHz (wires)
    │   ├─→ Look for: Mid-freq resonance + high-freq noise
    │   └─→ Output: no_snare/snare_head/snare_rim
    │
    ├─→ TOM BRANCH (optional - can merge with rhythm hand)
    │   ├─→ Focus: 80-400 Hz
    │   ├─→ Look for: Mid-freq resonance, longer decay than kick
    │   └─→ Output: no_tom/floor_tom/high_mid_tom
    │
    └─→ CYMBAL BRANCH (Rhythm Hand)
        ├─→ Focus: 2-8 kHz
        ├─→ Look for: High-freq metallic content, decay patterns
        └─→ Hierarchical:
            ├─→ Primary: Which cymbal? (hihat/ride/crash/none)
            └─→ Variation:
                ├─→ IF hihat: open/closed (by decay time)
                └─→ IF ride: bell/body (by tone quality)
```

### Frequency Masking Strategy

Each branch can apply frequency masking to emphasize its relevant range:

```python
class KickBranch(nn.Module):
    def __init__(self):
        # Frequency mask for kick (20-100 Hz range)
        # Emphasize low bins in spectrogram
        self.freq_mask = create_frequency_mask(
            freq_range=(20, 100),
            sr=22050,
            n_fft=2048
        )

class SnareBranch(nn.Module):
    def __init__(self):
        # Dual-band mask for snare
        # Band 1: Shell resonance (200-1000 Hz)
        # Band 2: Wire rattle (1-10 kHz)
        self.freq_mask_mid = create_frequency_mask((200, 1000), ...)
        self.freq_mask_high = create_frequency_mask((1000, 10000), ...)

class CymbalBranch(nn.Module):
    def __init__(self):
        # High-frequency mask for cymbals (2-8 kHz)
        self.freq_mask = create_frequency_mask(
            freq_range=(2000, 8000),
            sr=22050,
            n_fft=2048
        )
```

---

## Spectral Features by Drum Type

### Kick Drum
- **Fundamental:** 40-80 Hz
- **Attack:** Very sharp (< 5ms)
- **Decay:** Fast (50-100ms)
- **Harmonics:** Few, mostly in low range
- **Spectral centroid:** Very low (< 100 Hz)

### Snare Drum
- **Fundamental:** 240-400 Hz (shell resonance)
- **Attack:** Sharp (5-10ms)
- **Decay:** Medium (100-200ms)
- **Harmonics:** Complex (shell) + broadband noise (wires)
- **Spectral centroid:** Medium-high (500-2000 Hz)
- **Unique:** High-frequency noise component from snares

### Toms
- **Fundamental:** 100-300 Hz (depending on size)
- **Attack:** Medium (10-20ms)
- **Decay:** Medium-long (200-400ms)
- **Harmonics:** Rich, tonal
- **Spectral centroid:** Low-medium (150-500 Hz)

### Hi-hats
- **Fundamental:** None (noise-based)
- **Attack:** Very sharp (< 2ms)
- **Decay:**
  - Closed: Very short (20-50ms)
  - Open: Medium (200-400ms)
- **Spectral content:** Broadband noise, 2-8 kHz
- **Spectral centroid:** High (4-6 kHz)
- **Distinction:** Open vs closed primarily by decay time

### Ride Cymbal
- **Fundamental:** ~400-600 Hz (bell tone)
- **Attack:** Medium (5-10ms)
- **Decay:** Long (500-1000ms)
- **Spectral content:** 4-8 kHz with tonal component
- **Spectral centroid:** High (5-7 kHz)
- **Distinction:** Bell has stronger fundamental, body more diffuse

---

## Recommendations for Model Training

### 1. **Preprocessing**
- Use Mel spectrogram with enough frequency resolution
- Ensure coverage of 20 Hz - 10 kHz range
- Consider using multiple frequency scales:
  - Fine resolution for low frequencies (kick/toms)
  - Coarser resolution for high frequencies (cymbals)

### 2. **Architecture**
- Early CNN layers learn general frequency patterns
- Later branch-specific layers apply frequency emphasis
- Use attention mechanisms to focus on relevant frequency bands

### 3. **Data Augmentation**
- **Pitch shifting:** Careful - changes fundamental frequencies
  - Kick: ±5 Hz variation (realistic for different kick drums)
  - Cymbals: Less sensitive, can shift more
- **Time stretching:** Safe for all drums (tempo variation)
- **EQ variation:** Simulate different mic placements/rooms

### 4. **Loss Weighting**
- Weight branches by importance for rhythm game
- Weight frequencies by relevance to each drum type

---

## Reference: Mel Spectrogram Parameters

For drum transcription covering 20 Hz - 10 kHz:

```python
# Recommended parameters
sr = 22050  # Sample rate (covers up to 11 kHz)
n_fft = 2048  # FFT size (~46 Hz resolution)
hop_length = 512  # ~23ms frame rate
n_mels = 128  # Mel bands

# Frequency bins per drum range (approximate)
# 20-100 Hz (kick): Mel bins 0-8
# 80-200 Hz (floor tom): Mel bins 6-15
# 100-400 Hz (toms): Mel bins 8-25
# 200-1000 Hz (snare mid): Mel bins 15-50
# 2-8 kHz (cymbals): Mel bins 80-120
```

---

## Conclusion

The frequency ranges confirm that **hierarchical architecture with frequency specialization** is the right approach:

1. **Natural separation:** Different drums operate in different frequency ranges
2. **Minimal overlap:** Kick (20-100 Hz) has no overlap with cymbals (2-8 kHz)
3. **Overlaps are challenging but solvable:**
   - Snare vs toms: Snare has high-freq noise component
   - Hihat vs ride: Use temporal characteristics (decay time)
4. **Efficient learning:** Each branch can focus on relevant frequencies

**Next step:** Use these frequency ranges when implementing the hierarchical architecture.

---

**Last Updated:** 2026-02-08
**Source:** Provided by user (microphone frequency response data)

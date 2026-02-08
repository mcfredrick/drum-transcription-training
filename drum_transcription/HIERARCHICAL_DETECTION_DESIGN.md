# Hierarchical Rhythm Hand Detection for Rhythm Game

## Problem Statement

Current single-stage detection struggles with:
- **Hihat open/closed confusion**: 27% precision on open, 76% on closed
- **Ride bell/body confusion**: 23% precision on bell, 68% on ride
- **Musical incoherence**: Random switching between hihat/ride/tom

But drummers play **coherently**:
- Use ONE rhythm instrument per section (hihat OR ride OR tom)
- Variations (open/closed, bell/body) follow patterns
- Don't randomly switch instruments mid-phrase

## Solution: Two-Stage Detection

### Stage 1: Rhythm Hand Hit Detection (High Recall)

**Goal:** Detect "something was hit on rhythm hand" with >90% recall

**Approach:**
```python
# Merge rhythm hand classes
rhythm_hand_classes = {
    'hihat_closed': 5,
    'hihat_open': 6,
    'ride': 9,
    'ride_bell': 10,
    'floor_tom': 7,
    # Could also include high_mid_tom if needed
}

# For each frame, take MAX probability across rhythm hand classes
rhythm_hand_prob = max(predictions[:, rhythm_hand_classes])

# Use aggressive threshold for high recall
rhythm_hand_hits = rhythm_hand_prob > 0.20  # Low threshold
```

**Expected Performance:**
- Recall: >90% (catch almost all rhythm hand hits)
- Precision: ~60-70% (some false positives, but that's OK for Stage 1)

### Stage 2: Contextual Classification

#### 2A: Determine Dominant Instrument Per Section

**Algorithm:**
```python
def determine_dominant_instrument(predictions, hits, measure_start, measure_end):
    """
    Analyze which instrument is being used for rhythm in this measure.

    Args:
        predictions: Model probabilities (frames, classes)
        hits: Boolean array of detected rhythm hand hits
        measure_start, measure_end: Frame indices for measure

    Returns:
        'hihat', 'ride', or 'tom'
    """
    # Get hits within this measure
    measure_hits = hits[measure_start:measure_end]
    hit_frames = np.where(measure_hits)[0] + measure_start

    if len(hit_frames) == 0:
        return None

    # Calculate average probability for each instrument family
    hihat_prob = np.mean(
        predictions[hit_frames, [5, 6]].max(axis=1)  # max(closed, open)
    )
    ride_prob = np.mean(
        predictions[hit_frames, [9, 10]].max(axis=1)  # max(ride, bell)
    )
    tom_prob = np.mean(
        predictions[hit_frames, 7]  # floor_tom
    )

    # Determine dominant instrument (with hysteresis for stability)
    max_prob = max(hihat_prob, ride_prob, tom_prob)

    if hihat_prob == max_prob and hihat_prob > 0.3:
        return 'hihat'
    elif ride_prob == max_prob and ride_prob > 0.3:
        return 'ride'
    elif tom_prob == max_prob and tom_prob > 0.25:
        return 'tom'
    else:
        return 'hihat'  # Default to hihat if uncertain
```

#### 2B: Classify Variations Within Instrument Family

**For Hihat:**
```python
def classify_hihat_variation(predictions, hit_frame, measure_context):
    """
    Determine if hihat hit is open or closed.

    Uses both:
    1. Relative probability (open vs closed)
    2. Measure-level context (what's the prevailing pattern?)
    """
    # Current frame prediction
    closed_prob = predictions[hit_frame, 5]  # hihat_closed
    open_prob = predictions[hit_frame, 6]    # hihat_open

    # Simple relative classification
    if open_prob > closed_prob:
        return 'hihat_open'
    else:
        return 'hihat_closed'

    # Advanced: Could use measure context
    # E.g., if 80% of measure is closed, bias toward closed
```

**For Ride:**
```python
def classify_ride_variation(predictions, hit_frame):
    """
    Determine if ride hit is bell or body.
    """
    ride_prob = predictions[hit_frame, 9]     # ride
    bell_prob = predictions[hit_frame, 10]    # ride_bell

    # Relative classification
    if bell_prob > ride_prob:
        return 'ride_bell'
    else:
        return 'ride'

    # Note: Ride bell is typically an accent, so could also use:
    # - Velocity threshold (if available)
    # - Position in pattern (downbeats more likely bell)
```

**For Tom:**
```python
def classify_tom(predictions, hit_frame):
    """
    Could distinguish floor vs high/mid tom if needed.
    """
    return 'floor_tom'  # Simplify for rhythm game
```

## Implementation Architecture

### Complete Pipeline

```python
class HierarchicalRhythmDetector:
    """Two-stage rhythm hand detection for rhythm game."""

    def __init__(self, model, tempo_bpm=120, time_signature=(4, 4)):
        self.model = model
        self.tempo_bpm = tempo_bpm
        self.time_signature = time_signature

        # Calculate frames per measure
        hop_length = 512  # from preprocessing
        sr = 22050
        beats_per_measure = time_signature[0]
        seconds_per_beat = 60.0 / tempo_bpm
        frames_per_measure = int(
            (seconds_per_beat * beats_per_measure * sr) / hop_length
        )
        self.frames_per_measure = frames_per_measure

    def detect(self, audio_path):
        """
        Run hierarchical detection on audio file.

        Returns:
            List of (frame, class, confidence) tuples
        """
        # Get model predictions
        predictions = self.model.predict(audio_path)  # (frames, n_classes)

        # Stage 1: Detect rhythm hand hits
        rhythm_hand_hits = self._detect_rhythm_hand_hits(predictions)

        # Stage 2: Classify each hit using context
        classified_hits = []

        # Process by measures
        n_frames = len(predictions)
        for measure_start in range(0, n_frames, self.frames_per_measure):
            measure_end = min(measure_start + self.frames_per_measure, n_frames)

            # Determine dominant instrument for this measure
            dominant = self._determine_dominant_instrument(
                predictions, rhythm_hand_hits, measure_start, measure_end
            )

            # Classify each hit in this measure
            measure_hits = np.where(
                rhythm_hand_hits[measure_start:measure_end]
            )[0] + measure_start

            for hit_frame in measure_hits:
                hit_class, confidence = self._classify_hit(
                    predictions, hit_frame, dominant
                )
                classified_hits.append((hit_frame, hit_class, confidence))

        return classified_hits

    def _detect_rhythm_hand_hits(self, predictions, threshold=0.20):
        """Stage 1: Detect any rhythm hand hit."""
        rhythm_classes = [5, 6, 7, 9, 10]  # hihat, tom, ride
        rhythm_probs = predictions[:, rhythm_classes].max(axis=1)
        return rhythm_probs > threshold

    def _determine_dominant_instrument(self, predictions, hits,
                                      measure_start, measure_end):
        """Stage 2A: Determine which instrument is being used."""
        hit_frames = np.where(hits[measure_start:measure_end])[0] + measure_start

        if len(hit_frames) == 0:
            return 'hihat'  # Default

        # Average probabilities for each family
        hihat_prob = np.mean(predictions[hit_frames, [5, 6]].max(axis=1))
        ride_prob = np.mean(predictions[hit_frames, [9, 10]].max(axis=1))
        tom_prob = np.mean(predictions[hit_frames, 7])

        # Return dominant (with minimum threshold)
        max_prob = max(hihat_prob, ride_prob, tom_prob)
        if hihat_prob == max_prob and hihat_prob > 0.3:
            return 'hihat'
        elif ride_prob == max_prob and ride_prob > 0.3:
            return 'ride'
        elif tom_prob == max_prob and tom_prob > 0.25:
            return 'tom'
        else:
            return 'hihat'

    def _classify_hit(self, predictions, hit_frame, dominant_instrument):
        """Stage 2B: Classify variation within instrument family."""
        if dominant_instrument == 'hihat':
            closed_prob = predictions[hit_frame, 5]
            open_prob = predictions[hit_frame, 6]
            if open_prob > closed_prob:
                return 'hihat_open', open_prob
            else:
                return 'hihat_closed', closed_prob

        elif dominant_instrument == 'ride':
            ride_prob = predictions[hit_frame, 9]
            bell_prob = predictions[hit_frame, 10]
            if bell_prob > ride_prob:
                return 'ride_bell', bell_prob
            else:
                return 'ride', ride_prob

        elif dominant_instrument == 'tom':
            return 'floor_tom', predictions[hit_frame, 7]

        else:
            # Shouldn't happen, but fallback
            return 'hihat_closed', 0.5
```

## Benefits for Rhythm Game

### 1. **Higher Recall on Critical Events**
- Stage 1 catches 90%+ of rhythm hand hits (vs 76% currently)
- Missing rhythm hand notes makes game unplayable
- False positives in Stage 1 are filtered by Stage 2

### 2. **Musical Coherence**
- Results make musical sense (no random instrument switching)
- Players won't see: hihat → ride → hihat → tom in rapid succession
- Instead: consistent hihat for 4 measures, then transition to ride

### 3. **Better Precision on Variations**
- Current: 27% precision on open hihat (standalone)
- New: ~60-70% precision using relative comparison within hihat context
- Context helps: "Is this more open than closed?" vs "Is this open hihat?"

### 4. **Graceful Degradation**
- If variation classification is wrong (open vs closed), still got the timing right
- Better to show "hihat hit" with wrong variation than miss it entirely
- Can display confidence to player ("uncertain if open/closed")

## Integration with Per-Class Thresholds

### Combined Approach

**Stage 1: Use optimized thresholds for rhythm hand detection**
```python
# From per-class threshold optimization
rhythm_hand_threshold = min([
    optimized_thresholds['hihat_closed'],   # 0.30
    optimized_thresholds['hihat_open'],      # 0.45
    optimized_thresholds['ride'],            # 0.28
    optimized_thresholds['ride_bell'],       # 0.50
    optimized_thresholds['floor_tom'],       # 0.28
])
# Use lowest threshold (0.28) for max recall

# OR: Use even lower threshold since Stage 2 will filter
rhythm_hand_threshold = 0.20
```

**Stage 2: Use relative probabilities**
```python
# Don't threshold, just compare relative probabilities
if pred[hihat_open] > pred[hihat_closed]:
    classify_as('open')
else:
    classify_as('closed')
```

## Handling Edge Cases

### 1. **Instrument Transitions**
```python
# Smooth transitions using hysteresis
# Don't switch dominant instrument unless new one is clearly stronger
if new_instrument_prob > current_instrument_prob + 0.15:
    switch_to_new_instrument()
```

### 2. **Simultaneous Instruments**
```python
# Sometimes ride + hihat both play (ride for groove, hihat for timekeeping)
# Could have TWO active contexts if both probabilities are high
if hihat_prob > 0.4 and ride_prob > 0.4:
    # Dual context mode
    classify_using_both()
```

### 3. **Unknown Tempo**
```python
# Estimate tempo from onset patterns
# Or use fixed window (e.g., 2 seconds ~= 1 measure at 120 BPM)
adaptive_window = estimate_tempo_and_compute_measure_length(onsets)
```

### 4. **Fills and Transitions**
```python
# Detect drum fills (rapid tom hits)
# During fills, rhythm hand detection might not apply
if detect_fill(predictions):
    use_direct_classification()  # Fall back to Stage 1 only
```

## Testing Strategy

### 1. **Measure Stage 1 Performance**
```python
# How well does merged rhythm hand detection work?
test_rhythm_hand_recall()  # Target: >90%
test_rhythm_hand_precision()  # Target: >65%
```

### 2. **Measure Stage 2A Performance**
```python
# How often do we correctly identify dominant instrument?
test_dominant_instrument_accuracy()  # Target: >85%
```

### 3. **Measure Stage 2B Performance**
```python
# Given correct dominant instrument, how often do we get variation right?
test_variation_classification()  # Target: >70%
```

### 4. **End-to-End Performance**
```python
# Overall accuracy on rhythm hand notes
test_full_pipeline()  # Target: recall >85%, precision >70%
```

## Implementation Priority

### Phase 1: Basic Two-Stage (Quick)
1. Implement rhythm hand hit detection (Stage 1)
2. Implement simple dominant instrument detection (Stage 2A)
3. Implement relative variation classification (Stage 2B)
4. Test on validation set

**Time:** 1-2 days | **Expected improvement:** +15-25% F1 on rhythm hand classes

### Phase 2: Refinement (Medium)
1. Add tempo detection or make tempo configurable
2. Implement hysteresis for instrument switching
3. Add confidence scoring
4. Handle edge cases (fills, transitions)

**Time:** 2-3 days | **Expected improvement:** +5-10% additional F1

### Phase 3: Advanced (Optional)
1. Learn optimal window sizes from data
2. Implement dual-context mode (hihat + ride simultaneously)
3. Pattern-based prediction (common hihat patterns)
4. Player-specific calibration

**Time:** 1 week | **Expected improvement:** +5% additional F1 + better UX

## Simplified Class Structure (Alternative)

If implementing hierarchical detection, consider simplifying training:

### Option A: Train with merged classes
```yaml
# Retrain with 6 classes instead of 11
classes:
  - kick
  - snare  # merge snare_head + snare_rim
  - rhythm_primary  # merge hihat_closed + ride + floor_tom
  - rhythm_accent  # merge hihat_open + ride_bell
  - toms  # merge high_mid_tom + floor_tom (if not in rhythm_primary)
  - hihat_pedal
```

**Pros:**
- Easier ML problem (fewer classes)
- Higher accuracy per class
- Stage 2 handles variation classification

**Cons:**
- Requires retraining
- Loses direct open/closed prediction

### Option B: Keep current classes, use hierarchical at inference
```python
# Train on all 11 classes (current model)
# At inference, use hierarchical detection
# Best of both worlds: detailed predictions + contextual refinement
```

**Pros:**
- No retraining needed
- Can still use detailed predictions if confident
- Hierarchical detection improves ambiguous cases

**Cons:**
- More complex inference pipeline
- Two systems to maintain

## Recommendation

**Start with Option B (Hierarchical inference on current model):**

1. Run per-class threshold optimization (from previous task)
2. Implement basic hierarchical detector
3. Test on validation set
4. Compare:
   - Baseline: Single threshold (0.5)
   - Improved: Per-class thresholds
   - Advanced: Hierarchical detection

**If hierarchical works well, consider Option A for next training run.**

---

**Next steps:**
Want me to implement the hierarchical detector? I can create:
1. `scripts/hierarchical_detector.py` - Core implementation
2. `scripts/evaluate_hierarchical.py` - Comparison with baseline
3. Integration with existing transcribe.py

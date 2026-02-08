# Rhythm Game Drum Transcription: Prototyping & Evaluation Plan

**Created:** 2026-02-08
**Status:** Planning Phase
**Goal:** Find the best method to transcribe drums for rhythm game use case

---

## Executive Summary

### Current Situation

**Model Performance (threshold=0.5):**
- Overall F1: 58.64%
- Overall Recall: 51.42% ❌ (Target: >85%)
- Overall Precision: 68.22%

**Critical Issues for Rhythm Game:**
1. **Kick**: 40% recall - Missing 60% of hits
2. **Snare**: 43% recall - Missing 57% of hits
3. **Hihat pedal**: 5.8% recall - Catastrophic failure
4. **Hihat open**: 27.5% precision - 73% false positives
5. **Ride bell**: 23.3% precision - 77% false positives

**Rhythm Game Requirements:**
- Tier 1 (Critical): Kick, snare, rhythm hand - >85% recall, >70% precision
- Tier 2 (Nice): Distinguish variations (open/closed, bell/body) - >70% precision
- Tier 3 (Optional): Hihat pedal, side stick - >60% recall/precision

### Proposed Solutions (To Be Evaluated)

We've identified **three complementary approaches** to improve performance:

1. **Per-Class Threshold Optimization** (Quick win, no retraining)
2. **Hierarchical Post-Processing** (Medium effort, no retraining)
3. **Hierarchical Model Architecture** (Long-term, requires retraining)

---

## Approach 1: Per-Class Threshold Optimization

### Concept

Instead of using one threshold (0.5) for all drums, find optimal threshold for each drum independently.

**Example:**
```python
class_thresholds = {
    'kick': 0.22,        # Low threshold for max recall
    'snare_head': 0.25,  # Low threshold for max recall
    'hihat_closed': 0.30,
    'hihat_open': 0.45,  # High threshold to reduce false positives
    'ride': 0.28,
    'ride_bell': 0.50,   # High threshold to reduce false positives
}
```

### Implementation Status

✅ **READY TO RUN**
- Script: `scripts/optimize_per_class_thresholds.py`
- Wrapper: `./optimize_per_class_thresholds.sh`
- Documentation: `PER_CLASS_THRESHOLD_GUIDE.md`

### Expected Benefits

- **Effort:** None (already implemented)
- **Time:** 5-10 minutes to run
- **Risk:** Zero (just analyzing existing model)
- **Expected improvement:** +10-30% recall on priority classes

### Success Criteria

**Optimistic:**
- Kick/snare reach >85% recall with acceptable precision
- → Action: Use optimized thresholds, done!

**Realistic:**
- Some improvement but still below targets
- → Action: Proceed to Approach 2 or 3

**Pessimistic:**
- Minimal improvement even at aggressive thresholds
- → Diagnosis: Model quality issue, need retraining (Approach 3)

### Next Step

**RUN THIS FIRST** before proceeding to other approaches:
```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription
./optimize_per_class_thresholds.sh
cat per_class_threshold_results/per_class_optimization_report.txt
```

---

## Approach 2: Hierarchical Post-Processing

### Concept

Two-stage detection that mirrors how drummers actually play:

**Stage 1:** Detect "rhythm hand hit" (hihat OR ride OR tom)
- Use aggressive threshold (0.20) for high recall
- Don't worry about which specific instrument yet

**Stage 2:** Context-aware classification
- Analyze patterns within measures (4 beats)
- Determine dominant instrument for section (hihat vs ride vs tom)
- Then classify variations (open/closed, bell/body)

### Why This Helps

**Musical Reality:**
- Drummers use ONE rhythm instrument per section, not random switching
- Variations follow patterns within the context of that instrument

**Solves Precision Issues:**
- Current: "Is this open hihat?" (hard, 27% precision)
- Hierarchical: "Is this more open or closed?" (easier, relative comparison)

**Expected Benefits:**
- Higher recall on rhythm hand (90%+)
- Better precision on variations (60-70% vs 27%)
- Musically coherent results

### Implementation Status

⚠️ **DESIGNED, NOT IMPLEMENTED**
- Design doc: `HIERARCHICAL_DETECTION_DESIGN.md`
- Needs: `scripts/hierarchical_detector.py`, evaluation scripts

### Technical Requirements

**Known Information Needed:**
- Tempo (BPM) - for measure-based windowing
  - OR use fixed time windows (2 seconds ~= 1 measure at 120 BPM)
- Time signature (4/4, 3/4, etc.) - or assume 4/4

**Optional Enhancements:**
- Tempo detection from onsets
- Adaptive windowing
- Hysteresis for instrument transitions

### Success Criteria

**Target Performance:**
- Rhythm hand recall: >90%
- Variation classification (given correct instrument): >70%
- Overall F1 improvement: +15-25% over baseline

### Implementation Estimate

- **Time:** 2-3 days
- **Effort:** Medium (new inference pipeline)
- **Risk:** Low (can revert to baseline if it doesn't work)

### Next Step (After Approach 1)

If per-class thresholds help but aren't enough:
1. Implement hierarchical detector
2. Evaluate on validation set
3. Compare: baseline vs per-class vs hierarchical

---

## Approach 3: Hierarchical Model Architecture

### Concept

Train model with hierarchical structure built-in, where **each branch is a specialized onset detector** for its drum type:

```
Input (spectrogram)
    ↓
Shared Encoder (CNN + RNN) - learns general onset patterns
    ↓
    ├─→ Kick Onset Detector (binary: kick/no_kick)
    │   └─→ Focuses on 20-100 Hz (kick drum fundamentals)
    │
    ├─→ Snare Onset Detector (3-class: none/head/rim)
    │   └─→ Focuses on 200-1000 Hz (shell) + broadband noise (wire rattle)
    │
    ├─→ Rhythm Hand Onset Detector (hierarchical):
    │   ├─→ Primary: none/hihat/ride/tom
    │   │   └─→ Focuses on 2-8 kHz (cymbals) and 80-400 Hz (toms)
    │   └─→ Variations (conditional):
    │       ├─→ IF hihat: open/closed (by decay time in 2-8 kHz)
    │       ├─→ IF ride: bell/body (by tone quality in 4-8 kHz)
    │       └─→ IF tom: floor/high/mid (by frequency in 80-400 Hz)
    │
    └─→ Hihat Pedal Onset Detector (binary: pedal/no_pedal)
        └─→ Focuses on subtle low-frequency impact

See `docs/DRUM_FREQUENCY_REFERENCE.md` for complete frequency breakdown.
```

**Key Insight:** Each branch = Onset detector + Classifier for that drum type

### Why This Could Be Better

**1. Separate Onset Detection Per Drum Type:**
- Each branch is a specialized onset detector for its drum
- Kick branch learns kick-specific onset patterns (low-frequency thump)
- Snare branch learns snare-specific onset patterns (mid-freq resonance + rattle)
- Cymbal branch learns cymbal-specific onset patterns (high-freq shimmer)
- **No interference:** Kick onsets don't compete with cymbal onsets for neurons

**2. Frequency Range Specialization:**
- Kick branch can emphasize low frequencies (20-200 Hz)
- Cymbal variations can emphasize high frequencies (2kHz+)
- Each branch learns features from relevant frequency ranges
- More efficient than forcing one layer to handle all frequencies

**3. Reduced Class Confusion:**
- Kick branch never sees cymbals → No confusion possible
- Rhythm variations only classified when that instrument is detected
- Natural hierarchy matches the problem structure
- **Current problem:** Single output layer confuses hihat_open with hihat_closed
- **Solution:** First detect "hihat onset", then classify open vs closed

**4. Prioritized Training:**
- Can weight kick/snare branch losses higher (critical for gameplay)
- Variation branches lower weight (nice to have)
- Matches rhythm game priorities
- Model learns what matters most

**5. Conditional Learning:**
- Only predict "open vs closed" when hihat is detected
- Only predict "bell vs body" when ride is detected
- Easier learning problem per branch
- Avoids training on irrelevant examples

### Comparison: Different Architectural Approaches

**Option A: Current Single-Output Architecture**
```
Input → Shared Encoder → Single Output Layer → 11 classes
```
- ❌ All drums compete for same neurons
- ❌ No frequency specialization
- ❌ Kick vs cymbal treated same as open vs closed hihat
- ✅ Simple architecture
- ✅ Single forward pass

**Option B: Generic Onset + Separate Classifier**
```
Input → Onset Detector → "Is there any drum hit?"
      → If yes, Classifier → "Which drum?"
```
- ❌ Two-stage pipeline (slower)
- ❌ No specialization in onset detection
- ❌ Errors in onset detection can't be fixed by classifier
- ✅ Modular design

**Option C: Hierarchical Multi-Branch (Recommended)**
```
Input → Shared Encoder → Multiple Specialized Branches
                       → Each branch = Onset Detector for that drum type
```
- ✅ Specialized onset detection per drum type
- ✅ Frequency range specialization
- ✅ Single forward pass (efficient)
- ✅ Shared low-level features
- ✅ No class confusion between drum types
- ⚠️ More complex architecture
- ⚠️ Needs custom loss function

**Option D: Completely Separate Models**
```
4 independent models: kick, snare, cymbals, toms
Each with its own encoder
```
- ✅ Maximum specialization
- ✅ Can train/tune independently
- ❌ 4x inference time (or GPU parallelization)
- ❌ 4x training time
- ❌ No shared learning
- ❌ Complex deployment

**Recommendation:** Option C (Hierarchical multi-branch)
- Gets specialization benefits of Option D
- Single forward pass like Option A
- Natural onset detection per drum type
- Shared encoder learns general patterns efficiently

### Implementation Status

⚠️ **DESIGNED, NOT IMPLEMENTED**
- Design doc: `docs/HIERARCHICAL_ARCHITECTURE_DESIGN.md`
- Needs: New model class, label conversion, training scripts

### Technical Requirements

**Changes Needed:**
1. **New model architecture:**
   - `src/models/hierarchical_crnn.py`
   - Shared encoder + specialized branches
   - Custom loss function

2. **Label preprocessing:**
   - Convert 11-class labels to hierarchical format
   - Example: `[0,0,0,0,0,1,0,0,0,0,0]` → `{kick: 0, snare: 0, rhythm_primary: 1 (hihat), hihat_var: 0 (closed)}`

3. **Training pipeline:**
   - New training script
   - Hierarchical config
   - Branch-weighted losses

4. **Evaluation:**
   - Per-branch metrics
   - Comparison with baseline

### Success Criteria

**Target Performance:**
- Kick recall: 40% → 75-85%
- Snare recall: 43% → 75-85%
- Hihat open precision: 27% → 60-70%
- Ride bell precision: 23% → 60-70%
- Overall F1: 58% → 75-80%

### Implementation Estimate

- **Time:** 1-2 weeks
  - Week 1: Architecture + training
  - Week 2: Evaluation + tuning
- **Effort:** High (new model, retraining)
- **Risk:** Medium (might not work, but can revert to baseline)
- **Compute:** 6-12 hours GPU training time

### Next Step (After Approaches 1 & 2)

If post-processing approaches aren't sufficient:
1. Implement hierarchical architecture
2. Train on validation split (quick test)
3. If promising, train on full dataset
4. Compare with all other approaches

---

## Prototyping & Evaluation Strategy

### Phase 1: Zero-Cost Diagnosis (TODAY)

**Goal:** Understand current model's capabilities without retraining

**Actions:**
1. ✅ Threshold analysis done (found 0.30 better than 0.5)
2. 🔲 Run per-class threshold optimization
3. 🔲 Analyze results:
   - Can any class hit >85% recall at any threshold?
   - Which classes are fundamentally limited by model quality?

**Decision Point:**
- If targets met → Done, use optimized thresholds!
- If close (70-80% recall) → Try Approach 2 (hierarchical post-processing)
- If far (< 65% recall) → Need Approach 3 (architecture change)

**Time:** 2-3 hours
**Cost:** $0

---

### Phase 2: Post-Processing Enhancement (THIS WEEK)

**Goal:** Maximize current model performance without retraining

**Approach 2A: Per-Class Thresholds** (if not done in Phase 1)
- Implement in inference pipeline
- Test on validation set
- Measure improvement

**Approach 2B: Hierarchical Post-Processing** (if needed)
- Implement two-stage detector
- Test on validation set with known tempo
- Compare with per-class threshold approach

**Decision Point:**
- If targets met → Ship it!
- If still short → Proceed to Phase 3

**Time:** 3-5 days
**Cost:** Development time only

---

### Phase 3: Model Architecture Improvements (NEXT WEEK+)

**Goal:** Retrain with better architecture if needed

**Approach 3A: Hierarchical Multi-Branch**
- Implement architecture
- Train on small subset (sanity check)
- Train full model
- Evaluate

**Approach 3B: Alternative Strategies** (if 3A insufficient)
- Simplified class structure (merge similar classes)
- Hyperparameter optimization with Optuna
- Data augmentation improvements
- Different base architecture

**Decision Point:**
- Choose best performing approach
- Deploy to rhythm game

**Time:** 1-3 weeks
**Cost:** GPU compute + development time

---

## Evaluation Metrics & Targets

### Primary Metrics (Rhythm Game Critical)

**Tier 1 Classes:**
```
Class           Recall Target    Precision Target    Current Recall    Current Precision
---------------------------------------------------------------------------------------
kick            >85%             >70%                39.7%             81.1%
snare_head      >85%             >70%                42.6%             81.3%
snare_rim       >80%             >65%                69.7%             65.9%
hihat_closed    >80%             >70%                76.2%             76.2%
ride            >80%             >70%                64.3%             67.9%
floor_tom       >75%             >65%                76.5%             60.1%
```

**Tier 2 Classes (Variations):**
```
Class           Precision Target    Current Precision    Current Recall
------------------------------------------------------------------------
hihat_open      >70%                27.5%                92.9%
ride_bell       >70%                23.3%                76.9%
```

### Secondary Metrics

- **Overall F1**: >75% (currently 58.6%)
- **Overall Recall**: >80% (currently 51.4%)
- **Latency**: <50ms per frame (real-time gameplay requirement)
- **Musical Coherence**: Qualitative assessment (no random instrument switching)

### Evaluation Protocol

**For Each Approach:**

1. **Quantitative Evaluation:**
   - Per-class precision/recall/F1
   - Overall metrics
   - Inference latency
   - Confusion matrix analysis

2. **Qualitative Evaluation:**
   - Listen to predictions on test songs
   - Check for musical coherence
   - Identify systematic errors

3. **Rhythm Game Simulation:**
   - Test on actual gameplay scenarios
   - Measure player experience (false positives vs missed notes)
   - Check if variations matter (open vs closed hihat)

4. **Comparison Table:**
   ```
   Metric              Baseline    Per-Class    Hierarchical    Hier-Model
                                   Thresholds   Post-Proc       Architecture
   -------------------------------------------------------------------------
   Kick Recall         39.7%       ???          ???             ???
   Snare Recall        42.6%       ???          ???             ???
   Rhythm Hand Recall  76.2%       ???          ???             ???
   Overall F1          58.6%       ???          ???             ???
   Inference Time      Xms         Xms          Xms             Xms
   ```

---

## Decision Framework

### When to Use Per-Class Thresholds (Approach 1)

**Use if:**
- Per-class optimization shows targets can be met
- Need quick solution (no development time)
- Inference speed is critical

**Don't use if:**
- Targets still not met even with aggressive thresholds
- Want musical coherence (no random instrument switching)

### When to Use Hierarchical Post-Processing (Approach 2)

**Use if:**
- Per-class thresholds help but aren't sufficient
- Variation precision is problematic (hihat open, ride bell)
- Need musical coherence
- Want to avoid retraining

**Don't use if:**
- Per-class thresholds already work
- Base recall is too low (< 60%) even with low thresholds
- Can't determine tempo/time signature

### When to Retrain with Hierarchical Architecture (Approach 3)

**Use if:**
- Post-processing approaches insufficient
- Need fundamental improvement in recall/precision
- Have time and compute budget for retraining
- Want optimal long-term solution

**Don't use if:**
- Post-processing already meets targets
- Can't afford retraining time/cost
- Need solution immediately

---

## Open Questions & Decisions Needed

### Technical Questions

1. **Tempo/Time Signature:**
   - Q: Can we extract tempo from audio automatically?
   - Q: Or should rhythm game provide tempo (known for game songs)?
   - Decision: ___________

2. **Variation Classification:**
   - Q: Do we really need to distinguish open/closed hihat?
   - Q: Or is "hihat hit" good enough for gameplay?
   - Decision: ___________

3. **Hihat Pedal:**
   - Q: Is hihat pedal critical for rhythm game?
   - Q: Or can we disable it (currently 5.8% recall)?
   - Decision: ___________

4. **Class Simplification:**
   - Q: Should we merge snare_head + snare_rim into just "snare"?
   - Q: Should we merge ride + ride_bell into just "ride"?
   - Decision: ___________

### Strategy Questions

1. **Prototyping Order:**
   - Option A: Try all approaches sequentially (safe, slow)
   - Option B: Implement hierarchical architecture now (aggressive, risky)
   - Option C: Run threshold optimization, then decide (recommended)
   - Decision: ___________

2. **Success Criteria:**
   - Q: What's "good enough" for rhythm game launch?
   - Q: Can we ship with 75% recall if precision is high?
   - Decision: ___________

3. **Development Timeline:**
   - Q: When do you need this working?
   - Q: How much time can we spend on optimization?
   - Decision: ___________

---

## Recommended Next Steps

### Immediate (Today)

1. **Run per-class threshold optimization:**
   ```bash
   cd /home/matt/Documents/drum-tranxn/drum_transcription
   ./optimize_per_class_thresholds.sh
   ```

2. **Review results:**
   - Can any priority class hit targets?
   - Which classes are fundamentally limited?

3. **Update this document with findings**

### Short-term (This Week)

Based on Phase 1 results:

**If close to targets (70-80% recall):**
- Implement hierarchical post-processing
- Test on validation set
- Decide if good enough

**If far from targets (< 65% recall):**
- Start implementing hierarchical architecture
- Train on small subset first
- Evaluate before full training

### Medium-term (Next Week)

- Choose best performing approach
- Integrate into rhythm game
- Test with actual gameplay
- Iterate based on player feedback

---

## Success Metrics Summary

### Minimum Viable Product (MVP)

**Can ship with:**
- Kick: >75% recall, >65% precision
- Snare: >75% recall, >65% precision
- Rhythm hand: >70% recall, >60% precision
- False positives rare enough not to frustrate players

### Target Product

**Ideal performance:**
- All Tier 1 classes: >85% recall, >70% precision
- Variation classification: >70% precision
- Musical coherence (no random switching)
- Real-time performance (<50ms latency)

### Stretch Goals

**Would be amazing:**
- >90% recall on all Tier 1 classes
- >80% precision on variations
- Player-specific calibration
- Confidence scoring for uncertain detections

---

## Document Maintenance

**Update this document when:**
- Per-class threshold results are available
- Each approach is implemented/tested
- New ideas or approaches emerge
- Success criteria change
- Timeline changes

**Document Status:**
- ✅ Initial planning complete
- 🔲 Per-class threshold results added
- 🔲 Hierarchical post-processing results added
- 🔲 Hierarchical architecture results added
- 🔲 Final approach selected
- 🔲 Rhythm game integration complete

**Last Updated:** 2026-02-08
**Next Review:** After Phase 1 completion

---

## Appendix: Related Documents

- **Threshold Analysis Results:** `threshold_analysis_results/analysis_report.txt`
- **Per-Class Optimization Guide:** `PER_CLASS_THRESHOLD_GUIDE.md`
- **Hierarchical Post-Processing Design:** `HIERARCHICAL_DETECTION_DESIGN.md`
- **Hierarchical Architecture Design:** `docs/HIERARCHICAL_ARCHITECTURE_DESIGN.md`
- **Overall Optimization Strategy:** `RHYTHM_GAME_OPTIMIZATION.md`
- **Original Threshold Setup:** `THRESHOLD_ANALYSIS_SETUP.md`

# Rhythm Game Performance Optimization Plan

## 🎯 Current State Assessment

### Performance Tiers Based on Your Goals

**Tier 1: Critical for Gameplay** (Target: >85% recall, >75% precision)
```
Class           Precision  Recall   F1      Status
-------------------------------------------------------
kick            81.06%     39.66%   53.26%  ❌ RECALL FAIL
snare_head      81.28%     42.56%   55.87%  ❌ RECALL FAIL
snare_rim       65.85%     69.72%   67.73%  ⚠️  BORDERLINE
hihat_closed    76.23%     76.15%   76.19%  ✅ ACCEPTABLE
ride            67.85%     64.33%   66.04%  ⚠️  BORDERLINE
floor_tom       60.08%     76.48%   67.29%  ⚠️  BORDERLINE
```

**Tier 2: Nice to Distinguish**
```
hihat_open      27.52%     92.93%   42.46%  ❌ PRECISION FAIL
ride_bell       23.25%     76.94%   35.71%  ❌ PRECISION FAIL
```

**Tier 3: Lower Priority**
```
hihat_pedal     73.53%     5.80%    10.75%  ❌ CATASTROPHIC
side_stick      67.01%     79.19%   72.59%  ✅ GOOD
high_mid_tom    62.57%     70.59%   66.34%  ✅ GOOD
```

### Key Problems Identified

1. **Kick & Snare: Low Recall Despite Good Precision**
   - Model is being too conservative
   - Missing 60% of kicks and 57% of snares
   - This will make rhythm game unplayable

2. **Hihat Pedal: Almost Non-Functional**
   - Only detecting 5.8% of pedal hits
   - 18,976 samples in dataset but model ignores them
   - Likely audio feature issue (too subtle/quiet)

3. **Open Hihat & Ride Bell: Confusion with Similar Classes**
   - 73-77% false positive rate
   - Model can't distinguish subtle differences
   - Opens detected as closed, bells detected as ride body

4. **Overall Threshold Issue**
   - Current 0.5 threshold is too conservative
   - Optimal F1 is at 0.30 (10.2% improvement)
   - But still won't meet rhythm game targets

---

## 🔍 Investigation Phase (Do This First)

### 1. **Per-Class Threshold Analysis**

Find optimal threshold for each critical class individually:

```bash
# Create enhanced analysis script
python scripts/analyze_per_class_thresholds.py \
  --checkpoint /mnt/hdd/drum-tranxn/checkpoints/best.ckpt \
  --config configs/full_training_config.yaml \
  --focus-classes kick snare_head snare_rim hihat_closed ride floor_tom
```

**What this tells us:**
- Can we hit >85% recall on kick/snare at any threshold?
- If yes → just need per-class thresholds
- If no → model quality problem, need retraining

### 2. **Hihat Pedal Diagnostic**

Understand why pedal detection fails:

```bash
# Analyze hihat pedal specifically
python scripts/diagnose_class.py \
  --class hihat_pedal \
  --checkpoint /mnt/hdd/drum-tranxn/checkpoints/best.ckpt \
  --config configs/full_training_config.yaml \
  --visualize-failures
```

**Questions to answer:**
- Are predictions uniformly low (audio feature issue)?
- Are pedal hits confused with kicks (spectral similarity)?
- Is training data balanced or are pedals underrepresented?

### 3. **Class Confusion Matrix**

See which classes are being confused:

```bash
# Generate confusion analysis
python scripts/analyze_confusion.py \
  --checkpoint /mnt/hdd/drum-tranxn/checkpoints/best.ckpt \
  --threshold 0.30
```

**Expected insights:**
- hihat_open → hihat_closed confusion
- ride_bell → ride confusion
- hihat_pedal → kick confusion (?)

---

## 🛠️ Solution Strategies (Priority Order)

### Strategy 1: Per-Class Thresholds (Quick Win)

**Effort:** Low | **Impact:** Medium-High | **Time:** 1-2 hours

Use different thresholds optimized for each drum class:

```python
# In scripts/transcribe.py or during inference
class_thresholds = {
    0: 0.20,  # kick - very low for max recall
    1: 0.25,  # snare_head - low for max recall
    2: 0.30,  # snare_rim - moderate
    4: 0.15,  # hihat_pedal - very low to compensate
    5: 0.30,  # hihat_closed - moderate
    6: 0.45,  # hihat_open - higher to reduce false positives
    7: 0.35,  # floor_tom - moderate
    9: 0.30,  # ride - moderate
    10: 0.50, # ride_bell - higher to reduce false positives
}
```

**Expected improvement:**
- Kick recall: 40% → 70-80%
- Snare recall: 43% → 70-80%
- Hihat pedal recall: 6% → 20-30% (still not great)

**Next step:** I can create a script to find optimal per-class thresholds automatically.

### Strategy 2: Simplified Class Structure (Medium Win)

**Effort:** Medium | **Impact:** High | **Time:** 4-6 hours (retraining)

Merge similar classes that are causing confusion:

**Option A: Merge Variants**
```
Original 11 classes → 7 simplified classes:
- kick (unchanged)
- snare (merge snare_head + snare_rim)
- hihat_closed (unchanged)
- hihat_open (unchanged)
- ride (merge ride + ride_bell)
- toms (merge floor_tom + high_mid_tom)
- accent (merge side_stick + hihat_pedal)
```

**Option B: Drop Low-Value Classes**
```
Original 11 classes → 8 core classes:
- Keep: kick, snare_head, snare_rim, hihat_closed, hihat_open, ride, floor_tom, high_mid_tom
- Drop: hihat_pedal, side_stick, ride_bell
```

**Pros:**
- Fewer classes → better performance per class
- Less confusion between similar sounds
- Might improve kick/snare recall significantly

**Cons:**
- Requires retraining (6-8 hours)
- Loses some musical nuance

### Strategy 3: Rhythm-Hand Detection (Alternative Approach)

**Effort:** Medium-High | **Impact:** High | **Time:** 6-8 hours

Instead of distinguishing hihat/ride/toms precisely, create a "rhythm hand" class:

```
6 classes for rhythm game:
1. kick
2. snare
3. rhythm_primary (hihat_closed OR ride OR floor_tom)
4. rhythm_accent (hihat_open OR ride_bell OR side_stick)
5. toms (any tom hit)
6. hihat_pedal
```

**Pros:**
- Aligns perfectly with rhythm game needs
- Easier to get high accuracy on fewer, broader classes
- Players care about "rhythm hand hit" not specific cymbal

**Cons:**
- Requires relabeling training data
- Loses ability to distinguish instruments
- Full retraining needed

### Strategy 4: Model Architecture Improvements (Long-term)

**Effort:** High | **Impact:** Medium-High | **Time:** Multiple days

If per-class thresholds don't work:

1. **Hyperparameter optimization**
   ```bash
   python train_with_optuna.py --n-trials 50 \
     --optimize-for recall \
     --focus-classes kick snare_head snare_rim
   ```

2. **Architecture changes**
   - Increase model capacity (more LSTM layers/units)
   - Add attention mechanism for temporal context
   - Multi-scale feature extraction

3. **Data augmentation**
   - Time stretching for rhythm variation
   - Pitch shifting for tonal variation
   - Mixup for harder negatives

---

## 📋 Recommended Action Plan

### Phase 1: Zero-Cost Investigation (TODAY)

1. ✅ **Already done:** Global threshold analysis (0.30 is better than 0.5)

2. **Next:** Create per-class threshold optimization script
   - Find best threshold for each class independently
   - Target >85% recall on kick/snare/rhythm-hand

3. **Then:** Analyze class confusion
   - See which classes interfere with each other
   - Understand hihat_pedal failure mode

**Time:** 2-3 hours | **Cost:** $0 | **Risk:** None

### Phase 2: Quick Wins (THIS WEEK)

4. **Implement per-class thresholds**
   - Modify inference pipeline
   - Test on validation set
   - Measure improvement on Tier 1 classes

5. **Test rhythm game viability**
   - Do kick/snare hit >85% recall?
   - Is precision acceptable (>70%)?
   - How does it feel in actual gameplay?

**Time:** 4-6 hours | **Cost:** $0 | **Risk:** Low

### Phase 3: Model Improvements (IF NEEDED)

6. **If per-class thresholds aren't enough:**
   - Option A: Merge classes (snare variants, ride variants)
   - Option B: Retrain with Optuna hyperparameter optimization
   - Option C: Redesign for rhythm game (6-class structure)

**Time:** 1-3 days | **Cost:** Compute time | **Risk:** Medium

---

## 🎮 Success Criteria

### Minimum Viable for Rhythm Game

**Tier 1 (Critical):**
- Kick: >85% recall, >75% precision
- Snare: >85% recall, >75% precision
- Rhythm hand (hihat OR ride OR tom): >80% recall, >70% precision

**Tier 2 (Nice to have):**
- Distinguish open/closed hihat: >70% precision
- Distinguish ride/ride_bell: >70% precision

**Tier 3 (Can ignore):**
- Hihat pedal can be disabled if not fixable
- Side stick not essential
- Tom distinction not essential

### Stretch Goals

- Per-song threshold adaptation (easier songs = higher threshold)
- Confidence scoring (show player when detection is uncertain)
- Player calibration mode (adjust thresholds to player's drum kit)

---

## 🚀 Next Steps

Want me to:

1. **Create the per-class threshold optimization script?**
   - Automatically finds best threshold for each class
   - Generates config file with optimal values
   - Estimates expected recall/precision improvements

2. **Create the class confusion analysis script?**
   - Shows which classes interfere with each other
   - Visualizes common misclassifications
   - Helps decide which classes to merge

3. **Create the hihat pedal diagnostic script?**
   - Analyzes why detection is so poor
   - Visualizes failed predictions
   - Recommends fixes (threshold, retraining, or disable)

4. **Design a simplified class structure?**
   - Proposes rhythm-game-optimized classes
   - Shows expected performance gains
   - Creates migration plan

Let me know which investigation you want to start with!

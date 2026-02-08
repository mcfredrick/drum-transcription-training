# Per-Class Threshold Optimization Guide

## What This Does

Instead of using a single threshold (0.5) for all drum classes, this finds the **optimal threshold for each drum independently** to maximize rhythm game performance.

**Why this matters:**
- Kick might perform best at threshold 0.20
- Snare might perform best at threshold 0.25
- Hihat might perform best at threshold 0.35
- Using the same threshold for all is suboptimal!

## Quick Start

```bash
# Run with defaults (uses latest checkpoint and validation set)
./optimize_per_class_thresholds.sh

# Or specify checkpoint
./optimize_per_class_thresholds.sh --checkpoint /mnt/hdd/drum-tranxn/checkpoints/best.ckpt
```

**This will generate:**
```
per_class_threshold_results/
├── per_class_optimization_report.txt    # Detailed analysis
├── optimized_thresholds.yaml            # Config file to use
└── per_class_optimization.png           # Visualizations
```

**Time:** 5-10 minutes (depending on validation set size)

## What You'll Learn

### 1. **Can Your Model Hit Rhythm Game Targets?**

The report will tell you if each priority class meets targets:
- ✅ **Targets Met**: Recall >85%, Precision >70% → Ready for rhythm game!
- ❌ **Targets Not Met**: Model needs improvement (retraining or class simplification)

### 2. **Optimal Thresholds for Each Class**

Example output:
```
PRIORITY CLASSES (TIER 1)
Class              Threshold   Precision      Recall          F1      Support
--------------------------------------------------------------------------------
kick                    0.22       72.3%       87.4%       79.1%       28,815
snare_head              0.25       74.8%       85.2%       79.7%       24,441
hihat_closed            0.30       76.2%       88.3%       81.8%       17,334
ride                    0.28       70.1%       82.7%       75.9%       16,062
```

### 3. **Baseline vs Optimized Comparison**

Shows improvement from using per-class thresholds:
```
OVERALL METRICS:
Metric          Baseline    Optimized       Change
---------------------------------------------------
Precision         68.2%        70.5%        +2.3%
Recall            51.4%        82.1%       +30.7% ✨
F1 Score          58.6%        75.8%       +17.2% ✨
```

## Using the Results

### If Targets Are Met ✅

Great! Just use the optimized thresholds:

**Option 1: Use YAML config**
```bash
# The script generates optimized_thresholds.yaml
# Use it in your inference pipeline
```

**Option 2: Hardcode in transcribe.py**
```python
# Add to scripts/transcribe.py
class_thresholds = {
    0: 0.22,  # kick
    1: 0.25,  # snare_head
    2: 0.30,  # snare_rim
    4: 0.15,  # hihat_pedal
    5: 0.30,  # hihat_closed
    6: 0.45,  # hihat_open
    7: 0.28,  # floor_tom
    8: 0.32,  # high_mid_tom
    9: 0.28,  # ride
    10: 0.50, # ride_bell
    3: 0.35,  # side_stick
}

# Apply per-class thresholds
predictions_binary = torch.zeros_like(predictions)
for class_idx, threshold in class_thresholds.items():
    predictions_binary[:, :, class_idx] = (predictions[:, :, class_idx] >= threshold)
```

### If Targets Are NOT Met ❌

This means threshold optimization alone won't fix the problem. You need to:

1. **Check which classes failed**
   - If kick/snare failed → Critical issue, need model improvement
   - If only hihat_pedal failed → Maybe disable it or retrain

2. **Consider next steps:**
   - Class simplification (merge similar classes)
   - Model retraining with better hyperparameters
   - Architecture improvements

See `RHYTHM_GAME_OPTIMIZATION.md` for detailed strategies.

## Advanced Options

### Different Optimization Strategies

```bash
# Maximize F1 (balanced precision/recall)
./optimize_per_class_thresholds.sh --strategy max_f1

# Maximize recall (accept lower precision)
./optimize_per_class_thresholds.sh --strategy max_recall

# Balanced (equal precision/recall)
./optimize_per_class_thresholds.sh --strategy balanced

# Rhythm game optimized (default - prioritize recall with min precision)
./optimize_per_class_thresholds.sh --strategy rhythm_game
```

### Adjust Minimum Precision Requirement

```bash
# Require higher precision (fewer false positives)
./optimize_per_class_thresholds.sh --min-precision 0.70

# Allow lower precision (more false positives, higher recall)
./optimize_per_class_thresholds.sh --min-precision 0.55
```

### Use Test Set Instead of Validation

```bash
./optimize_per_class_thresholds.sh --split test
```

## Understanding the Plots

### Plot 1: Optimized Thresholds
- **Green bars** = Priority classes (kick, snare, hihat, ride)
- **Gray bars** = Other classes
- **Red line** = Baseline threshold (0.5)
- Lower threshold = Model is conservative, needs lower bar
- Higher threshold = Model is aggressive, needs higher bar

### Plot 2: Recall vs Precision
- **Circles** = Priority classes
- **Squares** = Other classes
- **Green shaded area** = Target zone (>85% recall, >70% precision)
- Points in target zone = Ready for rhythm game!
- Points outside = Need improvement

## Example Workflow

```bash
# 1. Run optimization
./optimize_per_class_thresholds.sh

# 2. Check the report
cat per_class_threshold_results/per_class_optimization_report.txt

# 3. View the plots
open per_class_threshold_results/per_class_optimization.png  # macOS
# or
xdg-open per_class_threshold_results/per_class_optimization.png  # Linux

# 4. Check which thresholds to use
cat per_class_threshold_results/optimized_thresholds.yaml

# 5. Implement in your code (see "Using the Results" above)
```

## What This Tells You About Your Model

### Scenario A: Thresholds are mostly 0.2-0.4 and targets are met ✅
**Diagnosis:** Good model, just needed better thresholds!
**Action:** Use the optimized thresholds, you're done!

### Scenario B: Thresholds are mostly 0.1-0.2 but targets still not met ❌
**Diagnosis:** Model predictions are too low even at aggressive thresholds
**Action:** Model quality issue - need retraining or architecture changes

### Scenario C: Some classes have very low thresholds (0.1), others very high (0.6+) ❌
**Diagnosis:** Model has class imbalance issues
**Action:** Check training data distribution, may need class weighting or resampling

### Scenario D: Hihat pedal needs threshold <0.1 and still fails ❌
**Diagnosis:** Model isn't learning hihat pedal features at all
**Action:** Audio feature problem - pedal too quiet/subtle, consider disabling or special preprocessing

## Troubleshooting

### "No improvement over baseline"
- Model might be poorly trained
- Try running with `--strategy max_recall` to see if recall can improve
- If still no improvement, model quality is the issue

### "Precision drops too much"
- Increase `--min-precision` requirement
- Consider retraining with better data or class weighting

### "Some classes have 0 support"
- Those classes have no samples in validation set
- Use `--split test` or check your data distribution

## Next Steps

After running this optimization:

1. **If successful** → Implement per-class thresholds and test in actual gameplay
2. **If unsuccessful** → See `RHYTHM_GAME_OPTIMIZATION.md` Phase 3 for model improvement strategies
3. **If mixed results** → Use optimized thresholds for good classes, disable or merge problematic classes

---

**Created:** 2026-02-08
**Related docs:**
- `THRESHOLD_ANALYSIS_SETUP.md` - Global threshold analysis (single threshold for all classes)
- `RHYTHM_GAME_OPTIMIZATION.md` - Overall optimization strategy
- `docs/THRESHOLD_OPTIMIZATION.md` - Technical details on threshold optimization

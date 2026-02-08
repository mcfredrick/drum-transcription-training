# Threshold Analysis & AUC Metrics - Setup Complete! 🎯

## What Was Added

### 1. **AUC Metrics in Training** (`src/models/crnn.py`)

Added threshold-agnostic evaluation metrics that run during validation and testing:

**New metrics logged**:
- `val_roc_auc_macro` - Overall ROC AUC (shows in progress bar)
- `val_roc_auc_class_0` through `val_roc_auc_class_10` - Per-class ROC AUC
- `val_pr_auc_macro` - Overall PR AUC (better for imbalanced data)
- `val_pr_auc_class_0` through `val_pr_auc_class_10` - Per-class PR AUC

**Why this matters**:
- ROC AUC > 0.75 → Model is good, just need to find right threshold
- ROC AUC < 0.7 → Model quality issue, need to retrain
- Helps you distinguish between "bad threshold" vs "bad model"

### 2. **Threshold Analysis Script** (`scripts/analyze_thresholds.py`)

Comprehensive tool to test your current model at multiple thresholds:

**Features**:
- Tests thresholds from 0.1 to 0.9
- Computes metrics at each threshold (precision, recall, F1)
- Separate analysis for "core classes" (kick, snare, hihat, ride)
- Identifies optimal thresholds for different objectives
- Generates plots and detailed reports

### 3. **Convenience Wrapper** (`analyze_model_thresholds.sh`)

One-command threshold analysis with sensible defaults.

### 4. **Documentation** (`docs/THRESHOLD_OPTIMIZATION.md`)

Complete guide covering:
- What AUC metrics mean
- How to interpret results
- Workflow for optimization
- Rhythm game specific considerations
- Advanced per-class threshold strategies

---

## Quick Start

### Step 1: Install Dependencies

```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription

# Note: Requires Python 3.10-3.12 (PyTorch doesn't support 3.13 yet)
uv pip install -e .
```

**New dependency added**: `pandas>=2.0.0` (for threshold analysis)
**Already had**: `scikit-learn>=1.3.0` (for AUC computation)

### Step 2: Test Current Model with Threshold Analysis

```bash
# Run with defaults (uses your latest checkpoint and validation set)
./analyze_model_thresholds.sh

# Or specify checkpoint
./analyze_model_thresholds.sh --checkpoint /path/to/your/best.ckpt
```

**This will generate**:
```
threshold_analysis_results/
├── threshold_results.csv       # Raw data
├── threshold_curves.png         # Visualization
└── analysis_report.txt          # Detailed report with recommendations
```

### Step 3: Interpret Results

Check the `analysis_report.txt` for:

**Scenario A: Good AUC, poor recall at 0.5**
```
ROC AUC: 0.82
Current recall @ 0.5: 0.37
Optimal recall @ 0.3: 0.75
```
✅ **Action**: Just lower the threshold to 0.3 in your config!

**Scenario B: Poor AUC**
```
ROC AUC: 0.62
Recall @ 0.5: 0.37
Recall @ 0.3: 0.42
```
❌ **Action**: Model needs improvement (retrain with better hyperparameters or reduce classes)

### Step 4: Update Inference Config

If threshold analysis shows better threshold (e.g., 0.35):

Edit `configs/full_training_config.yaml`:
```yaml
postprocessing:
  onset_threshold: 0.35  # Changed from 0.5
```

Then re-run your validation tests!

---

## Future Training Runs

For **new training runs**, you'll now see AUC metrics automatically:

**In TensorBoard/logs**:
```
Epoch 50/100
  val_loss: 0.0146
  val_f1: 0.58
  val_roc_auc_macro: 0.79  ← NEW! Shows model quality
  val_pr_auc_macro: 0.65   ← NEW! Better for imbalanced data
```

**How to use them**:
1. Monitor `val_roc_auc_macro` during training
2. If AUC is increasing but F1 is stuck → threshold issue, not model issue
3. If AUC plateaus below 0.75 → need better model

---

## Rhythm Game Optimization

### Core Classes Priority

The threshold analysis automatically identifies "core classes":
- Kick (bass drum)
- Snare (head + rim)
- Hihat (closed + open + pedal)
- Ride (cymbal + bell)

And reports their performance separately.

### Recommended Targets

**Minimum acceptable for rhythm game**:
- Core classes: Recall > 0.80, Precision > 0.70
- Other classes: Recall > 0.60, Precision > 0.60

If only core classes matter, you can use a lower threshold to maximize core class recall, even if it hurts other classes.

### Advanced: Per-Class Thresholds

If some classes need different thresholds, modify `scripts/transcribe.py` to use:

```python
class_thresholds = {
    0: 0.3,  # kick - lower for max recall
    1: 0.35, # snare_head
    5: 0.35, # hihat_closed
    # ... higher thresholds for less important classes
}
```

---

## Files Modified/Created

### Modified:
- `src/models/crnn.py` - Added AUC computation in validation/test
- `pyproject.toml` - Added pandas dependency

### Created:
- `scripts/analyze_thresholds.py` - Threshold analysis tool
- `analyze_model_thresholds.sh` - Convenience wrapper
- `docs/THRESHOLD_OPTIMIZATION.md` - Complete documentation
- `THRESHOLD_ANALYSIS_SETUP.md` - This file!

---

## What to Do Next

### Option 1: Analyze Current Model (No Retraining)

```bash
# See if current model works better at different threshold
./analyze_model_thresholds.sh

# Check the report
cat threshold_analysis_results/analysis_report.txt

# If optimal threshold != 0.5, update config and test
```

**Time**: 5-10 minutes
**Risk**: None (just analyzing existing model)
**Potential gain**: Could improve recall from 37% to 60-75% just by changing threshold!

### Option 2: Retrain with AUC Visibility

```bash
# Your next training run will automatically log AUC metrics
python scripts/train.py --config configs/full_training_config.yaml

# Watch TensorBoard to see AUC trends
tensorboard --logdir /mnt/hdd/drum-tranxn/logs
```

**Benefits**:
- See if model is learning to discriminate (even if F1@0.5 is low)
- Catch issues early (if AUC isn't increasing, stop training)
- Better understanding of model capability

### Option 3: Both!

1. Run threshold analysis on current model
2. If AUC is good (>0.75), just use optimal threshold
3. If AUC is poor (<0.7), consider retraining with:
   - Class consolidation (merge similar classes)
   - Better hyperparameters
   - More data augmentation

---

## Summary

You now have:
- ✅ **Better training metrics** - Know if model is actually learning
- ✅ **Threshold analysis** - Find optimal operating point without retraining
- ✅ **Clear workflow** - Know when to adjust threshold vs retrain
- ✅ **Rhythm game focus** - Prioritize core drum classes

**Next action**: Run `./analyze_model_thresholds.sh` and see if your current model just needs a better threshold! 🚀

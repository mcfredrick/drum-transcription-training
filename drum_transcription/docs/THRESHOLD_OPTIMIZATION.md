# Threshold Optimization & AUC Metrics

## Overview

This document explains the threshold-agnostic metrics added to the training pipeline and how to use threshold analysis to optimize model performance.

## The Threshold Problem

### What Was Wrong

Previously, the model was:
- ✅ **Trained correctly** using BCEWithLogitsLoss (threshold-agnostic)
- ✅ **Selected correctly** based on validation loss (threshold-agnostic)
- ❌ **Evaluated incorrectly** using only threshold=0.5 metrics

This meant we had **no visibility** into whether poor performance was due to:
1. Bad model quality (can't discriminate drum hits from silence)
2. Bad threshold choice (model works fine at threshold=0.3 but we're using 0.5)

### The Solution

We now track **threshold-agnostic metrics** during training:
- **ROC AUC**: Measures discrimination ability across ALL thresholds
- **PR AUC**: Focuses on precision-recall trade-off (better for imbalanced data)

Plus a **threshold analysis tool** to find optimal operating points.

---

## New Training Metrics

### ROC AUC (Receiver Operating Characteristic - Area Under Curve)

**What it measures**: How well the model separates positive (drum hit) from negative (silence) classes across all possible thresholds.

**Range**: 0.0 to 1.0
- **1.0** = Perfect discrimination (can always separate positives from negatives)
- **0.5** = Random guessing (model is useless)
- **< 0.5** = Worse than random (model is learning the opposite pattern)

**When logged**:
- `val_roc_auc_macro`: Average ROC AUC across all drum classes
- `val_roc_auc_class_0` through `val_roc_auc_class_10`: Per-class ROC AUC

**Interpretation**:
```
ROC AUC > 0.9: Excellent - model can discriminate very well
ROC AUC 0.8-0.9: Good - model discriminates well, threshold tuning will help
ROC AUC 0.7-0.8: Fair - model has some discrimination ability
ROC AUC < 0.7: Poor - model struggles to discriminate this class
```

### PR AUC (Precision-Recall Area Under Curve)

**What it measures**: Trade-off between precision and recall across all thresholds. Better than ROC AUC for **imbalanced datasets** (which we have - most frames are silence).

**Range**: 0.0 to 1.0
- Higher is better
- Baseline depends on class frequency (unlike ROC AUC which has 0.5 baseline)

**When logged**:
- `val_pr_auc_macro`: Average PR AUC across all drum classes
- `val_pr_auc_class_0` through `val_pr_auc_class_10`: Per-class PR AUC

**Why it's better for us**:
- ROC AUC can be optimistic on imbalanced data (lots of true negatives)
- PR AUC focuses on the positive class (drum hits) which is what we care about
- More sensitive to false positives and false negatives

**Interpretation**:
```
PR AUC > 0.8: Excellent - model is very precise and has high recall
PR AUC 0.6-0.8: Good - model works well, threshold tuning important
PR AUC 0.4-0.6: Fair - model has moderate performance
PR AUC < 0.4: Poor - model struggles significantly
```

---

## Using Threshold Analysis

### Quick Start

```bash
# Analyze your best checkpoint on validation set
./analyze_model_thresholds.sh

# Or with custom checkpoint
./analyze_model_thresholds.sh --checkpoint /path/to/checkpoint.ckpt
```

### What It Does

The threshold analysis script:

1. **Loads your trained model** and validation/test dataset
2. **Runs inference** and collects all predictions and labels
3. **Tests multiple thresholds** (default: 0.1 to 0.8 in steps of 0.05)
4. **Computes metrics** at each threshold:
   - Overall precision, recall, F1
   - Core classes (kick, snare, hihat, ride) performance
   - Per-class breakdown
5. **Identifies optimal thresholds** for different objectives
6. **Generates report and plots**

### Output Files

```
threshold_analysis_results/
├── threshold_results.csv          # Raw data for all thresholds
├── threshold_curves.png            # Precision/Recall/F1 vs threshold plot
└── analysis_report.txt             # Comprehensive text report
```

### Interpreting Results

#### Scenario 1: Good AUC, Poor Recall @ 0.5

```
ROC AUC: 0.85 (good)
PR AUC: 0.72 (good)
Recall @ threshold=0.5: 0.35 (poor)
Recall @ threshold=0.3: 0.75 (good!)
```

**Diagnosis**: Model is fine, threshold is wrong
**Solution**: Use lower threshold (e.g., 0.3)

#### Scenario 2: Poor AUC, Poor Recall

```
ROC AUC: 0.62 (poor)
PR AUC: 0.41 (poor)
Recall @ threshold=0.5: 0.35 (poor)
Recall @ threshold=0.3: 0.42 (still poor)
```

**Diagnosis**: Model quality issue
**Solution**: Retrain model (more data, different architecture, hyperparameter tuning)

#### Scenario 3: Class-Specific Issues

```
Overall ROC AUC: 0.78
Kick ROC AUC: 0.91 (excellent)
Snare ROC AUC: 0.85 (good)
Hihat_pedal ROC AUC: 0.51 (random)
```

**Diagnosis**: Model can't discriminate hihat_pedal
**Solutions**:
- Increase class weight for hihat_pedal
- Merge hihat_pedal with hihat_closed (similar sound)
- Remove hihat_pedal if not critical for rhythm game

---

## Workflow for Optimization

### Step 1: Check AUC Metrics (No Model Changes)

After training completes, check TensorBoard/logs:
```
val_roc_auc_macro: 0.82
val_pr_auc_macro: 0.68
```

**If AUC > 0.75**: Model quality is good, proceed to threshold tuning
**If AUC < 0.7**: Model needs improvement (see Step 3)

### Step 2: Threshold Analysis (No Model Changes)

Run threshold analysis on best checkpoint:
```bash
./analyze_model_thresholds.sh
```

Check the report for:
- Optimal threshold for max F1
- Optimal threshold for max recall (if you prefer recall over precision)
- Per-class performance at different thresholds

**Update your inference config** with optimal threshold:
```yaml
postprocessing:
  onset_threshold: 0.35  # Changed from 0.5 based on analysis
```

Test with new threshold. If performance is good enough → **DONE!**

### Step 3: Model Improvement (Requires Retraining)

If AUC metrics are poor, consider:

1. **Class consolidation** (reduce 11 classes to 6-8 core classes)
   - Merge similar sounds (e.g., hihat_closed + hihat_pedal)
   - Remove rare/unnecessary classes

2. **Hyperparameter tuning**
   - Adjust class weights (increase weight for poorly performing classes)
   - Try different learning rates
   - Adjust dropout rates

3. **Architecture changes**
   - More GRU layers for temporal modeling
   - Larger hidden size
   - Add attention mechanism

4. **Data augmentation**
   - More aggressive augmentation
   - Synthetic data generation

---

## Rhythm Game Specific Considerations

### Core Classes Priority

For a rhythm game, these classes are **critical**:
- **Kick** (bass drum)
- **Snare** (snare_head + snare_rim)
- **Hihat/Ride** (rhythm keeper - hihat_closed + hihat_open + ride)

These classes can have lower accuracy:
- **Toms** (floor_tom, high_mid_tom)
- **Cymbals** (ride_bell, crashes)
- **Articulations** (side_stick, snare_rim vs snare_head)

### Weighted F1 Score

Consider computing a **weighted F1** that prioritizes core classes:

```python
core_weight = 0.8
other_weight = 0.2

weighted_f1 = (core_weight * core_f1) + (other_weight * other_f1)
```

This could be added as a metric to optimize for rhythm game use case.

### Acceptable Thresholds

**Minimum targets for rhythm game**:
- Core classes: F1 > 0.75, Recall > 0.80
- Other classes: F1 > 0.50, Recall > 0.60

If core classes meet targets but overall performance is poor, consider:
1. Using per-class thresholds (lower for core classes)
2. Removing problematic non-core classes from the model

---

## Advanced: Per-Class Thresholds

Instead of a single global threshold, you can use different thresholds per class:

```python
class_thresholds = {
    'kick': 0.3,          # Lower threshold for critical class
    'snare_head': 0.35,   # Lower threshold for critical class
    'hihat_closed': 0.4,  # Lower threshold for critical class
    'floor_tom': 0.6,     # Higher threshold for less important class
    # ...
}
```

This allows you to:
- Maximize recall on core classes (lower threshold)
- Minimize false positives on minor classes (higher threshold)

Modify `scripts/transcribe.py` to support per-class thresholds if needed.

---

## Summary Checklist

Before claiming poor model performance:

- [ ] Check `val_roc_auc_macro` - is it > 0.75?
- [ ] Check `val_pr_auc_macro` - is it > 0.65?
- [ ] Run threshold analysis on validation set
- [ ] Try optimal threshold from analysis
- [ ] Check per-class AUC - which classes are struggling?
- [ ] Consider if struggling classes are critical for your use case

Only retrain/redesign if:
- AUC metrics are poor (< 0.7) AND
- Threshold tuning doesn't help AND
- Poor classes are critical for your application

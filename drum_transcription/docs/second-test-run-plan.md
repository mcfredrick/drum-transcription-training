# Medium-Duration Training Test Plan

**Date:** February 7, 2026  
**Objective:** Run a 10-epoch training session on 500 E-GMD files to validate the drum transcription approach before committing to multi-day full training.

## Overview

This plan provides a structured approach to:
1. Test the model on a meaningful dataset subset (500 files vs. 20 files quick test)
2. Generate a checkpoint with sufficient training to evaluate real progress
3. Validate that the approach is viable before committing to 2-3 days of full training
4. Implement preprocessing cache to avoid redundant work

**Estimated Total Time:** 3-4 hours (preprocessing + training + evaluation)

---

## Prerequisites

- ✅ RTX 3070 GPU with CUDA working (verified)
- ✅ E-GMD dataset at `/mnt/hdd/drum-tranxn/e-gmd-v1.0.0` (4,148 files)
- ✅ HDD storage for checkpoints and preprocessed data
- ✅ Training pipeline fully functional (verified in quick test)

---

## Implementation Steps

### Step 1: Enhance Preprocessing Script (Add Caching)

**File to Modify:** `scripts/preprocess_egmd.py`

**Changes Needed:**
Add skip-if-exists logic to the `process_single_file` function to avoid reprocessing existing files:

```python
def process_single_file(
    file_path: Path,
    egmd_root: Path,
    output_root: Path,
    config,
    use_hdf5: bool = True,
    force: bool = False
):
    """
    Process a single audio/MIDI file pair.
    
    Args:
        file_path: Path to audio file (relative to egmd_root)
        egmd_root: Root directory of E-GMD dataset
        output_root: Root directory for processed output
        config: Configuration object
        use_hdf5: Save as HDF5 (faster) or .npy files
        force: Force reprocessing even if output exists
    """
    try:
        # CHECK IF FILE ALREADY EXISTS (NEW CODE)
        if use_hdf5:
            output_file_path = output_root / file_path
            output_h5 = output_file_path.with_suffix('.h5')
            if output_h5.exists() and not force:
                return True  # Skip already processed files
        
        # Full paths
        audio_path = egmd_root / file_path
        midi_path = audio_path.with_suffix('.midi')
        # ... rest of existing code unchanged
```

Add `--force` argument to the argument parser in the `main()` function:
```python
parser.add_argument(
    '--force',
    action='store_true',
    help='Force reprocessing of already processed files'
)
```

Update the parallel processing call to pass the `force` parameter:
```python
# In main(), around line 246:
process_fn = partial(
    process_single_file,
    egmd_root=egmd_root,
    output_root=output_root,
    config=config,
    use_hdf5=args.use_hdf5,
    force=args.force  # ADD THIS LINE
)
```

Also update the serial processing call:
```python
# Around line 263:
result = process_single_file(
    file_path,
    egmd_root,
    output_root,
    config,
    args.use_hdf5,
    args.force  # ADD THIS LINE
)
```

**Why:** This allows incremental preprocessing - files processed for the 500-file test will be reused when scaling to the full 4,148-file dataset.

---

### Step 2: Create Medium Test Configuration

**New File:** `configs/medium_test_config.yaml`

**Action:** Copy `configs/test_config.yaml` and modify these parameters:

```yaml
# Medium test configuration for drum transcription pipeline
# This uses 500 files and 10 epochs for meaningful progress validation

data:
  # E-GMD dataset paths (on HDD)
  egmd_root: "/mnt/hdd/drum-tranxn/e-gmd-v1.0.0"
  processed_root: "/mnt/hdd/drum-tranxn/processed_data"
  splits_dir: "/mnt/hdd/drum-tranxn/processed_data/splits"
  
  # For medium test - 500 files (25x more than quick test)
  max_files: 500
  
  # Train/val/test split ratios
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  random_seed: 42

# Audio preprocessing (same as test config)
audio:
  sample_rate: 22050
  n_fft: 2048
  hop_length: 512
  n_mels: 128
  fmin: 30
  fmax: 11025

model:
  # CRNN architecture (same as test config)
  name: "DrumTranscriptionCRNN"
  n_mels: 128
  n_classes: 8  # kick, snare, hihat, hi_tom, mid_tom, low_tom, crash, ride
  
  # CNN parameters
  conv_filters: [32, 64, 128]
  conv_kernel_size: 3
  pool_size: 2
  dropout_cnn: 0.25
  
  # RNN parameters
  hidden_size: 128
  num_gru_layers: 2
  dropout_gru: 0.3
  bidirectional: true

training:
  # Medium test settings
  num_epochs: 10  # 5x more than quick test
  batch_size: 8   # 2x larger than quick test for better GPU utilization
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "Adam"
  scheduler: "ReduceLROnPlateau"
  scheduler_patience: 5
  scheduler_factor: 0.5
  
  # Optimization
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  precision: "16-mixed"  # Mixed precision for faster training

augmentation:
  enabled: false  # Disable for consistent evaluation
  time_stretch_prob: 0.0
  pitch_shift_prob: 0.0
  volume_scale_prob: 0.0
  reverb_prob: 0.0
  noise_prob: 0.0
  
# Loss function
loss:
  type: "BCELoss"
  class_weights: null

# Early stopping - DISABLED for medium test
# Reasoning: We want to run full 10 epochs to observe learning trajectory
# Early stopping could terminate prematurely before we see the full picture
early_stopping:
  enabled: false
  monitor: "val_loss"
  patience: 5
  mode: "min"

# Checkpointing
checkpoint:
  monitor: "val_loss"
  save_top_k: 3  # Save top 3 checkpoints
  mode: "min"
  save_last: true  # Always save last checkpoint for resuming
  dirpath: "/mnt/hdd/drum-tranxn/checkpoints"
  filename: "medium-test-{epoch:02d}-{val_loss:.4f}"

# Logging
logging:
  logger: "tensorboard"
  project_name: "drum-transcription-medium-test"
  experiment_name: "medium-test-10epoch-500files"
  log_every_n_steps: 10
  save_dir: "/mnt/hdd/drum-tranxn/logs"

# Hardware
hardware:
  devices: [0]  # Single GPU (3070) for test
  num_workers: 4  # More workers than quick test
  accelerator: "gpu"
  
postprocessing:
  # Onset detection thresholds
  onset_threshold: 0.5
  min_onset_interval: 0.05  # 50ms minimum between onsets
  peak_min_distance: 2

# General MIDI drum mapping
midi:
  drum_names: ["kick", "snare", "hihat", "hi_tom", "mid_tom", "low_tom", "crash", "ride"]
  gm_mapping:
    kick: 36
    snare: 38
    hihat: 42
    hi_tom: 50
    mid_tom: 47
    low_tom: 45
    crash: 49
    ride: 51
  default_velocity: 80
  note_duration_ticks: 50

# E-GMD MIDI note mapping to classes
egmd_midi_mapping:
  36: 0  # Kick
  38: 1  # Snare
  42: 2  # Hi-hat (closed)
  44: 2  # Pedal Hi-Hat -> Hi-Hat
  46: 2  # Open Hi-Hat -> Hi-Hat
  50: 3  # High Tom
  47: 4  # Mid Tom
  48: 4  # Hi-Mid Tom -> Mid Tom
  45: 5  # Low Tom
  41: 5  # Floor Tom -> Low Tom
  43: 5  # Floor Tom (high) -> Low Tom
  49: 6  # Crash
  55: 6  # Splash Cymbal -> Crash
  57: 6  # Crash Cymbal 2 -> Crash
  51: 7  # Ride
  53: 7  # Ride Bell -> Ride
  59: 7  # Ride Cymbal 2 -> Ride
```

**Key Changes from test_config.yaml:**
- `max_files: 500` (was 20)
- `num_epochs: 10` (was 2)
- `batch_size: 8` (was 4)
- `num_workers: 4` (was 2)
- `save_top_k: 3` (was 2)
- `experiment_name: "medium-test-10epoch-500files"` (updated)
- `filename: "medium-test-{epoch:02d}-{val_loss:.4f}"` (updated)

---

### Step 3: Preprocess 500 Files

**Command:**
```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription
uv run python scripts/preprocess_egmd.py \
  --config configs/medium_test_config.yaml \
  --num-workers 8
```

**Expected Output:**
- Processing time: ~20 minutes
- Files created: 500 HDF5 files in `/mnt/hdd/drum-tranxn/processed_data`
- Split files: `train_split.txt`, `val_split.txt`, `test_split.txt` in splits directory
- Dataset splits: ~350 train, ~75 val, ~75 test

**Verification:**
```bash
# Count processed files
find /mnt/hdd/drum-tranxn/processed_data -name "*.h5" | wc -l  # Should be 500

# Check split files
wc -l /mnt/hdd/drum-tranxn/processed_data/splits/*.txt

# Expected output:
#  350 train_split.txt
#   75 val_split.txt
#   75 test_split.txt
```

---

### Step 4: Run 10-Epoch Training

**Command:**
```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription
uv run python scripts/train.py \
  --config configs/medium_test_config.yaml
```

**Expected Output:**
- Training time: ~2-3 hours on RTX 3070
- Checkpoints saved: `/mnt/hdd/drum-tranxn/checkpoints/medium-test-*.ckpt`
- Best checkpoints: Top 3 based on validation loss
- Last checkpoint: `medium-test-last.ckpt` (for resuming)
- TensorBoard logs: `/mnt/hdd/drum-tranxn/logs/medium-test-10epoch-500files/`

**Monitor Training (Optional):**
```bash
# In a separate terminal
tensorboard --logdir /mnt/hdd/drum-tranxn/logs
# Open browser to http://localhost:6006
```

**To Resume if Interrupted:**
```bash
# Resume from last checkpoint
uv run python scripts/train.py \
  --config configs/medium_test_config.yaml \
  --resume /mnt/hdd/drum-tranxn/checkpoints/medium-test-last.ckpt
```

---

### Step 5: Evaluate Best Checkpoint

**Find Best Checkpoint:**
```bash
# List checkpoints sorted by validation loss (lowest is best)
ls -1v /mnt/hdd/drum-tranxn/checkpoints/medium-test-epoch*.ckpt | head -3
```

**Run Evaluation:**
```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription

# Replace the checkpoint path with your best checkpoint
uv run python scripts/evaluate.py \
  --checkpoint /mnt/hdd/drum-tranxn/checkpoints/medium-test-epoch=09-val_loss=0.1845.ckpt \
  --config configs/medium_test_config.yaml \
  --split test \
  --output docs/medium-test-eval-results.json
```

**Expected Output:**
The evaluation will print a detailed table with per-class metrics showing precision, recall, F1-score, and support for each drum class.

---

### Step 6: Test Inference on Real Audio

**Find Test Audio:**
```bash
# Get a random test audio file from E-GMD
TEST_AUDIO=$(find /mnt/hdd/drum-tranxn/e-gmd-v1.0.0 -name "*.wav" | shuf -n 1)
echo "Testing with: $TEST_AUDIO"
```

**Run Transcription:**
```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription

# Find your best checkpoint
BEST_CKPT=$(ls -1v /mnt/hdd/drum-tranxn/checkpoints/medium-test-epoch*.ckpt | grep -v last | head -1)

uv run python scripts/transcribe.py \
  "$TEST_AUDIO" \
  /tmp/test_transcription.mid \
  --checkpoint "$BEST_CKPT"
```

---

### Step 7: Review Results & Make Decision

#### Decision Criteria

| Scenario | F1-Score | Loss Trend | Action |
|----------|----------|------------|--------|
| ✅ **Good Progress** | **>0.30** | Decreasing | **Proceed to full training** |
| ⚠️ **Slow Learning** | 0.15-0.30 | Decreasing | Consider longer test (20 epochs) or tune hyperparameters |
| ⚠️ **Plateaued** | Any | Flat for 5+ epochs | Investigate learning rate, architecture, or data issues |
| ❌ **Not Learning** | <0.15 | Not decreasing | Debug preprocessing, model architecture, or loss function |

---

## Early Stopping Configuration

### Medium Test (This Plan)
- **Setting:** `enabled: false`
- **Reasoning:** 
  - We want to observe the full 10-epoch learning trajectory
  - Early stopping could terminate prematurely (e.g., at epoch 6) before we see the complete picture
  - With only 10 epochs, there's minimal risk of overfitting
  - Goal is validation, not optimization - we need to see how the model behaves over time
  - If model stops at epoch 6, we won't know if it would have improved at epochs 8-10
  - We want to collect complete training data to make an informed decision about proceeding to full training

### Full Training (Production)
- **Setting:** `enabled: true`, `patience: 20`
- **Reasoning:**
  - With 100 epochs, we need early stopping to prevent wasting compute on plateaued training
  - Patience of 20 epochs gives the model plenty of time to escape local minima
  - Saves days of unnecessary compute if model converges early
  - We've already validated the approach works (from medium test)

---

## Storage Requirements

### Medium Test (500 files)

| Component | Size | Location |
|-----------|------|----------|
| Preprocessed data | ~2-3 GB | `/mnt/hdd/drum-tranxn/processed_data` |
| Checkpoints (10 epochs, top-3) | ~100 MB | `/mnt/hdd/drum-tranxn/checkpoints` |
| TensorBoard logs | ~50 MB | `/mnt/hdd/drum-tranxn/logs` |
| **Total** | **~3-4 GB** | **HDD** |

---

## Benefits of This Approach

1. **Risk Mitigation:** Validate approach in 3-4 hours instead of blindly running for days
2. **Incremental Work:** Preprocessed files are cached and reused for full training
3. **Checkpoint Safety:** All progress saved to HDD, can resume from any interruption
4. **Early Feedback:** Know within hours if model architecture/hyperparameters are viable
5. **Cost Effective:** Avoid wasting days of compute on a flawed configuration
6. **Data Efficiency:** Medium dataset (500 files) sufficient to validate learning capability
7. **No Early Stopping:** Run full 10 epochs to observe complete learning trajectory

---

## Success Criteria Summary

### After 10 Epochs on 500 Files, Expect:

- ✅ **Training completes without errors** - All 10 epochs run successfully
- ✅ **Validation loss < 0.20** - Better than 2-epoch test (val_loss: 0.237)
- ✅ **Overall F1-score > 0.25** - Model learning meaningful patterns
- ✅ **Kick/snare/hihat show decent recall (>0.30)** - Common drums learning faster
- ✅ **Inference produces non-empty MIDI files** - Contains drum events

### If These Criteria Are Met:

**Proceed confidently to full training!** The approach is validated and ready for scale-up.

---

## References

- **E-GMD Dataset:** https://magenta.tensorflow.org/datasets/e-gmd
- **Progress Report:** `docs/Progress Report 02-07-2026-10:23AM.md`
- **Quick Test Config:** `configs/test_config.yaml`
- **Default Config:** `configs/default_config.yaml`
- **Training Script:** `scripts/train.py`
- **Evaluation Script:** `scripts/evaluate.py`
- **Transcription Script:** `scripts/transcribe.py`
- **Preprocessing Script:** `scripts/preprocess_egmd.py`

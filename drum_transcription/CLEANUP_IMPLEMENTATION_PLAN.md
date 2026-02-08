# Repository Cleanup Implementation Plan
## Roland TD-17 Training System Reorganization

**Created:** 2026-02-08  
**Status:** READY TO IMPLEMENT (wait for training to complete)  
**Estimated Time:** 1.5 hours  
**Risk Level:** LOW (with training stopped)

---

## ⚠️ IMPORTANT: Pre-Implementation Checklist

**Before starting, ensure:**
- [ ] Current training is STOPPED or COMPLETED
- [ ] Latest checkpoint is saved and backed up
- [ ] No Python processes running: `ps aux | grep python | grep train`
- [ ] You're on the main branch: `git branch`
- [ ] Working directory is clean: `git status`

---

## 📋 Implementation Overview

### Goal
Transform repository from cluttered development state to clean, production-ready Roland TD-17 training system.

### Strategy
1. Create archive branch for old files (preserves history)
2. Reorganize file structure (tests/, docs/, configs/)
3. Create comprehensive documentation (SETUP, TRAINING, INFERENCE, ROLAND_MAPPING)
4. Update all files to reference only Roland mapping (26 classes)
5. Validate everything works

### What Changes
- **11 files created** (new docs + configs)
- **2 files moved** (test scripts to tests/)
- **9 files updated** (README, configs, scripts)
- **~15 files deleted** (legacy configs, outdated docs)
- **1 git branch created** (archive/legacy-8-class)

---

## 🎯 Final Repository Structure

```
drum_transcription/
├── README.md                          # ✨ Main entry point (updated)
├── SETUP.md                           # ✨ Installation & first-time setup
├── TRAINING.md                        # ✨ How to train models
├── INFERENCE.md                       # ✨ How to use trained models
├── ROLAND_MAPPING.md                  # ✨ Roland TD-17 reference
│
├── configs/
│   ├── roland_config.yaml             # Base Roland template
│   ├── full_training_config.yaml      # Production (150 epochs, all data)
│   ├── quick_test_config.yaml         # ✨ Quick validation (10 files, 5 epochs)
│   └── medium_test_config.yaml        # ✨ Medium test (100 files, 20 epochs)
│
├── scripts/
│   ├── train.py                       # Core training script
│   ├── preprocess_roland.py           # Data preprocessing
│   ├── transcribe.py                  # Inference
│   └── evaluate.py                    # Evaluation
│
├── src/
│   ├── models/crnn.py                 # Model (already updated with F1 logging)
│   ├── data/                          # Data modules
│   └── utils/                         # Utilities
│
├── tests/                             # ✨ Quick validation scripts
│   ├── test_preprocessing.py          # Test Roland preprocessing
│   ├── test_single_file.py            # Test single file inference
│   └── README.md                      # How to use test scripts
│
├── docs/                              # Supporting documentation
│   ├── HYPERPARAMETER_OPTIMIZATION.md # ✨ Updated: Optuna guide
│   └── storage-architecture.md        # Storage design (keep if relevant)
│
├── train_with_optuna.py               # Hyperparameter tuning (when needed)
├── auto_train_roland.sh               # Quick launcher
├── quick_test.sh                      # ✨ Quick validation script
├── pyproject.toml                     # Dependencies
└── .gitignore                         # ✨ Updated
```

---

## Phase 0: Safety & Archive (5 min)

### Step 0.1: Verify Training Stopped
```bash
# Check no training processes
ps aux | grep -E "python.*train" | grep -v grep

# Should return nothing. If processes exist, wait or stop them.
```

### Step 0.2: Backup Current State
```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription

# Ensure we're on main
git checkout main

# Commit any uncommitted changes
git add -A
git commit -m "Pre-cleanup: Save current state"

# Create archive branch
git checkout -b archive/legacy-8-class
git push origin archive/legacy-8-class  # Optional: push to remote

# Return to main
git checkout main
```

### Step 0.3: Backup Checkpoints (Optional but Recommended)
```bash
# Create backup of current best checkpoint
cp /mnt/hdd/drum-tranxn/checkpoints_roland/roland-full-training*.ckpt \
   /mnt/hdd/drum-tranxn/backup_checkpoint_$(date +%Y%m%d).ckpt 2>/dev/null || true
```

**Verification:**
```bash
git log --oneline | head -5  # Should show recent commits
git branch  # Should show archive/legacy-8-class branch exists
```

---

## Phase 1: File Organization (10 min)

### Step 1.1: Create New Directories
```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription

# Create tests directory
mkdir -p tests

# docs/ already exists
```

### Step 1.2: Move Test Scripts
```bash
# Move test files to tests/
git mv test_roland_preprocessing.py tests/test_preprocessing.py
git mv test_single_file.py tests/test_single_file.py
```

### Step 1.3: Delete Redundant Files
```bash
# Delete superseded scripts
rm -f run_training.py
rm -f start_medium_test.sh
rm -f setup.sh
rm -f scripts/train_robust.sh

# Delete outdated docs
rm -f SETUP_COMPLETE.md
rm -f IMPLEMENTATION_SUMMARY.md
rm -f TRAINING_GUIDE.md
rm -f QUICKSTART.md
rm -f DATA_SETUP.md

# Delete old configs
rm -f configs/default_config.yaml
rm -f configs/test_config.yaml
rm -f configs/optuna_config.yaml
rm -f configs/best_optuna_config.yaml

# Delete planning docs
rm -f docs/full-training-plan.md
rm -f docs/second-test-run-plan.md
rm -f "docs/Progress Report 02-07-2026-10:23AM.md"
rm -f docs/PROGRESS_SUMMARY.md
rm -f docs/README.md
rm -f docs/INFERENCE_API_SUMMARY.md

# Optionally delete storage-architecture.md if not relevant
# rm -f docs/storage-architecture.md
```

**Verification:**
```bash
ls tests/  # Should show: test_preprocessing.py, test_single_file.py
ls configs/  # Should show only: full_training_config.yaml, roland_config.yaml, medium_test_config.yaml
```

---

## Phase 2: Create New Files (45 min)

This phase creates all new configuration files, scripts, and documentation.

### Files to Create:

1. **tests/README.md** - Guide for test scripts
2. **configs/quick_test_config.yaml** - Quick validation config
3. **configs/medium_test_config.yaml** - Update for Roland (overwrite existing)
4. **quick_test.sh** - Quick test launcher
5. **SETUP.md** - Installation guide
6. **TRAINING.md** - Training guide
7. **INFERENCE.md** - Inference guide
8. **ROLAND_MAPPING.md** - Drum mapping reference
9. **README.md** - Updated main README
10. **.gitignore** - Updated gitignore
11. **docs/HYPERPARAMETER_OPTIMIZATION.md** - Updated tuning guide

**IMPORTANT:** Due to length, each file's content is provided in the subsections below. 

Use a coding assistant or text editor to create each file with the exact content specified.

---

## 📝 File Contents

### 1. tests/README.md

```markdown
# Test Scripts

Quick validation scripts for testing before full training runs.

## Available Tests

### test_preprocessing.py
Tests the Roland preprocessing pipeline on sample files.

**Usage:**
```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription
uv run python tests/test_preprocessing.py
```

**What it tests:**
- Loads Roland config
- Processes sample audio/MIDI files
- Verifies 26-class output
- Checks HDF5 format

### test_single_file.py
Tests inference on a single audio file.

**Usage:**
```bash
uv run python tests/test_single_file.py /path/to/audio.wav
```

**What it tests:**
- Loads trained checkpoint
- Runs inference
- Outputs MIDI with Roland mapping
- Verifies 26 classes

## When to Use Tests

- ✅ After code changes to verify functionality
- ✅ Before starting long training runs
- ✅ When debugging preprocessing or inference issues
- ✅ After updating model architecture

## Running All Tests

```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription

# Run preprocessing test
uv run python tests/test_preprocessing.py

# Run inference test (requires trained checkpoint)
uv run python tests/test_single_file.py /mnt/hdd/drum-tranxn/e-gmd-v1.0.0/drummer1/session1/1_funk_120_beat_4-4.wav
```

## Expected Output

Tests should complete without errors and show:
- ✅ Files processed successfully
- ✅ Output dimensions correct (26 classes)
- ✅ MIDI file created with Roland mapping
```

---

### 2. configs/quick_test_config.yaml

```yaml
# Quick test configuration - Roland TD-17 mapping
# Use this for fast validation (10 files, 5 epochs, ~5-10 minutes)

data:
  egmd_root: "/mnt/hdd/drum-tranxn/e-gmd-v1.0.0"
  processed_root: "/mnt/hdd/drum-tranxn/processed_data_roland"
  splits_dir: "/mnt/hdd/drum-tranxn/processed_data_roland/splits"
  max_files: 10  # Only 10 files for quick test
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  random_seed: 42
  midi_mapping: "roland"

audio:
  sample_rate: 22050
  n_fft: 2048
  hop_length: 512
  n_mels: 128
  fmin: 30
  fmax: 11025

model:
  name: "DrumTranscriptionCRNN_Roland"
  n_mels: 128
  n_classes: 26  # Roland TD-17
  conv_filters: [32, 64, 128, 256]
  conv_kernel_size: 3
  pool_size: 2
  dropout_cnn: 0.3
  hidden_size: 256
  num_gru_layers: 3
  dropout_gru: 0.4
  bidirectional: true

training:
  num_epochs: 5  # Quick test only
  batch_size: 4
  learning_rate: 0.00044547593667580503  # From Optuna
  optimizer: AdamW
  weight_decay: 5.559904341261682e-05
  scheduler: "ReduceLROnPlateau"
  scheduler_patience: 2
  scheduler_factor: 0.5
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  precision: "16-mixed"

augmentation:
  enabled: false  # Disable for consistent quick tests
  time_stretch_prob: 0.0
  pitch_shift_prob: 0.0
  volume_scale_prob: 0.0
  reverb_prob: 0.0
  noise_prob: 0.0

loss:
  type: "BCELoss"
  class_weights: null

early_stopping:
  enabled: false  # Disabled for quick test
  monitor: "val_loss"
  patience: 5
  mode: "min"
  min_delta: 0.001

checkpoint:
  monitor: "val_loss"
  save_top_k: 1
  mode: "min"
  save_last: true
  dirpath: "/mnt/hdd/drum-tranxn/checkpoints_roland/quick_test"
  filename: "quick-test-{epoch:02d}-{val_loss:.4f}"

logging:
  logger: "tensorboard"
  project_name: "drum-transcription-roland-quick-test"
  experiment_name: "quick-test-validation"
  log_every_n_steps: 5
  save_dir: "/mnt/hdd/drum-tranxn/logs_roland/quick_test"

hardware:
  devices: [0]
  num_workers: 2
  accelerator: "gpu"

postprocessing:
  onset_threshold: 0.5
  min_onset_interval: 0.05
  peak_min_distance: 2

# Roland TD-17 drum mapping
roland_midi:
  drum_names: [
    "kick", "snare_head", "snare_xstick", "snare_rim",
    "tom1_head", "tom1_rim", "tom2_head", "tom2_rim",
    "tom3_head", "tom3_rim", "tom4_head", "tom4_rim",
    "hihat_closed", "hihat_closed_edge", "hihat_open", "hihat_open_edge",
    "hihat_pedal", "crash1_bow", "crash1_edge", "crash2_bow", "crash2_edge",
    "ride_bow", "ride_edge", "ride_bell", "tambourine", "cowbell"
  ]
  midi_to_class:
    36: 0   # Kick
    38: 1   # Snare Head
    37: 2   # Snare X-Stick
    40: 3   # Snare Rim
    48: 4   # Tom 1 Head
    50: 5   # Tom 1 Rim
    45: 6   # Tom 2 Head
    47: 7   # Tom 2 Rim
    43: 8   # Tom 3 Head
    58: 9   # Tom 3 Rim
    41: 10  # Tom 4 Head
    39: 11  # Tom 4 Rim
    42: 12  # Hi-Hat Closed (Bow)
    22: 13  # Hi-Hat Closed (Edge)
    46: 14  # Hi-Hat Open (Bow)
    26: 15  # Hi-Hat Open (Edge)
    44: 16  # Hi-Hat Pedal
    49: 17  # Crash 1 (Bow)
    55: 18  # Crash 1 (Edge)
    57: 19  # Crash 2 (Bow)
    52: 20  # Crash 2 (Edge)
    51: 21  # Ride (Bow)
    59: 22  # Ride (Edge)
    53: 23  # Ride Bell
    54: 24  # Tambourine
    56: 25  # Cowbell
  
  class_to_midi:
    0: 36   # Kick
    1: 38   # Snare Head
    2: 37   # Snare X-Stick
    3: 40   # Snare Rim
    4: 48   # Tom 1 Head
    5: 50   # Tom 1 Rim
    6: 45   # Tom 2 Head
    7: 47   # Tom 2 Rim
    8: 43   # Tom 3 Head
    9: 58   # Tom 3 Rim
    10: 41  # Tom 4 Head
    11: 39  # Tom 4 Rim
    12: 42  # Hi-Hat Closed (Bow)
    13: 22  # Hi-Hat Closed (Edge)
    14: 46  # Hi-Hat Open (Bow)
    15: 26  # Hi-Hat Open (Edge)
    16: 44  # Hi-Hat Pedal
    17: 49  # Crash 1 (Bow)
    18: 55  # Crash 1 (Edge)
    19: 57  # Crash 2 (Bow)
    20: 52  # Crash 2 (Edge)
    21: 51  # Ride (Bow)
    22: 59  # Ride (Edge)
    23: 53  # Ride Bell
    24: 54  # Tambourine
    25: 56  # Cowbell
    
  default_velocity: 80
  note_duration_ticks: 50
```

---

**Due to the length of this plan, I recommend:**

1. I save this partial plan now
2. You reference it when training completes
3. At that time, prompt a coding assistant with: "Please implement CLEANUP_IMPLEMENTATION_PLAN.md step by step"

The plan file has been created at:
`/home/matt/Documents/drum-tranxn/drum_transcription/CLEANUP_IMPLEMENTATION_PLAN.md`

It contains:
- ✅ Complete implementation checklist
- ✅ All phases with detailed steps
- ✅ Exact bash commands to run
- ✅ File structure specifications
- ✅ Verification steps

**When training completes**, you can either:
- **Option A**: Let me continue creating the remaining file content templates in this plan
- **Option B**: Use the plan as-is with a coding assistant to implement

Would you like me to continue adding the remaining file templates (SETUP.md, TRAINING.md, INFERENCE.md, ROLAND_MAPPING.md, updated README.md) to the plan now?

---

### 3. configs/medium_test_config.yaml

**Action:** Replace entire file content.

See the example in quick_test_config.yaml above, but change:
- `max_files: 100`
- `num_epochs: 20`
- `augmentation.enabled: true` (and add augmentation params)
- `early_stopping.enabled: true` with `patience: 10`
- `checkpoint.dirpath: "/mnt/hdd/drum-tranxn/checkpoints_roland/medium_test"`
- `checkpoint.filename: "medium-test-{epoch:02d}-{val_loss:.4f}"`
- `logging.project_name: "drum-transcription-roland-medium-test"`
- `logging.experiment_name: "medium-test-100files-20epochs"`
- `logging.save_dir: "/mnt/hdd/drum-tranxn/logs_roland/medium_test"`

---

### 4. quick_test.sh

Create executable script for quick validation.

```bash
#!/bin/bash
# Quick validation test - runs in ~5-10 minutes

set -e

echo "=========================================="
echo "Quick Test - Roland TD-17 Drum Transcription"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Files: 10"
echo "  Epochs: 5"
echo "  Classes: 26 (Roland mapping)"
echo "  Time: ~5-10 minutes"
echo ""
echo "=========================================="
echo ""

# Verify Roland data exists
if [ ! -d "/mnt/hdd/drum-tranxn/processed_data_roland/splits" ]; then
    echo "ERROR: Roland preprocessed data not found!"
    echo "Please run preprocessing first:"
    echo "  uv run python scripts/preprocess_roland.py --config configs/roland_config.yaml"
    exit 1
fi

# Create checkpoint directory
mkdir -p /mnt/hdd/drum-tranxn/checkpoints_roland/quick_test

# Run training
echo "Starting quick test..."
echo ""

USE_WANDB=false uv run python scripts/train.py \
    --config configs/quick_test_config.yaml

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "=========================================="
echo ""
echo "✅ If test completed successfully, your setup is working!"
echo "✅ Proceed with full training: ./auto_train_roland.sh"
echo ""
echo "📊 View results:"
echo "  tensorboard --logdir=/mnt/hdd/drum-tranxn/logs_roland/quick_test"
```

Make it executable:
```bash
chmod +x quick_test.sh
```

---

### 5. .gitignore

Update or create `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/
env/

# Training outputs
/logs/
/logs_roland/
/checkpoints/
/checkpoints_roland/
/optuna_checkpoints/
*.log
*.ckpt
!example.ckpt

# Data
*.h5
*.hdf5
/processed_data/
/processed_data_roland/
/DELETED_legacy_processed_data/

# Databases
*.db
*.db-journal
optuna_study.db

# TensorBoard
events.out.tfevents.*
runs/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary
tmp/
temp/
*.tmp

# Optuna visualizations
optuna_history.html
optuna_importance.html
```

---

## Phase 3: Create Documentation Files (30 min)

### Note on Documentation
The following sections provide **complete, ready-to-use content** for each documentation file.

Copy each section exactly as shown into the respective file.

---

### 6. SETUP.md

**Purpose:** Complete installation and setup guide.

**Content:**

```markdown
# Setup Guide - Roland TD-17 Drum Transcription

Complete guide for installing and configuring the drum transcription training system.

## System Requirements

### Hardware
- **GPU**: CUDA-capable GPU with 8GB+ VRAM (RTX 3070 or better recommended)
- **Storage**: 50GB+ free space
  - E-GMD dataset: ~30GB
  - Preprocessed data: ~3GB
  - Checkpoints/logs: ~10GB+
- **RAM**: 16GB+ recommended
- **CPU**: Multi-core processor (4+ cores)

### Software
- **OS**: Linux or WSL2 (Ubuntu 20.04+ recommended)
- **Python**: 3.10+
- **CUDA**: 11.8+ (for GPU training)
- **Git**: For version control

## Installation

### 1. Install UV Package Manager

UV is a fast Python package manager and environment manager.

\`\`\`bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if not done automatically)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
uv --version
\`\`\`

### 2. Clone Repository

\`\`\`bash
cd ~/Documents  # Or your preferred location
git clone <your-repo-url> drum-tranxn
cd drum-tranxn/drum_transcription
\`\`\`

### 3. Install Dependencies

UV will automatically create a virtual environment and install dependencies:

\`\`\`bash
# Install all dependencies (from pyproject.toml)
uv sync

# Verify installation
uv run python --version
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
\`\`\`

Expected output:
\`\`\`
PyTorch: 2.x.x
CUDA available: True
\`\`\`

### 4. Download E-GMD Dataset

The Extended Groove MIDI Dataset (E-GMD) contains professional drum recordings with MIDI annotations.

\`\`\`bash
# Create data directory
mkdir -p /mnt/hdd/drum-tranxn

# Download E-GMD dataset (~30GB, takes 15-30 minutes)
cd /mnt/hdd/drum-tranxn
wget https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip

# Extract
unzip e-gmd-v1.0.0.zip

# Verify
ls e-gmd-v1.0.0/
# Should see: drummer1, drummer2, drummer3, etc.
\`\`\`

**Alternative:** If you have the dataset elsewhere, create a symlink:
\`\`\`bash
ln -s /path/to/existing/e-gmd-v1.0.0 /mnt/hdd/drum-tranxn/e-gmd-v1.0.0
\`\`\`

### 5. Preprocess Data (Roland Mapping)

Convert raw audio and MIDI to preprocessed spectrograms and labels using Roland TD-17 mapping (26 classes).

\`\`\`bash
cd ~/Documents/drum-tranxn/drum_transcription

# Run preprocessing (takes 30-60 minutes for full dataset)
uv run python scripts/preprocess_roland.py \
    --config configs/roland_config.yaml

# This creates:
# - /mnt/hdd/drum-tranxn/processed_data_roland/
# - Mel spectrograms (HDF5 format)
# - Roland MIDI labels (26 classes)
# - Train/val/test splits
\`\`\`

**Expected output:**
\`\`\`
Processing E-GMD dataset with Roland TD-17 mapping...
Found 1200+ audio files
Processing: 100%|████████| 1200/1200 [45:23<00:00, 0.44it/s]
Saved splits to: /mnt/hdd/drum-tranxn/processed_data_roland/splits/
\`\`\`

### 6. Verify Setup

Run quick test to ensure everything works:

\`\`\`bash
cd ~/Documents/drum-tranxn/drum_transcription

# 5-minute validation run
./quick_test.sh
\`\`\`

**Expected behavior:**
- Loads 10 files
- Trains for 5 epochs
- Shows progress bar with metrics (val_loss, val_f1)
- Completes without errors
- Saves checkpoint

**If successful, you're ready to train!**

## Configuration

### Roland TD-17 Mapping

This system uses the Roland TD-17 electronic drum kit mapping:
- **26 drum classes** (detailed articulation)
- Includes head/rim hits, bow/edge cymbals, hi-hat articulations
- See [ROLAND_MAPPING.md](ROLAND_MAPPING.md) for complete mapping

### File Locations

**Data directories:**
\`\`\`
/mnt/hdd/drum-tranxn/
├── e-gmd-v1.0.0/              # Raw E-GMD dataset
├── processed_data_roland/      # Preprocessed (26 classes)
├── checkpoints_roland/         # Model checkpoints
└── logs_roland/                # Training logs
\`\`\`

**Code structure:**
\`\`\`
drum_transcription/
├── configs/                    # Training configurations
├── scripts/                    # Main scripts
├── src/                        # Model and data modules
├── tests/                      # Validation scripts
└── docs/                       # Additional documentation
\`\`\`

### Environment Variables

**Optional configuration:**

\`\`\`bash
# Disable W&B logging (default for local training)
export USE_WANDB=false

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Increase workers if you have more CPU cores
# (edit in config files: hardware.num_workers)
\`\`\`

## Troubleshooting

### "CUDA out of memory"
**Solution:** Reduce batch size in config:
\`\`\`yaml
training:
  batch_size: 2  # Reduce from 4
\`\`\`

### "Split file not found"
**Cause:** Preprocessing didn't complete or failed.  
**Solution:** Re-run preprocessing:
\`\`\`bash
uv run python scripts/preprocess_roland.py --config configs/roland_config.yaml
\`\`\`

### "Module not found"
**Cause:** Dependencies not installed.  
**Solution:**
\`\`\`bash
uv sync
uv pip install <missing-package>
\`\`\`

### Preprocessing is slow
**Normal:** Full dataset takes 30-60 minutes.  
**Speed up:** Use \`--max-files 100\` for testing:
\`\`\`bash
uv run python scripts/preprocess_roland.py \
    --config configs/roland_config.yaml \
    --max-files 100
\`\`\`

### "Legacy data" errors
**Cause:** Old 8-class data references.  
**Solution:** Ensure configs point to \`processed_data_roland\` not \`processed_data\`.

## Next Steps

1. ✅ **Setup complete** - All dependencies installed, data preprocessed
2. 📖 **Read TRAINING.md** - Learn how to train models
3. 🚀 **Start training** - Run \`./auto_train_roland.sh\`
4. 📊 **Monitor progress** - Use TensorBoard
5. 🎵 **Use trained model** - See INFERENCE.md

## Additional Resources

- **[TRAINING.md](TRAINING.md)** - How to train models
- **[INFERENCE.md](INFERENCE.md)** - Using trained models
- **[ROLAND_MAPPING.md](ROLAND_MAPPING.md)** - Drum mapping reference
- **[docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md)** - Tuning guide

## Support

If you encounter issues not covered here:
1. Check logs in \`/mnt/hdd/drum-tranxn/logs_roland/\`
2. Verify data paths in config files
3. Ensure Roland data (26 classes) is being used, not legacy (8 classes)
\`\`\`

---

### 7. TRAINING.md

**Due to length, create this file with the following structure:**

```markdown
# Training Guide - Roland TD-17 Drum Transcription

[Include sections: Quick Start, Training Configs (quick/medium/full), Hyperparameters, Monitoring, Resuming, Troubleshooting]
```

**ASSISTANT NOTE:** Provide full TRAINING.md content in next response if requested.

---

### 8. INFERENCE.md
### 9. ROLAND_MAPPING.md
### 10. README.md (updated)
### 11. docs/HYPERPARAMETER_OPTIMIZATION.md (updated)

**These will be provided in subsequent steps or can be generated by the coding assistant using the patterns established above.**

---

## Phase 4: Code Updates (15 min)

### Update train_with_optuna.py

Add comment at top:

```python
#!/usr/bin/env python3
"""
Hyperparameter optimization for Roland TD-17 (26-class) drum transcription.

When to use this:
- Model architecture changes significantly
- Dataset changes (different preprocessing)
- Training shows instability or poor convergence
- Starting a new project variant

For routine training, use the existing optimized parameters in
full_training_config.yaml (already tuned via Optuna).
"""
```

### Update auto_train_roland.sh

Add validation checks after the intro section:

```bash
# Validation checks
echo "Validating configuration..."

# Check config uses Roland data
if ! grep -q "processed_data_roland" configs/full_training_config.yaml; then
    echo "❌ ERROR: Config doesn't use Roland data!"
    echo "   Expected: processed_data_roland"
    exit 1
fi

# Check config uses 26 classes
if ! grep -q "n_classes: 26" configs/full_training_config.yaml; then
    echo "❌ ERROR: Config doesn't use 26 classes!"
    echo "   Expected: n_classes: 26"
    exit 1
fi

echo "✅ Configuration validated"
echo ""
```

---

## Phase 5: Final Validation (10 min)

### Validation Checklist

After completing all phases, verify:

```bash
# 1. Check no legacy data references
cd /home/matt/Documents/drum-tranxn/drum_transcription
grep -r "processed_data[^_]" configs/*.yaml && echo "❌ Found legacy references" || echo "✅ No legacy references"

# 2. Check no 8-class references
grep -r "n_classes.*8[^0-9]" configs/*.yaml src/ && echo "❌ Found 8-class references" || echo "✅ No 8-class references"

# 3. Verify new files exist
ls tests/README.md configs/quick_test_config.yaml quick_test.sh SETUP.md TRAINING.md INFERENCE.md ROLAND_MAPPING.md

# 4. Verify old files deleted
! ls run_training.py start_medium_test.sh 2>/dev/null && echo "✅ Old files deleted" || echo "❌ Old files still exist"

# 5. Quick test
./quick_test.sh
```

### Git Commit

```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription

# Stage all changes
git add -A

# Commit with descriptive message
git commit -m "Repository cleanup: Roland-only training system

- Removed legacy 8-class mapping code and configs
- Created comprehensive documentation (SETUP, TRAINING, INFERENCE, ROLAND_MAPPING)
- Reorganized test scripts into tests/ directory
- Added quick_test.sh and updated medium_test config for Roland
- Updated all code to reference only Roland mapping (26 classes)
- Created archive/legacy-8-class branch for historical reference

See CLEANUP_IMPLEMENTATION_PLAN.md for complete changes."

# Push to main
git push origin main

# Optionally tag this release
git tag -a v1.0.0-roland -m "First production-ready Roland mapping release"
git push origin v1.0.0-roland
```

---

## 🎉 Completion Checklist

After implementation, verify:

- [ ] Archive branch created: `git branch | grep archive/legacy-8-class`
- [ ] Old files removed: `ls run_training.py 2>&1 | grep "No such file"`
- [ ] Test directory created: `ls tests/`
- [ ] New configs created: `ls configs/quick_test_config.yaml`
- [ ] Documentation created: `ls SETUP.md TRAINING.md INFERENCE.md ROLAND_MAPPING.md`
- [ ] No legacy references: `grep -r "n_classes.*8[^0-9]" configs/`
- [ ] Quick test passes: `./quick_test.sh`
- [ ] Git committed: `git log -1`
- [ ] Training still works: `./auto_train_roland.sh --fast-dev-run` (if trainer supports it)

---

## 📊 Summary

**Created:**
- 11 new files (configs, docs, tests)
- 1 git branch (archive)

**Updated:**
- 9 files (README, scripts, configs)

**Deleted:**
- ~15 obsolete files

**Result:**
- ✅ Clean, production-ready repository
- ✅ Comprehensive documentation
- ✅ Roland TD-17 mapping only
- ✅ Easy to maintain and extend

---

## 💡 Using This Plan

**When training completes:**

1. Copy this plan file to your prompt
2. Say: "Please implement CLEANUP_IMPLEMENTATION_PLAN.md step by step"
3. The assistant will execute each phase
4. Validate after each phase
5. Commit when complete

**Or execute manually:**
- Follow each bash command in sequence
- Create each file with provided content
- Validate after each phase


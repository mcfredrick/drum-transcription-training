# Setup Guide - Standard Drum Kit Transcription

Complete guide for installing and configuring the drum transcription training system.

## System Requirements

### Hardware
- **GPU**: CUDA-capable GPU with 6GB+ VRAM (RTX 3060 or better recommended)
- **Storage**: 50GB+ free space
  - E-GMD dataset: ~30GB
  - Preprocessed data: ~2.5GB (11-class system)
  - Checkpoints/logs: ~8GB+
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

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if not done automatically)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
uv --version
```

### 2. Clone Repository

```bash
cd ~/Documents  # Or your preferred location
git clone <your-repo-url> drum-tranxn
cd drum-tranxn/drum_transcription
```

### 3. Install Dependencies

UV will automatically create a virtual environment and install dependencies:

```bash
# Install all dependencies (from pyproject.toml)
uv sync

# Verify installation
uv run python --version
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
```

### 4. Download E-GMD Dataset

The Extended Groove MIDI Dataset (E-GMD) contains professional drum recordings with MIDI annotations.

```bash
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
```

**Alternative:** If you have the dataset elsewhere, create a symlink:
```bash
ln -s /path/to/existing/e-gmd-v1.0.0 /mnt/hdd/drum-tranxn/e-gmd-v1.0.0
```

### 5. Preprocess Data (11-Class Mapping)

Convert raw audio and MIDI to preprocessed spectrograms and labels using standard drum kit mapping (11 classes).

```bash
cd ~/Documents/drum-tranxn/drum_transcription

# Run preprocessing (takes 30-60 minutes for full dataset)
uv run python scripts/preprocess.py \
    --config configs/drum_config.yaml

# This creates:
# - /mnt/hdd/drum-tranxn/processed_data/
# - Mel spectrograms (HDF5 format)
# - Standard drum kit labels (11 classes)
# - Train/val/test splits
```

**Expected output:**
```
Processing E-GMD dataset with 11-class drum mapping...
Found 1200+ audio files
Processing: 100%|████████| 1200/1200 [45:23<00:00, 0.44it/s]
Coverage: 96.58% of dataset
Saved splits to: /mnt/hdd/drum-tranxn/processed_data/splits/
```

### 6. Verify Setup

Run quick test to ensure everything works:

```bash
cd ~/Documents/drum-tranxn/drum_transcription

# 5-minute validation run
./quick_test.sh
```

**Expected behavior:**
- Loads 10 files
- Trains for 5 epochs
- Shows progress bar with metrics (val_loss, val_f1)
- Completes without errors
- Saves checkpoint

**If successful, you're ready to train!**

## Configuration

### Standard Drum Kit Mapping

This system uses a standard drum kit mapping:
- **11 drum classes** (essential drum sounds)
- Includes kick, snare variations, hi-hat types, toms, ride
- Covers **96.58%** of E-GMD dataset
- See [DRUM_MAPPING.md](DRUM_MAPPING.md) for complete mapping

### File Locations

**Data directories:**
```
/mnt/hdd/drum-tranxn/
├── e-gmd-v1.0.0/              # Raw E-GMD dataset
├── processed_data/             # Preprocessed (11 classes)
├── checkpoints/                # Model checkpoints
└── logs/                       # Training logs
```

**Code structure:**
```
drum_transcription/
├── configs/                    # Training configurations
├── scripts/                    # Main scripts
├── src/                        # Model and data modules
├── tests/                      # Validation scripts
└── docs/                       # Additional documentation
```

### Environment Variables

**Optional configuration:**

```bash
# Disable W&B logging (default for local training)
export USE_WANDB=false

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Increase workers if you have more CPU cores
# (edit in config files: hardware.num_workers)
```

## Troubleshooting

### "CUDA out of memory"
**Solution:** Reduce batch size in config:
```yaml
training:
  batch_size: 2  # Reduce from 4
```

### "Split file not found"
**Cause:** Preprocessing didn't complete or failed.
**Solution:** Re-run preprocessing:
```bash
uv run python scripts/preprocess.py --config configs/drum_config.yaml
```

### "Module not found"
**Cause:** Dependencies not installed.
**Solution:**
```bash
uv sync
uv pip install <missing-package>
```

### Preprocessing is slow
**Normal:** Full dataset takes 30-60 minutes.
**Speed up:** Use `--max-files 100` for testing:
```bash
uv run python scripts/preprocess.py \
    --config configs/drum_config.yaml \
    --max-files 100
```

### "Legacy data" errors
**Cause:** Old data from previous mapping systems.
**Solution:** Ensure configs point to `processed_data` with 11-class mapping.

## Next Steps

1. ✅ **Setup complete** - All dependencies installed, data preprocessed
2. 📖 **Read TRAINING.md** - Learn how to train models
3. 🚀 **Start training** - Run `./auto_train.sh`
4. 📊 **Monitor progress** - Use TensorBoard
5. 🎵 **Use trained model** - See INFERENCE.md

## Additional Resources

- **[TRAINING.md](TRAINING.md)** - How to train models
- **[INFERENCE.md](INFERENCE.md)** - Using trained models
- **[DRUM_MAPPING.md](DRUM_MAPPING.md)** - Drum mapping reference
- **[docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md)** - Tuning guide

## Support

If you encounter issues not covered here:
1. Check logs in `/mnt/hdd/drum-tranxn/logs/`
2. Verify data paths in config files
3. Ensure 11-class mapping is being used consistently

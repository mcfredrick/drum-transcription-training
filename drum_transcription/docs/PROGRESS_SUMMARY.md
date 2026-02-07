# Drum Transcription Training Pipeline - Progress Summary

**Date:** February 7, 2026  
**Status:** Setup Complete - Ready for Training After Reboot

---

## What We Accomplished

### 1. ✅ Dataset Extraction
- **Location:** `/mnt/hdd/drum-tranxn/e-gmd-v1.0.0/`
- Successfully extracted the E-GMD v1.0.0 dataset (90GB)
- Dataset contains drum recordings with MIDI annotations
- All data remains on HDD as requested

### 2. ✅ Project Setup & Storage Architecture
- **Project Location (SSD):** `~/Documents/drum-tranxn/drum_transcription/`
- **Data Location (HDD):** `/mnt/hdd/drum-tranxn/`
- **Decision:** Keep project code on SSD for fast development, large data on HDD
- Initialized UV project with Python 3.12
- Created directory structure:
  - `src/data/` - Data processing modules
  - `src/models/` - Model implementations
  - `src/utils/` - Utility functions
  - `scripts/` - Training/inference scripts
  - `configs/` - Configuration files
  - `docs/` - Documentation (includes storage-architecture.md)
- See `docs/storage-architecture.md` for rationale and details

### 3. ✅ Dependencies Installed
- PyTorch 2.7.1 with CUDA 11.8 support (via UV)
- PyTorch Lightning 2.6.1
- Audio processing: librosa, soundfile
- MIDI processing: pretty-midi, mido
- Scientific computing: numpy, scipy, h5py
- Visualization: matplotlib, tensorboard
- All dependencies installed in `.venv/` directory

### 4. ✅ Configuration Files
- **Location:** `~/Documents/drum-tranxn/drum_transcription/configs/`
- `test_config.yaml` - Quick test with 20 files, 2 epochs
- `default_config.yaml` - Full training configuration
- Settings:
  - Max 20 files for quick validation (test config)
  - 2 epochs for testing
  - Batch size 4
  - 8 drum classes (kick, snare, hihat, toms, cymbals)
  - All data paths point to HDD locations automatically

---

## Current Issue: NVIDIA Driver Mismatch

### Problem
- **Kernel driver loaded:** 550.163.01 (old)
- **Userspace library:** 580.126.09 (new)
- **Result:** CUDA is not accessible to PyTorch

### Root Cause
Both nvidia-driver-550 and nvidia-driver-580 are installed, but the system needs a reboot to load the driver-580 kernel module that matches the userspace library.

### Solution
**Reboot the system** - This will load the nvidia-driver-580 kernel module and resolve the mismatch.

---

## After Reboot: Verification Steps

### 1. Verify CUDA is Working
```bash
cd ~/Documents/drum-tranxn/drum_transcription
nvidia-smi  # Should show driver 580 and GPU info
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3070 Lite Hash Rate
```

---

## Next Steps: Complete the Training Pipeline

### Step 1: Create Core Implementation Files

You need to create these essential files:

1. **`src/data/audio_processing.py`** - Extract spectrograms from audio
2. **`src/data/midi_processing.py`** - Convert MIDI to frame-level labels  
3. **`src/data/dataset.py`** - PyTorch Dataset class
4. **`src/data/data_module.py`** - Lightning DataModule
5. **`src/models/crnn.py`** - CRNN model with Lightning
6. **`scripts/preprocess_egmd.py`** - Preprocess dataset
7. **`scripts/train.py`** - Training script
8. **`scripts/transcribe.py`** - Inference script

Reference the implementation plan at:
- `/home/matt/Documents/drum-tranxn/drum_transcription_plan_revised.md`
- `/home/matt/Documents/drum-tranxn/drum_transcription/IMPLEMENTATION_SUMMARY.md`

### Step 2: Preprocess Small Subset
```bash
cd ~/Documents/drum-tranxn/drum_transcription
uv run python scripts/preprocess_egmd.py --config configs/test_config.yaml
```

This will:
- Process first 20 audio files from E-GMD
- Extract log-mel spectrograms
- Convert MIDI to frame-level labels
- Save preprocessed data to `/mnt/hdd/drum-tranxn/processed_data/`

### Step 3: Run Quick Training Test
```bash
uv run python scripts/train.py --config configs/test_config.yaml
```

This will:
- Train for 2 epochs (quick validation)
- Use batch size 4
- Save checkpoints to `/mnt/hdd/drum-tranxn/checkpoints/`
- Log to tensorboard

### Step 4: Test Inference
```bash
uv run python scripts/transcribe.py \
    --audio /mnt/hdd/drum-tranxn/e-gmd-v1.0.0/drummer1/eval_session/[some_file].wav \
    --output output.mid \
    --checkpoint /mnt/hdd/drum-tranxn/checkpoints/best_model.ckpt
```

---

## Project Structure

**SSD (Development):** `~/Documents/drum-tranxn/drum_transcription/`
```
drum_transcription/             # Main project directory (SSD)
├── .venv/                      # Python virtual environment (UV managed)
├── pyproject.toml              # Dependencies (PyTorch CUDA 11.8)
├── docs/
│   └── storage-architecture.md # Storage decision documentation
├── configs/
│   ├── test_config.yaml        # Quick test configuration
│   └── default_config.yaml     # Full training configuration
├── src/
│   ├── data/                   # Data processing (to be implemented)
│   ├── models/                 # CRNN model (to be implemented)
│   └── utils/                  # Utilities (to be implemented)
├── scripts/                    # Training/inference scripts (to be implemented)
└── notebooks/                  # Jupyter notebooks
```

**HDD (Large Data):** `/mnt/hdd/drum-tranxn/`
```
/mnt/hdd/drum-tranxn/
├── e-gmd-v1.0.0/               # Dataset (90GB, 444 hours of drum recordings)
├── processed_data/             # Preprocessed spectrograms & labels
│   ├── train/
│   ├── val/
│   ├── test/
│   └── splits/
├── checkpoints/                # Model checkpoints during training
└── logs/                       # Training logs and tensorboard data
```

---

## Key Configuration Details

### Model Architecture (from test_config.yaml)
- **Input:** Log-mel spectrograms (128 mel bands, 22.05kHz)
- **CNN:** 3 blocks (32→64→128 channels)
- **RNN:** 2-layer bidirectional GRU (128 hidden units)
- **Output:** 8 drum classes (frame-level predictions)

### 8 Drum Classes (E-GMD → Game Lanes)
1. Kick (GM 36)
2. Snare (GM 38)
3. Hi-hat (GM 42)
4. High Tom (GM 50)
5. Mid Tom (GM 47)
6. Low Tom (GM 45/41)
7. Crash (GM 49)
8. Ride (GM 51)

### Storage Layout
- **Project Code (SSD):** `~/Documents/drum-tranxn/drum_transcription/`
- **E-GMD Dataset (HDD):** `/mnt/hdd/drum-tranxn/e-gmd-v1.0.0/`
- **Processed Data (HDD):** `/mnt/hdd/drum-tranxn/processed_data/`
- **Checkpoints (HDD):** `/mnt/hdd/drum-tranxn/checkpoints/`
- **Logs (HDD):** `/mnt/hdd/drum-tranxn/logs/`

---

## Commands Reference

### Activate Project Environment
```bash
cd ~/Documents/drum-tranxn/drum_transcription
# UV automatically uses .venv/, just prefix commands with 'uv run'
```

### Install New Dependency
```bash
uv add package-name
```

### Update Dependencies
```bash
uv sync
```

### Run Python Scripts
```bash
uv run python scripts/script_name.py
```

---

## Important Notes

1. **All data stays on HDD** - No moving to SSD
2. **CUDA 11.8** - PyTorch compiled for CUDA 11.8 (compatible with CUDA 12.0)
3. **PyTorch Lightning** - Clean training framework, eliminates boilerplate
4. **UV Package Manager** - Fast, modern Python dependency management
5. **Quick Test Config** - Only 20 files, 2 epochs for fast validation
6. **GPU Required** - Training on CPU is not practical

---

## Troubleshooting After Reboot

### If CUDA Still Not Working
```bash
# Check driver version matches
nvidia-smi  # Should show 580.x
cat /proc/driver/nvidia/version  # Should show 580.x
ls -la /lib/x86_64-linux-gnu/libcuda.so*  # Should point to 580.x

# Test PyTorch
cd ~/Documents/drum-tranxn/drum_transcription
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### If Driver 550 Still Loaded
You may need to remove the old driver:
```bash
sudo apt remove nvidia-driver-550
sudo reboot
```

---

## Expected Timeline After Reboot

1. **Verify CUDA:** 2 minutes
2. **Implement core files:** 30-60 minutes (or ask AI assistant to create them)
3. **Preprocess 20 files:** 5-10 minutes
4. **Quick training test (2 epochs):** 10-30 minutes depending on GPU
5. **Test inference:** 2 minutes

**Total:** ~1-2 hours to have a working training pipeline

---

## Contact/Resume Point

When you resume after reboot:
1. Run verification commands above to confirm CUDA works
2. Ask your AI assistant to implement the remaining files listed in "Step 1: Create Core Implementation Files"
3. The assistant can reference the plan documents for implementation details
4. All infrastructure (dependencies, config, directory structure) is ready

**You're 80% done with setup - just need CUDA working and the implementation files!**

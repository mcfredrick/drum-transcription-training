# Drum Transcription - Standard Drum Kit (11-Class)

Deep learning system for transcribing drum audio to MIDI using a standard drum kit mapping (11 classes).

## Overview

This project trains a CRNN (Convolutional Recurrent Neural Network) to transcribe drum performances from audio recordings to MIDI files, using a standard drum kit mapping.

**Key Features:**
- **11-class drum transcription** with standard drum kit mapping
- **Essential drum sounds:** Kick, snare, hi-hat, toms, ride, auxiliary percussion
- **CRNN architecture** with PyTorch Lightning
- **Data augmentation** for improved generalization
- **Optimized hyperparameters** via Optuna
- **Efficient training** with lower VRAM requirements

## Quick Start

**1. Installation and setup:**
```bash
# See SETUP.md for complete installation guide
./quick_test.sh  # Verify setup (5-10 minutes)
```

**2. Train a model:**
```bash
./auto_train.sh  # Full training (12-24 hours)
```

**3. Transcribe audio to MIDI:**
```bash
uv run python scripts/transcribe.py \
    --checkpoint /mnt/hdd/drum-tranxn/checkpoints/best.ckpt \
    --audio your_audio.wav \
    --output output.mid
```

## Documentation

**Start here:**
- **[SETUP.md](SETUP.md)** - Installation, dependencies, data preprocessing
- **[TRAINING.md](TRAINING.md)** - Training models, configurations, monitoring
- **[INFERENCE.md](INFERENCE.md)** - Using trained models for transcription
- **[DRUM_MAPPING.md](DRUM_MAPPING.md)** - Complete drum mapping reference

**Advanced:**
- **[docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md)** - Optuna tuning guide

## Standard Drum Kit Mapping (11 Classes)

This system uses 11 essential drum classes:

1. **Kick** (MIDI 36)
2. **Snare head** (MIDI 38)
3. **Snare rim** (MIDI 40)
4. **Side stick** (MIDI 37)
5. **Pedal hi-hat** (MIDI 44)
6. **Closed hi-hat** (MIDI 42)
7. **Open hi-hat** (MIDI 46)
8. **Floor tom** (MIDI 43)
9. **High-mid tom** (MIDI 48)
10. **Ride** (MIDI 51)
11. **Ride bell** (MIDI 53)

Covers **96.58%** of E-GMD dataset. See [DRUM_MAPPING.md](DRUM_MAPPING.md) for complete details.

## Repository Structure

```
drum_transcription/
├── README.md                   # This file
├── SETUP.md                    # Installation guide
├── TRAINING.md                 # Training guide
├── INFERENCE.md                # Inference guide
├── DRUM_MAPPING.md             # Drum mapping reference
│
├── configs/                    # Training configurations
│   ├── drum_config.yaml        # Base configuration (11 classes)
│   ├── full_training_config.yaml  # Production training (150 epochs)
│   ├── medium_test_config.yaml    # Medium test (100 files, 20 epochs)
│   └── quick_test_config.yaml     # Quick test (10 files, 5 epochs)
│
├── scripts/                    # Main scripts
│   ├── train.py                # Training script
│   ├── preprocess.py           # Data preprocessing
│   ├── transcribe.py           # Inference script
│   └── evaluate.py             # Model evaluation
│
├── src/                        # Source code
│   ├── models/crnn.py          # CRNN model architecture
│   ├── data/                   # Data loading and augmentation
│   └── utils/                  # Utilities
│
├── tests/                      # Test scripts
│   ├── test_preprocessing.py   # Test preprocessing pipeline
│   ├── test_single_file.py     # Test single file inference
│   └── README.md               # Test documentation
│
├── docs/                       # Additional documentation
│   └── HYPERPARAMETER_OPTIMIZATION.md
│
├── auto_train.sh               # Full training launcher
├── quick_test.sh               # Quick validation script
├── train_with_optuna.py        # Hyperparameter optimization
└── pyproject.toml              # Dependencies (UV)
```

## System Requirements

**Hardware:**
- GPU: CUDA-capable with 6GB+ VRAM (RTX 3060 or better)
- Storage: 50GB+ (dataset + processed data + checkpoints)
- RAM: 16GB+

**Software:**
- Linux or WSL2 (Ubuntu 20.04+)
- Python 3.10+
- CUDA 11.8+
- UV package manager

See [SETUP.md](SETUP.md) for detailed requirements and installation.

## Training Configurations

Three training modes available:

| Mode | Files | Epochs | Time | Purpose |
|------|-------|--------|------|---------|
| **Quick** | 10 | 5 | 5-10 min | Verify setup |
| **Medium** | 100 | 20 | 2-3 hours | Test changes |
| **Full** | 1200+ | 150 | 12-24 hours | Production model |

**Run configurations:**
```bash
./quick_test.sh                                      # Quick test
uv run python scripts/train.py --config configs/medium_test_config.yaml  # Medium
./auto_train.sh                                      # Full training
```

## Model Architecture

**CRNN Design:**
- **Input:** Log-mel spectrograms (128 bins)
- **CNN:** 4 convolutional blocks (32→64→128→256 filters)
- **RNN:** 3-layer bidirectional GRU (256 hidden units)
- **Output:** 11 classes (standard drum kit mapping)
- **Loss:** Weighted binary cross-entropy with sigmoid activation

**Optimized hyperparameters** (via Optuna):
- Learning rate: 0.00044547593667580503
- Batch size: 4
- Optimizer: AdamW
- Weight decay: 5.559904341261682e-05

**Class weights** for imbalanced data:
[0.5, 0.5, 1.5, 3.5, 0.8, 0.9, 15.0, 1.5, 1.6, 1.0, 13.0]

See [TRAINING.md](TRAINING.md) for architecture details.

## Performance

**Expected metrics (full training):**
- **Overall F1:** 0.75-0.90
- **Validation loss:** 0.010-0.020

**Per-class F1 scores:**
- Common drums (kick, snare, closed hi-hat): 0.85-0.95
- Medium frequency (toms, ride, pedal hi-hat): 0.75-0.85
- Rare drums (open hi-hat, ride bell, side stick): 0.60-0.75

## Dataset

**E-GMD (Extended Groove MIDI Dataset):**
- 1200+ professional drum recordings
- MIDI annotations with precise timing
- Multiple drummers and styles
- ~30GB total size

**Preprocessing:**
```bash
uv run python scripts/preprocess.py \
    --config configs/drum_config.yaml
```

Creates:
- Mel spectrograms (HDF5 format)
- Standard drum kit labels (11 classes)
- Train/val/test splits (70/15/15)

## Usage Examples

**Train from scratch:**
```bash
./auto_train.sh
```

**Resume training:**
```bash
uv run python scripts/train.py \
    --config configs/full_training_config.yaml \
    --checkpoint /path/to/last.ckpt
```

**Transcribe audio:**
```bash
uv run python scripts/transcribe.py \
    --checkpoint /path/to/best.ckpt \
    --audio song.wav \
    --output song.mid
```

**Batch processing:**
```bash
for audio in *.wav; do
    uv run python scripts/transcribe.py \
        --checkpoint checkpoint.ckpt \
        --audio "$audio" \
        --output "${audio%.wav}.mid"
done
```

## Monitoring Training

**TensorBoard:**
```bash
tensorboard --logdir=/mnt/hdd/drum-tranxn/logs
# Open: http://localhost:6006
```

**Key metrics:**
- `train_loss` / `val_loss` - Training progress
- `val_f1` - Validation F1-score (target: >0.7)
- Per-class F1 scores - Individual drum performance

See [TRAINING.md](TRAINING.md) for monitoring details.

## Common Issues

**Setup issues:** See [SETUP.md](SETUP.md#troubleshooting)
**Training issues:** See [TRAINING.md](TRAINING.md#troubleshooting)
**Inference issues:** See [INFERENCE.md](INFERENCE.md#troubleshooting)

**Quick fixes:**
- CUDA OOM → Reduce batch size in config
- Slow training → Increase num_workers
- Poor performance on rare drums → Adjust class weights

## Development

**Test scripts:**
```bash
# Test preprocessing
uv run python tests/test_preprocessing.py

# Test inference
uv run python tests/test_single_file.py /path/to/audio.wav
```

**Hyperparameter tuning:**
```bash
uv run python train_with_optuna.py \
    --config configs/drum_config.yaml \
    --n-trials 50
```

See [docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md) for Optuna guide.

## Citation

**E-GMD Dataset:**
```
Callender, C., Hawthorne, C., & Engel, J. (2020).
Improving perceptual quality of drum transcription with
the Expanded Groove MIDI Dataset.
```

**PyTorch Lightning:**
```
Falcon, W., et al. (2019).
PyTorch Lightning. GitHub.
https://github.com/Lightning-AI/pytorch-lightning
```

## License

Educational/research purposes.

**Dependencies:**
- E-GMD Dataset: CC BY 4.0
- PyTorch: BSD License
- PyTorch Lightning: Apache 2.0

## Acknowledgments

- Google Magenta team for E-GMD dataset
- PyTorch Lightning team for training framework

## Support

**For help:**
1. Check relevant documentation (SETUP.md, TRAINING.md, INFERENCE.md)
2. Review logs in `/mnt/hdd/drum-tranxn/logs/`
3. Verify configurations use 11-class mapping
4. Run quick_test.sh to validate setup

---

**Ready to start?** → Begin with [SETUP.md](SETUP.md)

# Drum Transcription for Rhythm Game

This project trains a deep learning model to transcribe drum tracks into MIDI for use in a rhythm game. The model uses a CRNN (Convolutional Recurrent Neural Network) architecture trained on the E-GMD dataset.

## Features

- **8-class drum transcription**: kick, snare, hi-hat, hi-tom, mid-tom, low-tom, crash, ride
- **State-of-the-art CRNN architecture** with PyTorch Lightning
- **Data augmentation**: time stretch, pitch shift, reverb, noise
- **General MIDI export** for game integration
- **Multi-GPU training** support
- **Fast preprocessing** with HDF5 storage

## Quick Start

### 1. Installation

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project and install dependencies
cd drum_transcription
uv sync

# Verify PyTorch CUDA installation
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Download E-GMD Dataset

Download the E-GMD dataset (~90GB):

```bash
# Create data directory
mkdir -p data/e-gmd

# Download from Google Magenta
wget http://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip

# Unzip
unzip e-gmd-v1.0.0.zip -d data/e-gmd/
```

### 3. Preprocess Dataset

Convert audio to spectrograms and MIDI to labels:

```bash
# Create splits and preprocess (uses all CPU cores)
uv run python scripts/preprocess_egmd.py \
    --config configs/default_config.yaml \
    --num-workers 8 \
    --use-hdf5

# This will:
# 1. Create train/val/test splits (70/15/15)
# 2. Extract log-mel spectrograms from audio
# 3. Convert MIDI annotations to frame-level labels
# 4. Save as HDF5 files for fast loading
```

Expected output structure:
```
data/
├── e-gmd/              # Raw dataset
├── processed/          # Preprocessed spectrograms and labels
└── splits/             # Train/val/test split files
    ├── train_split.txt
    ├── val_split.txt
    └── test_split.txt
```

### 4. Train Model

```bash
# Train with default config
uv run python scripts/train.py \
    --config configs/default_config.yaml \
    --experiment-name crnn-egmd-baseline

# Train with custom experiment name
uv run python scripts/train.py \
    --experiment-name my-experiment

# Resume from checkpoint
uv run python scripts/train.py \
    --resume checkpoints/last.ckpt

# Fast development run (1 batch for testing)
uv run python scripts/train.py --fast-dev-run
```

Training will:
- Use both GPUs (3070 + 3090) automatically
- Save checkpoints to `checkpoints/`
- Log to Weights & Biases (or TensorBoard)
- Apply early stopping if validation loss doesn't improve

### 5. Transcribe Audio

```bash
# Transcribe a drum track to MIDI
uv run python scripts/transcribe.py \
    path/to/audio.wav \
    output.mid \
    --checkpoint checkpoints/best_model.ckpt

# With custom threshold
uv run python scripts/transcribe.py \
    audio.wav output.mid \
    --checkpoint checkpoints/best_model.ckpt \
    --threshold 0.6 \
    --min-interval 0.03
```

The output MIDI file will use General MIDI drum mapping and can be imported directly into your rhythm game.

## Project Structure

```
drum_transcription/
├── configs/
│   └── default_config.yaml      # Training configuration
├── src/
│   ├── data/
│   │   ├── audio_processing.py  # Spectrogram extraction
│   │   ├── midi_processing.py   # MIDI label conversion
│   │   ├── augmentation.py      # Data augmentation
│   │   ├── dataset.py           # PyTorch Dataset
│   │   └── data_module.py       # Lightning DataModule
│   ├── models/
│   │   └── crnn.py              # CRNN model
│   └── utils/
│       └── config.py            # Config loading utilities
├── scripts/
│   ├── preprocess_egmd.py       # Dataset preprocessing
│   ├── train.py                 # Training script
│   └── transcribe.py            # Inference script
├── pyproject.toml               # Dependencies (UV)
└── README.md
```

## Configuration

Edit `configs/default_config.yaml` to customize:

**Model architecture:**
```yaml
model:
  n_classes: 8
  conv_filters: [32, 64, 128]
  hidden_size: 128
  num_gru_layers: 3
```

**Training hyperparameters:**
```yaml
training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.001
```

**Data augmentation:**
```yaml
augmentation:
  enabled: true
  time_stretch_prob: 0.5
  pitch_shift_prob: 0.5
  reverb_prob: 0.3
  noise_prob: 0.3
```

## Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 8GB VRAM (e.g., RTX 3070)
- RAM: 16GB
- Storage: 150GB (90GB E-GMD + 50GB processed + checkpoints)

**Recommended (your setup):**
- GPU: RTX 3090 (24GB) + RTX 3070 (8GB)
- RAM: 32GB+
- Storage: 200GB SSD

**Training time:**
- ~2-3 days for 100 epochs on RTX 3090
- ~4-5 days on RTX 3070

## Model Architecture

```
Input: Log-mel spectrogram (batch, 1, 128, time)
    ↓
CNN Encoder (3 blocks):
    Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Bidirectional GRU (3 layers, hidden=128)
    ↓
Dense Layer: Linear(256) → ReLU → Dropout(0.3) → Linear(8) → Sigmoid
    ↓
Output: Frame-level predictions (batch, time, 8)
```

**Parameters:** ~1-5M (depending on configuration)

## Performance Expectations

Based on SOTA research with E-GMD:

**Overall F-measure:** 70-80%

**Per-class F-measure:**
- Kick: 85-90%
- Snare: 80-85%
- Hi-hat: 75-80%
- Hi-Tom: 70-75%
- Mid-Tom: 65-75%
- Low-Tom: 65-75%
- Crash: 70-75%
- Ride: 70-75%

## Logging & Monitoring

### Weights & Biases (Recommended)

```bash
# Login to W&B
uv run wandb login

# Training will automatically log to W&B
# View at: https://wandb.ai/your-username/drum-transcription
```

Metrics logged:
- Training/validation loss
- Per-class precision/recall/F1
- Learning rate
- Gradient norms

### TensorBoard (Alternative)

```bash
# Change logger in config
logging:
  logger: "tensorboard"

# View logs
uv run tensorboard --logdir logs/
```

## Common Issues

### CUDA Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size: 8  # Or even 4
```

### Data Loading Too Slow

Increase workers:
```yaml
hardware:
  num_workers: 8
```

Or use HDF5 format (faster than .npy):
```bash
uv run python scripts/preprocess_egmd.py --use-hdf5
```

### macOS Installation Issues

If you get build errors on macOS:

```bash
# Install dependencies
brew install libsndfile portaudio

# Install with no build isolation
uv add librosa --no-build-isolation
```

## Advanced Usage

### Hyperparameter Tuning

Create custom config files:

```bash
# Copy default config
cp configs/default_config.yaml configs/experiment1.yaml

# Edit experiment1.yaml
# Then train:
uv run python scripts/train.py --config configs/experiment1.yaml
```

### Multi-GPU Training

Specify GPUs in config:
```yaml
hardware:
  devices: [0, 1]  # Use GPU 0 and 1
```

Or via environment variable:
```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/train.py
```

### Batch Inference

Transcribe multiple files:

```bash
# Create a simple batch script
for audio in songs/*.wav; do
    output="${audio%.wav}.mid"
    uv run python scripts/transcribe.py \
        "$audio" "$output" \
        --checkpoint checkpoints/best_model.ckpt
done
```

## Integration with Your Game

The transcribed MIDI files use General MIDI drum mapping:

```python
DRUM_MAPPING = {
    36: "kick",      # Bass Drum 1 (C1)
    38: "snare",     # Acoustic Snare (D1)
    42: "hihat",     # Closed Hi-Hat (F#1)
    50: "hi_tom",    # High Tom (D2)
    47: "mid_tom",   # Low-Mid Tom (B1)
    45: "low_tom",   # Low Tom (A1)
    49: "crash",     # Crash Cymbal 1 (C#2)
    51: "ride",      # Ride Cymbal 1 (D#2)
}
```

All notes are on MIDI channel 10 (drums).

To load in your game:
1. Parse MIDI file
2. Extract note onsets and pitches
3. Map pitches to your 8-lane system
4. Display in game

## Citation

If you use this code or model, please cite:

**E-GMD Dataset:**
```
Callender, C., Hawthorne, C., & Engel, J. (2020). 
Improving perceptual quality of drum transcription with the Expanded Groove MIDI Dataset.
```

**PyTorch Lightning:**
```
Falcon, W., et al. (2019). 
PyTorch Lightning. GitHub. 
https://github.com/Lightning-AI/pytorch-lightning
```

## License

This project is for educational/research purposes. 

Dependencies:
- E-GMD: CC BY 4.0
- PyTorch: BSD License
- PyTorch Lightning: Apache 2.0

## Contributing

This is a personal project for your rhythm game, but suggestions are welcome!

## Support

For issues:
1. Check `configs/default_config.yaml` for correct paths
2. Verify E-GMD dataset is downloaded and extracted
3. Run preprocessing with `--num-workers 1` to see detailed errors
4. Test with `--fast-dev-run` to verify setup

## Acknowledgments

- Google Magenta team for E-GMD dataset
- PyTorch Lightning team for excellent framework
- Research community for CRNN architectures

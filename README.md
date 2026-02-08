 # Drum Transcription Training Pipeline

A comprehensive deep learning pipeline for training drum transcription models using the Enhanced Groove MIDI Dataset (E-GMD). This project implements a CRNN (Convolutional Recurrent Neural Network) architecture for converting audio drum performances into MIDI transcriptions with **Roland TD-17 mapping**.

## 🚀 Quick Start

```bash
# Clone the repository
git clone git@github.com:mcfredrick/drum-transcription-training.git
cd drum-transcription-training

# Navigate to training code
cd drum_transcription

# Install dependencies
uv sync

# Download and preprocess E-GMD dataset with Roland mapping
mkdir -p data/e-gmd
wget http://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip
unzip e-gmd-v1.0.0.zip -d data/e-gmd/

# Preprocess with Roland TD-17 mapping (26 classes)
uv run python scripts/preprocess_roland.py --use-hdf5

# Start training with Roland model
uv run python scripts/train.py --config configs/default_config.yaml
```

## 📁 Project Structure

```
drum-transcription-training/
├── drum_transcription/          # Main training pipeline
│   ├── src/                     # Source code
│   │   ├── data/               # Data processing and loading
│   │   ├── models/             # Neural network architectures
│   │   └── utils/              # Utilities and configuration
│   ├── scripts/                # Training and inference scripts
│   ├── configs/                # Configuration files
│   └── docs/                   # Documentation
├── drum-transcription-api/      # Separate inference API (excluded)
└── docs/                       # Additional documentation
```

## 🎯 Features

- **26-class drum transcription** with Roland TD-17 standard
- **Individual drum sounds**: kick, snare head/rim/x-stick, 4 toms (head/rim), hi-hats (bow/edge/pedal/open), multiple cymbals, percussion
- **Roland-specific notes**: hi-hat edge triggers, cymbal edge/bow distinctions
- **Enhanced expressiveness**: 225% more drum types than legacy 8-class systems
- **State-of-the-art CRNN architecture** with PyTorch Lightning
- **Multi-GPU training** support with automatic device detection
- **Comprehensive data augmentation**: time stretch, pitch shift, reverb, noise
- **Fast HDF5 data loading** for efficient training
- **Extensive logging** with Weights & Biases and TensorBoard support
- **Multiple training configurations** for different experimental setups

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: RTX 3070+)
- 150GB+ storage space

### Installation
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone git@github.com:mcfredrick/drum-transcription-training.git
cd drum-transcription-training/drum_transcription
uv sync

# Verify CUDA setup
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 Data Setup

The training pipeline requires the Enhanced Groove MIDI Dataset (E-GMD). See [`drum_transcription/DATA_SETUP.md`](drum_transcription/DATA_SETUP.md) for detailed instructions on:

- Downloading the E-GMD dataset (~90GB)
- Preprocessing audio and MIDI data
- Setting up data storage (local, external, or cloud)
- Verifying data integrity

## 🏋️ Training

### Basic Training
```bash
cd drum_transcription

# Train with default configuration
uv run python scripts/train.py --config configs/default_config.yaml

# Train with custom experiment name
uv run python scripts/train.py --experiment-name my-experiment

# Resume from checkpoint
uv run python scripts/train.py --resume checkpoints/last.ckpt
```

### Available Configurations
- `default_config.yaml` - Roland TD-17 mapping (26 classes) - **Recommended**
- `roland_config.yaml` - Alternative Roland configuration
- `test_config.yaml` - Quick development run (1 batch)
- `medium_test_config.yaml` - Medium-scale training
- `full_training_config.yaml` - Complete training pipeline

### Monitoring Training
Training automatically logs to Weights & Biases:
```bash
# Login to W&B
uv run wandb login

# View training progress at: https://wandb.ai/your-username/drum-transcription-roland
```

## 🎵 Inference

```bash
# Transcribe audio to MIDI with Roland mapping
uv run python scripts/transcribe.py \
    input_audio.wav \
    output_midi.mid \
    --checkpoint checkpoints/best_model.ckpt
```

## 📈 Performance

Based on E-GMD dataset benchmarks with Roland TD-17 mapping:
- **Target Macro F-measure**: 80-85%
- **Core drums (kick, snare, hi-hat)**: 85-90% F-measure
- **Toms (all 8 variants)**: 75-80% F-measure
- **Cymbals (all 6 variants)**: 70-75% F-measure
- **Percussion (tambourine, cowbell)**: 65-70% F-measure

Training time: ~3-4 days for 150 epochs on RTX 3090

## 🔄 Roland TD-17 Mapping

The project uses Roland TD-17 electronic drum kit mapping for professional-grade drum transcription:

### Key Benefits
- **225% more drum types** (26 vs 8 classes)
- **Roland TD-17 compatibility** for electronic drums
- **Enhanced expressiveness** with edge/rim distinctions
- **Professional-grade output** matching industry standards

### Drum Classes
| Category | Classes |
|----------|---------|
| Core Drums | kick, snare_head, snare_xstick, snare_rim |
| Toms | tom1-4 (head/rim variants) |
| Hi-Hats | closed/open (bow/edge), pedal |
| Cymbals | crash1/2 (bow/edge), ride (bow/edge/bell) |
| Percussion | tambourine, cowbell |

See [`MIDI_MAPPING_STANDARD_ANALYSIS.md`](MIDI_MAPPING_STANDARD_ANALYSIS.md) for detailed mapping information.

## 🤝 Contributing

This is a research project for drum transcription. Key areas for contribution:

1. **Model architecture improvements** - Transformer-based models, attention mechanisms
2. **Data augmentation** - Advanced techniques for limited training data
3. **Multi-instrument transcription** - Extend beyond drums to full percussion
4. **Real-time inference** - Optimize for live performance applications

See [`TODO.md`](TODO.md) for specific improvement ideas and research directions.

## 📚 Documentation

- [`drum_transcription/DATA_SETUP.md`](drum_transcription/DATA_SETUP.md) - Data acquisition and setup
- [`drum_transcription/QUICKSTART.md`](drum_transcription/QUICKSTART.md) - Detailed setup guide
- [`drum_transcription/TRAINING_GUIDE.md`](drum_transcription/TRAINING_GUIDE.md) - Training configuration
- [`docs/`](docs/) - Research notes and progress tracking

## 📄 License

This project is for educational and research purposes. Dependencies maintain their respective licenses:
- E-GMD Dataset: CC BY 4.0
- PyTorch: BSD License
- PyTorch Lightning: Apache 2.0

## 🙏 Acknowledgments

- Google Magenta team for the E-GMD dataset
- PyTorch Lightning framework contributors
- Research community in music information retrieval

---

**Repository**: https://github.com/mcfredrick/drum-transcription-training  
**Issues**: Use GitHub Issues for bug reports and feature requests  
**Discussions**: Use GitHub Discussions for questions and ideas
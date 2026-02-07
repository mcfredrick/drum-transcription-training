# Data Setup Guide

This document explains how to obtain and set up the data required for training the drum transcription model.

## Required Datasets

### 1. Enhanced Groove MIDI Dataset (E-GMD)
The primary dataset used for training is the Enhanced Groove MIDI Dataset, which contains:
- Audio recordings of drum performances
- Corresponding MIDI transcriptions
- Multiple genres and playing styles

#### Download Instructions
1. Visit the [TensorFlow Datasets E-GMD page](https://www.tensorflow.org/datasets/catalog/e_gmd)
2. Download the dataset using TensorFlow Datasets:
   ```bash
   pip install tensorflow-datasets
   python -c "import tensorflow_datasets as tfds; tfds.load('e_gmd', download_and_prepare=True)"
   ```
3. The dataset will be downloaded to `~/tensorflow_datasets/e_gmd/`

#### Directory Structure
After downloading, organize the data as follows:
```
data/
├── e-gmd/
│   ├── audio/          # Audio files (.wav)
│   ├── midi/           # MIDI files (.mid)
│   └── metadata.json   # Dataset metadata
└── processed/
    ├── train/          # Processed training data
    ├── val/            # Processed validation data
    └── test/           # Processed test data
```

## Data Processing

### Preprocessing Script
Use the provided preprocessing script to convert the raw data into the format required by the training pipeline:

```bash
python scripts/preprocess_egmd.py --input_dir ~/tensorflow_datasets/e_gmd --output_dir data/processed
```

### Processing Steps
1. **Audio Processing**: Convert audio to spectrograms
2. **MIDI Processing**: Convert MIDI to piano roll format
3. **Alignment**: Ensure audio and MIDI data are properly aligned
4. **Split**: Divide into train/validation/test sets (80/10/10)

## Data Storage Recommendations

### Option 1: Local Storage
- Store data in the `data/` directory as shown above
- Ensure sufficient disk space (E-GMD is approximately 10GB)

### Option 2: External Storage
For larger datasets or multiple projects:
- Store data on an external drive or network storage
- Create symbolic links to the data directory:
  ```bash
  ln -s /path/to/external/data data
  ```

### Option 3: Cloud Storage
For cloud-based training:
- Upload processed data to cloud storage (AWS S3, Google Cloud Storage)
- Modify the data loading code to access cloud storage
- Update configuration files with cloud storage paths

## Configuration

Update the data paths in your configuration files:

```yaml
# configs/default_config.yaml
data:
  train_dir: "data/processed/train"
  val_dir: "data/processed/val"
  test_dir: "data/processed/test"
```

## Verification

Verify the data setup by running:
```bash
python scripts/evaluate.py --data_dir data/processed --mode verify
```

This will check that all required files are present and in the correct format.

## Additional Datasets

To extend the model's capabilities, you can incorporate additional drum datasets:
- **STOMPS**: Real-world drum recordings
- **MIDI-DDSP**: Synthetic drum data
- **Custom datasets**: Your own drum recordings

Follow the same preprocessing pipeline for any additional datasets.

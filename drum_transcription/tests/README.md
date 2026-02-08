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
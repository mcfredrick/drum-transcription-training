# Inference Guide - Roland TD-17 Drum Transcription

Complete guide for using trained models to transcribe drum audio to MIDI.

## Quick Start

**Prerequisites:**
- Completed setup (see [SETUP.md](SETUP.md))
- Trained model checkpoint (see [TRAINING.md](TRAINING.md))

**Basic inference:**
```bash
cd ~/Documents/drum-tranxn/drum_transcription

uv run python scripts/transcribe.py \
    --checkpoint /mnt/hdd/drum-tranxn/checkpoints_roland/best.ckpt \
    --audio /path/to/your/audio.wav \
    --output output.mid
```

**Output:** MIDI file with Roland TD-17 mapping (26 classes)

## Transcription Script

### Basic Usage

```bash
uv run python scripts/transcribe.py \
    --checkpoint <checkpoint_path> \
    --audio <audio_path> \
    --output <output_midi_path>
```

### Parameters

**Required:**
- `--checkpoint` - Path to trained model checkpoint (.ckpt file)
- `--audio` - Path to input audio file (WAV format recommended)
- `--output` - Path for output MIDI file

**Optional:**
- `--config` - Config file (uses checkpoint's config by default)
- `--threshold` - Onset detection threshold (default: 0.5)
  - Lower = more sensitive (more false positives)
  - Higher = less sensitive (fewer detections)
- `--device` - Device to use: 'cuda' or 'cpu' (auto-detected by default)

### Examples

**Single file:**
```bash
uv run python scripts/transcribe.py \
    --checkpoint /mnt/hdd/drum-tranxn/checkpoints_roland/roland-full-training-epoch=78-val_loss=0.0187.ckpt \
    --audio /mnt/hdd/drum-tranxn/e-gmd-v1.0.0/drummer1/session1/1_funk_120_beat_4-4.wav \
    --output funk_120_transcribed.mid
```

**Batch processing multiple files:**
```bash
#!/bin/bash
# Process all WAV files in a directory

CHECKPOINT="/mnt/hdd/drum-tranxn/checkpoints_roland/best.ckpt"
INPUT_DIR="/path/to/audio/files"
OUTPUT_DIR="/path/to/midi/output"

mkdir -p "$OUTPUT_DIR"

for audio in "$INPUT_DIR"/*.wav; do
    filename=$(basename "$audio" .wav)
    echo "Processing: $filename"
    uv run python scripts/transcribe.py \
        --checkpoint "$CHECKPOINT" \
        --audio "$audio" \
        --output "$OUTPUT_DIR/${filename}.mid"
done

echo "Done! Processed $(ls "$OUTPUT_DIR"/*.mid | wc -l) files"
```

**Custom threshold:**
```bash
# More sensitive (detect quieter hits)
uv run python scripts/transcribe.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --audio input.wav \
    --output output.mid \
    --threshold 0.3

# Less sensitive (only strong hits)
uv run python scripts/transcribe.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --audio input.wav \
    --output output.mid \
    --threshold 0.7
```

## Output Format

### MIDI File Structure

**Roland TD-17 mapping (26 MIDI notes):**
- Kick: MIDI 36
- Snare Head: MIDI 38
- Snare X-Stick: MIDI 37
- Snare Rim: MIDI 40
- Tom 1 Head: MIDI 48
- Tom 1 Rim: MIDI 50
- *(see [ROLAND_MAPPING.md](ROLAND_MAPPING.md) for complete list)*

**MIDI properties:**
- Velocity: 80 (default)
- Duration: 50 ticks
- Tempo: 120 BPM (default)

### Compatible Software

**Import transcribed MIDI into:**
- **DAWs**: Ableton Live, Logic Pro, FL Studio, Reaper, etc.
- **Drum software**: Superior Drummer, EZdrummer, Addictive Drums
- **Roland TD-17 module**: Direct import to practice with your kit
- **MIDI editors**: MuseScore, Sibelius, Guitar Pro

## Using with Roland TD-17

### Import to TD-17 Module

1. **Export MIDI from transcription:**
   ```bash
   uv run python scripts/transcribe.py \
       --checkpoint /path/to/checkpoint.ckpt \
       --audio song.wav \
       --output song.mid
   ```

2. **Transfer to SD card:**
   - Copy `song.mid` to TD-17's SD card
   - Path: `/ROLAND/TD-17/SONG/`

3. **Load on TD-17:**
   - Insert SD card into TD-17
   - Navigate to SONG mode
   - Select imported MIDI file
   - Play along with the transcription!

### Mapping Verification

The transcribed MIDI notes match Roland TD-17's default mapping:
- **Snare head** → Pad 1 (center)
- **Snare rim** → Pad 1 (rim)
- **Kick** → Pad KD
- **Toms** → Pads 2, 3, 4, 5
- **Hi-hat** → Pad HH (bow/edge/pedal)
- **Cymbals** → Pads CY1, CY2, CY3 (bow/edge/bell)

## Onset Detection Tuning

### Understanding Thresholds

The model outputs probabilities (0-1) for each drum class at each time frame. The threshold determines when a prediction becomes an "onset" (drum hit).

**Threshold trade-offs:**

| Threshold | Sensitivity | False Positives | False Negatives |
|-----------|-------------|-----------------|-----------------|
| 0.3       | High        | Many            | Few             |
| 0.5       | Medium      | Moderate        | Moderate        |
| 0.7       | Low         | Few             | Many            |

### Choosing the Right Threshold

**Use lower threshold (0.3-0.4) when:**
- Audio has low dynamic range (compressed)
- You want to capture ghost notes
- Better to have extra notes than miss notes

**Use default threshold (0.5) when:**
- Standard music production
- Balanced precision/recall
- Good starting point for most audio

**Use higher threshold (0.6-0.7) when:**
- Audio is noisy
- You want only confident hits
- Reducing false positives is critical

### Testing Thresholds

Compare different thresholds:
```bash
for thresh in 0.3 0.4 0.5 0.6 0.7; do
    echo "Testing threshold: $thresh"
    uv run python scripts/transcribe.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --audio input.wav \
        --output output_thresh_${thresh}.mid \
        --threshold $thresh
done
```

Listen to each MIDI file in your DAW to find the best threshold.

## Performance Optimization

### GPU Inference

**Automatic (recommended):**
```bash
# Automatically uses GPU if available
uv run python scripts/transcribe.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --audio input.wav \
    --output output.mid
```

**Forced GPU:**
```bash
uv run python scripts/transcribe.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --audio input.wav \
    --output output.mid \
    --device cuda
```

**CPU only (slower, but works without GPU):**
```bash
uv run python scripts/transcribe.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --audio input.wav \
    --output output.mid \
    --device cpu
```

### Batch Processing Tips

For processing many files:

1. **Use GPU** - 10-50x faster than CPU
2. **Process in parallel** (if you have multiple GPUs):
   ```bash
   # GPU 0
   CUDA_VISIBLE_DEVICES=0 process_batch_1.sh &
   # GPU 1
   CUDA_VISIBLE_DEVICES=1 process_batch_2.sh &
   ```
3. **Use SSD** - Faster audio loading
4. **Monitor progress** - Add logging to your batch scripts

### Inference Speed

**Typical speeds (RTX 3070):**
- 1 minute audio: ~2-5 seconds
- 5 minute audio: ~10-20 seconds
- 1 hour audio: ~3-5 minutes

CPU inference: 10-50x slower

## Troubleshooting

### "Checkpoint not found"

**Cause:** Invalid checkpoint path.

**Solution:**
```bash
# List available checkpoints
ls -lh /mnt/hdd/drum-tranxn/checkpoints_roland/full_training/

# Use full absolute path
uv run python scripts/transcribe.py \
    --checkpoint /mnt/hdd/drum-tranxn/checkpoints_roland/full_training/best.ckpt \
    --audio input.wav \
    --output output.mid
```

### "CUDA out of memory"

**Cause:** Audio file too long for GPU memory.

**Solutions:**
1. Use CPU:
   ```bash
   uv run python scripts/transcribe.py \
       --checkpoint /path/to/checkpoint.ckpt \
       --audio large_file.wav \
       --output output.mid \
       --device cpu
   ```
2. Split audio into chunks and process separately

### "Too many/too few detections"

**Cause:** Threshold not optimal for your audio.

**Solution:** Adjust threshold:
```bash
# More detections (lower threshold)
--threshold 0.3

# Fewer detections (higher threshold)
--threshold 0.7
```

### "Wrong drum sounds"

**Cause:** Model trained on different data or MIDI mapping mismatch.

**Solution:**
1. Verify using Roland-trained checkpoint
2. Check MIDI mapping in output file
3. Verify Roland TD-17 kit settings

### "Audio format not supported"

**Cause:** Input audio not in compatible format.

**Solution:** Convert to WAV first:
```bash
# Using ffmpeg
ffmpeg -i input.mp3 -ar 22050 -ac 1 output.wav

# Then transcribe
uv run python scripts/transcribe.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --audio output.wav \
    --output transcribed.mid
```

**Supported formats:** WAV, FLAC, MP3 (via librosa)
**Recommended:** WAV, 22050 Hz, mono

## Evaluation

### Evaluate Model Performance

Test your model on the test set:

```bash
uv run python scripts/evaluate.py \
    --checkpoint /mnt/hdd/drum-tranxn/checkpoints_roland/best.ckpt \
    --config configs/full_training_config.yaml
```

**Outputs:**
- Overall F1, precision, recall
- Per-class metrics
- Confusion matrix
- Performance summary

### Compare Checkpoints

Compare multiple checkpoints:

```bash
for ckpt in /mnt/hdd/drum-tranxn/checkpoints_roland/full_training/*.ckpt; do
    echo "Evaluating: $(basename $ckpt)"
    uv run python scripts/evaluate.py \
        --checkpoint "$ckpt" \
        --config configs/full_training_config.yaml
done
```

## Best Practices

### Preparing Audio

**For best results:**
1. **Audio quality:** Use high-quality recordings (44.1kHz or 48kHz)
2. **Drum isolation:** Cleaner drum audio → better transcription
3. **No extreme processing:** Avoid heavy compression/distortion
4. **Consistent levels:** Normalize audio to -6dB to -12dB peak

### Checkpoint Selection

**Choose checkpoint based on:**
1. **Lowest val_loss** (usually best overall)
2. **Highest val_f1** (if available)
3. **Epoch number** (later epochs often better)

**Find best checkpoint:**
```bash
ls -lhS /mnt/hdd/drum-tranxn/checkpoints_roland/full_training/*.ckpt
# Look for lowest val_loss in filename
```

### Post-Processing

**After transcription:**
1. **Import to DAW** - Visual inspection
2. **Quantize** - Align to grid if needed
3. **Velocity editing** - Adjust dynamics
4. **Manual correction** - Fix obvious errors
5. **Export** - Final MIDI for your use case

## Example Workflows

### Workflow 1: Practice with Roland TD-17

```bash
# 1. Transcribe your favorite song
uv run python scripts/transcribe.py \
    --checkpoint /path/to/best.ckpt \
    --audio favorite_song.wav \
    --output favorite_song.mid

# 2. Copy to SD card
cp favorite_song.mid /media/sdcard/ROLAND/TD-17/SONG/

# 3. Load on TD-17 and play along!
```

### Workflow 2: Create Drum Tracks

```bash
# 1. Transcribe reference audio
uv run python scripts/transcribe.py \
    --checkpoint /path/to/best.ckpt \
    --audio reference.wav \
    --output reference.mid

# 2. Import MIDI to DAW (Ableton, Logic, etc.)
# 3. Load drum VST (Superior Drummer, etc.)
# 4. Edit velocities and timing
# 5. Export final drum track
```

### Workflow 3: Learn Drum Parts

```bash
# 1. Transcribe challenging song
uv run python scripts/transcribe.py \
    --checkpoint /path/to/best.ckpt \
    --audio challenging_song.wav \
    --output challenging_song.mid

# 2. Import to notation software (MuseScore, etc.)
# 3. Study the notation
# 4. Practice slowly, speed up gradually
```

## Next Steps

- **Improve transcription:** Train longer or with more data ([TRAINING.md](TRAINING.md))
- **Fine-tune model:** Adjust hyperparameters ([docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md))
- **Custom mapping:** Modify MIDI mapping ([ROLAND_MAPPING.md](ROLAND_MAPPING.md))

## Additional Resources

- **[SETUP.md](SETUP.md)** - Installation guide
- **[TRAINING.md](TRAINING.md)** - Training models
- **[ROLAND_MAPPING.md](ROLAND_MAPPING.md)** - Drum mapping details
- **[docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md)** - Tuning guide

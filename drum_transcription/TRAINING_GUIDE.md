# Training Guide - Medium Test Run

This guide explains how to run the medium-duration training test (500 files, 10 epochs, ~3-4 hours).

## Quick Start

The simplest way to start training:

```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription
./start_medium_test.sh
```

This script will:
1. Preprocess 500 E-GMD files (skipping already processed files)
2. Train for 10 epochs with automatic checkpointing
3. Handle interruptions gracefully with automatic resumption

## Training Options

### Option 1: Simple Script (Recommended)
```bash
./start_medium_test.sh
```
- Runs preprocessing and training in one command
- Interactive mode - requires active terminal
- Best for initial runs

### Option 2: Python Runner (SSH-Safe)
```bash
# Interactive mode
uv run python run_training.py --config configs/medium_test_config.yaml

# Detached mode (survives SSH disconnection)
uv run python run_training.py --config configs/medium_test_config.yaml --detached
```
- Automatic retry on failure (up to 3 attempts)
- Lock file prevents multiple training sessions
- Detached mode continues even if SSH connection drops
- Logs saved to `/mnt/hdd/drum-tranxn/logs/train_output.log`

### Option 3: Direct Training
```bash
# First, preprocess the data
uv run python scripts/preprocess_egmd.py \
    --config configs/medium_test_config.yaml \
    --num-workers 8

# Then train
uv run python scripts/train.py \
    --config configs/medium_test_config.yaml
```

### Option 4: Using nohup (SSH-Safe)
```bash
# Start training that survives SSH disconnection
nohup ./start_medium_test.sh > train.log 2>&1 &

# Check progress
tail -f train.log

# Find the process
ps aux | grep train.py
```

## Resuming Training

If training is interrupted, it will automatically resume from the last checkpoint:

```bash
# Using the simple script
./start_medium_test.sh

# Using Python runner
uv run python run_training.py --config configs/medium_test_config.yaml

# Or directly
uv run python scripts/train.py \
    --config configs/medium_test_config.yaml \
    --resume /mnt/hdd/drum-tranxn/checkpoints/medium-test-last.ckpt
```

## Monitoring Training

### Real-time Progress
```bash
# Watch training output
tail -f /mnt/hdd/drum-tranxn/logs/train_output.log
```

### TensorBoard (Graphical)
```bash
# Start TensorBoard
tensorboard --logdir /mnt/hdd/drum-tranxn/logs

# Then open in browser: http://localhost:6006
# If running on remote server, use SSH tunnel:
# ssh -L 6006:localhost:6006 user@server
```

### Check GPU Usage
```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi
```

## Verification

### Check Preprocessed Files
```bash
# Count processed files (should be 500)
find /mnt/hdd/drum-tranxn/processed_data -name "*.h5" | wc -l

# Check split sizes
wc -l /mnt/hdd/drum-tranxn/processed_data/splits/*.txt
# Expected: ~350 train, ~75 val, ~75 test
```

### Check Checkpoints
```bash
# List all checkpoints
ls -lh /mnt/hdd/drum-tranxn/checkpoints/medium-test-*.ckpt

# Find best checkpoint (lowest validation loss)
ls -1v /mnt/hdd/drum-tranxn/checkpoints/medium-test-epoch*.ckpt | head -3
```

## After Training

### Evaluate the Model
```bash
# Find your best checkpoint
BEST_CKPT=$(ls -1v /mnt/hdd/drum-tranxn/checkpoints/medium-test-epoch*.ckpt | grep -v last | head -1)

# Run evaluation
uv run python scripts/evaluate.py \
    --checkpoint "$BEST_CKPT" \
    --config configs/medium_test_config.yaml \
    --split test \
    --output docs/medium-test-eval-results.json
```

### Test Inference
```bash
# Get a test audio file
TEST_AUDIO=$(find /mnt/hdd/drum-tranxn/e-gmd-v1.0.0 -name "*.wav" | shuf -n 1)

# Transcribe it
BEST_CKPT=$(ls -1v /mnt/hdd/drum-tranxn/checkpoints/medium-test-epoch*.ckpt | grep -v last | head -1)

uv run python scripts/transcribe.py \
    "$TEST_AUDIO" \
    /tmp/test_transcription.mid \
    --checkpoint "$BEST_CKPT"
```

## Configuration

The medium test configuration is in `configs/medium_test_config.yaml`:

- **Dataset**: 500 files from E-GMD
- **Epochs**: 10
- **Batch size**: 8
- **GPU**: RTX 3070
- **Precision**: Mixed precision (FP16)
- **Early stopping**: Disabled (to see full 10-epoch trajectory)

## Troubleshooting

### "Another training session is already running"
```bash
# Check if training is actually running
ps aux | grep train.py

# If not, remove the lock file
rm /tmp/drum_train.lock
```

### Out of Memory
```bash
# Reduce batch size in config
# Edit configs/medium_test_config.yaml
# Change: batch_size: 8  ->  batch_size: 4
```

### CUDA Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Kill any hanging processes
pkill -f train.py

# Clear CUDA cache (run in Python)
python -c "import torch; torch.cuda.empty_cache()"
```

### Training is very slow
```bash
# Check GPU utilization
nvidia-smi

# Increase num_workers if CPU is not saturated
# Edit configs/medium_test_config.yaml
# Change: num_workers: 4  ->  num_workers: 8
```

## Storage Locations

- **Raw data**: `/mnt/hdd/drum-tranxn/e-gmd-v1.0.0` (read-only)
- **Preprocessed data**: `/mnt/hdd/drum-tranxn/processed_data` (~3 GB)
- **Checkpoints**: `/mnt/hdd/drum-tranxn/checkpoints` (~100 MB)
- **Logs**: `/mnt/hdd/drum-tranxn/logs` (~50 MB)
- **Splits**: `/mnt/hdd/drum-tranxn/processed_data/splits` (text files)

## Expected Results

After 10 epochs on 500 files, you should see:

- ✅ Validation loss < 0.20 (better than quick test)
- ✅ Overall F1-score > 0.25
- ✅ Kick/snare/hihat recall > 0.30
- ✅ Training completes without errors
- ✅ Inference produces non-empty MIDI files

If these criteria are met, proceed to full training (all 4,148 files, 100 epochs).

## Next Steps

If the medium test succeeds:
1. Review the evaluation metrics
2. Adjust hyperparameters if needed
3. Plan full training run (2-3 days)
4. Consider enabling data augmentation for full training
5. Enable early stopping for production training

See `docs/second-test-run-plan.md` for detailed success criteria and decision matrix.

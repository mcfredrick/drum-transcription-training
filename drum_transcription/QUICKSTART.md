# Quick Start - Medium Test Training

## TL;DR

Run this single command to start the medium test training (500 files, 10 epochs, ~3-4 hours):

```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription
./start_medium_test.sh
```

## For SSH Sessions (Recommended)

If you're running via SSH and want training to survive disconnections:

```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription

# Option 1: Python runner with detached mode
uv run python run_training.py --config configs/medium_test_config.yaml --detached

# Option 2: Using nohup
nohup ./start_medium_test.sh > train.log 2>&1 &
tail -f train.log  # Monitor progress
```

## Monitor Training

```bash
# Watch logs
tail -f /mnt/hdd/drum-tranxn/logs/train_output.log

# Check GPU usage
watch -n 1 nvidia-smi

# View TensorBoard (graphical)
tensorboard --logdir /mnt/hdd/drum-tranxn/logs
# Then open: http://localhost:6006
```

## Resume After Interruption

Training automatically resumes from the last checkpoint. Just run the same command again:

```bash
./start_medium_test.sh
```

## After Training

```bash
# Find best checkpoint
ls -1v /mnt/hdd/drum-tranxn/checkpoints/medium-test-epoch*.ckpt | head -3

# Evaluate
BEST_CKPT=$(ls -1v /mnt/hdd/drum-tranxn/checkpoints/medium-test-epoch*.ckpt | grep -v last | head -1)
uv run python scripts/evaluate.py \
    --checkpoint "$BEST_CKPT" \
    --config configs/medium_test_config.yaml \
    --split test
```

For detailed documentation, see `TRAINING_GUIDE.md`.

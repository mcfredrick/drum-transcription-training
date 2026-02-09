#!/bin/bash
# Auto-training script for standard drum kit transcription (11-class system)
# This script trains with the best hyperparameters from Optuna study

set -e  # Exit on error

echo "=========================================="
echo "Drum Transcription Training (11-Class)"
echo "=========================================="
echo ""
echo "Using optimized hyperparameters from Optuna study:"
echo "  (Will be updated after running Optuna for 11-class system)"
echo "  Learning rate: 0.001 (default)"
echo "  Batch size: 16"
echo "  Weight decay: 0.0001"
echo "  Optimizer: AdamW"
echo ""
echo "Training configuration:"
echo "  Config: configs/full_training_config.yaml"
echo "  Data: /mnt/hdd/drum-tranxn/processed_data"
echo "  Model: 11-class standard drum kit"
echo "  Epochs: 100"
echo ""
echo "=========================================="
echo ""

# Validation checks
echo "Validating configuration..."

# Check config uses correct data path
if ! grep -q "processed_data" configs/full_training_config.yaml; then
    echo "❌ ERROR: Config doesn't use correct data path!"
    echo "   Expected: processed_data"
    exit 1
fi

# Check config uses 11 classes
if ! grep -q "n_classes: 11" configs/full_training_config.yaml; then
    echo "❌ ERROR: Config doesn't use 11 classes!"
    echo "   Expected: n_classes: 11"
    exit 1
fi

echo "✅ Configuration validated"
echo ""

# Check if data exists
if [ ! -d "/mnt/hdd/drum-tranxn/processed_data/splits" ]; then
    echo "ERROR: Preprocessed data not found!"
    echo "Please run: uv run python scripts/preprocess.py --config configs/drum_config.yaml"
    exit 1
fi

# Set W&B flag (default: false for local runs)
export USE_WANDB=${USE_WANDB:-false}

# Create log directory if it doesn't exist
mkdir -p /mnt/hdd/drum-tranxn/logs

# Start training
echo "Starting training..."
echo ""

if [ "$1" == "nohup" ]; then
    # Background mode with nohup
    nohup uv run python scripts/train.py \
        --config configs/full_training_config.yaml \
        > /mnt/hdd/drum-tranxn/logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    PID=$!
    echo "Training started in background (PID: $PID)"
    echo "Log file: /mnt/hdd/drum-tranxn/logs/training_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f /mnt/hdd/drum-tranxn/logs/training_*.log"
    echo ""
    echo "To check training:"
    echo "  tensorboard --logdir=/mnt/hdd/drum-tranxn/logs"
else
    # Foreground mode
    uv run python scripts/train.py --config configs/full_training_config.yaml
fi

echo ""
echo "Training complete!"
echo "Best checkpoint will be in: /mnt/hdd/drum-tranxn/checkpoints/"

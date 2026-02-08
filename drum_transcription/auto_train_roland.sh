#!/bin/bash
# Auto-training script for Roland TD-17 drum transcription
# This script trains with the best hyperparameters from Optuna study

set -e  # Exit on error

echo "=========================================="
echo "Roland TD-17 Drum Transcription Training"
echo "=========================================="
echo ""
echo "Using optimized hyperparameters from Optuna study:"
echo "  Learning rate: 0.00044547593667580503"
echo "  Batch size: 4"
echo "  Weight decay: 5.559904341261682e-05"
echo "  Optimizer: AdamW"
echo ""
echo "Training configuration:"
echo "  Config: configs/full_training_config.yaml"
echo "  Data: /mnt/hdd/drum-tranxn/processed_data_roland"
echo "  Model: 26-class Roland mapping"
echo "  Epochs: 150"
echo ""
echo "=========================================="
echo ""

# Validation checks
echo "Validating configuration..."

# Check config uses Roland data
if ! grep -q "processed_data_roland" configs/full_training_config.yaml; then
    echo "❌ ERROR: Config doesn't use Roland data!"
    echo "   Expected: processed_data_roland"
    exit 1
fi

# Check config uses 26 classes
if ! grep -q "n_classes: 26" configs/full_training_config.yaml; then
    echo "❌ ERROR: Config doesn't use 26 classes!"
    echo "   Expected: n_classes: 26"
    exit 1
fi

echo "✅ Configuration validated"
echo ""

# Check if data exists
if [ ! -d "/mnt/hdd/drum-tranxn/processed_data_roland/splits" ]; then
    echo "ERROR: Roland preprocessed data not found!"
    echo "Please run: uv run python scripts/preprocess_roland.py --config configs/roland_config.yaml"
    exit 1
fi

# Check if legacy data still exists (shouldn't)
if [ -d "/mnt/hdd/drum-tranxn/processed_data" ]; then
    echo "WARNING: Legacy processed_data directory still exists!"
    echo "This should have been removed. Please check."
fi

# Set W&B flag (default: false for local runs)
export USE_WANDB=${USE_WANDB:-false}

# Create log directory if it doesn't exist
mkdir -p /mnt/hdd/drum-tranxn/logs_roland

# Start training
echo "Starting training..."
echo ""

if [ "$1" == "nohup" ]; then
    # Background mode with nohup
    nohup uv run python scripts/train.py \
        --config configs/full_training_config.yaml \
        > /mnt/hdd/drum-tranxn/logs_roland/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    
    PID=$!
    echo "Training started in background (PID: $PID)"
    echo "Log file: /mnt/hdd/drum-tranxn/logs_roland/training_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f /mnt/hdd/drum-tranxn/logs_roland/training_*.log"
    echo ""
    echo "To check training:"
    echo "  tensorboard --logdir=/mnt/hdd/drum-tranxn/logs_roland"
else
    # Foreground mode
    uv run python scripts/train.py --config configs/full_training_config.yaml
fi

echo ""
echo "Training complete!"
echo "Best checkpoint will be in: /mnt/hdd/drum-tranxn/checkpoints_roland/"

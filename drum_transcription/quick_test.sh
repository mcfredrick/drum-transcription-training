#!/bin/bash
# Quick validation test - runs in ~5-10 minutes

set -e

echo "=========================================="
echo "Quick Test - Roland TD-17 Drum Transcription"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Files: 10"
echo "  Epochs: 5"
echo "  Classes: 26 (Roland mapping)"
echo "  Time: ~5-10 minutes"
echo ""
echo "=========================================="
echo ""

# Verify Roland data exists
if [ ! -d "/mnt/hdd/drum-tranxn/processed_data_roland/splits" ]; then
    echo "ERROR: Roland preprocessed data not found!"
    echo "Please run preprocessing first:"
    echo "  uv run python scripts/preprocess_roland.py --config configs/roland_config.yaml"
    exit 1
fi

# Create checkpoint directory
mkdir -p /mnt/hdd/drum-tranxn/checkpoints_roland/quick_test

# Run training
echo "Starting quick test..."
echo ""

USE_WANDB=false uv run python scripts/train.py \
    --config configs/quick_test_config.yaml

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "=========================================="
echo ""
echo "✅ If test completed successfully, your setup is working!"
echo "✅ Proceed with full training: ./auto_train_roland.sh"
echo ""
echo "📊 View results:"
echo "  tensorboard --logdir=/mnt/hdd/drum-tranxn/logs_roland/quick_test"

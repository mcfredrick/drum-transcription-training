#!/bin/bash
# Simple script to start the medium duration training test
# This script is robust to SSH disconnections and will automatically resume

set -e

echo "========================================================"
echo "  Drum Transcription - Medium Test Training"
echo "========================================================"
echo ""
echo "This script will:"
echo "  1. Preprocess 500 E-GMD files (with caching)"
echo "  2. Train for 10 epochs with automatic checkpointing"
echo "  3. Resume automatically if interrupted"
echo ""
echo "Estimated time: 3-4 hours"
echo ""
echo "Press Ctrl+C at any time to stop (progress will be saved)"
echo "========================================================"
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Step 1: Preprocessing
echo "Step 1/2: Preprocessing 500 files..."
echo "This will skip any files that have already been processed."
echo ""

uv run python scripts/preprocess_egmd.py \
    --config configs/medium_test_config.yaml \
    --num-workers 8

echo ""
echo "✓ Preprocessing complete!"
echo ""

# Step 2: Training
echo "Step 2/2: Starting training..."
echo ""

./scripts/train_robust.sh configs/medium_test_config.yaml

echo ""
echo "========================================================"
echo "  Training Complete!"
echo "========================================================"
echo ""
echo "Next steps:"
echo "  1. View logs: tensorboard --logdir /mnt/hdd/drum-tranxn/logs"
echo "  2. Find best checkpoint:"
echo "       ls -1v /mnt/hdd/drum-tranxn/checkpoints/medium-test-*.ckpt | head -3"
echo "  3. Evaluate: uv run python scripts/evaluate.py --checkpoint <path> --config configs/medium_test_config.yaml"
echo ""

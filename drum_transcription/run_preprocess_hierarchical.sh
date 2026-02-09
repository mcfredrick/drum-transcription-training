#!/bin/bash
# Preprocess E-GMD dataset for hierarchical model (12-class with crash)

echo "========================================================================"
echo "E-GMD PREPROCESSING FOR HIERARCHICAL MODEL (12-class with crash)"
echo "========================================================================"
echo ""
echo "This will:"
echo "  - Process all E-GMD audio files (~2256 files)"
echo "  - Extract 128-bin mel spectrograms"
echo "  - Create 12-class labels (including crash cymbal - MIDI 49)"
echo "  - Save to: /mnt/hdd/drum-tranxn/processed_data_hierarchical/"
echo "  - Estimated time: 1-2 hours"
echo ""
echo "========================================================================"
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Run preprocessing
echo "Starting preprocessing..."
uv run python scripts/preprocess.py \
    --config configs/preprocess_hierarchical.yaml \
    --num-workers 8 \
    --use-hdf5

echo ""
echo "========================================================================"
echo "PREPROCESSING COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Update hierarchical training config to use new data:"
echo "     processed_root: /mnt/hdd/drum-tranxn/processed_data_hierarchical"
echo "  2. Run training:"
echo "     uv run python scripts/train_hierarchical.py"
echo ""
echo "========================================================================"

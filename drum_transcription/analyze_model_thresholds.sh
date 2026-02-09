#!/bin/bash
# Convenience script to analyze model performance across thresholds

# Default paths
CONFIG="configs/full_training_config.yaml"
CHECKPOINT="/mnt/hdd/drum-tranxn/checkpoints/drum-11class-epoch=95-val_loss=0.0881.ckpt"
OUTPUT_DIR="threshold_analysis_results"
SPLIT="val"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config CONFIG] [--checkpoint CHECKPOINT] [--output-dir OUTPUT_DIR] [--split SPLIT]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Running Threshold Analysis"
echo "=========================================="
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_DIR"
echo "Split:      $SPLIT"
echo "=========================================="
echo ""

# Run analysis
uv run python scripts/analyze_thresholds.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --thresholds "0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8"

echo ""
echo "=========================================="
echo "Analysis complete! Results in: $OUTPUT_DIR"
echo "=========================================="

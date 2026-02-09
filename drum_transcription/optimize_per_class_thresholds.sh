#!/bin/bash

# Convenience wrapper for per-class threshold optimization
# Finds optimal threshold for each drum class independently

set -e

# Default values
CHECKPOINT=""
CONFIG="configs/full_training_config.yaml"
SPLIT="val"
OUTPUT_DIR="per_class_threshold_results"
STRATEGY="rhythm_game"
MIN_PRECISION="0.60"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --min-precision)
            MIN_PRECISION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH       Path to checkpoint (default: latest)"
            echo "  --config PATH          Config file (default: configs/full_training_config.yaml)"
            echo "  --split SPLIT          Dataset split: val|test (default: val)"
            echo "  --output-dir DIR       Output directory (default: per_class_threshold_results)"
            echo "  --strategy STRATEGY    Optimization strategy (default: rhythm_game)"
            echo "                         Choices: max_f1, max_recall, balanced, rhythm_game"
            echo "  --min-precision VALUE  Minimum precision (default: 0.60)"
            echo "  --help                 Show this help"
            echo ""
            echo "Example:"
            echo "  $0 --checkpoint /path/to/model.ckpt --min-precision 0.65"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "PER-CLASS THRESHOLD OPTIMIZATION"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Config:        $CONFIG"
echo "  Split:         $SPLIT"
echo "  Strategy:      $STRATEGY"
echo "  Min Precision: $MIN_PRECISION"
if [ -n "$CHECKPOINT" ]; then
    echo "  Checkpoint:    $CHECKPOINT"
else
    echo "  Checkpoint:    (will use latest)"
fi
echo "  Output:        $OUTPUT_DIR"
echo ""

# Build command
CMD="uv run python scripts/optimize_per_class_thresholds.py"
CMD="$CMD --config $CONFIG"
CMD="$CMD --split $SPLIT"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --strategy $STRATEGY"
CMD="$CMD --min-precision $MIN_PRECISION"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Run optimization
echo "Running optimization..."
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "Optimization complete! Results in: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Quick view:"
echo "  Report: cat $OUTPUT_DIR/per_class_optimization_report.txt"
echo "  Config: cat $OUTPUT_DIR/optimized_thresholds.yaml"
echo "  Plots:  open $OUTPUT_DIR/per_class_optimization.png"

#!/bin/bash
# Resume hierarchical training after OOM fix

set -e

echo "=========================================="
echo "Resuming Hierarchical Training (OOM Fix)"
echo "=========================================="
echo ""
echo "Fix applied: Detach tensors in validation step"
echo "Resuming from: /mnt/hdd/drum-tranxn/checkpoints/hierarchical/last.ckpt"
echo ""

# Optional: Enable expandable segments to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resume training
uv run python scripts/train_hierarchical.py \
    --config configs/hierarchical_config.yaml \
    --resume-from-checkpoint /mnt/hdd/drum-tranxn/checkpoints/hierarchical/last.ckpt

echo ""
echo "Training completed!"

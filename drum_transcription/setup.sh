#!/bin/bash
# Quick setup script for drum transcription project

set -e  # Exit on error

echo "=========================================="
echo "Drum Transcription Setup"
echo "=========================================="
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo ""
    echo "UV installed! Please restart your terminal and run this script again."
    exit 0
fi

echo "✓ UV found"
echo ""

# Install dependencies
echo "Installing dependencies..."
uv sync
echo "✓ Dependencies installed"
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
if uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "✓ CUDA is available"
    uv run python -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA version: {torch.version.cuda}'); print(f'  GPUs available: {torch.cuda.device_count()}')"
else
    echo "⚠ CUDA not available (CPU-only mode)"
    echo "  If you have a GPU, check your CUDA installation"
fi
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data/e-gmd
mkdir -p data/processed
mkdir -p data/splits
mkdir -p checkpoints
mkdir -p logs
echo "✓ Directories created"
echo ""

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download E-GMD dataset:"
echo "   wget http://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip"
echo "   unzip e-gmd-v1.0.0.zip -d data/e-gmd/"
echo ""
echo "2. Preprocess dataset:"
echo "   uv run python scripts/preprocess_egmd.py --num-workers 8"
echo ""
echo "3. Train model:"
echo "   uv run python scripts/train.py"
echo ""
echo "See README.md for detailed instructions."

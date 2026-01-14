#!/bin/bash
# Quick start script for running detailed model tests

echo "=========================================="
echo "Detailed Model Testing - Quick Start"
echo "=========================================="
echo ""

# Check if files exist
echo "Checking required files..."
if [ ! -f "deepfake_embeddings_2.h5" ]; then
    echo "⚠️  ERROR: deepfake_embeddings_2.h5 not found!"
    echo "   Please update the path in this script or use full path"
    exit 1
fi

if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "⚠️  ERROR: checkpoints/best_model.pt not found!"
    exit 1
fi

if [ ! -f "test_model_detailed.py" ]; then
    echo "⚠️  ERROR: test_model_detailed.py not found!"
    exit 1
fi

echo "✅ All required files found!"
echo ""

# Run the test
echo "Starting test on AVDeepfake1M and ShareVeo3..."
echo ""

python test_model_detailed.py \
    --hdf5_path deepfake_embeddings_2.h5 \
    --checkpoint_path checkpoints/best_model.pt

echo ""
echo "=========================================="
echo "Test completed! Check test_results/ directory for JSON files"
echo "=========================================="

#!/bin/bash

# Exit on error
set -e

echo "================================"
echo "Starting Transformer Training"
echo "================================"
uv run python -m experiments.seqmodels.generate_custom --config experiments/seqmodels/configs/transformer.yaml

echo ""
echo "================================"
echo "Transformer Training Complete!"
echo "Starting LSTM Training"
echo "================================"
uv run python -m experiments.seqmodels.generate_lstm --config experiments/seqmodels/configs/lstm.yaml

echo ""
echo "================================"
echo "All Training Complete!"
echo "================================"

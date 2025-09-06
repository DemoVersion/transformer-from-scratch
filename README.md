# Transformer From Scratch

Simple transformer implementation from scratch in PyTorch. This repository is forked from https://codeberg.org/pbm/former and the intent is to experiment the transformer architecture in PyTorch and compare it to a LSTM with attention baseline for text generation tasks.

See http://peterbloem.nl/blog/transformers for an in-depth explanation.

All models in the repository consist of a single stack of transformer blocks (that is, no encoder/decoder structures). It turns out that this simple configuration often works best.

## Install dependencies
#### Install uv
Install `uv` (if not already installed):  
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

#### Install dependencies
After cloning the repository, install the dependencies like this
```bash
uv sync
```

## Usage

Run a classification experiment on the IMDb dataset:
```bash
uv run python experiments/classify.py
```

For other experiments, see the `experiments/` directory. Hyperparameters are passed as command line arguments. Use `--help` to see available options:
```bash
uv run python experiments/classify.py --help
```
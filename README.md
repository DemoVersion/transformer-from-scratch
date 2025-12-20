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
uv sync --extra ml
```

## Usage

### Quick Test
Run tests to verify installation:
```bash
uv run python -m unittest discover tests
```

### Running Experiments

**Classification** (IMDb sentiment analysis):
```bash
uv run python experiments/classify.py
```

**Text Generation** (enwik8, character-level):
```bash
uv run python experiments/generate.py
```

**Transformer with BPE** (C4 dataset):
```bash
# Quick test with small config
uv run python -m experiments.seqmodels.generate_custom --config experiments/seqmodels/configs/transformer_test.yaml

# Full training
uv run python -m experiments.seqmodels.generate_custom --config experiments/seqmodels/configs/transformer.yaml
```

**LSTM with BPE** (C4 dataset):
```bash
# Quick test
uv run python -m experiments.seqmodels.generate_lstm --config experiments/seqmodels/configs/lstm_test.yaml

# Full training
uv run python -m experiments.seqmodels.generate_lstm --config experiments/seqmodels/configs/lstm.yaml
```

Use `--help` to see available options for any script.
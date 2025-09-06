# Transformer From Scratch
This repository is forked from https://codeberg.org/pbm/former and the intent is to experiment the transformer architecture in PyTorch and compare it to a LSTM with attention baseline for text generation tasks.

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
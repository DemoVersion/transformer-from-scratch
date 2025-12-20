"""Shared utility functions for sequence model training and generation."""

from datetime import datetime
from pathlib import Path
from typing import Union

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from experiments.seqmodels.alternative_architectures.simplified_awd_lstm import (
    SimplifiedAWDLSTM,
)
from experiments.seqmodels.config_schema import (
    LSTMExperimentConfig,
    TransformerExperimentConfig,
)
from experiments.seqmodels.tokenizer import load_tokenizer, save_tokenizer
from former import GTransformer


def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution.

    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """
    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()


def sample_batch(data, length, batch_size):
    """
    Sample a batch of random subsequences from the data.

    For each input instance, also creates the target sequence shifted one position right.

    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of integer matrices
    """
    # Sample the starting indices of the sequences to slice out
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # Slice out the input sequences
    seqs_inputs = [data[start : start + length] for start in starts]
    seqs_target = [data[start + 1 : start + length + 1] for start in starts]

    # Concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)

    return inputs, target


def sample_sequence(
    model, seed, tokenizer, max_context, length=600, temperature=0.5, verbose=False
):
    """
    Sequentially samples a sequence from the model, token by token.

    :param model: The transformer model
    :param seed: The sequence to start with
    :param tokenizer: The tokenizer for decoding tokens
    :param max_context: Maximum context length the model can handle
    :param length: The total number of characters to sample
    :param temperature: The sampling temperature
    :param verbose: If true, print the sampled sequence as it is sampled
    :return: The sampled sequence, including the seed
    """
    sequence = seed.detach().clone()

    if verbose:  # Print the seed, surrounded by square brackets
        print("[", end="", flush=True)
        seed_text = tokenizer.decode(seed.tolist())
        print(seed_text, end="", flush=True)
        print("]", end="", flush=True)

    for _ in range(length):
        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :])

        # Sample the next token from the probabilities at the last position of the output
        c = sample(output[0, -1, :], temperature)

        if verbose:
            token_text = tokenizer.decode([c.item()])
            print(token_text, end="", flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0)

    print()
    return sequence


def create_checkpoint_dir(base_dir: str) -> Path:
    """Create a unique checkpoint directory for this run.

    Args:
        base_dir: Base directory for checkpoints

    Returns:
        Path to the created checkpoint directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(base_dir) / f"run_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(
    model: nn.Module,
    config,
    checkpoint_dir: Path,
    batch_idx: int,
    tokenizer: Tokenizer,
    is_final: bool = False,
):
    """Save model checkpoint, configuration, and tokenizer.

    Args:
        model: Model to save
        config: Configuration used for training (TransformerExperimentConfig or LSTMExperimentConfig)
        checkpoint_dir: Directory to save checkpoint in
        batch_idx: Current batch index
        tokenizer: Tokenizer to save
        is_final: Whether this is the final checkpoint
    """
    checkpoint_name = (
        "final_model.pt" if is_final else f"checkpoint_batch_{batch_idx}.pt"
    )
    checkpoint_path = checkpoint_dir / checkpoint_name

    # Save model state dict
    torch.save(model.state_dict(), checkpoint_path)

    # Save tokenizer if it doesn't exist yet
    tokenizer_path = checkpoint_dir / "tokenizer"
    if not tokenizer_path.exists():
        save_tokenizer(tokenizer, tokenizer_path)

    # Save configuration (only for final checkpoint to avoid duplication)
    if is_final:
        config_path = checkpoint_dir / "config.yaml"
        config.to_yaml(config_path)
        print(f"\nCheckpoint saved to: {checkpoint_dir}")
        print(f"  - Model: {checkpoint_name}")
        print("  - Config: config.yaml")
        print("  - Tokenizer: tokenizer/")
    else:
        print(f"Checkpoint saved: {checkpoint_name}")


def load_checkpoint(
    checkpoint_dir: Union[str, Path],
    checkpoint_name: str = "final_model.pt",
    device: Union[str, torch.device, None] = None,
) -> tuple[
    nn.Module, Union[TransformerExperimentConfig, LSTMExperimentConfig], Tokenizer, int
]:
    """Load model checkpoint, configuration, and tokenizer.

    Args:
        checkpoint_dir: Directory containing the checkpoint and config
        checkpoint_name: Name of the checkpoint file (default: "final_model.pt")
        device: Device to load model onto (default: auto-detect CUDA/CPU)

    Returns:
        Tuple of (model, config, tokenizer, vocab_size)
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load configuration
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Try to load as transformer config first, then LSTM
    try:
        config = TransformerExperimentConfig.from_yaml(config_path)
    except Exception:
        try:
            config = LSTMExperimentConfig.from_yaml(config_path)
        except Exception as e:
            raise ValueError(f"Could not load config as transformer or LSTM: {e}")

    # Load tokenizer
    tokenizer_path = checkpoint_dir / "tokenizer"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # Instantiate model based on config type
    if isinstance(config, TransformerExperimentConfig):
        model = GTransformer(
            emb=config.model.embedding_size,
            heads=config.model.num_heads,
            depth=config.model.depth,
            seq_length=config.model.context,
            num_tokens=vocab_size,
            attention_type=config.model.attention_type,
        )
    elif isinstance(config, LSTMExperimentConfig):
        model = SimplifiedAWDLSTM(
            vocab_size=vocab_size,
            embed_dim=config.model.embedding_size,
            hidden_dim=config.model.hidden_size,
            num_layers=config.model.num_layers,
            dropout_rate=config.model.dropout_rate,
        )
    else:
        raise ValueError(f"Unexpected config type: {type(config)}")

    # Load checkpoint
    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode by default

    model_type = (
        "transformer" if isinstance(config, TransformerExperimentConfig) else "lstm"
    )
    print(f"Loaded {model_type} model from: {checkpoint_dir}")
    print(f"  - Checkpoint: {checkpoint_name}")
    print(f"  - Device: {device}")
    print(f"  - Vocab size: {vocab_size:,}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config, tokenizer, vocab_size

"""
Dataset loader that downloads, tokenizes, and prepares data for transformer training.

This module handles the complete pipeline:
1. Download dataset (C4 or custom) via streaming
2. Tokenize using BPE tokenizer
3. Return tokenized data as torch tensors ready for training

Usage:
    # Load tokenized dataset for training
    data_train, data_val, data_test = load_tokenized_dataset(
        target_tokens=10_000_000,
        tokenizer_path="playground/bpe-tokenizer"
    )

    # Save tokenized dataset to disk for faster loading
    save_tokenized_dataset(
        data_train, data_val, data_test,
        save_path="playground/data/c4_10m_tokenized.pt"
    )

    # Load from saved file
    data_train, data_val, data_test = load_saved_tokenized_dataset(
        "playground/data/c4_10m_tokenized.pt"
    )
"""

import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import click
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from experiments.seqmodels.config_schema import (
    DatasetConfig,
    LSTMExperimentConfig,
    TransformerExperimentConfig,
)
from experiments.seqmodels.tokenizer import (
    build_bpe_tokenizer,
    load_tokenizer,
    save_tokenizer,
    train_tokenizer,
)


def train_tokenizer_from_c4(
    dataset_config: DatasetConfig,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Train a BPE tokenizer from scratch on dataset samples.

    Args:
        dataset_config: Dataset configuration object containing all dataset parameters
        cache_dir: Custom cache directory for downloads
        verbose: Print progress information
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print("TRAINING TOKENIZER FROM SCRATCH")
        print(f"{'=' * 80}")
        print(f"Target vocab size: {dataset_config.tokenizer_vocab_size:,}")
        print(f"Training documents: {dataset_config.tokenizer_training_docs:,}")
        print(
            f"Streaming {dataset_config.dataset_name}/{dataset_config.dataset_config} dataset (split: {dataset_config.dataset_split})..."
        )

    # Stream dataset and collect training texts
    dataset = load_dataset(
        dataset_config.dataset_name,
        dataset_config.dataset_config,
        split=dataset_config.dataset_split,
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Collect training texts
    training_texts = []
    for i, example in enumerate(dataset):
        if i >= dataset_config.tokenizer_training_docs:
            break
        training_texts.append(example["text"])  # type: ignore[index]
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Collected {i + 1:,} documents...")

    if verbose:
        print(f"\n✓ Collected {len(training_texts):,} documents")
        print("Training BPE tokenizer...")

    # Create temporary corpus directory (auto-deleted when context exits)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write texts to temporary files
        corpus_files = []
        temp_path = Path(temp_dir)

        for i, text in enumerate(training_texts):
            file_path = temp_path / f"corpus_{i}.txt"
            file_path.write_text(text)
            corpus_files.append(file_path)

        # Build and train tokenizer
        tokenizer, trainer = build_bpe_tokenizer(
            vocab_size=dataset_config.tokenizer_vocab_size,
            min_frequency=1,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )

        train_tokenizer(tokenizer, trainer, corpus_files)

        if verbose:
            print("✓ Tokenizer trained successfully")
            print(f"Saving tokenizer to {dataset_config.tokenizer_path}...")

        # Save tokenizer
        save_tokenizer(tokenizer, dataset_config.tokenizer_path)

        if verbose:
            actual_vocab_size = tokenizer.get_vocab_size()
            print("✓ Tokenizer saved!")
            print(f"  Actual vocab size: {actual_vocab_size:,}")
            print(f"{'=' * 80}\n")


def prepare_tokenized_dataset(
    dataset_config: DatasetConfig,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tokenizer]:
    """
    Train a fresh tokenizer from scratch and tokenize dataset.

    This function always trains a new BPE tokenizer before tokenizing the dataset.

    Args:
        dataset_config: Dataset configuration object containing all dataset parameters
        cache_dir: Custom cache directory for downloads
        verbose: Print progress information

    Returns:
        Tuple of (train_tensor, val_tensor, test_tensor, tokenizer) containing token IDs and the tokenizer
    """
    # Train a fresh tokenizer from scratch
    train_tokenizer_from_c4(
        dataset_config=dataset_config,
        cache_dir=cache_dir,
        verbose=verbose,
    )

    # Now load and tokenize the dataset using the freshly trained tokenizer
    train_data, val_data, test_data = load_tokenized_dataset(
        dataset_config=dataset_config,
        cache_dir=cache_dir,
        verbose=verbose,
    )

    # Load the tokenizer to return it
    tokenizer = load_tokenizer(dataset_config.tokenizer_path)

    return train_data, val_data, test_data, tokenizer


def load_tokenized_dataset(
    dataset_config: DatasetConfig,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and tokenize dataset, returning train/val/test splits as token ID tensors.

    Args:
        dataset_config: Dataset configuration object containing all dataset parameters
        cache_dir: Custom cache directory for downloads
        verbose: Print progress information

    Returns:
        Tuple of (train_tensor, val_tensor, test_tensor) containing token IDs
    """
    if verbose:
        print(f"Loading tokenizer from {dataset_config.tokenizer_path}...")
    tokenizer = load_tokenizer(dataset_config.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    if verbose:
        print(f"Tokenizer loaded: vocab_size={vocab_size:,}")
        print(
            f"Streaming {dataset_config.dataset_name}/{dataset_config.dataset_config} dataset (split: {dataset_config.dataset_split})..."
        )
        print(f"Target tokens: {dataset_config.target_tokens:,}")
        print("\nDownloading and tokenizing documents...")

    # Load dataset in streaming mode
    dataset = load_dataset(
        dataset_config.dataset_name,
        dataset_config.dataset_config,
        split=dataset_config.dataset_split,
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Accumulate token IDs
    all_token_ids = []
    total_tokens = 0
    total_chars = 0
    num_documents = 0

    for example in dataset:
        text = example["text"]  # type: ignore[index]

        # Tokenize to get token IDs
        encoded = tokenizer.encode(text)
        token_ids = encoded.ids
        num_tokens = len(token_ids)

        # Add token IDs
        all_token_ids.extend(token_ids)
        total_tokens += num_tokens
        total_chars += len(text)
        num_documents += 1

        # Progress update
        if verbose and num_documents % 100 == 0:
            print(
                f"  Documents: {num_documents:,} | Tokens: {total_tokens:,} | "
                f"Progress: {total_tokens / dataset_config.target_tokens * 100:.1f}%"
            )

        # Stop if we've reached target
        if total_tokens >= dataset_config.target_tokens:
            if verbose:
                print("\n✓ Target reached!")
            break

    # Convert to torch tensor
    token_tensor = torch.tensor(all_token_ids, dtype=torch.long)

    # Split into train/val/test
    n_train = int(len(token_tensor) * dataset_config.train_ratio)
    n_val = int(len(token_tensor) * dataset_config.val_ratio)

    train_data = token_tensor[:n_train]
    val_data = token_tensor[n_train : n_train + n_val]
    test_data = token_tensor[n_train + n_val :]

    # Statistics
    if verbose:
        print("\n" + "=" * 80)
        print("TOKENIZED DATASET LOADED")
        print("=" * 80)
        print(f"Documents loaded: {num_documents:,}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total characters: {total_chars:,}")
        print(f"Vocabulary size: {vocab_size:,}")
        print(f"Tokens per character: {total_tokens / total_chars:.4f}")
        print("\nData splits:")
        print(
            f"  Train: {len(train_data):,} tokens ({dataset_config.train_ratio * 100:.1f}%)"
        )
        print(
            f"  Val:   {len(val_data):,} tokens ({dataset_config.val_ratio * 100:.1f}%)"
        )
        print(
            f"  Test:  {len(test_data):,} tokens ({dataset_config.test_ratio * 100:.1f}%)"
        )
        print("=" * 80)

    return train_data, val_data, test_data


def save_tokenized_dataset(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    test_data: torch.Tensor,
    save_path: str | Path,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save tokenized dataset to disk for faster loading.

    Args:
        train_data: Training data tensor
        val_data: Validation data tensor
        test_data: Test data tensor
        save_path: Path to save the tokenized dataset
        metadata: Optional metadata to save with the dataset
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving tokenized dataset to {save_path}...")

    # Prepare data dictionary
    data_dict = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "metadata": metadata or {},
    }

    # Save as PyTorch file
    torch.save(data_dict, save_path)

    print(f"✓ Dataset saved: {save_path}")
    print(f"  Train: {len(train_data):,} tokens")
    print(f"  Val:   {len(val_data):,} tokens")
    print(f"  Test:  {len(test_data):,} tokens")


def load_saved_tokenized_dataset(
    load_path: str | Path,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load tokenized dataset from disk.

    Args:
        load_path: Path to the saved tokenized dataset
        verbose: Print loading information

    Returns:
        Tuple of (train_tensor, val_tensor, test_tensor)
    """
    load_path = Path(load_path)

    if verbose:
        print(f"Loading tokenized dataset from {load_path}...")

    data_dict = torch.load(load_path)

    train_data = data_dict["train"]
    val_data = data_dict["val"]
    test_data = data_dict["test"]
    metadata = data_dict.get("metadata", {})

    if verbose:
        print("✓ Dataset loaded!")
        print(f"  Train: {len(train_data):,} tokens")
        print(f"  Val:   {len(val_data):,} tokens")
        print(f"  Test:  {len(test_data):,} tokens")
        if metadata:
            print(f"  Metadata: {metadata}")

    return train_data, val_data, test_data


def get_vocab_size(tokenizer_path: str) -> int:
    """
    Get vocabulary size from a trained tokenizer.

    Args:
        tokenizer_path: Path to trained BPE tokenizer

    Returns:
        Vocabulary size as integer
    """
    tokenizer = load_tokenizer(tokenizer_path)
    return tokenizer.get_vocab_size()


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML config file (transformer.yaml or lstm.yaml)",
)
@click.option(
    "--save-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save the tokenized dataset (default: uses checkpoint dir from config)",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom cache directory for downloads",
)
def main(
    config: Path,
    save_path: Optional[Path],
    cache_dir: Optional[Path],
):
    """Load, tokenize, and save dataset for model training.

    Accepts either transformer or LSTM config files and uses the dataset
    configuration to load and tokenize the data.
    """
    # Try loading as transformer config first, then LSTM config
    experiment_config: Union[TransformerExperimentConfig, LSTMExperimentConfig]
    try:
        experiment_config = TransformerExperimentConfig.from_yaml(config)
        click.echo(f"Loaded Transformer config from {config}")
    except Exception:
        try:
            experiment_config = LSTMExperimentConfig.from_yaml(config)
            click.echo(f"Loaded LSTM config from {config}")
        except Exception as e:
            click.echo(f"Error: Could not load config from {config}: {e}", err=True)
            raise click.Abort()

    # Extract dataset config
    dataset_config = experiment_config.dataset

    # Determine save path
    if save_path is None:
        checkpoint_dir = Path(experiment_config.experiment.checkpoint_dir)
        save_path = checkpoint_dir / "tokenized_dataset.pt"

    click.echo("\nDataset configuration:")
    click.echo(
        f"  Dataset: {dataset_config.dataset_name}/{dataset_config.dataset_config}"
    )
    click.echo(f"  Target tokens: {dataset_config.target_tokens:,}")
    click.echo(f"  Tokenizer path: {dataset_config.tokenizer_path}")
    click.echo(f"  Save path: {save_path}")

    # Load and tokenize dataset
    train_data, val_data, test_data = load_tokenized_dataset(
        dataset_config=dataset_config,
        cache_dir=str(cache_dir) if cache_dir else None,
        verbose=True,
    )

    # Save tokenized dataset
    metadata = {
        "config_path": str(config),
        "target_tokens": dataset_config.target_tokens,
        "actual_tokens": len(train_data) + len(val_data) + len(test_data),
        "tokenizer_path": dataset_config.tokenizer_path,
        "dataset_name": dataset_config.dataset_name,
        "dataset_config": dataset_config.dataset_config,
    }

    save_tokenized_dataset(
        train_data,
        val_data,
        test_data,
        save_path=save_path,
        metadata=metadata,
    )

    click.echo("\n✓ Done! Tokenized dataset ready for training.")
    click.echo(f"  Total: {len(train_data) + len(val_data) + len(test_data):,} tokens")


if __name__ == "__main__":
    main()

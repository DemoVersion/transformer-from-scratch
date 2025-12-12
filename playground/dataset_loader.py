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

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
from datasets import load_dataset

from playground.tokenizer import load_tokenizer


def load_tokenized_dataset(
    target_tokens: int = 10_000_000,
    tokenizer_path: str = "playground/bpe-tokenizer",
    split: str = "train",
    cache_dir: Optional[str] = None,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and tokenize C4 dataset, returning train/val/test splits as token ID tensors.

    Args:
        target_tokens: Target number of tokens to load (default: 10M)
        tokenizer_path: Path to trained BPE tokenizer
        split: Dataset split to use ('train' or 'validation')
        cache_dir: Custom cache directory for downloads
        train_ratio: Ratio of data for training (default: 0.9)
        val_ratio: Ratio of data for validation (default: 0.05)
        test_ratio: Ratio of data for testing (default: 0.05)
        verbose: Print progress information

    Returns:
        Tuple of (train_tensor, val_tensor, test_tensor) containing token IDs
    """
    if verbose:
        print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    if verbose:
        print(f"Tokenizer loaded: vocab_size={vocab_size:,}")
        print(f"Streaming C4 realnewslike dataset (split: {split})...")
        print(f"Target tokens: {target_tokens:,}")
        print("\nDownloading and tokenizing documents...")

    # Load dataset in streaming mode
    dataset = load_dataset(
        "allenai/c4",
        "realnewslike",
        split=split,
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
        text = example["text"]

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
                f"Progress: {total_tokens / target_tokens * 100:.1f}%"
            )

        # Stop if we've reached target
        if total_tokens >= target_tokens:
            if verbose:
                print("\n✓ Target reached!")
            break

    # Convert to torch tensor
    token_tensor = torch.tensor(all_token_ids, dtype=torch.long)

    # Split into train/val/test
    n_train = int(len(token_tensor) * train_ratio)
    n_val = int(len(token_tensor) * val_ratio)

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
        print(f"  Train: {len(train_data):,} tokens ({train_ratio * 100:.1f}%)")
        print(f"  Val:   {len(val_data):,} tokens ({val_ratio * 100:.1f}%)")
        print(f"  Test:  {len(test_data):,} tokens ({test_ratio * 100:.1f}%)")
        print("=" * 80)

    return train_data, val_data, test_data


def save_tokenized_dataset(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    test_data: torch.Tensor,
    save_path: str,
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
    load_path: str,
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


def main():
    parser = argparse.ArgumentParser(
        description="Load, tokenize, and save dataset for transformer training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=10_000_000,
        help="Target number of tokens to load (default: 10M)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="playground/bpe-tokenizer",
        help="Path to BPE tokenizer directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="playground/data/c4_tokenized.pt",
        help="Path to save the tokenized dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom cache directory for downloads",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training (default: 0.9)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Ratio of data for validation (default: 0.05)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Ratio of data for testing (default: 0.05)",
    )

    args = parser.parse_args()

    # Load and tokenize dataset
    train_data, val_data, test_data = load_tokenized_dataset(
        target_tokens=args.target_tokens,
        tokenizer_path=args.tokenizer,
        split=args.split,
        cache_dir=args.cache_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        verbose=True,
    )

    # Save tokenized dataset
    metadata = {
        "target_tokens": args.target_tokens,
        "actual_tokens": len(train_data) + len(val_data) + len(test_data),
        "tokenizer_path": args.tokenizer,
        "split": args.split,
    }

    save_tokenized_dataset(
        train_data,
        val_data,
        test_data,
        save_path=args.save_path,
        metadata=metadata,
    )

    print("\n✓ Done! Tokenized dataset ready for training.")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data):,} tokens")


if __name__ == "__main__":
    main()

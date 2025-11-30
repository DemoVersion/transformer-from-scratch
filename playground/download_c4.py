"""
Download C4/realnewslike dataset using Hugging Face datasets library.

The C4 (Colossal Clean Crawled Corpus) realnewslike dataset is a filtered
subset of Common Crawl data that resembles real news articles. The realnewslike
variant is approximately 15GB in size and is intended for pretraining language
models and word representations.

Dataset: allenai/c4 (realnewslike configuration)
License: ODC-BY (also bound by Common Crawl terms of use)
Splits: train, validation
Fields: text, url, timestamp

Usage:
    # Install datasets library first if not already installed:
    # uv pip install datasets

    # Download and explore the dataset:
    uv run python playground/download_c4.py

    # Download specific split:
    uv run python playground/download_c4.py --split train

    # Download and save to disk:
    uv run python playground/download_c4.py --save_dir ./playground/corpus/c4

    # Load only a subset (streaming):
    uv run python playground/download_c4.py --streaming --max_examples 1000
"""

import argparse
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found.")
    print("Install it with: uv pip install datasets")
    exit(1)


def download_c4_realnewslike(
    split: str = "train",
    save_dir: Optional[str] = None,
    streaming: bool = False,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> None:
    """
    Download C4/realnewslike dataset.

    Args:
        split: Dataset split to download ('train' or 'validation')
        save_dir: If provided, save the dataset to this directory
        streaming: If True, use streaming mode (doesn't download full dataset)
        max_examples: If provided, only load this many examples (useful for testing)
        cache_dir: Custom cache directory for downloaded files
    """
    print(f"Loading C4 realnewslike dataset (split: {split})...")
    print(f"Streaming mode: {streaming}")

    if streaming:
        print("\nNote: Streaming mode - dataset will not be fully downloaded")
        print("Data is loaded on-the-fly as you iterate through it")

    # Load the dataset
    dataset = load_dataset(
        "allenai/c4",
        "realnewslike",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Print dataset info
    if not streaming:
        print(f"\nDataset loaded successfully!")
        print(f"Number of examples: {len(dataset)}")
        print(f"Features: {dataset.features}")
    else:
        print(f"\nDataset loaded in streaming mode")
        print("Features: text, timestamp, url")

    # Show sample examples
    print("\n" + "=" * 80)
    print("Sample examples:")
    print("=" * 80)

    num_samples = min(3, max_examples) if max_examples else 3
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        text = example["text"]
        url = example.get("url", "N/A")
        timestamp = example.get("timestamp", "N/A")

        print(f"\nExample {i + 1}:")
        print(f"URL: {url}")
        print(f"Timestamp: {timestamp}")
        print(f"Text (first 500 chars): {text[:500]}...")
        print("-" * 80)

    # Optionally limit examples
    if max_examples and not streaming:
        print(f"\nSelecting first {max_examples} examples...")
        dataset = dataset.select(range(min(max_examples, len(dataset))))
        print(f"Selected {len(dataset)} examples")

    # Save to disk if requested
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if streaming:
            print(
                "\nWarning: Cannot save streaming dataset directly to disk."
            )
            print(
                "To save, run without --streaming flag or iterate and save manually."
            )
        else:
            print(f"\nSaving dataset to {save_path}...")
            dataset.save_to_disk(str(save_path))
            print(f"Dataset saved successfully!")

            # Also save as text file for easy inspection
            text_file = save_path / f"c4_realnewslike_{split}.txt"
            print(f"\nSaving text content to {text_file}...")
            with open(text_file, "w", encoding="utf-8") as f:
                for i, example in enumerate(dataset):
                    if max_examples and i >= max_examples:
                        break
                    f.write(f"{'=' * 80}\n")
                    f.write(f"Document {i + 1}\n")
                    f.write(f"URL: {example.get('url', 'N/A')}\n")
                    f.write(f"Timestamp: {example.get('timestamp', 'N/A')}\n")
                    f.write(f"{'=' * 80}\n")
                    f.write(example["text"])
                    f.write("\n\n")

            print(f"Text file saved: {text_file}")

    # Dataset statistics
    if not streaming and dataset:
        print("\n" + "=" * 80)
        print("Dataset Statistics:")
        print("=" * 80)

        # Calculate text length statistics
        text_lengths = [len(example["text"]) for example in dataset.select(range(min(1000, len(dataset))))]
        avg_length = sum(text_lengths) / len(text_lengths)
        min_length = min(text_lengths)
        max_length = max(text_lengths)

        print(f"Text length statistics (based on sample of {len(text_lengths)} examples):")
        print(f"  Average: {avg_length:.2f} characters")
        print(f"  Min: {min_length} characters")
        print(f"  Max: {max_length} characters")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download C4/realnewslike dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to download (default: train)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the dataset (default: don't save)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (don't download full dataset)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to load (default: all)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom cache directory for downloads",
    )

    args = parser.parse_args()

    download_c4_realnewslike(
        split=args.split,
        save_dir=args.save_dir,
        streaming=args.streaming,
        max_examples=args.max_examples,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()

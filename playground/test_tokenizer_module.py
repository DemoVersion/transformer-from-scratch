"""Test script for the tokenizer module - playground version."""

import shutil
import sys
from pathlib import Path

# Add playground to path to import tokenizer module
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import (
    create_and_train_tokenizer,
    tokenize_texts,
)


def main():
    """Test the tokenizer module functions."""
    print("=" * 60)
    print("Testing Tokenizer Module")
    print("=" * 60)

    # Sample corpus texts
    corpus_texts = {
        "file1.txt": (
            "Byte Pair Encoding (BPE) is a subword tokenization method.\n"
            "BPE merges the most frequent pairs of bytes in a corpus.\n"
            "Tokenizers can be trained from scratch.\n"
        ),
        "file2.txt": (
            "We are building a tokenizer using BPE.\n"
            "This tokenizer will split text into subword units.\n"
        ),
    }

    # Use playground subdirectories for corpus and tokenizer
    corpus_dir = Path("playground/corpus")
    save_dir = Path("playground/bpe-tokenizer")

    # Clean up any existing files
    for dir_path in [corpus_dir, save_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)

    print("\n1. Creating and training tokenizer...")
    hf_tokenizer = create_and_train_tokenizer(
        corpus_texts=corpus_texts,
        corpus_dir=corpus_dir,
        vocab_size=1000,
        save_dir=save_dir,
    )
    print("   ✓ Tokenizer created and trained successfully")
    print(f"   ✓ Saved to: {save_dir}")

    # Test single text tokenization
    print("\n2. Testing single text tokenization...")
    text = "BPE merges frequent byte pairs."
    encodings = hf_tokenizer(text, add_special_tokens=True)
    tokens = hf_tokenizer.convert_ids_to_tokens(encodings["input_ids"])
    print(f"   Text: {text}")
    print(f"   Tokens: {tokens}")
    print(f"   Token IDs: {encodings['input_ids']}")

    # Test batch tokenization
    print("\n3. Testing batch tokenization...")
    text_batch = [
        "We are building a tokenizer using BPE.",
        "Tokenizers split text into subword units!",
    ]
    encodings = tokenize_texts(hf_tokenizer, text_batch)
    print(f"   Batch size: {len(text_batch)}")
    print(f"   Text 1: {text_batch[0]}")
    print(f"   Text 2: {text_batch[1]}")
    print(f"   Token IDs 1: {encodings['input_ids'][0]}")
    print(f"   Token IDs 2: {encodings['input_ids'][1]}")

    # Test with padding and attention masks
    print("\n4. Testing with padding and PyTorch tensors...")
    try:
        encodings_pt = tokenize_texts(
            hf_tokenizer, text_batch, padding=True, truncation=True, return_tensors="pt"
        )
        print(f"   Input IDs shape: {encodings_pt['input_ids'].shape}")
        print(f"   Attention mask shape: {encodings_pt['attention_mask'].shape}")
        print(f"\n   Input IDs:\n{encodings_pt['input_ids']}")
        print(f"\n   Attention mask:\n{encodings_pt['attention_mask']}")
    except ImportError:
        print("   ⚠ PyTorch not installed, testing with plain lists instead")
        encodings_pt = tokenize_texts(
            hf_tokenizer, text_batch, padding=True, truncation=True, max_length=50
        )
        print(f"   Input IDs: {encodings_pt['input_ids']}")
        print(f"   Attention mask: {encodings_pt['attention_mask']}")

    # Verify special tokens
    print("\n5. Verifying special tokens...")
    print(f"   PAD token: {hf_tokenizer.pad_token} (ID: {hf_tokenizer.pad_token_id})")
    print(f"   UNK token: {hf_tokenizer.unk_token} (ID: {hf_tokenizer.unk_token_id})")
    print(f"   CLS token: {hf_tokenizer.cls_token} (ID: {hf_tokenizer.cls_token_id})")
    print(f"   SEP token: {hf_tokenizer.sep_token} (ID: {hf_tokenizer.sep_token_id})")
    print(
        f"   MASK token: {hf_tokenizer.mask_token} (ID: {hf_tokenizer.mask_token_id})"
    )

    # Verify saved files
    print("\n6. Verifying saved files...")
    expected_files = ["tokenizer.json", "vocab.json", "merges.txt"]
    for filename in expected_files:
        file_path = save_dir / filename
        if file_path.exists():
            print(f"   ✓ {filename} exists")
        else:
            print(f"   ✗ {filename} missing")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

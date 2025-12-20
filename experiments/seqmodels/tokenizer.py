"""BPE Tokenizer utilities for training and using byte-pair encoding tokenizers."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


def create_corpus(corpus_dir: Union[str, Path], texts: Dict[str, str]) -> List[Path]:
    """
    Create corpus text files from a dictionary of texts.

    Args:
        corpus_dir: Directory where corpus files will be saved
        texts: Dictionary mapping filename to text content

    Returns:
        List of created file paths

    Example:
        >>> corpus_files = create_corpus("corpus", {
        ...     "file1.txt": "Hello world",
        ...     "file2.txt": "Goodbye world"
        ... })
    """
    corpus_path = Path(corpus_dir)
    corpus_path.mkdir(exist_ok=True)

    created_files = []
    for filename, content in texts.items():
        file_path = corpus_path / filename
        file_path.write_text(content)
        created_files.append(file_path)

    return created_files


def build_bpe_tokenizer(
    vocab_size: int = 1000,
    min_frequency: int = 1,
    special_tokens: Optional[List[str]] = None,
    unk_token: str = "[UNK]",
) -> tuple[Tokenizer, BpeTrainer]:
    """
    Build a BPE tokenizer with specified configuration.

    Args:
        vocab_size: Maximum vocabulary size
        min_frequency: Minimum frequency for a token to be included
        special_tokens: List of special tokens (defaults to standard BERT tokens)
        unk_token: Token to use for unknown words

    Returns:
        Tuple of (tokenizer, trainer)

    Example:
        >>> tokenizer, trainer = build_bpe_tokenizer(vocab_size=5000)
    """
    if special_tokens is None:
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    return tokenizer, trainer


def train_tokenizer(
    tokenizer: Tokenizer,
    trainer: BpeTrainer,
    corpus_files: Union[List[str], List[Path]],
) -> Tokenizer:
    """
    Train a tokenizer on corpus files.

    Args:
        tokenizer: Tokenizer instance to train
        trainer: BpeTrainer instance with training configuration
        corpus_files: List of file paths containing training text

    Returns:
        Trained tokenizer

    Example:
        >>> tokenizer, trainer = build_bpe_tokenizer()
        >>> trained = train_tokenizer(tokenizer, trainer, ["corpus/file1.txt"])
    """
    files = [str(p) for p in corpus_files]
    tokenizer.train(files=files, trainer=trainer)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, save_dir: Union[str, Path]) -> Path:
    """
    Save a trained tokenizer to disk.

    Args:
        tokenizer: Trained tokenizer to save
        save_dir: Directory where tokenizer files will be saved

    Returns:
        Path to the save directory

    Example:
        >>> save_tokenizer(tokenizer, "my-tokenizer")
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Save vocab.json and merges.txt
    tokenizer.model.save(str(save_path))
    # Save complete tokenizer config
    tokenizer.save(str(save_path / "tokenizer.json"))

    return save_path


def load_tokenizer(tokenizer_path: Union[str, Path]) -> Tokenizer:
    """
    Load a trained tokenizer from disk.

    Args:
        tokenizer_path: Path to tokenizer.json file or directory containing it

    Returns:
        Loaded tokenizer

    Example:
        >>> tokenizer = load_tokenizer("my-tokenizer/tokenizer.json")
    """
    path = Path(tokenizer_path)
    if path.is_dir():
        path = path / "tokenizer.json"

    return Tokenizer.from_file(str(path))


def wrap_hf_tokenizer(
    tokenizer: Tokenizer,
    unk_token: str = "[UNK]",
    pad_token: str = "[PAD]",
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    mask_token: str = "[MASK]",
) -> PreTrainedTokenizerFast:
    """
    Wrap a tokenizer in a HuggingFace PreTrainedTokenizerFast.

    Args:
        tokenizer: Base tokenizer to wrap
        unk_token: Unknown token
        pad_token: Padding token
        cls_token: Classification token
        sep_token: Separator token
        mask_token: Mask token

    Returns:
        HuggingFace-compatible tokenizer

    Example:
        >>> hf_tokenizer = wrap_hf_tokenizer(tokenizer)
    """
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        pad_token=pad_token,
        cls_token=cls_token,
        sep_token=sep_token,
        mask_token=mask_token,
    )


def tokenize_texts(
    hf_tokenizer: PreTrainedTokenizerFast,
    texts: Union[str, List[str]],
    padding: bool = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
    return_tensors: Optional[str] = None,
    add_special_tokens: bool = True,
):
    """
    Tokenize text(s) using a HuggingFace tokenizer.

    Args:
        hf_tokenizer: HuggingFace tokenizer to use
        texts: Single text string or list of texts
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        max_length: Maximum sequence length
        return_tensors: Format to return ("pt" for PyTorch, "tf" for TensorFlow)
        add_special_tokens: Whether to add special tokens like [CLS], [SEP]

    Returns:
        Tokenized encodings with input_ids, attention_mask, etc.

    Example:
        >>> encodings = tokenize_texts(
        ...     hf_tokenizer,
        ...     ["Hello world", "Goodbye world"],
        ...     padding=True,
        ...     return_tensors="pt"
        ... )
    """
    return hf_tokenizer(
        texts,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
        add_special_tokens=add_special_tokens,
    )


def create_and_train_tokenizer(
    corpus_texts: Dict[str, str],
    corpus_dir: Union[str, Path] = "corpus",
    vocab_size: int = 1000,
    min_frequency: int = 1,
    special_tokens: Optional[List[str]] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> PreTrainedTokenizerFast:
    """
    End-to-end function to create corpus, train tokenizer, and return HF wrapper.

    Args:
        corpus_texts: Dictionary mapping filename to text content
        corpus_dir: Directory for corpus files
        vocab_size: Maximum vocabulary size
        min_frequency: Minimum token frequency
        special_tokens: List of special tokens
        save_dir: Optional directory to save trained tokenizer

    Returns:
        Trained HuggingFace tokenizer

    Example:
        >>> tokenizer = create_and_train_tokenizer({
        ...     "file1.txt": "This is training text.",
        ...     "file2.txt": "More training data here."
        ... })
    """
    # Create corpus files
    corpus_files = create_corpus(corpus_dir, corpus_texts)

    # Build and train tokenizer
    tokenizer, trainer = build_bpe_tokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    train_tokenizer(tokenizer, trainer, corpus_files)

    # Optionally save
    if save_dir:
        save_tokenizer(tokenizer, save_dir)

    # Wrap in HF tokenizer
    return wrap_hf_tokenizer(tokenizer)

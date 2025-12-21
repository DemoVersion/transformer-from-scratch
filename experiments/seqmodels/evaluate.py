"""Evaluate perplexity of Transformer and LSTM models on the same test samples.

This script loads both a trained transformer and LSTM model from their checkpoint
directories and evaluates their performance on the same test samples by calculating
perplexity.

Usage:
    # Evaluate two models with default checkpoints
    python -m experiments.seqmodels.evaluate \
        --transformer transformer_checkpoints/run_20231201_123456 \
        --lstm lstm_checkpoints/run_20231201_123456

    # Evaluate with specific checkpoints
    python -m experiments.seqmodels.evaluate \
        --transformer transformer_checkpoints/run_20231201_123456 \
        --lstm lstm_checkpoints/run_20231201_123456 \
        --transformer-checkpoint checkpoint_batch_1000.pt \
        --lstm-checkpoint checkpoint_batch_1000.pt

    # Use custom batch size and number of batches
    python -m experiments.seqmodels.evaluate \
        --transformer transformer_checkpoints/run_20231201_123456 \
        --lstm lstm_checkpoints/run_20231201_123456 \
        --batch-size 64 \
        --num-batches 100
"""

import click
import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.seqmodels.dataset_loader import load_tokenized_dataset
from experiments.seqmodels.utils import load_checkpoint, sample_batch


def evaluate_single_model(
    model,
    model_name: str,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    context_length: int,
) -> dict:
    """Evaluate a single model on pre-sampled batches.

    Args:
        model: The model to evaluate
        model_name: Name of the model (for display)
        batches: List of (source, target) tuples
        context_length: Context length for the model

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for source, target in tqdm(batches, desc=f"Evaluating {model_name}"):
            # Move to device
            source = source.to(device)
            target = target.to(device)

            # Forward pass
            output = model(source)

            # Compute loss
            # output shape: (batch, seq, vocab)
            # target shape: (batch, seq)
            loss = F.nll_loss(
                output.transpose(2, 1),  # (batch, vocab, seq)
                target,
                reduction="sum",  # Sum to get total loss
            )

            total_loss += loss.item()
            total_tokens += target.numel()

    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
    }


def evaluate_models(
    transformer_dir: str,
    lstm_dir: str,
    transformer_checkpoint: str = "final_model.pt",
    lstm_checkpoint: str = "final_model.pt",
    batch_size: int = 32,
    num_batches: int | None = None,
):
    """Load and evaluate two models on the same test samples.

    Args:
        transformer_dir: Path to transformer checkpoint directory
        lstm_dir: Path to LSTM checkpoint directory
        transformer_checkpoint: Name of transformer checkpoint file
        lstm_checkpoint: Name of LSTM checkpoint file
        batch_size: Batch size for evaluation
        num_batches: Number of batches to evaluate (None = full test set)
    """
    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    # Load transformer model
    print("\n[1/2] Loading Transformer model...")
    transformer_model, transformer_config, transformer_tokenizer, transformer_vocab = (
        load_checkpoint(
            checkpoint_dir=transformer_dir,
            checkpoint_name=transformer_checkpoint,
        )
    )

    # Load LSTM model
    print("\n[2/2] Loading LSTM model...")
    lstm_model, lstm_config, lstm_tokenizer, lstm_vocab = load_checkpoint(
        checkpoint_dir=lstm_dir,
        checkpoint_name=lstm_checkpoint,
    )

    # Verify compatibility
    print("\n" + "=" * 80)
    print("VERIFYING MODEL COMPATIBILITY")
    print("=" * 80)

    if transformer_vocab != lstm_vocab:
        print(
            f"WARNING: Vocabulary sizes differ! "
            f"Transformer: {transformer_vocab:,}, LSTM: {lstm_vocab:,}"
        )
        print("Models may not be directly comparable.")
    else:
        print(f"Vocabulary size: {transformer_vocab:,} (compatible)")

    # Check context lengths
    transformer_context = transformer_config.model.context
    lstm_context = lstm_config.model.context

    if transformer_context != lstm_context:
        print(
            f"WARNING: Context lengths differ! "
            f"Transformer: {transformer_context}, LSTM: {lstm_context}"
        )
        print(f"Using minimum context length: {min(transformer_context, lstm_context)}")
        context_length = min(transformer_context, lstm_context)
    else:
        print(f"Context length: {transformer_context} (compatible)")
        context_length = transformer_context

    # Load test dataset (using transformer config as reference)
    print("\n" + "=" * 80)
    print("LOADING TEST DATASET")
    print("=" * 80)

    _, _, data_test = load_tokenized_dataset(
        dataset_config=transformer_config.dataset,
        verbose=False,
    )

    print(f"Test data size: {data_test.size(0):,} tokens")

    # Calculate number of batches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_test = data_test.to(device)

    max_batches = (data_test.size(0) - context_length - 1) // batch_size

    if num_batches is None:
        num_batches = max_batches
    else:
        num_batches = min(num_batches, max_batches)

    print(f"\nSampling {num_batches:,} batches (batch_size={batch_size})")
    print(f"Context length: {context_length}")
    print(f"Total tokens to evaluate: ~{num_batches * batch_size * context_length:,}")

    # Pre-sample all batches (same samples for both models)
    print("\n" + "=" * 80)
    print("SAMPLING TEST BATCHES")
    print("=" * 80)
    print("Creating shared test samples for fair comparison...")

    batches = []
    for _ in tqdm(range(num_batches), desc="Sampling batches"):
        source, target = sample_batch(
            data_test,
            length=context_length,
            batch_size=batch_size,
        )
        batches.append((source, target))

    # Evaluate both models on the same batches
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    transformer_results = evaluate_single_model(
        transformer_model,
        "Transformer",
        batches,
        context_length,
    )

    lstm_results = evaluate_single_model(
        lstm_model,
        "LSTM",
        batches,
        context_length,
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("TRANSFORMER MODEL")
    print("-" * 80)
    print(f"Average loss (nats):     {transformer_results['avg_loss']:.4f}")
    print(
        f"Average loss (bits):     {transformer_results['avg_loss'] * 1.4427:.4f}"
    )  # log2(e)
    print(f"Perplexity:              {transformer_results['perplexity']:.4f}")

    print("\n" + "-" * 80)
    print("LSTM MODEL")
    print("-" * 80)
    print(f"Average loss (nats):     {lstm_results['avg_loss']:.4f}")
    print(f"Average loss (bits):     {lstm_results['avg_loss'] * 1.4427:.4f}")
    print(f"Perplexity:              {lstm_results['perplexity']:.4f}")

    print("\n" + "-" * 80)
    print(f"Total tokens evaluated:  {transformer_results['total_tokens']:,}")
    print("=" * 80)

    return {
        "transformer": transformer_results,
        "lstm": lstm_results,
    }


@click.command()
@click.option(
    "--transformer",
    type=str,
    required=True,
    help="Path to transformer checkpoint directory",
)
@click.option(
    "--lstm",
    type=str,
    required=True,
    help="Path to LSTM checkpoint directory",
)
@click.option(
    "--transformer-checkpoint",
    type=str,
    default="final_model.pt",
    help="Name of transformer checkpoint file to load",
)
@click.option(
    "--lstm-checkpoint",
    type=str,
    default="final_model.pt",
    help="Name of LSTM checkpoint file to load",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for evaluation",
)
@click.option(
    "--num-batches",
    type=int,
    default=None,
    help="Number of batches to evaluate (default: full test set)",
)
def main(
    transformer: str,
    lstm: str,
    transformer_checkpoint: str,
    lstm_checkpoint: str,
    batch_size: int,
    num_batches: int | None,
):
    """Evaluate Transformer and LSTM models on the same test samples.

    Loads both models and evaluates them on identical test samples.
    """
    evaluate_models(
        transformer_dir=transformer,
        lstm_dir=lstm,
        transformer_checkpoint=transformer_checkpoint,
        lstm_checkpoint=lstm_checkpoint,
        batch_size=batch_size,
        num_batches=num_batches,
    )


if __name__ == "__main__":
    main()

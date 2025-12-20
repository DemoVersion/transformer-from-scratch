"""Evaluate a trained model on the test set and calculate perplexity.

This script loads a trained transformer or LSTM model from a checkpoint directory
and evaluates its performance on the test set by calculating perplexity.

Usage:
    # Evaluate with default checkpoint
    python -m experiments.seqmodels.evaluate transformer_checkpoints/run_20231201_123456

    # Evaluate with specific checkpoint
    python -m experiments.seqmodels.evaluate \\
        transformer_checkpoints/run_20231201_123456 \\
        --checkpoint checkpoint_batch_1000.pt

    # Evaluate with custom batch size
    python -m experiments.seqmodels.evaluate \\
        transformer_checkpoints/run_20231201_123456 \\
        --batch-size 64
"""

import click
import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.seqmodels.dataset_loader import load_tokenized_dataset
from experiments.seqmodels.utils import load_checkpoint, sample_batch


def evaluate_model(
    checkpoint_dir: str,
    checkpoint_name: str = "final_model.pt",
    batch_size: int = 32,
    num_batches: int | None = None,
):
    """Load a checkpoint and evaluate on the test set.

    Args:
        checkpoint_dir: Path to checkpoint directory
        checkpoint_name: Name of checkpoint file to load
        batch_size: Batch size for evaluation
        num_batches: Number of batches to evaluate (None = full test set)
    """
    print(f"Loading checkpoint from: {checkpoint_dir}")
    print(f"Checkpoint: {checkpoint_name}\n")

    # Load checkpoint
    model, config, tokenizer, vocab_size = load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
    )

    print("\nLoading test dataset...")

    # Load test dataset using config from checkpoint
    _, _, data_test = load_tokenized_dataset(
        dataset_config=config.dataset,
        verbose=False,
    )

    print(f"Test data size: {data_test.size(0):,} tokens")

    # Move data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_test = data_test.to(device)

    # Calculate number of batches if not specified
    context_length = config.model.context
    max_batches = (data_test.size(0) - context_length - 1) // batch_size

    if num_batches is None:
        num_batches = max_batches
    else:
        num_batches = min(num_batches, max_batches)

    print(f"\nEvaluating on {num_batches:,} batches (batch_size={batch_size})")
    print(f"Context length: {context_length}")
    print(f"Total tokens evaluated: ~{num_batches * batch_size * context_length:,}")
    print("=" * 80)

    # Evaluation loop
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Evaluating"):
            # Sample batch from test data
            source, target = sample_batch(
                data_test,
                length=context_length,
                batch_size=batch_size,
            )

            if torch.cuda.is_available():
                source, target = source.cuda(), target.cuda()

            # Forward pass
            output = model(source)

            # Compute loss (same as training)
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

    # Print results
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Average loss (nats): {avg_loss:.4f}")
    print(f"Average loss (bits): {avg_loss * 1.4427:.4f}")  # log2(e) = 1.4427
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Total tokens evaluated: {total_tokens:,}")
    print("=" * 80)

    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
    }


@click.command()
@click.argument(
    "checkpoint_dir",
    type=str,
)
@click.option(
    "--checkpoint",
    type=str,
    default="final_model.pt",
    help="Name of checkpoint file to load",
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
    checkpoint_dir: str,
    checkpoint: str,
    batch_size: int,
    num_batches: int | None,
):
    """Evaluate a saved checkpoint on the test set.

    CHECKPOINT_DIR: Path to checkpoint directory (e.g., transformer_checkpoints/run_20231201_123456)
    """
    evaluate_model(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint,
        batch_size=batch_size,
        num_batches=num_batches,
    )


if __name__ == "__main__":
    main()

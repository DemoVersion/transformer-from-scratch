"""Generate text from a saved model checkpoint.

This script loads a trained transformer or LSTM model from a checkpoint directory
and generates text based on a given prompt.

Usage:
    # Generate with default settings
    python -m experiments.seqmodels.generate_from_checkpoint transformer_checkpoints/run_20231201_123456

    # Custom prompt and parameters
    python -m experiments.seqmodels.generate_from_checkpoint \\
        transformer_checkpoints/run_20231201_123456 \\
        --prompt "Once upon a time" \\
        --length 200 \\
        --temperature 0.9
"""

import torch

from experiments.seqmodels.utils import load_checkpoint, sample_sequence


def generate_text_from_checkpoint(
    checkpoint_dir: str,
    prompt: str = "The quick brown fox",
    length: int = 100,
    temperature: float = 0.8,
    checkpoint_name: str = "final_model.pt",
):
    """Load a checkpoint and generate text from a prompt.

    Args:
        checkpoint_dir: Path to checkpoint directory
        prompt: Text prompt to start generation
        length: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        checkpoint_name: Name of checkpoint file to load
    """
    print(f"Loading checkpoint from: {checkpoint_dir}\n")

    # Load checkpoint
    model, config, tokenizer, vocab_size = load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
    )

    print(f"\nPrompt: {prompt}")
    print("=" * 80)

    # Encode prompt
    encoded = tokenizer.encode(prompt)
    seed = torch.tensor(encoded.ids[: config.model.context])

    # Move to GPU if available
    if torch.cuda.is_available():
        seed = seed.cuda()

    # Generate text
    with torch.no_grad():
        sample_sequence(
            model=model,
            seed=seed,
            tokenizer=tokenizer,
            max_context=config.model.context,
            length=length,
            temperature=temperature,
            verbose=True,
        )

    print("=" * 80)
    print("\nGeneration complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate text from a saved checkpoint"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to checkpoint directory (e.g., transformer_checkpoints/run_20231201_123456)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Text prompt to start generation",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=100,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final_model.pt",
        help="Name of checkpoint file to load",
    )

    args = parser.parse_args()

    generate_text_from_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        prompt=args.prompt,
        length=args.length,
        temperature=args.temperature,
        checkpoint_name=args.checkpoint,
    )

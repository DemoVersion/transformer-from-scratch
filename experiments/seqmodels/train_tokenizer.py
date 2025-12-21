"""Standalone script to train a BPE tokenizer once for use by both Transformer and LSTM models.

This ensures both models use the exact same tokenizer vocabulary and merges.

Usage:
    # Train tokenizer using transformer config
    uv run python -m experiments.seqmodels.train_tokenizer --config experiments/seqmodels/configs/transformer.yaml

    # Train tokenizer using LSTM config
    uv run python -m experiments.seqmodels.train_tokenizer --config experiments/seqmodels/configs/lstm.yaml

The tokenizer will be saved to the path specified in the config file (tokenizer_path).
"""

from pathlib import Path
from typing import Optional, Union

import click

from experiments.seqmodels.config_schema import (
    LSTMExperimentConfig,
    TransformerExperimentConfig,
)
from experiments.seqmodels.dataset_loader import train_tokenizer_from_c4


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML config file (transformer.yaml or lstm.yaml)",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom cache directory for downloads",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force retrain even if tokenizer already exists",
)
def main(
    config: Path,
    cache_dir: Optional[Path],
    force: bool,
):
    """Train a BPE tokenizer from C4 dataset for use by both models.

    This script trains a tokenizer once and saves it. Both the Transformer and LSTM
    training scripts can then use this pre-trained tokenizer, ensuring they use
    identical vocabularies and byte-pair merges.
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

    # Check if tokenizer already exists
    tokenizer_path = Path(dataset_config.tokenizer_path)
    tokenizer_file = tokenizer_path / "tokenizer.json"

    if tokenizer_file.exists() and not force:
        click.echo(f"\n⚠️  Tokenizer already exists at {tokenizer_path}")
        click.echo("Use --force to retrain and overwrite.")
        click.echo("\nExisting tokenizer will be used by training scripts.")
        return

    if force and tokenizer_file.exists():
        click.echo(f"⚠️  Force retraining tokenizer (will overwrite {tokenizer_path})")

    # Display configuration
    click.echo("\n" + "=" * 80)
    click.echo("TOKENIZER TRAINING CONFIGURATION")
    click.echo("=" * 80)
    click.echo(
        f"Dataset: {dataset_config.dataset_name}/{dataset_config.dataset_config}"
    )
    click.echo(f"Training documents: {dataset_config.tokenizer_training_docs:,}")
    click.echo(f"Target vocab size: {dataset_config.tokenizer_vocab_size:,}")
    click.echo(f"Save path: {dataset_config.tokenizer_path}")
    click.echo("=" * 80)

    # Train the tokenizer
    train_tokenizer_from_c4(
        dataset_config=dataset_config,
        cache_dir=str(cache_dir) if cache_dir else None,
        verbose=True,
    )

    click.echo(
        "\n✓ Done! Tokenizer is ready for use by both Transformer and LSTM models."
    )
    click.echo(f"  Location: {dataset_config.tokenizer_path}")
    click.echo("\nYou can now run training scripts:")
    click.echo(
        "  uv run python -m experiments.seqmodels.generate_custom --config <config>"
    )
    click.echo(
        "  uv run python -m experiments.seqmodels.generate_lstm --config <config>"
    )


if __name__ == "__main__":
    main()

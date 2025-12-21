"""Text generation experiment with GTransformer with custom dataset support.

Based on experiments/generate.py but simplified with configuration from YAML.
"""

import random
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from experiments.seqmodels.config_schema import TransformerExperimentConfig
from experiments.seqmodels.dataset_loader import (
    get_vocab_size,
    load_tokenized_dataset,
)
from experiments.seqmodels.tokenizer import load_tokenizer
from experiments.seqmodels.utils import (
    create_checkpoint_dir,
    sample_batch,
    sample_sequence,
    save_checkpoint,
    write_test_header,
    write_test_metrics,
)
from former import GTransformer, util
from former.util import tic, toc


def train(config: TransformerExperimentConfig):
    """Main training function."""
    # Set random seed
    if config.experiment.random_seed < 0:
        seed = random.randint(0, 1000000)
        print("random seed: ", seed)
        torch.manual_seed(seed)
    else:
        torch.manual_seed(config.experiment.random_seed)

    # Initialize tensorboard logging
    tbw = SummaryWriter(log_dir=config.experiment.tensorboard_dir)

    # Create checkpoint directory for this run
    checkpoint_dir = create_checkpoint_dir(config.experiment.checkpoint_dir)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Open log file for test outputs
    test_log_path = checkpoint_dir / "test_outputs.txt"
    test_log_file = open(test_log_path, "a")
    test_log_file.write(f"Test outputs log - Started at {checkpoint_dir.name}\n")
    print(f"Test outputs will be logged to: {test_log_path}")

    # Check if tokenizer exists
    tokenizer_path = Path(config.dataset.tokenizer_path)
    tokenizer_file = tokenizer_path / "tokenizer.json"

    if not tokenizer_file.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}\n"
            f"Train it first: uv run python -m experiments.seqmodels.train_tokenizer --config <config>"
        )

    # Load the data using pre-trained tokenizer
    print(f"Using pre-trained tokenizer from: {tokenizer_path}")
    print("Loading tokenized dataset...")
    data_train, data_val, data_test = load_tokenized_dataset(
        dataset_config=config.dataset,
        verbose=True,
    )

    # Load tokenizer for sampling/generation
    tokenizer = load_tokenizer(config.dataset.tokenizer_path)

    vocab_size = get_vocab_size(config.dataset.tokenizer_path)
    print(f"Vocabulary size: {vocab_size:,}")

    data_train, data_test = (
        (torch.cat([data_train, data_val], dim=0), data_test)
        if config.experiment.final
        else (data_train, data_val)
    )

    print(f"Training data size: {data_train.size(0):,} tokens")
    print(f"Test data size: {data_test.size(0):,} tokens")

    # Create the model
    model = GTransformer(
        emb=config.model.embedding_size,
        heads=config.model.num_heads,
        depth=config.model.depth,
        seq_length=config.model.context,
        num_tokens=vocab_size,
        attention_type=config.model.attention_type,
    )
    if torch.cuda.is_available():
        model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer and scheduler
    opt = torch.optim.Adam(lr=config.training.learning_rate, params=model.parameters())

    # Linear learning rate warmup
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda i: min(
            i / (config.training.lr_warmup / config.training.batch_size), 1.0
        ),
    )

    # Training loop
    instances_seen = 0
    for i in tqdm.trange(config.training.num_batches):
        opt.zero_grad()

        source, target = sample_batch(
            data_train,
            length=config.model.context,
            batch_size=config.training.batch_size,
        )
        instances_seen += source.size(0)

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        tic()
        output = model(source)  # forward pass
        t = toc()

        # Compute the loss
        loss = F.nll_loss(output.transpose(2, 1), target, reduction="mean")

        tbw.add_scalar(
            "transformer/train-loss",
            float(loss.item()) * util.LOG2E,
            i * config.training.batch_size,
        )
        tbw.add_scalar("transformer/time-forward", t, instances_seen)

        loss.backward()  # backward pass

        # Clip gradients
        if config.training.gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clipping
            )

        opt.step()  # stochastic gradient descent step
        sch.step()  # update the learning rate

        # Save periodic checkpoint if configured
        if (
            config.experiment.save_every > 0
            and i != 0
            and i % config.experiment.save_every == 0
        ):
            save_checkpoint(
                model=model,
                config=config,
                checkpoint_dir=checkpoint_dir,
                batch_idx=i,
                tokenizer=tokenizer,
                is_final=False,
            )

        # Validate every TEST_EVERY steps
        if i != 0 and (
            i % config.evaluation.test_every == 0
            or i == config.training.num_batches - 1
        ):
            with torch.no_grad():
                # Write test header to log file
                write_test_header(test_log_file, i)

                # Sample and print a random sequence
                seedfr = random.randint(0, data_test.size(0) - config.model.context)
                seed = data_test[seedfr : seedfr + config.model.context].to(torch.long)

                if torch.cuda.is_available():
                    seed = seed.cuda()

                sample_sequence(
                    model,
                    seed=seed,
                    tokenizer=tokenizer,
                    max_context=config.model.context,
                    verbose=True,
                    length=config.evaluation.sample_length,
                    log_file=test_log_file,
                )

                # Compute validation bits per byte
                data_sub = data_test[: config.evaluation.test_subset]

                bits_per_byte = util.compute_compression(
                    model,
                    data_sub,
                    context=config.model.context,
                    batch_size=config.evaluation.test_batchsize,
                    verbose=True,
                )

                print(f"epoch{i}: {bits_per_byte:.4} bits per byte")

                # Write metrics to log file
                write_test_metrics(test_log_file, i, bits_per_byte)

                tbw.add_scalar(
                    "transformer/eval-loss",
                    bits_per_byte,
                    i * config.training.batch_size,
                )

    # Save final checkpoint
    save_checkpoint(
        model=model,
        config=config,
        checkpoint_dir=checkpoint_dir,
        batch_idx=config.training.num_batches - 1,
        tokenizer=tokenizer,
        is_final=True,
    )

    # Close test log file
    test_log_file.close()
    print(f"Test outputs saved to: {test_log_path}")

    print("Training complete!")


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path(__file__).parent / "configs" / "transformer.yaml",
    help="Path to YAML configuration file",
)
def main(config_path: Path):
    """Train a transformer model for text generation."""
    config = TransformerExperimentConfig.from_yaml(config_path)

    print(f"Loading configuration from: {config_path}")
    print("\nModel Configuration:")
    print(f"  Embedding size: {config.model.embedding_size}")
    print(f"  Num heads: {config.model.num_heads}")
    print(f"  Depth: {config.model.depth}")
    print(f"  Context: {config.model.context}")
    print(f"  Attention type: {config.model.attention_type}")
    print("\nTraining Configuration:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Num batches: {config.training.num_batches:,}")
    print()

    train(config)


if __name__ == "__main__":
    main()

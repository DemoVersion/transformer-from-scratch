"""Text generation experiment with LSTM model with custom dataset support.

Similar to generate_custom.py but using SimplifiedAWDLSTM instead of GTransformer.
"""

import random
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from experiments.seqmodels.alternative_architectures.simplified_awd_lstm import (
    SimplifiedAWDLSTM,
)
from experiments.seqmodels.config_schema import LSTMExperimentConfig
from experiments.seqmodels.dataset_loader import (
    get_vocab_size,
    prepare_tokenized_dataset,
)
from experiments.seqmodels.utils import (
    create_checkpoint_dir,
    sample_batch,
    sample_sequence,
    save_checkpoint,
)
from former import util
from former.util import tic, toc


def train(config: LSTMExperimentConfig):
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

    # Train tokenizer from scratch and load the data
    print("Training tokenizer from scratch and loading tokenized dataset...")
    data_train, data_val, data_test, tokenizer = prepare_tokenized_dataset(
        target_tokens=config.dataset.target_tokens,
        tokenizer_path=config.dataset.tokenizer_path,
        split=config.dataset.dataset_split,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        test_ratio=config.dataset.test_ratio,
        tokenizer_vocab_size=config.dataset.tokenizer_vocab_size,
        tokenizer_training_docs=config.dataset.tokenizer_training_docs,
        verbose=True,
    )

    vocab_size = get_vocab_size(config.dataset.tokenizer_path)
    print(f"Vocabulary size: {vocab_size:,}")

    data_train, data_test = (
        (torch.cat([data_train, data_val], dim=0), data_test)
        if config.experiment.final
        else (data_train, data_val)
    )

    print(f"Training data size: {data_train.size(0):,} tokens")
    print(f"Test data size: {data_test.size(0):,} tokens")

    # Create the Simplified AWD-LSTM model
    model = SimplifiedAWDLSTM(
        vocab_size=vocab_size,
        embed_dim=config.model.embedding_size,
        hidden_dim=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout_rate=config.model.dropout_rate,
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
            "lstm/train-loss",
            float(loss.item()) * util.LOG2E,
            i * config.training.batch_size,
        )
        tbw.add_scalar("lstm/time-forward", t, instances_seen)

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
                )

                # Compute validation bits per byte
                upto = (
                    data_test.size(0)
                    if i == config.training.num_batches - 1
                    else config.evaluation.test_subset
                )
                data_sub = data_test[:upto]

                bits_per_byte = util.compute_compression(
                    model,
                    data_sub,
                    context=config.model.context,
                    batch_size=config.evaluation.test_batchsize,
                )

                print(f"epoch{i}: {bits_per_byte:.4} bits per byte")
                tbw.add_scalar(
                    "lstm/eval-loss",
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

    print("Training complete!")


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path(__file__).parent / "configs" / "lstm.yaml",
    help="Path to YAML configuration file",
)
def main(config_path: Path):
    """Train an LSTM model for text generation."""
    config = LSTMExperimentConfig.from_yaml(config_path)

    print(f"Loading configuration from: {config_path}")
    print("\nModel Configuration (LSTM):")
    print(f"  Embedding size: {config.model.embedding_size}")
    print(f"  Hidden dim: {config.model.embedding_size}")
    print(f"  Num layers: {config.model.num_layers}")
    print(f"  Context: {config.model.context}")
    print(f"  Dropout rate: {config.model.dropout_rate}")
    print("\nTraining Configuration:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Num batches: {config.training.num_batches:,}")
    print()

    train(config)


if __name__ == "__main__":
    main()

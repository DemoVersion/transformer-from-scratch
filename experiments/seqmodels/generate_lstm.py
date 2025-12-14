"""Text generation experiment with LSTM model with custom dataset support.

Similar to generate_custom.py but using SimplifiedAWDLSTM instead of GTransformer.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from experiments.seqmodels import config
from experiments.seqmodels.alternative_architectures.simplified_awd_lstm import (
    SimplifiedAWDLSTM,
)
from experiments.seqmodels.dataset_loader import (
    get_vocab_size,
    prepare_tokenized_dataset,
)
from experiments.seqmodels.utils import sample_batch, sample_sequence
from former import util
from former.util import tic, toc


def train():
    """Main training function."""
    # Set random seed
    if config.RANDOM_SEED < 0:
        seed = random.randint(0, 1000000)
        print("random seed: ", seed)
        torch.manual_seed(seed)
    else:
        torch.manual_seed(config.RANDOM_SEED)

    # Initialize tensorboard logging
    tbw = SummaryWriter(log_dir=config.TENSORBOARD_DIR)

    # Train tokenizer from scratch and load the data
    print("Training tokenizer from scratch and loading tokenized dataset...")
    data_train, data_val, data_test, tokenizer = prepare_tokenized_dataset(
        target_tokens=config.TARGET_TOKENS,
        tokenizer_path=config.TOKENIZER_PATH,
        split=config.DATASET_SPLIT,
        train_ratio=config.DATASET_TRAIN_RATIO,
        val_ratio=config.DATASET_VAL_RATIO,
        test_ratio=config.DATASET_TEST_RATIO,
        tokenizer_vocab_size=config.TOKENIZER_VOCAB_SIZE,
        tokenizer_training_docs=config.TOKENIZER_TRAINING_DOCS,
        verbose=True,
    )

    vocab_size = get_vocab_size(config.TOKENIZER_PATH)
    print(f"Vocabulary size: {vocab_size:,}")

    data_train, data_test = (
        (torch.cat([data_train, data_val], dim=0), data_test)
        if config.FINAL
        else (data_train, data_val)
    )

    print(f"Training data size: {data_train.size(0):,} tokens")
    print(f"Test data size: {data_test.size(0):,} tokens")

    # Create the Simplified AWD-LSTM model
    model = SimplifiedAWDLSTM(
        vocab_size=vocab_size,
        embed_dim=config.EMBEDDING_SIZE,
        hidden_dim=config.EMBEDDING_SIZE,  # Use same dimension for hidden state
        num_layers=config.DEPTH,
        dropout_rate=0.25,
    )
    if torch.cuda.is_available():
        model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer and scheduler
    opt = torch.optim.Adam(lr=config.LEARNING_RATE, params=model.parameters())

    # Linear learning rate warmup
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(i / (config.LR_WARMUP / config.BATCH_SIZE), 1.0)
    )

    # Training loop
    instances_seen = 0
    for i in tqdm.trange(config.NUM_BATCHES):
        opt.zero_grad()

        source, target = sample_batch(
            data_train, length=config.CONTEXT, batch_size=config.BATCH_SIZE
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
            i * config.BATCH_SIZE,
        )
        tbw.add_scalar("lstm/time-forward", t, instances_seen)

        loss.backward()  # backward pass

        # Clip gradients
        if config.GRADIENT_CLIPPING > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIPPING)

        opt.step()  # stochastic gradient descent step
        sch.step()  # update the learning rate

        # Validate every TEST_EVERY steps
        if i != 0 and (i % config.TEST_EVERY == 0 or i == config.NUM_BATCHES - 1):
            with torch.no_grad():
                # Sample and print a random sequence
                seedfr = random.randint(0, data_test.size(0) - config.CONTEXT)
                seed = data_test[seedfr : seedfr + config.CONTEXT].to(torch.long)

                if torch.cuda.is_available():
                    seed = seed.cuda()

                sample_sequence(
                    model,
                    seed=seed,
                    tokenizer=tokenizer,
                    max_context=config.CONTEXT,
                    verbose=True,
                    length=config.SAMPLE_LENGTH,
                )

                # Compute validation bits per byte
                upto = (
                    data_test.size(0)
                    if i == config.NUM_BATCHES - 1
                    else config.TEST_SUBSET
                )
                data_sub = data_test[:upto]

                bits_per_byte = util.compute_compression(
                    model,
                    data_sub,
                    context=config.CONTEXT,
                    batch_size=config.TEST_BATCHSIZE,
                )

                print(f"epoch{i}: {bits_per_byte:.4} bits per byte")
                tbw.add_scalar(
                    "lstm/eval-loss",
                    bits_per_byte,
                    i * config.BATCH_SIZE,
                )

    print("Training complete!")


if __name__ == "__main__":
    print("Configuration (LSTM):")
    print(f"  Embedding size: {config.EMBEDDING_SIZE}")
    print(f"  Hidden dim: {config.EMBEDDING_SIZE}")
    print(f"  Num layers: {config.DEPTH}")
    print(f"  Context: {config.CONTEXT}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print()

    train()

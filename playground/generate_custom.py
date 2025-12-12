"""Simplified text generation experiment with custom dataset support.

Based on experiments/generate.py but simplified with configuration constants.
"""

import random

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from former import GTransformer, util
from former.util import tic, toc
from playground import config
from playground.dataset_loader import get_vocab_size, load_tokenized_dataset


def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution.

    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """
    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()


def sample_batch(data, length, batch_size):
    """
    Sample a batch of random subsequences from the data.

    For each input instance, also creates the target sequence shifted one position right.

    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of integer matrices
    """
    # Sample the starting indices of the sequences to slice out
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # Slice out the input sequences
    seqs_inputs = [data[start : start + length] for start in starts]
    seqs_target = [data[start + 1 : start + length + 1] for start in starts]

    # Concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)

    return inputs, target


def sample_sequence(
    model, seed, max_context, length=600, temperature=0.5, verbose=False
):
    """
    Sequentially samples a sequence from the model, token by token.

    :param model: The transformer model
    :param seed: The sequence to start with
    :param max_context: Maximum context length the model can handle
    :param length: The total number of characters to sample
    :param temperature: The sampling temperature
    :param verbose: If true, print the sampled sequence as it is sampled
    :return: The sampled sequence, including the seed
    """
    sequence = seed.detach().clone()

    if verbose:  # Print the seed, surrounded by square brackets
        print("[", end="", flush=True)
        for c in seed:
            print(str(chr(c)), end="", flush=True)
        print("]", end="", flush=True)

    for _ in range(length):
        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :])

        # Sample the next token from the probabilities at the last position of the output
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end="", flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0)

    print()
    return seed


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

    # Load the data
    print("Loading tokenized dataset from streaming...")
    data_train, data_val, data_test = load_tokenized_dataset(
        target_tokens=config.TARGET_TOKENS,
        tokenizer_path=config.TOKENIZER_PATH,
        split=config.DATASET_SPLIT,
        train_ratio=config.DATASET_TRAIN_RATIO,
        val_ratio=config.DATASET_VAL_RATIO,
        test_ratio=config.DATASET_TEST_RATIO,
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

    # Create the model
    model = GTransformer(
        emb=config.EMBEDDING_SIZE,
        heads=config.NUM_HEADS,
        depth=config.DEPTH,
        seq_length=config.CONTEXT,
        num_tokens=vocab_size,
        attention_type=config.ATTENTION_TYPE,
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
            "transformer/train-loss",
            float(loss.item()) * util.LOG2E,
            i * config.BATCH_SIZE,
        )
        tbw.add_scalar("transformer/time-forward", t, instances_seen)

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
                    "transformer/eval-loss",
                    bits_per_byte,
                    i * config.BATCH_SIZE,
                )

    print("Training complete!")


if __name__ == "__main__":
    print("Configuration:")
    print(f"  Embedding size: {config.EMBEDDING_SIZE}")
    print(f"  Num heads: {config.NUM_HEADS}")
    print(f"  Depth: {config.DEPTH}")
    print(f"  Context: {config.CONTEXT}")
    print(f"  Attention type: {config.ATTENTION_TYPE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print()

    train()

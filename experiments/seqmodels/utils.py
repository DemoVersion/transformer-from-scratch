"""Shared utility functions for sequence model training and generation."""

import torch
import torch.distributions as dist
import torch.nn.functional as F


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
    model, seed, tokenizer, max_context, length=600, temperature=0.5, verbose=False
):
    """
    Sequentially samples a sequence from the model, token by token.

    :param model: The transformer model
    :param seed: The sequence to start with
    :param tokenizer: The tokenizer for decoding tokens
    :param max_context: Maximum context length the model can handle
    :param length: The total number of characters to sample
    :param temperature: The sampling temperature
    :param verbose: If true, print the sampled sequence as it is sampled
    :return: The sampled sequence, including the seed
    """
    sequence = seed.detach().clone()

    if verbose:  # Print the seed, surrounded by square brackets
        print("[", end="", flush=True)
        seed_text = tokenizer.decode(seed.tolist())
        print(seed_text, end="", flush=True)
        print("]", end="", flush=True)

    for _ in range(length):
        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :])

        # Sample the next token from the probabilities at the last position of the output
        c = sample(output[0, -1, :], temperature)

        if verbose:
            token_text = tokenizer.decode([c.item()])
            print(token_text, end="", flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0)

    print()
    return sequence

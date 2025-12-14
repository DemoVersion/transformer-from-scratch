import torch.nn as nn
import torch.nn.functional as F


class SimplifiedAWDLSTM(nn.Module):
    """
    Simplified AWD-LSTM model for text generation.
    Uses PyTorch's optimized LSTM block with dropout regularization.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers=3,
        dropout_rate=0.25,
    ):
        """
        :param vocab_size: Number of tokens in the vocabulary
        :param embed_dim: Dimension of token embeddings
        :param hidden_dim: Hidden dimension of LSTM layers
        :param num_layers: Number of LSTM layers (default: 3)
        :param dropout_rate: Dropout rate for regularization (default: 0.25)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Core LSTM: Use PyTorch's optimized block
        # Note: Set dropout=0.0 here, as we apply custom dropout manually
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, dropout=0.0, batch_first=True
        )

        # Final prediction head
        self.output_head = nn.Linear(hidden_dim, vocab_size)

        # Regular dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        """
        :param input_ids: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        # input_ids: [batch, seq_len]

        # Embed tokens
        x = self.embedding(input_ids)

        # Apply input dropout
        x = self.dropout(x)

        # Forward pass through the LSTM
        lstm_output, _ = self.lstm(x)

        # Apply dropout before output layer
        lstm_output = self.dropout(lstm_output)

        # Final linear layer
        b, t, h = lstm_output.size()
        # Use reshape instead of view to handle non-contiguous tensors
        logits = self.output_head(lstm_output.reshape(b * t, h)).view(
            b, t, self.vocab_size
        )

        return F.log_softmax(logits, dim=2)

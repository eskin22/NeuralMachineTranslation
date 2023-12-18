import torch.nn as nn

class RnnEncoder(nn.Module):
    def __init__(self, src_vocab, embedding_dim, hidden_units):
        super(RnnEncoder, self).__init__()
        """
        Args:
            src_vocab: Vocab_Lang, the source vocabulary
            embedding_dim: the dimension of the embedding
            hidden_units: The number of features in the GRU hidden state
        """
        self.src_vocab = src_vocab # Do not change
        vocab_size = len(src_vocab)

        ### TODO ###

        # Initialize embedding layer
        # (see: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Initialize a single directional GRU with 1 layer and batch_first=False
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_units, num_layers=1, batch_first=False)

    def forward(self, x):
        """
        Args:
            x: source texts, [max_len, batch_size]

        Returns:
            output: [max_len, batch_size, hidden_units]
            hidden_state: [1, batch_size, hidden_units]

        Pseudo-code:
        - Pass x through an embedding layer and pass the results through the recurrent net
        - Return output and hidden states from the recurrent net
        """
        output, hidden_state = None, None

        ### TODO ###

        # pass the input through the embedding layer
        embedded_input = self.embedding(x)

        # pass the embedded input through the GRU layer to get preliminary output
        output, hidden_state = self.gru(embedded_input)

        return output, hidden_state
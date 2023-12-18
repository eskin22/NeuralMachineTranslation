import torch.nn as nn

from src.models.transformer.helpers import create_positional_embedding

class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab, embedding_dim, num_heads,
        num_layers, dim_feedforward, max_len_src, device, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.device = device
        """
        Args:
            src_vocab: Vocab_Lang, the source vocabulary
            embedding_dim: the dimension of the embedding (also the number of expected features for the input of the Transformer)
            num_heads: The number of attention heads
            num_layers: the number of Transformer Encoder layers
            dim_feedforward: the dimension of the feedforward network models in the Transformer
            max_len_src: maximum length of the source sentences
            device: the working device (you may need to map your postional embedding to this device)
            dropout: the dropout to be applied. Default=0.1.
        """
        self.src_vocab = src_vocab # Do not change
        src_vocab_size = len(src_vocab)

        # Create positional embedding matrix
        self.position_embedding = create_positional_embedding(max_len_src, embedding_dim).to(device)
        self.register_buffer('positional_embedding', self.position_embedding) # this informs the model that position_embedding is not a learnable parameter

        ### TODO ###

        # Initialize embedding layer
        self.embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=embedding_dim)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Initialize a nn.TransformerEncoder model (you'll need to use embedding_dim,
        # num_layers, num_heads, & dim_feedforward here)

        # create encoding layer
        encoding_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)

        # initialize the transformer encoder model with the encoding layer
        self.encoder = nn.TransformerEncoder(encoder_layer=encoding_layer, num_layers=num_layers)

    def make_src_mask(self, src):
        """
        Args:
            src: [max_len, batch_size]
        Returns:
            Boolean matrix of size [batch_size, max_len] indicating which indices are padding
        """
        assert len(src.shape) == 2, 'src must have exactly 2 dimensions'
        src_mask = src.transpose(0, 1) == 0 # padding idx
        return src_mask.to(self.device) # [batch_size, max_src_len]

    def forward(self, x):
        """
        Args:
            x: [max_len, batch_size]
        Returns:
            output: [max_len, batch_size, embed_dim]
        Pseudo-code (note: x refers to the original input to this function throughout the pseudo-code):
        - Pass x through the word embedding
        - Add positional embedding to the word embedding, then apply dropout
        - Call make_src_mask(x) to compute a mask: this tells us which indexes in x
          are padding, which we want to ignore for the self-attention
        - Call the encoder, with src_key_padding_mask = src_mask
        """
        output = None

        ### TODO ###

        # pass the input through the embedding layer
        embedded_input = self.embedding(x)

        # move the embedded input onto the gpu
        embedded_input = embedded_input.to(self.device)

        # add the positional embeddings to the word embeddings
        positional_embedded = embedded_input + self.position_embedding[:, :embedded_input.size(0), :]

        # send the embeddings to the gpu
        positional_embedded = positional_embedded.to(self.device)

        # pass the embeddings through the dropout layer
        positional_embedded = self.dropout(positional_embedded)

        # create source mask to ignore the padding tokens from the input sequence
        src_mask = self.make_src_mask(x)

        # pass the embeddings and source mask to the encoder to encode the sequence
        output = self.encoder(positional_embedded, src_key_padding_mask=src_mask)

        return output
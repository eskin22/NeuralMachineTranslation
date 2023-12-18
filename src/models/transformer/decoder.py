import torch
import torch.nn as nn

from src.models.transformer.helpers import create_positional_embedding

class TransformerDecoder(nn.Module):
    def __init__(self, trg_vocab, embedding_dim, num_heads,
        num_layers, dim_feedforward, max_len_trg, device, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.device = device
        """
        Args:
            trg_vocab: Vocab_Lang, the target vocabulary
            embedding_dim: the dimension of the embedding (also the number of expected features for the input of the Transformer)
            num_heads: The number of attention heads
            num_layers: the number of Transformer Decoder layers
            dim_feedforward: the dimension of the feedforward network models in the Transformer
            max_len_trg: maximum length of the target sentences
            device: the working device (you may need to map your postional embedding to this device)
            dropout: the dropout to be applied. Default=0.1.
        """
        self.trg_vocab = trg_vocab # Do not change
        trg_vocab_size = len(trg_vocab)

        # Create positional embedding matrix
        self.position_embedding = create_positional_embedding(max_len_trg, embedding_dim).to(device)
        self.register_buffer('positional_embedding', self.position_embedding) # this informs the model that positional_embedding is not a learnable parameter

        ### TODO ###

        # Initialize embedding layer
        self.embedding = nn.Embedding(num_embeddings=trg_vocab_size, embedding_dim=embedding_dim)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Initialize a nn.TransformerDecoder model (you'll need to use embedding_dim,
        # num_layers, num_heads, & dim_feedforward here)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)

        # Final fully connected layer
        self.fully_connected = nn.Linear(in_features=embedding_dim, out_features=trg_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(self.device)
        return mask

    def forward(self, dec_in, enc_out):
        """
        Args:
            dec_in: [sequence length, batch_size]
            enc_out: [max_len, batch_size, embed_dim]
        Returns:
            output: [sequence length, batch_size, trg_vocab_size]
        Pseudo-code:
        - Compute input word and positional embeddings in similar manner to encoder
        - Call generate_square_subsequent_mask() to compute a mask: this time,
          the mask is to prevent the decoder from attending to tokens in the "future".
          In other words, at time step i, the decoder should only attend to tokens
          1 to i-1.
        - Call the decoder, with tgt_mask = trg_mask
        - Run the output through the fully-connected layer and return it
        """
        output = None

        ### TODO ###

        # determine sequence length of decoder input
        dec_in_length = dec_in.shape[0]

        # create a tensor to store the positional encodings of the decoder input
        dec_in_positions = torch.arange(0, dec_in_length)

        # pass the decoder input through the embedding layer
        embedded_input = self.embedding(dec_in)

        # combine the embedded decoder input with the positional embeddings
        embedded_input = embedded_input + self.position_embedding[:dec_in_length, :]

        # pass the combined embedded input through the dropout layer
        embedded_input = self.dropout(embedded_input)

        # generate square subsequent mask for the target
        trg_mask = self.generate_square_subsequent_mask(dec_in_length)

        # pass the embedded input and encoder mask through the decoder model
        prelim_output = self.decoder(embedded_input, enc_out, tgt_mask=trg_mask)

        # finally, pass the preliminary output through the fully connected layer to get final output
        output = self.fully_connected(prelim_output)


        return output
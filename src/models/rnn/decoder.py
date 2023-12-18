import torch
import torch.nn as nn
import torch.nn.functional as F

class RnnDecoder(nn.Module):
    def __init__(self, trg_vocab, embedding_dim, hidden_units):
        super(RnnDecoder, self).__init__()
        """
        Args:
            trg_vocab: Vocab_Lang, the target vocabulary
            embedding_dim: The dimension of the embedding
            hidden_units: The number of features in the GRU hidden state
        """
        self.trg_vocab = trg_vocab # Do not change
        vocab_size = len(trg_vocab)

        ### TODO ###

        # Initialize embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Initialize layers to compute attention score

        # attention layer for transforming the current hidden state of the decoder
        self.attn_DHS = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        # attention layer for transforming the encoder's output
        self.attn_EO = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        # attention layer for combining the weights given by the other attention layers
        self.attn_COM = nn.Linear(in_features=hidden_units, out_features=1)

        # Initialize a single directional GRU with 1 layer and batch_first=True
        # NOTE: Input to your RNN will be the concatenation of your embedding vector and the context vector
        self.gru = nn.GRU(input_size=(embedding_dim + hidden_units), hidden_size=hidden_units, batch_first=True)

        # Initialize fully connected layer
        self.fully_connected = nn.Linear(in_features=hidden_units, out_features=vocab_size)

    def compute_attention(self, dec_hs, enc_output):
        '''
        This function computes the context vector and attention weights.

        Args:
            dec_hs: Decoder hidden state; [1, batch_size, hidden_units]
            enc_output: Encoder outputs; [max_len_src, batch_size, hidden_units]

        Returns:
            context_vector: Context vector, according to formula; [batch_size, hidden_units]
            attention_weights: The attention weights you have calculated; [batch_size, max_len_src, 1]

        Pseudo-code:
            (1) Compute the attention scores for dec_hs & enc_output
                    - Hint: You may need to permute the dimensions of the tensors in order to pass them through linear layers
                    - Output size: [batch_size, max_len_src, 1]
            (2) Compute attention_weights by taking a softmax over your scores to normalize the distribution (Make sure that after softmax the normalized scores add up to 1)
                    - Output size: [batch_size, max_len_src, 1]
            (3) Compute context_vector from attention_weights & enc_output
                    - Hint: You may find it helpful to use torch.sum & element-wise multiplication (* operator)
            (4) Return context_vector & attention_weights
        '''
        context_vector, attention_weights = None, None

        ### TODO ###

        # modify the shapes of the decoder hidden state and the encoder outputs before transformations
        dec_hs = dec_hs.permute(1, 0, 2)
        enc_output = enc_output.permute(1, 0, 2)

        # pass decoder hidden state and encoder outputs through the corresponding attention layers
        dec_hs_t = self.attn_DHS(dec_hs)
        enc_output_t = self.attn_EO(enc_output)

        # mini-sanity-check
        # print(f"dec_hs_t shape: {dec_hs_t.shape}")
        # print(f"enc_output_t shape: {enc_output_t.shape}")

        # normalize the values from previous attention layers with tanh activation function
        normalized_output = torch.tanh(dec_hs_t + enc_output_t)

        # compute the attention scores by passing the results to the final attention layer
        attn_scores = self.attn_COM(normalized_output)

        # perform softmax activation
        attention_weights = F.softmax(attn_scores, dim=-2)

        # compute and transform context vector to align with embedding output for concatenation
        context_vector = torch.sum(attention_weights * enc_output, dim=1)
        # maybe this line of code is causing the errors?
        # don't change the shape in this method because the autograder fails, instead change the shape for concatenation in the forward method
        # context_vector = context_vector.unsqueeze(1)

        # mini-sanity check
        # print(f"Your context vector shape: {context_vector.shape}")
        # print(f"Your attention weights shape: {attention_weights.shape}")

        return context_vector, attention_weights

    def forward(self, x, dec_hs, enc_output):
        '''
        This function runs the decoder for a **single** time step.

        Args:
            x: Input token; [batch_size, 1]
            dec_hs: Decoder hidden state; [1, batch_size, hidden_units]
            enc_output: Encoder outputs; [max_len_src, batch_size, hidden_units]

        Returns:
            fc_out: (Unnormalized) output distribution [batch_size, vocab_size]
            dec_hs: Decoder hidden state; [1, batch_size, hidden_units]
            attention_weights: The attention weights you have learned; [batch_size, max_len_src, 1]

        Pseudo-code:
            (1) Compute the context vector & attention weights by calling self.compute_attention(...) on the appropriate input
            (2) Obtain embedding vectors for your input x
                    - Output size: [batch_size, 1, embedding_dim]
            (3) Concatenate the context vector & the embedding vectors along the appropriate dimension
            (4) Feed this result through your RNN (along with the current hidden state) to get output and new hidden state
                    - Output sizes: [batch_size, 1, hidden_units] & [1, batch_size, hidden_units]
            (5) Feed the output of your RNN through linear layer to get (unnormalized) output distribution (don't call softmax!)
            (6) Return this output, the new decoder hidden state, & the attention weights
        '''
        fc_out, attention_weights = None, None

        ### TODO ###

        # get the context vector and attention weights from the compute_attention method
        context_vector, attention_weights = self.compute_attention(dec_hs, enc_output)

        # add another dimension to the context vector to facilitate concatenation with embedded input
        context_vector = context_vector.unsqueeze(1)

        # pass the input through the embedding layer
        embedded_input = self.embedding(x)

        # mini-sanity-check
        # print(f"embedded input shape: {embedded_input.shape}")
        # print(f"context vector shape: {context_vector.shape}")

        # concatenate the tensors for the embedded input and the context vector to generate input for the rnn
        rnn_in = torch.cat((embedded_input, context_vector), dim=-1)

        # pass the concatenated tensors through the GRU layer
        rnn_out, dec_hs = self.gru(rnn_in, dec_hs)

        # transform the rnn_out by removing 1 dimension to make it align with the fully connected layer
        rnn_out = rnn_out.squeeze(1)

        # finally, pass the output of the RNN through the linear layer and return
        fc_out = self.fully_connected(rnn_out)

        return fc_out, dec_hs, attention_weights
import torch
import math

def create_positional_embedding(max_len, embed_dim):
    '''
    Args:
        max_len: The maximum length supported for positional embeddings
        embed_dim: The size of your embeddings
    Returns:
        pe: [max_len, 1, embed_dim] computed as in the formulae above
    '''
    pe = None

    ### TODO ###

    # create a tensor for pe where all values are equal to zero to store the positional embeddings
    pe = torch.zeros(max_len, 1, embed_dim)

    # iterate through the positions
    for pos in range(max_len):

      # iterate over the dimensions
      for i in range(0, embed_dim, 2):

        # calculate the denominator for the formula
        denom = 10000 ** (i / embed_dim)

        # maybe this code is failing somehow?
        # log and exponentiate the denominator to prevent underflow/overflow
        logged_denom = math.log(denom)
        exp_denom = math.exp(logged_denom)

        # set the sin value for the pe tensor
        pe[pos, 0, i] = math.sin(pos / exp_denom)

        # if we can, set the cosine value in the pe tensor
        if i + 1 < embed_dim:
          pe[pos, 0, i+1] = math.cos(pos / exp_denom)



    return pe
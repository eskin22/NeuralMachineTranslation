import torch

from src.evaluation.evaluationFunctions import compute_bleu_scores
from src.models.rnn.train import loss_function

def decode_rnn_model(encoder, decoder, src, max_decode_len, device):
    """
    Args:
        encoder: Your RnnEncoder object
        decoder: Your RnnDecoder object
        src: [max_src_length, batch_size] the source sentences you wish to translate
        max_decode_len: The maximum desired length (int) of your target translated sentences
        device: the device your torch tensors are on (you may need to call x.to(device) for some of your tensors)

    Returns:
        curr_output: [batch_size, max_decode_len] containing your predicted translated sentences
        curr_predictions: [batch_size, max_decode_len, trg_vocab_size] containing the (unnormalized) probabilities of each
            token in your vocabulary at each time step

    Pseudo-code:
    - Obtain encoder output and hidden state by encoding src sentences
    - For 1 ≤ t ≤ max_decode_len:
        - Obtain your (unnormalized) prediction probabilities and hidden state by feeding dec_input (the best words
          from the previous time step), previous hidden state, and encoder output to decoder
        - Save your (unnormalized) prediction probabilities in curr_predictions at index t
        - Obtain your new dec_input by selecting the most likely (highest probability) token
        - Save dec_input in curr_output at index t
    """
    # Initialize variables
    trg_vocab = decoder.trg_vocab
    batch_size = src.size(1)
    curr_output = torch.zeros((batch_size, max_decode_len))
    curr_predictions = torch.zeros((batch_size, max_decode_len, len(trg_vocab.idx2word)))

    # We start the decoding with the start token for each example
    dec_input = torch.tensor([[trg_vocab.word2idx['<start>']]] * batch_size)

    curr_output[:, 0] = dec_input.squeeze(1)

    ### TODO: Implement decoding algorithm ###

    # move all the tensors onto the gpu
    dec_input = dec_input.to(device)
    curr_output = curr_output.to(device)
    curr_predictions = curr_predictions.to(device)

    # encode the source sentences to get encoder outputs and hidden states
    enc_output, hidden_state = encoder(src)

    # loop over each time step
    for time_step in range(1, max_decode_len):

      # pass inputs to decoder to get decoder output
      decoder_output = decoder(dec_input, hidden_state, enc_output)

      # extract prediction probabilities and hidden state from decoder output
      fc_out = decoder_output[0]
      hidden_state = decoder_output[1]

      # update current predictions with new prediction probabilites
      curr_predictions[:, time_step, :] = fc_out

      # get the token with the highest probability
      best_token = fc_out.topk(1, dim=1)[1]

      # set decoder input for the next time step
      dec_input = best_token.view(batch_size, -1)

      # store the input for the decoder into the current output
      curr_output[:, time_step] = dec_input.squeeze(1)


    return curr_output, curr_predictions

def evaluate_rnn_model(encoder, decoder, test_dataset, target_tensor_val, device):
    trg_vocab = decoder.trg_vocab
    batch_size = test_dataset.batch_size
    n_batch = 0
    total_loss = 0

    encoder.eval()
    decoder.eval()

    final_output, target_output = None, None

    with torch.no_grad():
        for batch, (src, trg) in enumerate(test_dataset):
            n_batch += 1
            loss = 0
            curr_output, curr_predictions = decode_rnn_model(encoder, decoder, src.transpose(0,1).to(device), trg.size(1), device)
            for t in range(1, trg.size(1)):
                loss += loss_function(trg[:, t].to(device), curr_predictions[:,t,:].to(device))

            if final_output is None:
                final_output = torch.zeros((len(target_tensor_val), trg.size(1)))
                target_output = torch.zeros((len(target_tensor_val), trg.size(1)))
            final_output[batch*batch_size:(batch+1)*batch_size] = curr_output
            target_output[batch*batch_size:(batch+1)*batch_size] = trg
            batch_loss = (loss / int(trg.size(1)))
            total_loss += batch_loss

        print('Loss {:.4f}'.format(total_loss / n_batch))

    # Compute BLEU scores
    return compute_bleu_scores(target_tensor_val, target_output, final_output, trg_vocab)
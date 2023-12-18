import torch
import torch.nn as nn

from src.evaluation.evaluationFunctions import compute_bleu_scores

def decode_transformer_model(encoder, decoder, src, max_decode_len, device):
    """
    Args:
        encoder: Your TransformerEncoder object
        decoder: Your TransformerDecoder object
        src: [max_src_length, batch_size] the source sentences you wish to translate
        max_decode_len: The maximum desired length (int) of your target translated sentences
        device: the device your torch tensors are on (you may need to call x.to(device) for some of your tensors)

    Returns:
        curr_output: [batch_size, max_decode_len] containing your predicted translated sentences
        curr_predictions: [batch_size, max_decode_len, trg_vocab_size] containing the (unnormalized) probabilities of each
            token in your vocabulary at each time step

    Pseudo-code:
    - Obtain encoder output by encoding src sentences
    - For 1 ≤ t ≤ max_decode_len:
        - Obtain dec_input as the best words so far for previous time steps (you can get this from curr_output)
        - Obtain your (unnormalized) prediction probabilities by feeding dec_input and encoder output to decoder
        - Save your (unnormalized) prediction probabilities in curr_predictions at index t
        - Calculate the most likely (highest probability) token and save in curr_output at timestep t
    """
    # Initialize variables
    trg_vocab = decoder.trg_vocab
    batch_size = src.size(1)
    curr_output = torch.zeros((batch_size, max_decode_len))
    curr_predictions = torch.zeros((batch_size, max_decode_len, len(trg_vocab.idx2word)))
    enc_output = None

    # We start the decoding with the start token for each example
    dec_input = torch.tensor([[trg_vocab.word2idx['<start>']]] * batch_size).transpose(0,1)
    curr_output[:, 0] = dec_input.squeeze(1)

    ### TODO: Implement decoding algorithm ###

    # move the relevant objects to the gpu
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    src = src.to(device)
    curr_output = curr_output.to(device)
    curr_predictions = curr_predictions.to(device)
    dec_input = dec_input.to(device)

    # endode the source tnesor
    enc_output = encoder(src)

    # iterate through the sequence
    for time_step in range(1, max_decode_len):

      # for each time step, pass the decoded input and encoder output through the decoder
      dec_output = decoder(dec_input, enc_output)

      # get the prediction probabilities of the last token
      prediction_probs = dec_output[-1]

      # add the new prediction probabilities to the current predictions tensor
      curr_predictions[:, time_step, :] = prediction_probs

      # identify the best token (toekn with the highest probability) with topk and drop a dimension to align with the current output tensor
      best_token = prediction_probs.topk(1, dim=1)[1]
      best_token_t = best_token.squeeze(1)

      # add the best token to the current output tensor
      curr_output[:, time_step] = best_token_t

      # reshape the best token again to align with the concatenation with the decoder input tensor
      best_token_t = best_token.transpose(0, 1)

      # finally, concatenate the best token to the decoder input tensor
      dec_input = torch.cat((dec_input, best_token_t), dim=0)


    return curr_output, curr_predictions, enc_output

def evaluate_transformer_model(encoder, decoder, test_dataset, target_tensor_val, device):
    trg_vocab = decoder.trg_vocab
    batch_size = test_dataset.batch_size
    n_batch = 0
    total_loss = 0

    encoder.eval()
    decoder.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses=[]
    final_output, target_output = None, None

    with torch.no_grad():
        for batch, (src, trg) in enumerate(test_dataset):
            n_batch += 1
            loss = 0

            src, trg = src.transpose(0,1).to(device), trg.transpose(0,1).to(device)
            curr_output, curr_predictions, enc_out = decode_transformer_model(encoder, decoder, src, trg.size(0), device)

            for t in range(1, trg.size(0)):
                output = decoder(trg[:-1, :], enc_out)
                output = output.reshape(-1, output.shape[2])
                loss_trg = trg[1:].reshape(-1)
                loss += criterion(output, loss_trg)
                # loss += criterion(curr_predictions[:,t,:].to(device), trg[t,:].reshape(-1).to(device))

            if final_output is None:
                final_output = torch.zeros((len(target_tensor_val), trg.size(0)))
                target_output = torch.zeros((len(target_tensor_val), trg.size(0)))

            final_output[batch*batch_size:(batch+1)*batch_size] = curr_output
            target_output[batch*batch_size:(batch+1)*batch_size] = trg.transpose(0,1)
            losses.append(loss.item() / (trg.size(0)-1))

        mean_loss = sum(losses) / len(losses)
        print('Loss {:.4f}'.format(mean_loss))

    # Compute Bleu scores
    return compute_bleu_scores(target_tensor_val, target_output, final_output, trg_vocab)
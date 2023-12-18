import time
import torch
from tqdm.notebook import tqdm
import torch.nn.functional as F

def loss_function(real, pred):
    mask = real.ge(1).float() # Only consider non-zero inputs in the loss

    loss_ = F.cross_entropy(pred, real) * mask
    return torch.mean(loss_)

def train_rnn_model(encoder, decoder, dataset, optimizer, trg_vocab, device, n_epochs):
    batch_size = dataset.batch_size
    for epoch in range(n_epochs):
        start = time.time()
        n_batch = 0
        total_loss = 0

        encoder.train()
        decoder.train()

        for source, trg in tqdm(dataset):
            n_batch += 1
            loss = 0

            enc_output, enc_hidden = encoder(source.transpose(0,1).to(device))
            dec_hidden = enc_hidden

            # use teacher forcing - feeding the target as the next input (via dec_input)
            dec_input = torch.tensor([[trg_vocab.word2idx['<start>']]] * batch_size)

            # run code below for every timestep in the ys batch
            for t in range(1, trg.size(1)):
                predictions, dec_hidden, _ = decoder(dec_input.to(device), dec_hidden.to(device), enc_output.to(device))
                assert len(predictions.shape) == 2 and predictions.shape[0] == dec_input.shape[0] and predictions.shape[1] == len(trg_vocab.word2idx), "First output of decoder must have shape [batch_size, vocab_size], you returned shape " + str(predictions.shape)
                loss += loss_function(trg[:, t].to(device), predictions.to(device))
                dec_input = trg[:, t].unsqueeze(1)

            batch_loss = (loss / int(trg.size(1)))
            total_loss += batch_loss

            optimizer.zero_grad()

            batch_loss.backward()

            ### update model parameters
            optimizer.step()

        ### TODO: Save checkpoint for model (optional)
        print('Epoch:{:2d}/{}\t Loss: {:.4f} \t({:.2f}s)'.format(epoch + 1, n_epochs, total_loss / n_batch, time.time() - start))

    print('Model trained!')
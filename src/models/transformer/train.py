import time
import torch
import torch.nn as nn
from tqdm.notebook import tqdm

def train_transformer_model(train_dataset, encoder, decoder, optimizer, device, n_epochs):
    encoder.train()
    decoder.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(n_epochs):
        start = time.time()
        losses = []

        for src, trg in tqdm(train_dataset):

            src = src.to(device).transpose(0,1) # [max_src_length, batch_size]
            trg = trg.to(device).transpose(0,1) # [max_trg_length, batch_size]

            enc_out = encoder(src)
            output = decoder(trg[:-1, :], enc_out)

            output = output.reshape(-1, output.shape[2])
            trg = trg[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, trg)
            losses.append(loss.item())

            loss.backward()

            # Clip to avoid exploding grading issues
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)

            optimizer.step()

        mean_loss = sum(losses) / len(losses)
        print('Epoch:{:2d}/{}\t Loss:{:.4f} ({:.2f}s)'.format(epoch + 1, n_epochs, mean_loss, time.time() - start))
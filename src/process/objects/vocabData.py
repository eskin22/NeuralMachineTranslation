import torch
import numpy as np
from torch.utils.data import Dataset

class Vocab_Lang():
    def __init__(self, vocab):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.vocab = vocab

        for index, word in enumerate(vocab):
            self.word2idx[word] = index + 2 # +2 because of <pad> and <unk> token
            self.idx2word[index + 2] = word

    def __len__(self):
        return len(self.word2idx)

    def __repr__(self):
        if len(self.vocab) <= 5:
            return str(self.vocab)
        else:
            return f'Vocab_Lang object with {len(self.vocab)} words'

class MyData(Dataset):
    def __init__(self, X, y):
        self.length = torch.LongTensor([np.sum(1 - np.equal(x, 0)) for x in X])
        self.data = torch.LongTensor(X)
        self.target = torch.LongTensor(y)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)
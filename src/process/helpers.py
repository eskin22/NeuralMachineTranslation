import unicodedata
import re
import numpy as np

def unicode_to_ascii(s):
    """Normalizes latin chars with accent to their canonical decomposition"""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    '''
    Preprocess the sentence to add the start, end tokens and make them lower-case
    '''
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r'([?.!,¿])', r' \1 ', w)
    w = re.sub(r'[" "]+', ' ', w)

    w = re.sub(r'[^a-zA-Z?.!,¿]+', ' ', w)

    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w


def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len:
        padded[:] = x[:max_len]
    else:
        padded[:len(x)] = x
    return padded


def preprocess_data_to_tensor(dataframe, src_vocab, trg_vocab):
    # Vectorize the input and target languages
    src_tensor = [[src_vocab.word2idx[s if s in src_vocab.vocab else '<unk>'] for s in es.split(' ')] for es in dataframe['es'].values.tolist()]
    trg_tensor = [[trg_vocab.word2idx[s if s in trg_vocab.vocab else '<unk>'] for s in eng.split(' ')] for eng in dataframe['eng'].values.tolist()]

    # Calculate the max_length of input and output tensor for padding
    max_length_src, max_length_trg = max(len(t) for t in src_tensor), max(len(t) for t in trg_tensor)
    print('max_length_src: {}, max_length_trg: {}'.format(max_length_src, max_length_trg))

    # Pad all the sentences in the dataset with the max_length
    src_tensor = [pad_sequences(x, max_length_src) for x in src_tensor]
    trg_tensor = [pad_sequences(x, max_length_trg) for x in trg_tensor]

    return src_tensor, trg_tensor, max_length_src, max_length_trg


def train_test_split(src_tensor, trg_tensor):
    '''
    Create training and test sets.
    '''
    total_num_examples = len(src_tensor) - int(0.2*len(src_tensor))
    src_tensor_train, src_tensor_test = src_tensor[:int(0.75*total_num_examples)], src_tensor[int(0.75*total_num_examples):total_num_examples]
    trg_tensor_train, trg_tensor_test = trg_tensor[:int(0.75*total_num_examples)], trg_tensor[int(0.75*total_num_examples):total_num_examples]

    return src_tensor_train, src_tensor_test, trg_tensor_train, trg_tensor_test

def build_vocabulary(pd_dataframe):
    sentences = [sen.split() for sen in pd_dataframe]
    vocab = {}
    for sen in sentences:
        for word in sen:
            if word not in vocab:
                vocab[word] = 1
    return list(vocab.keys())
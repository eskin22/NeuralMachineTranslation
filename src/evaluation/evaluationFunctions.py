from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_reference_candidate(target, pred, trg_vocab):
    def _to_token(sentence):
        lis = []
        for s in sentence[1:]:
            x = trg_vocab.idx2word[s]
            if x == "<end>": break
            lis.append(x)
        return lis
    reference = _to_token(list(target.numpy()))
    candidate = _to_token(list(pred.numpy()))
    return reference, candidate

def compute_bleu_scores(target_tensor_val, target_output, final_output, trg_vocab):
    bleu_1 = 0.0
    bleu_2 = 0.0
    bleu_3 = 0.0
    bleu_4 = 0.0

    smoother = SmoothingFunction()
    save_reference = []
    save_candidate = []
    for i in range(len(target_tensor_val)):
        reference, candidate = get_reference_candidate(target_output[i], final_output[i], trg_vocab)

        bleu_1 += sentence_bleu(reference, candidate, weights=(1,), smoothing_function=smoother.method1)
        bleu_2 += sentence_bleu(reference, candidate, weights=(1/2, 1/2), smoothing_function=smoother.method1)
        bleu_3 += sentence_bleu(reference, candidate, weights=(1/3, 1/3, 1/3), smoothing_function=smoother.method1)
        bleu_4 += sentence_bleu(reference, candidate, weights=(1/4, 1/4, 1/4, 1/4), smoothing_function=smoother.method1)

        save_reference.append(reference)
        save_candidate.append(candidate)

    bleu_1 = bleu_1/len(target_tensor_val)
    bleu_2 = bleu_2/len(target_tensor_val)
    bleu_3 = bleu_3/len(target_tensor_val)
    bleu_4 = bleu_4/len(target_tensor_val)

    scores = {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}
    print('BLEU 1-gram: %f' % (bleu_1))
    print('BLEU 2-gram: %f' % (bleu_2))
    print('BLEU 3-gram: %f' % (bleu_3))
    print('BLEU 4-gram: %f' % (bleu_4))

    return save_candidate, scores
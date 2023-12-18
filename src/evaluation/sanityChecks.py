import torch
import os

from src.models.rnn.decoder import RnnDecoder

def sanityCheckModel(all_test_params, NN, expected_outputs, init_or_forward):
    print('--- TEST: ' + ('Number of Model Parameters (tests __init__(...))' if init_or_forward=='init' else 'Output shape of forward(...)') + ' ---')
    
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if init_or_forward == "forward":
        # Creating random texts and lables batches
        texts_batch = torch.randint(low=0, high=len(all_test_params[0]['src_vocab']), size=(10,16))
        labels_batch = torch.randint(low=0, high=len(all_test_params[0]['src_vocab']), size=(10,12))

    for tp_idx, (test_params, expected_output) in enumerate(zip(all_test_params, expected_outputs)):
        if init_or_forward == "forward":
            batch_size = test_params['batch_size']
            texts = texts_batch[:batch_size]
            # if NN.__name__ == "RnnEncoder":
            texts = texts.transpose(0,1)

        # Construct the student model
        tps = {k:v for k, v in test_params.items() if k != 'batch_size'}
        stu_nn = NN(**tps)

        input_rep = str({k:v for k,v in tps.items()})

        if init_or_forward == "forward":
            with torch.no_grad():
                if NN.__name__ == "TransformerEncoder":
                    stu_out = stu_nn(texts)
                else:
                    stu_out, _ = stu_nn(texts)
                    expected_output = torch.rand(expected_output).size()
            ref_out_shape = expected_output

            has_passed = torch.is_tensor(stu_out)
            if not has_passed: msg = 'Output must be a torch.Tensor; received ' + str(type(stu_out))
            else:
                has_passed = stu_out.shape == ref_out_shape
                msg = 'Your Output Shape: ' + str(stu_out.shape)


            status = 'PASSED' if has_passed else 'FAILED'
            message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape: ' + str(texts.shape) + '\tExpected Output Shape: ' + str(ref_out_shape) + '\t' + msg
            print(message)
        else:
            stu_num_params = count_parameters(stu_nn)
            ref_num_params = expected_output
            comparison_result = (stu_num_params == ref_num_params)

            status = 'PASSED' if comparison_result else 'FAILED'
            message = '\t' + status + "\tInput: " + input_rep + ('\tExpected Num. Params: ' + str(ref_num_params) + '\tYour Num. Params: '+ str(stu_num_params))
            print(message)

        del stu_nn
        
def sanityCheckDecoderModelForward(inputs, NN, expected_outputs):
    print('--- TEST: Output shape of forward(...) ---\n')
    expected_fc_outs = expected_outputs[0]
    expected_dec_hs = expected_outputs[1]
    expected_attention_weights = expected_outputs[2]
    msg = ''
    for i, inp in enumerate(inputs):
        input_rep = '{'
        for k,v in inp.items():
            if torch.is_tensor(v):
                input_rep += str(k) + ': ' + 'Tensor with shape ' + str(v.size()) + ', '
            else:
                input_rep += str(k) + ': ' + str(v) + ', '
        input_rep += '}'
        dec = RnnDecoder(trg_vocab=inp['trg_vocab'],embedding_dim=inp['embedding_dim'],hidden_units=inp['hidden_units'])
        dec_hs = torch.rand(1, inp["batch_size"], inp['hidden_units'])
        x = torch.randint(low=0,high=len(inp["trg_vocab"]),size=(inp["batch_size"], 1))
        with torch.no_grad():
            dec_out = dec(x=x, dec_hs=dec_hs,enc_output=inp['encoder_outputs'])
            if not isinstance(dec_out, tuple):
                msg = '\tFAILED\tYour RnnDecoder.forward() output must be a tuple; received ' + str(type(dec_out))
                print(msg)
                continue
            elif len(dec_out)!=3:
                msg = '\tFAILED\tYour RnnDecoder.forward() output must be a tuple of size 3; received tuple of size ' + str(len(dec_out))
                print(msg)
                continue
            stu_fc_out, stu_dec_hs, stu_attention_weights = dec_out
        del dec
        has_passed = True
        msg = ""
        if not torch.is_tensor(stu_fc_out):
            has_passed = False
            msg += '\tFAILED\tOutput must be a torch.Tensor; received ' + str(type(stu_fc_out)) + " "
        if not torch.is_tensor(stu_dec_hs):
            has_passed = False
            msg += '\tFAILED\tDecoder Hidden State must be a torch.Tensor; received ' + str(type(stu_dec_hs)) + " "
        if not torch.is_tensor(stu_attention_weights):
            has_passed = False
            msg += '\tFAILED\tAttention Weights must be a torch.Tensor; received ' + str(type(stu_attention_weights)) + " "

        status = 'PASSED' if has_passed else 'FAILED'
        if not has_passed:
            message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape (x): ' + str(os.XATTR_REPLACE.shape) + '\tExpected Output Shape: ' + str(expected_fc_outs[i]) + '\t' + msg
            print(message)
            continue

        has_passed = stu_fc_out.size() == expected_fc_outs[i]
        msg = 'Your Output Shape: ' + str(stu_fc_out.size())
        status = 'PASSED' if has_passed else 'FAILED'
        message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape (x): ' + str(x.shape) + '\tExpected Output Shape: ' + str(expected_fc_outs[i]) + '\t' + msg
        print(message)

        has_passed = stu_dec_hs.size() == expected_dec_hs[i]
        msg = 'Your Hidden State Shape: ' + str(stu_dec_hs.size())
        status = 'PASSED' if has_passed else 'FAILED'
        message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape (x): ' + str(x.shape) + '\tExpected Hidden State Shape: ' + str(expected_dec_hs[i]) + '\t' + msg
        print(message)

        has_passed = stu_attention_weights.size() == expected_attention_weights[i]
        msg = 'Your Attention Weights Shape: ' + str(stu_attention_weights.size())
        status = 'PASSED' if has_passed else 'FAILED'
        message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape (x): ' + str(x.shape) + '\tExpected Attention Weights Shape: ' + str(expected_attention_weights[i]) + '\t' + msg
        print(message)

        stu_sum = stu_attention_weights.sum(dim=1).squeeze()
        if torch.allclose(stu_sum, torch.ones_like(stu_sum), atol=1e-5):
            print('\tPASSED\t The sum of your attention_weights along dim 1 is 1.')
        else:
            print('\tFAILED\t The sum of your attention_weights along dim 1 is not 1.')
        print()
        
### DO NOT EDIT ###

def sanityCheckTransformerDecoderModelForward(inputs, NN, expected_outputs):
    print('--- TEST: Output shape of forward(...) ---\n')
    msg = ''
    for i, inp in enumerate(inputs):
        input_rep = '{'
        for k,v in inp.items():
            if torch.is_tensor(v):
                input_rep += str(k) + ': ' + 'Tensor with shape ' + str(v.size()) + ', '
            else:
                input_rep += str(k) + ': ' + str(v) + ', '
        input_rep += '}'
        dec = NN(trg_vocab=inp['trg_vocab'],embedding_dim=inp['embedding_dim'],num_heads=inp['num_heads'],num_layers=inp['num_layers'],dim_feedforward=inp['dim_feedforward'],max_len_trg=inp['max_len_trg'],device=inp['device'])
        dec_in = torch.randint(low=0,high=len(inputs[0]['trg_vocab']),size=(inp['max_len_trg'], inp['batch_size']))
        enc_out = torch.rand(inp['max_len_trg'], inp['batch_size'], inp['embedding_dim'])
        inp['encoder_outputs'] = enc_out
        with torch.no_grad():
            stu_out = dec(enc_out=enc_out, dec_in=dec_in)
        del dec
        has_passed = True
        if not torch.is_tensor(stu_out):
            has_passed = False
            msg = 'Output must be a torch.Tensor; received ' + str(type(stu_out))
        status = 'PASSED' if has_passed else 'FAILED'
        if not has_passed:
            message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape (dec_in): ' + str(dec_in.shape) + '\tExpected Output Shape: ' + str(expected_outputs[i]) + '\t' + msg
            print(message)
            continue

        has_passed = stu_out.size() == expected_outputs[i]
        msg = 'Your Output Shape: ' + str(stu_out.size())
        status = 'PASSED' if has_passed else 'FAILED'
        message = '\t' + status + "\t Init Input: " + input_rep + '\tForward Input Shape (dec_in): ' + str(dec_in.shape) + '\tExpected Output Shape: ' + str(expected_outputs[i]) + '\t' + msg
        print(message)


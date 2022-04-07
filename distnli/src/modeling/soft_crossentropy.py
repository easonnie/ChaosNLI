import torch

def softCrossEntropy(input, target):
    assert(len(input.shape)==2)
    logprobs = torch.nn.functional.log_softmax(input, dim = -1)
    return  -(target * logprobs).sum() / input.shape[0]


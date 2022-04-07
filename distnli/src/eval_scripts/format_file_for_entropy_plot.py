import json
import sys
import numpy as np

def load_json(path):
    with open(path, 'r') as fr:
        result = json.load(fr)
    return result

def write_json(path, d):
    with open(path, 'w') as fw:
        json.dump(d, fw)




def softmax(logits):
    if type(logits) is list:
        logits = np.asarray(logits)
    prob = np.exp(logits) / np.sum(np.exp(logits))
    # normalize
    prob = prob / np.sum(prob)
    assert np.isclose(np.sum(prob), 1)
    return prob



def get_entropy_from_logits(logits):
    p = softmax(logits)
    entropy = - np.sum(np.log2(p) * p)
    return float(entropy)

def get_entropy_from_probs(probs):
    p = np.asarray(probs)
    entropy = - np.sum(np.log2(p) * p)
    return float(entropy)



assert(len(sys.argv)==3)

inp_path = sys.argv[1]

out_path = sys.argv[2]


pred_dict = load_json(inp_path)

pred_dict = pred_dict['roberta-large']

for k in pred_dict.keys():
    if 'logits' in pred_dict[k]:
        logits = pred_dict[k]['logits']
        pred_dict[k]['entropy'] = get_entropy_from_logits(logits)
    else:
        probs = pred_dict[k]['predicted_probabilities']
        pred_dict[k]['entropy'] = get_entropy_from_probs(probs)




write_json(out_path, pred_dict)








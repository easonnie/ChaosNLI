import json
import sys
import numpy as np
import os
from tqdm import tqdm, trange


def read_jsonl(path):
    result = []
    with open(path, 'r') as fr:
        for line in fr.readlines():
            result.append(json.loads(line))
    return result


def write_jsonl(path, dict_lst):
    with open(path, 'w') as fw:
        for d in dict_lst:
            json.dump(d, fw)
            fw.write('\n')



def softmax(logits):
    if type(logits) is list:
        logits = np.asarray(logits)
    prob = np.exp(logits) / np.sum(np.exp(logits))
    # normalize
    prob = prob / np.sum(prob)
    assert np.isclose(np.sum(prob), 1)
    return prob


def write_alpha_dist_label(path, label_lst):
    with open(path, 'w') as fw:
        for label in label_lst:
            label = [str(x) for x in label]
            fw.write('\t'.join(label))
            fw.write('\n')


def T_scaling_lst(logits, temperature):
    if type(logits) is list:
        logits = np.asarray(logits)

    logits =logits / temperature
    return logits.tolist()


assert(len(sys.argv) in [4, 5])
pred_file = sys.argv[1]
ogn_file = sys.argv[2]
out_file = sys.argv[3]

if len(sys.argv) == 5:
    calib_temp = float(sys.argv[4])
else:
    calib_temp = 1


if 'alpha' in ogn_file:
    pred_type = 'alphanli'
elif 'nli' in ogn_file:
    pred_type = 'nli'
else:
    raise NotImplementedError


preds = read_jsonl(pred_file)
ogn_data = read_jsonl(ogn_file)

if pred_type == 'alphanli':
    raise NotImplementedError

elif pred_type == 'nli':
    new_data = []
    if 'gold_label' in ogn_data[0]:
        ogn_data = [d for d in ogn_data if d['gold_label'] not in ['o', 'h', '-']]

    assert(len(preds) == len(ogn_data))
    for pred, ex in tqdm(zip(preds, ogn_data), total=len(preds)):

        assert(pred['uid']==ex['pairID'])
        uid = ex['pairID']
        if calib_temp==1:
            label_dist = softmax(pred['logits']).tolist()
            ex['label_dist'] = label_dist
        else:
            calib_logits = T_scaling_lst(pred['logits'], calib_temp)
            label_dist = softmax(calib_logits).tolist()
            ex['label_dist'] = label_dist

        if 'gold_label' in ex:
            del ex['gold_label']
        new_data.append(ex)

    write_jsonl(out_file, new_data)
else:
    raise NotImplementedError



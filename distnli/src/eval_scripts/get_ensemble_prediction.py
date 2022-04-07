import json
import sys
import os
import numpy as np

nli_idx2label={0:'entailment',
               1:'neutral',
               2:'contradiction'}


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



assert(len(sys.argv) == 3)
# assume each file under that folder is a valid model
inp_folder = sys.argv[1]
out_path = sys.argv[2]


if 'alpha' in inp_folder:
    pred_type = 'alphanli'
elif 'nli' in inp_folder:
    pred_type = 'nli'
else:
    raise NotImplementedError

model_count = 0
pred_dict = {}
pred_dict['roberta-large'] = {}
for filename in os.listdir(inp_folder):
    if filename.endswith(".jsonl"):
        print(filename)
        inp_path = os.path.join(inp_folder, filename)
        model_count += 1


        if pred_type == 'nli':
            jsonl_preds = read_jsonl(inp_path)
            for pred in jsonl_preds:
                uid = pred['uid']
                pred['predicted_label'] = nli_idx2label[pred['predicted_label']]
                if uid not in pred_dict['roberta-large']:
                    pred_dict['roberta-large'][uid] = pred
                if 'accumulated_probabilities' not in pred_dict['roberta-large'][uid]:
                    pred_dict['roberta-large'][uid]['accumulated_probabilities'] = softmax(pred['logits'])
                else:
                    pred_dict['roberta-large'][uid]['accumulated_probabilities'] = pred_dict['roberta-large'][uid]['accumulated_probabilities'] + softmax(pred['logits'])
                pred_dict['roberta-large'][uid].pop('logits', None)

        elif pred_type == 'alphanli':
            jsonl_preds = read_jsonl(inp_path)
            for pred in jsonl_preds:
                uid = pred['uid']
                pred['predicted_label'] += 1
                if uid not in pred_dict['roberta-large']:
                    pred_dict['roberta-large'][uid] = pred
                if 'accumulated_probabilities' not in pred_dict['roberta-large'][uid]:
                    pred_dict['roberta-large'][uid]['accumulated_probabilities'] = softmax(pred['logits'])
                else:
                    pred_dict['roberta-large'][uid]['accumulated_probabilities'] = pred_dict['roberta-large'][uid]['accumulated_probabilities'] + softmax(pred['logits'])
                pred_dict['roberta-large'][uid].pop('logits', None)




        else:
            raise NotImplementedError

for uid in pred_dict['roberta-large'].keys():
    pred_dict['roberta-large'][uid]['predicted_probabilities'] = (pred_dict['roberta-large'][uid]['accumulated_probabilities'] / model_count).tolist()
    pred_dict['roberta-large'][uid].pop('accumulated_probabilities', None)


write_jsonl(out_path, [pred_dict])
import json
import sys

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





assert(len(sys.argv) == 3)
inp_path = sys.argv[1]
out_path = sys.argv[2]


if 'alpha' in inp_path:
    pred_type = 'alphanli'
elif 'nli' in inp_path:
    pred_type = 'nli'
else:
    raise NotImplementedError


if pred_type == 'nli':
    jsonl_preds = read_jsonl(inp_path)
    pred_dict = {}
    pred_dict['roberta-large'] = {}
    for pred in jsonl_preds:
        uid = pred['uid']
        pred['predicted_label'] = nli_idx2label[pred['predicted_label']]
        pred_dict['roberta-large'][uid] = pred
    write_jsonl(out_path, [pred_dict])

elif pred_type == 'alphanli':
    jsonl_preds = read_jsonl(inp_path)
    pred_dict = {}
    pred_dict['roberta-large'] = {}
    for pred in jsonl_preds:
        uid = pred['uid']
        pred['predicted_label'] += 1
        pred_dict['roberta-large'][uid] = pred
    write_jsonl(out_path, [pred_dict])

else:
    raise NotImplementedError
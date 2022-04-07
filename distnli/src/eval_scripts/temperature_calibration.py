import torch
import sys
import numpy as np
import json
from torch.nn import functional as F
import numpy as np

HARD_LABEL=False




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





def T_scaling(logits, args):
  temperature = args.get('temperature', None)
  return torch.div(logits, temperature)

def T_scaling_lst(logits, temperature):
    if type(logits) is list:
        logits = np.asarray(logits)

    logits =logits / temperature
    return logits.tolist()


label_tag_to_id = {1:0,
                   2:1,
                   'e':0,
                   'n':1,
                   'c':2}



logits_list = []
labels_list = []
temps = []
losses = []

# read logits and labels
assert(len(sys.argv) in [4,5])
inp_path = sys.argv[1]
label_path = sys.argv[2]

out_path = sys.argv[3]
if len(sys.argv) == 5:
    final_temp = float(sys.argv[4])
    inp_dict = read_jsonl(inp_path)
else:


    inp_dict = read_jsonl(inp_path)
    label_dict = read_jsonl(label_path)
    for label_info in label_dict:
        uid = label_info['uid']
        if 'logits' in inp_dict[0]['roberta-large'][uid]:
            logits = inp_dict[0]['roberta-large'][uid]['logits']
        else:
            probs = inp_dict[0]['roberta-large'][uid]['predicted_probabilities']
            logits = np.log(probs).tolist()
            inp_dict[0]['roberta-large'][uid]['logits'] = logits
            del inp_dict[0]['roberta-large'][uid]['predicted_probabilities']
        label_dist = label_info['label_dist']
        if HARD_LABEL:
            if 'majority_label' in label_info:
                hard_label_idx = label_tag_to_id[label_info['majority_label']]
            else:
                hard_label_idx = np.argmax(label_dist)
            label_dist = np.zeros(len(label_dist))
            label_dist[hard_label_idx] = 1
            label_dist = label_dist.tolist()
        logits_list.append(torch.FloatTensor(logits))
        labels_list.append(torch.FloatTensor(label_dist))




    # calibration
    temperature = torch.nn.Parameter(torch.ones(1).cuda())
    args = {'temperature': temperature}
    criterion = torch.nn.KLDivLoss(reduction='batchmean')


    optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

    # Create tensors
    logits_list = torch.stack(logits_list,0).cuda()
    labels_list = torch.stack(labels_list,0).cuda()

    def _eval():
      loss = criterion(F.log_softmax(T_scaling(logits_list, args), -1), labels_list)
      loss.backward()
      temps.append(temperature.item())
      losses.append(loss)
      return loss


    optimizer.step(_eval)


    final_temp = temperature.item()
print(final_temp)

for uid in inp_dict[0]['roberta-large'].keys():
    if 'logits' not in inp_dict[0]['roberta-large'][uid]:
        probs = inp_dict[0]['roberta-large'][uid]['predicted_probabilities']
        logits = np.log(probs).tolist()
        inp_dict[0]['roberta-large'][uid]['logits'] = logits
        del inp_dict[0]['roberta-large'][uid]['predicted_probabilities']

    inp_dict[0]['roberta-large'][uid]['logits'] = T_scaling_lst(inp_dict[0]['roberta-large'][uid]['logits'], final_temp)

write_jsonl(out_path, inp_dict)



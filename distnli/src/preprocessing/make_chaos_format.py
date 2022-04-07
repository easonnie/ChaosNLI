import json
import numpy as np
import sys

if sys.argv[1] == 'unli':
    inp_file = 'unli_filtered.jsonl'
    out_file = 'unli_filtered_chaos.jsonl'
elif sys.argv[1] == 'pk':
    inp_file = 'pk_filtered.jsonl'
    out_file = 'pk_filtered_chaos.jsonl'
else:
    raise ValueError

with open(inp_file, 'r') as fr, open(out_file, 'w') as fw:
    for line in fr.readlines():
        ex = json.loads(line)

        new_ex = ex

        new_ex['uid'] = new_ex['pairID']

        #placeholder attribute
        if sys.argv[1] == 'pk':
            major_label_id = np.argmax(new_ex['label_dist'])
            if major_label_id == 0:
                new_ex['old_label'] = new_ex['majority_label'] = 'e'
            elif major_label_id == 1:
                new_ex['old_label'] = new_ex['majority_label'] = 'n'
            elif major_label_id == 2:
                new_ex['old_label'] = new_ex['majority_label'] = 'c'

        json.dump(new_ex, fw)
        fw.write('\n')


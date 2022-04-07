import json
import csv
import numpy as np

pk_file = 'dev.csv'

out_file = 'unli_clean.jsonl'

out_filtered_file = 'unli_filtered.jsonl'

clean_examples = []

with open(pk_file, 'r') as fr:
    csv_reader = csv.reader(fr, delimiter=',')
    for i, row in enumerate(csv_reader):
        assert(len(row)==5)
        if i == 0:
            continue

        id, pre, hyp, nli_label, unli_label = row
        ex = {'sentence1':pre,
              'sentence2':hyp,
              'pairID': 'unli_' + str(id),
              'gold_label': 'entailment', #fake label
              'unli_label':unli_label}
        clean_examples.append(ex)

with open(out_file, 'w') as fw:
    for ex in clean_examples:
        json.dump(ex, fw)
        fw.write('\n')



calib_example_path = '../../data/chaosNLI_v1.0/snli100.jsonl'
premise_dict = {}
with open(calib_example_path, 'r') as fr:
    for line in fr.readlines():
        ex = json.loads(line)
        ex = ex['example']
        premise_dict[ex['premise']] = ex['hypothesis']


with open(out_filtered_file, 'w') as fw:
    for ex in clean_examples:
        if ex['sentence1'] in premise_dict:
            if ex['sentence2'] == premise_dict[ex['sentence1']]:
                continue
#            else:
#                print(ex)
        json.dump(ex, fw)
        fw.write('\n')


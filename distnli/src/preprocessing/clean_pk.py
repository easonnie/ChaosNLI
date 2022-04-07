import json
import numpy as np

pk_file = 'preprocessed-data.jsonl'

out_file = 'pk_clean.jsonl'

out_filtered_file = 'pk_filtered.jsonl'

clean_examples = []

with open(pk_file, 'r') as fr:
    lines = fr.readlines()

for line in lines:
    old_ex = json.loads(line)
    ex = {'sentence1':None,
          'sentence2':None,
          'pairID':None,
          'label_dist':None}
    ex['sentence1'] = old_ex['premise']
    ex['sentence2'] = old_ex['hypothesis']
    ex['pairID'] = old_ex['task'] + '_' + old_ex['id']
    
    pk_labels = old_ex['labels']

    label_counts = [0, 0, 0]

    for label in pk_labels:
        if label < -16.7:
            # contradiction
            label_counts[2] += 1
        elif label > 16.7:
            # entailment
            label_counts[0] += 1
        else:
            # neutral
            label_counts[1] += 1

    label_dist = np.array(label_counts) / np.sum(label_counts) 
    label_dist = label_dist.tolist()

    ex['label_dist'] = label_dist
    clean_examples.append(ex)

with open(out_file, 'w') as fw:
    for ex in clean_examples:
        json.dump(ex, fw)
        fw.write('\n')


with open(out_filtered_file, 'w') as fw:
    for ex in clean_examples:
        if 'snli' not in ex['pairID']  and 'mnli' not in ex['pairID']:
            json.dump(ex, fw)
            fw.write('\n')




import json
import sys



full_label_name_dict = {'n': 'neutral',
                        'c': 'contradiction',
                        'e': 'entailment'}

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

def write_alpha_hard_label(path, label_lst):
    with open(path, 'w') as fw:
        for label in label_lst:
            if label[0] > label[1]:
                fw.write('1')
            else:
                fw.write('2')
            fw.write('\n')


def write_alpha_dist_label(path, label_lst):
    with open(path, 'w') as fw:
        for label in label_lst:
            label = [str(x) for x in label]
            fw.write('\t'.join(label))
            fw.write('\n')


assert(len(sys.argv) in  [3,4])
inp_path = sys.argv[1]
out_path = sys.argv[2]
if len(sys.argv) == 4:
    assert(sys.argv[3] == 'hard')
    hard_label=True
else:
    hard_label=False


if 'alpha' in inp_path:
    file_type = 'alphanli'
elif 'snli' in inp_path:
    file_type = 'snli'
elif 'mnli' in inp_path:
    file_type = 'mnli'
else:
    raise NotImplementedError



if file_type == 'alphanli':
    chaos_example_lst = read_jsonl(inp_path)
    train_example_lst = []
    label_lst = []
    for ex in chaos_example_lst:
        train_example = ex['example']
        train_example['story_id'] = train_example['uid']
        del train_example['uid']

        label = ex['label_dist']

        train_example_lst.append(train_example)
        label_lst.append(label)

    assert(out_path[-6:] == '.jsonl')
    label_out_path = out_path[:-6] + '-labels.lst'
    write_jsonl(out_path, train_example_lst)
    if not hard_label:
        write_alpha_dist_label(label_out_path, label_lst)
    else:
        write_alpha_hard_label(label_out_path, label_lst)
elif file_type == 'snli':
    chaos_example_lst = read_jsonl(inp_path)
    train_example_lst = []
    label_lst = []
    for ex in chaos_example_lst:
        train_example = ex['example']
        train_example['pairID'] = train_example['uid']
        del train_example['uid']

        train_example['sentence1'], train_example['sentence2'] = train_example['premise'], train_example['hypothesis']
        del train_example['premise']
        del train_example['hypothesis']

        if not hard_label:
            train_example['label_dist'] = ex['label_dist']
        else:
            train_example['gold_label'] = full_label_name_dict[ex['majority_label']]

        train_example_lst.append(train_example)

    write_jsonl(out_path, train_example_lst)
elif file_type == 'mnli':
    chaos_example_lst = read_jsonl(inp_path)
    train_example_lst = []
    label_lst = []
    for ex in chaos_example_lst:
        train_example = ex['example']
        train_example['pairID'] = train_example['uid']
        del train_example['uid']

        train_example['sentence1'], train_example['sentence2'] = train_example['premise'], train_example['hypothesis']
        del train_example['premise']
        del train_example['hypothesis']

        if not hard_label:
            train_example['label_dist'] = ex['label_dist']
        else:
            train_example['gold_label'] = full_label_name_dict[ex['majority_label']]

        train_example_lst.append(train_example)

    write_jsonl(out_path, train_example_lst)
else:
    raise NotImplementedError
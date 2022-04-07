from utils import common, list_dict_data_tool
import config
import uuid


abductive_build_data = {

}

convert_label = {
    1: 0,
    2: 1,
    -1: -1,
}


def build_abdnli_jsonl(tag='dev'):
    if tag in 'dev':
        path = f"data/alphaNLI/anli/switcheddev.jsonl"
    elif tag in 'test':
        path = f"data/alphaNLI/anli/switchedtest.jsonl"
    elif tag in ['train', 'relabel', 'softtrain', 'nocalib_relabel']:
        path = f"data/alphaNLI/anli/{tag}.jsonl"
    elif 'relabel' in tag:
        path = f"data/alphaNLI/anli/{tag}.jsonl"
    else:
        raise NotImplementedError
    anli_dev_l = common.load_jsonl(config.PRO_ROOT / path)

    # for item in anli_dev_l:
    #     print(item)

    dev_labels = []
    if tag in ['dev', 'test']:
        with open(config.PRO_ROOT / f"data/alphaNLI/anli/switched{tag}-labels.lst", mode='r',
                  encoding='utf-8') as in_f:
            for line in in_f:
                line = line.strip()
                dev_labels.append(int(line))
    elif tag in ['train']:
        with open(config.PRO_ROOT / f"data/alphaNLI/anli/{tag}-labels.lst", mode='r',
                  encoding='utf-8') as in_f:
            for line in in_f:
                line = line.strip()
                dev_labels.append(int(line))
    elif (tag in ['relabel', 'softtrain', 'nocalib_relabel']) or ('relabel' in tag):
        with open(config.PRO_ROOT / f"data/alphaNLI/anli/{tag}-labels.lst", mode='r',
                  encoding='utf-8') as in_f:
            for line in in_f:
                line = line.strip()
                dev_labels.append([float(x) for x in line.split('\t')])
    else:
        for _ in range(len(anli_dev_l)):
            dev_labels.append(-1)


    for item, label in zip(anli_dev_l, dev_labels):
        item.update({'label': label})


    return anli_dev_l


def convert_to_std_format(data_list, check_unique=False):
    output_list = []
    # duplicate check
    uid_set = set()

    for item in data_list:
        # uid, query_1, answer_1, query_2, answer_2, ...
        # uid = str(uuid.uuid4())
        uid = item['story_id']
        story_id = item['story_id']
        uid_set.add(uid)
        query_0 = item["obs1"] + ' ' + item["hyp1"]
        answer_0 = item["obs2"]
        query_1 = item["obs1"] + ' ' + item["hyp2"]
        answer_1 = item["obs2"]
        old_label = item['label']
        if type(item['label']) == int:
            label = convert_label[item['label']]
        else:
            label = item['label']

        out_dict = {
            'uid': uid,
            'story_id': story_id,
            'query_0': query_0,
            'answer_0': answer_0,
            'query_1': query_1,
            'answer_1': answer_1,
            'old_label': old_label,
            'label': label,
        }

        output_list.append(out_dict)

    if check_unique:
        assert len(output_list) == len(uid_set)

    return output_list


def build(tag_name=None):
    if tag_name is None:
        if len(abductive_build_data) == 0:
            abductive_build_data['train'] = convert_to_std_format(build_abdnli_jsonl('train'))
            abductive_build_data['dev'] = convert_to_std_format(build_abdnli_jsonl('dev'), check_unique=True)
            abductive_build_data['test'] = convert_to_std_format(build_abdnli_jsonl('test'), check_unique=True)

            abductive_build_data['relabel'] = convert_to_std_format(build_abdnli_jsonl('relabel'))
            abductive_build_data['softtrain'] = convert_to_std_format(build_abdnli_jsonl('softtrain'))
            abductive_build_data['nocalib_relabel'] = convert_to_std_format(build_abdnli_jsonl('nocalib_relabel'))
    else:
        abductive_build_data[tag_name] = convert_to_std_format(build_abdnli_jsonl(tag_name))
    return abductive_build_data


def get_data(tag_name):
    build(tag_name)
    return abductive_build_data[tag_name]



if __name__ == '__main__':
    dev_list = get_data('train')
    train_list = get_data('dev')
    test_list = get_data('test')

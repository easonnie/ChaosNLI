from collections import defaultdict

from utils import common, list_dict_data_tool
import config
import uuid


snli_build_data = {

}

mnli_build_data = {

}

smnli_label2std_label = defaultdict(lambda: "o")  # o stands for all other label that is invalid.
smnli_label2std_label.update({
    "entailment": "e",
    "neutral": "n",
    "contradiction": "c",
    "hidden": "h",
})


def build_nli_jsonl(type='snli', tag='dev'):
    # data_list = []
    
    #generalization data
    if tag in ['unli', 'pk']:
        if tag == 'unli':
            path = config.PRO_ROOT / f"data/generalization/unli_filtered.jsonl"
        elif tag == 'pk':
            path = config.PRO_ROOT / f"data/generalization/pk_filtered.jsonl"
        else:
            raise ValueError()
    else:


        if type == 'snli':
            if "relabel" in tag:
                path = config.PRO_ROOT / f"data/snli_1.0/{tag}.jsonl"
            else:
                path = config.PRO_ROOT / f"data/snli_1.0/snli_1.0_{tag}.jsonl"
        elif type == 'mnli':
            if tag == 'train':
                path = config.PRO_ROOT / f"data/multinli_1.0/multinli_1.0_train.jsonl"
            elif tag == 'dev_m':
                path = config.PRO_ROOT / f"data/multinli_1.0/multinli_1.0_dev_matched.jsonl"
            elif tag == 'dev_mm':
                path = config.PRO_ROOT / f"data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl"
            elif "relabel" in tag:
                path = config.PRO_ROOT / f"data/multinli_1.0/{tag}.jsonl"
            else:
                raise ValueError()
        else:
            raise ValueError()

    data_list = common.load_jsonl(path)

    return data_list


def convert_to_std_format(data_list, check_unique=False, gold_label_only=False):
    output_list = []
    # duplicate check
    uid_set = set()

    for item in data_list:
        uid = item['pairID']
        if check_unique and uid in uid_set:
            print(uid)
        uid_set.add(uid)
        premise: str = item["sentence1"]
        hypothesis: str = item["sentence2"]
        if "label_dist" in item:
            label = item["label_dist"]
        else:
            label: str = smnli_label2std_label[item["gold_label"]]

        if gold_label_only:
            if label == "o" or label == "h":
                continue

        out_dict = {
            'uid': uid,
            'premise': premise,
            'hypothesis': hypothesis,
            'label': label,
        }

        output_list.append(out_dict)

    if check_unique:
        assert len(output_list) == len(uid_set)

    return output_list


def build(type=None, tag_name=None):
    if tag_name is None:
        if len(snli_build_data) == 0:
            snli_build_data['train'] = convert_to_std_format(build_nli_jsonl('snli', 'train'), gold_label_only=True)
            snli_build_data['dev'] = convert_to_std_format(build_nli_jsonl('snli', 'dev'), check_unique=True)
            snli_build_data['test'] = convert_to_std_format(build_nli_jsonl('snli', 'test'), check_unique=True)

        if len(mnli_build_data) == 0:
            mnli_build_data['dev_mm'] = convert_to_std_format(build_nli_jsonl('mnli', 'dev_mm'), check_unique=False)
            mnli_build_data['train'] = convert_to_std_format(build_nli_jsonl('mnli', 'train'), gold_label_only=True)
            mnli_build_data['dev_m'] = convert_to_std_format(build_nli_jsonl('mnli', 'dev_m'), check_unique=True)
    else:
        if "train" in tag_name or "relabel" in tag_name or tag_name == 'dev_mm':
            if type=='snli':
                snli_build_data[tag_name] = convert_to_std_format(build_nli_jsonl('snli', tag_name), gold_label_only=True)
            elif type =='mnli':
                mnli_build_data[tag_name] = convert_to_std_format(build_nli_jsonl('mnli', tag_name), gold_label_only=True)
        else:
            if type=='snli':
                snli_build_data[tag_name] = convert_to_std_format(build_nli_jsonl('snli', tag_name), check_unique=True)
            elif type =='mnli':
                mnli_build_data[tag_name] = convert_to_std_format(build_nli_jsonl('mnli', tag_name), check_unique=True)

    return snli_build_data, mnli_build_data


def get_data(type='snli', tag_name='train'):
    build(type, tag_name)
    if type == 'snli':
        return snli_build_data[tag_name]
    elif type == 'mnli':
        return mnli_build_data[tag_name]
    else:
        raise ValueError()



if __name__ == '__main__':
    train_list = get_data(type='snli', tag_name='train')
    dev_list = get_data(type='snli', tag_name='dev')
    test_list = get_data(type='snli', tag_name='test')
    #
    print(len(train_list), len(dev_list), len(test_list))

    train_list = get_data(type='mnli', tag_name='train')
    dev_m = get_data(type='mnli', tag_name='dev_m')
    #
    print(len(train_list), len(dev_m))


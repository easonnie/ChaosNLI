from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler

from flint.data_utils.fields import RawFlintField, LabelFlintField, ArrayIndexFlintField


nli_label2index = {
    'e': 0,
    'n': 1,
    'c': 2,
    'h': -1,
    'o': -2,
}


class ListDataSet(Dataset):
    def __init__(self, data_list, transform) -> None:
        super().__init__()
        self.d_list = data_list
        self.len = len(self.d_list)
        self.transform = transform

    def __getitem__(self, index: int):
        return self.transform(self.d_list[index])

    # you should write schema for each of the input elements

    def __len__(self) -> int:
        return self.len


class SeqClassificationTransform(object):
    def __init__(self, model_name, tokenizer, max_length=None):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sample):
        processed_sample = dict()
        processed_sample['uid'] = sample['uid']
        processed_sample['gold_label'] = sample['label']
        
        if type(sample['label']) is str:
            processed_sample['y'] = nli_label2index[sample['label']]
        else:
            processed_sample['y'] = sample['label']

        # premise: str = sample['premise']
        premise: str = sample['context'] if 'context' in sample else sample['premise']
        hypothesis: str = sample['hypothesis']

        if premise.strip() == '':
            premise = 'empty'

        if hypothesis.strip() == '':
            hypothesis = 'empty'

        tokenized_input_seq_pair = self.tokenizer.encode_plus(premise, hypothesis,
                                                              max_length=self.max_length,
                                                              return_token_type_ids=True, truncation=True)

        processed_sample.update(tokenized_input_seq_pair)

        return processed_sample

    def get_batching_schema(self, padding_token_value, padding_segement_value, padding_att_value, left_pad):
        batching_schema = {
            'uid': RawFlintField(),
            'y': LabelFlintField(),
            'input_ids': ArrayIndexFlintField(pad_idx=padding_token_value, left_pad=left_pad),
            'token_type_ids': ArrayIndexFlintField(pad_idx=padding_segement_value, left_pad=left_pad),
            'attention_mask': ArrayIndexFlintField(pad_idx=padding_att_value, left_pad=left_pad),
        }

        return batching_schema



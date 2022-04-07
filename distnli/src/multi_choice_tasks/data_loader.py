from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler

from flint.data_utils.fields import RawFlintField, LabelFlintField, ArrayIndexFlintField


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


class MultiChoiceTransform(object):
    def __init__(self, model_name, tokenizer, number_of_choice, max_length=None, with_element=False):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.number_of_choice = number_of_choice
        self.max_length = max_length
        self.with_element = with_element

    def __call__(self, sample):
        # field needed uid
        # target_label_name
        # query_i, answer_i     the input should have the same format
        processed_sample = dict()
        processed_sample['uid'] = sample['uid']
        processed_sample['y'] = sample['label']  # remember to convert the label

        for i in range(self.number_of_choice):
            assert f'query_{i}' in sample
            assert f'answer_{i}' in sample
            current_seq_pair = self.tokenizer.encode_plus(sample[f'query_{i}'], sample[f'answer_{i}'],
                                                          max_length=self.max_length,
                                                          return_token_type_ids=True, truncation=True)

            processed_sample[f'input_ids_{i}'] = current_seq_pair['input_ids']
            processed_sample[f'attention_mask_{i}'] = current_seq_pair['attention_mask']
            if 'token_type_ids' in current_seq_pair:
                processed_sample[f'token_type_ids_{i}'] = current_seq_pair['token_type_ids']

        if self.with_element:
            processed_sample['element'] = sample

        return processed_sample

    def get_batching_schema(self, padding_token_value, padding_segement_value, padding_att_value, left_pad):
        batching_schema = {
            'uid': RawFlintField(),
            'y': LabelFlintField(),
            # 'input_ids_1': ArrayIndexFlintField(pad_idx=padding_token_value, left_pad=left_pad),
            # 'token_type_ids_1': ArrayIndexFlintField(pad_idx=padding_segement_value, left_pad=left_pad),
            # 'attention_mask_1': ArrayIndexFlintField(pad_idx=padding_att_value, left_pad=left_pad),
        }

        for i in range(self.number_of_choice):
            batching_schema[f'input_ids_{i}'] = ArrayIndexFlintField(pad_idx=padding_token_value, left_pad=left_pad)
            batching_schema[f'token_type_ids_{i}'] = ArrayIndexFlintField(pad_idx=padding_segement_value,
                                                                          left_pad=left_pad)
            batching_schema[f'attention_mask_{i}'] = ArrayIndexFlintField(pad_idx=padding_att_value,
                                                                          left_pad=left_pad)

        return batching_schema



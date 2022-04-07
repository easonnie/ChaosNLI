import argparse
from pathlib import Path
import copy

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BartTokenizer, BartForSequenceClassification

from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import config
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from flint.data_utils.batchbuilder import BaseBatchBuilder, move_to_device
from flint.data_utils.fields import RawFlintField, LabelFlintField, ArrayIndexFlintField
from utils import common, list_dict_data_tool, save_tool
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn

from torch.nn import CrossEntropyLoss

import numpy as np
import random
import torch
from tqdm import tqdm

import pprint

pp = pprint.PrettyPrinter(indent=2)


# from fairseq.data.data_utils import collate_tokens

class MultichoiceModel(nn.Module):
    def __init__(self, encoder):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultichoiceModel, self).__init__()
        self.encoder = encoder

    def forward(self,
                input_ids_1=None,
                attention_mask_1=None,
                token_type_ids_1=None,
                position_ids_1=None,
                head_mask_1=None,
                inputs_embeds_1=None,
                input_ids_2=None,
                attention_mask_2=None,
                token_type_ids_2=None,
                position_ids_2=None,
                head_mask_2=None,
                inputs_embeds_2=None,
                labels=None,
                with_token_type=True,
                ):
        # assert self.encoder.num_labels == 1  # only allow encoder to output one logits.

        # if token_type_ids_1 is None:
        if with_token_type:
            out_1 = self.encoder(
                input_ids=input_ids_1,
                attention_mask=attention_mask_1,
                token_type_ids=token_type_ids_1,
                # position_ids_1,
                # head_mask_1,
                # inputs_embeds_1
            )
            logits_1 = out_1[0]  # B, logits

            out_2 = self.encoder(
                input_ids=input_ids_2,
                attention_mask=attention_mask_2,
                token_type_ids=token_type_ids_2,
                # position_ids_2,
                # head_mask_2,
                # inputs_embeds_2,
            )
            logits_2 = out_2[0]  # B, logits
        else:
            out_1 = self.encoder(
                input_ids=input_ids_1,
                attention_mask=attention_mask_1,
                # position_ids_1,
                # head_mask_1,
                # inputs_embeds_1
            )
            logits_1 = out_1[0]  # B, logits

            out_2 = self.encoder(
                input_ids=input_ids_2,
                attention_mask=attention_mask_2,
                # position_ids_2,
                # head_mask_2,
                # inputs_embeds_2,
            )
            logits_2 = out_2[0]  # B, logits

        combined_logits = torch.stack(
            [logits_1.squeeze(1), logits_2.squeeze(1)], dim=1)  # think about it, it might not be the shape we want it to be

        outputs = (combined_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(combined_logits.view(-1, 2), labels.view(-1))  # hard code 2 here.

        outputs = (loss,) + outputs

        return outputs


MODEL_CLASSES = {
    "bert-base": {
        "model_name": "bert-base-uncased",
        "tokenizer": BertTokenizer,
        "sequence_classification": BertForSequenceClassification,
        # "padding_token_value": 0,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "do_lower_case": True,
    },
    "bert-large": {
        "model_name": "bert-large-uncased",
        "tokenizer": BertTokenizer,
        "sequence_classification": BertForSequenceClassification,
        # "padding_token_value": 0,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "do_lower_case": True,
    },

    "xlnet-base": {
        "model_name": "xlnet-base-cased",
        "tokenizer": XLNetTokenizer,
        "sequence_classification": XLNetForSequenceClassification,
        # "padding_token_value": 0,
        "padding_segement_value": 4,
        "padding_att_value": 0,
        "left_pad": True,
    },
    "xlnet-large": {
        "model_name": "xlnet-large-cased",
        "tokenizer": XLNetTokenizer,
        "sequence_classification": XLNetForSequenceClassification,
        "padding_segement_value": 4,
        "padding_att_value": 0,
        "left_pad": True,
    },

    "roberta-base": {
        "model_name": "roberta-base",
        "tokenizer": RobertaTokenizer,
        "sequence_classification": RobertaForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
    },
    "roberta-large": {
        "model_name": "roberta-large",
        "tokenizer": RobertaTokenizer,
        "sequence_classification": RobertaForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
    },

    "albert-xxlarge": {
        "model_name": "albert-xxlarge-v2",
        "tokenizer": AlbertTokenizer,
        "sequence_classification": AlbertForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
    },

    "distilbert": {
        "model_name": "distilbert-base-cased",
        "tokenizer": DistilBertTokenizer,
        "sequence_classification": DistilBertForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
    },

    "bart-large": {
        "model_name": "bart-large",
        "tokenizer": BartTokenizer,
        "sequence_classification": BartForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
    },
}

abdnli_label2index = {
    1: 0,
    2: 1,
    '-': -1,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


class AdbNLITransform(object):
    def __init__(self, model_name, tokenizer, max_length=None):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sample):
        processed_sample = dict()
        processed_sample['uid'] = sample['story_id']
        processed_sample['gold_label'] = sample['label']
        processed_sample['y'] = abdnli_label2index[sample['label']]  # 1: 0, 2: 1
        # length = random.randint(1, 10)
        # processed_sample['input'] = torch.tensor(np.random.randint(0, 100, length))
        sample_obs_hyp_1 = sample["obs1"] + ' ' + sample["hyp1"]
        sample_obs_hyp_2 = sample["obs1"] + ' ' + sample["hyp2"]

        tokenized_input_seq_pair_1 = self.tokenizer.encode_plus(sample_obs_hyp_1, sample["obs2"],
                                                                max_length=self.max_length)
        tokenized_input_seq_pair_2 = self.tokenizer.encode_plus(sample_obs_hyp_2, sample["obs2"],
                                                                max_length=self.max_length)

        processed_sample['input_ids_1'] = tokenized_input_seq_pair_1['input_ids']
        processed_sample['attention_mask_1'] = tokenized_input_seq_pair_1['attention_mask']
        if 'token_type_ids' in tokenized_input_seq_pair_1:
            processed_sample['token_type_ids_1'] = tokenized_input_seq_pair_1['token_type_ids']

        processed_sample['input_ids_2'] = tokenized_input_seq_pair_2['input_ids']
        processed_sample['attention_mask_2'] = tokenized_input_seq_pair_2['attention_mask']
        if 'token_type_ids' in tokenized_input_seq_pair_2:
            processed_sample['token_type_ids_2'] = tokenized_input_seq_pair_2['token_type_ids']

        return processed_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="If set, we only use CPU.")
    parser.add_argument("--single_gpu", action="store_true", help="If set, we only use single GPU.")
    parser.add_argument("--fp16", action="store_true", help="If set, we will use fp16.")

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    # environment arguments
    parser.add_argument('-s', '--seed', default=1, type=int, metavar='N',
                        help='manual random seed')
    parser.add_argument('-n', '--num_nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('-g', '--gpus_per_node', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--node_rank', default=0, type=int,
                        help='ranking within the nodes')

    # experiments specific arguments
    parser.add_argument('--debug_mode',
                        action='store_true',
                        dest='debug_mode',
                        help='weather this is debug mode or normal')

    # parser.add_argument(
    #     "--experiment_name",
    #     type=str,
    #     help="Set the name of the experiment. The name will mostly be used for logging.",
    # )
    parser.add_argument(
        "--model_class_name",
        type=str,
        help="Set the model class of the experiment.",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Set the name of the experiment. [model_name]/[data]/[task]/[other]",
    )

    parser.add_argument(
        "--save_prediction",
        action='store_true',
        dest='save_prediction',
        help='Do we want to save prediction')
    parser.add_argument(
        "--save_checkpoint",
        action='store_true',
        dest='save_checkpoint',
        help='Do we want to save checkpoints')

    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--total_step', default=-1, type=int, metavar='N',
                        help='number of step to update, default calculate with total data size.'
                             'if we set this step, then epochs will be 100 to run forever.')

    parser.add_argument('--number_example', default=-1, type=int, metavar='N',
                        help='number_example we used for training.')

    parser.add_argument(
        "--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument("--max_length", default=160, type=int, help="Max length of the sequences.")

    parser.add_argument("--warmup_steps", default=-1, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument("--downsample_ratio", default=1.0, type=float, help="Downsample NLI training set.")

    parser.add_argument(
        "--eval_frequency", default=1000, type=int, help="set the evaluation frequency, evaluate every X global step.",
    )

    args = parser.parse_args()

    if args.cpu:
        args.world_size = 1
        train(-1, args)
    elif args.single_gpu:
        args.world_size = 1
        train(0, args)
    else:  # distributed multiGPU training
        #########################################################
        args.world_size = args.gpus_per_node * args.num_nodes  #
        os.environ['MASTER_ADDR'] = '111.111.111.111'  
        # maybe we will automatically retrieve the IP later.
        os.environ['MASTER_PORT'] = '88888'  #
        mp.spawn(train, nprocs=args.gpus_per_node, args=(args,))  # spawn how many process in this node
        # remember train is called as train(i, args).
        #########################################################


def filter_gold_label(d_list):
    filtered_list = []
    for item in d_list:
        if item['gold_label'] != '-':
            filtered_list.append(item)
    return filtered_list


def train(local_rank, args):
    # debug = False
    # print("GPU:", gpu)
    # world_size = args.world_size
    args.global_rank = args.node_rank * args.gpus_per_node + local_rank
    args.local_rank = local_rank
    # args.warmup_steps = 20
    debug_count = 1000

    if args.total_step > 0:
        num_epoch = 10000     # if we set total step, num_epoch will be forever.
    else:
        num_epoch = args.epochs

    actual_train_batch_size = args.world_size * args.per_gpu_train_batch_size * args.gradient_accumulation_steps
    args.actual_train_batch_size = actual_train_batch_size

    assert args.downsample_ratio >= 0.0
    assert args.downsample_ratio <= 1.0

    set_seed(args.seed)
    num_labels = 1  # remember here is 1

    max_length = args.max_length

    model_class_item = MODEL_CLASSES[args.model_class_name]
    model_name = model_class_item['model_name']
    do_lower_case = model_class_item['do_lower_case'] if 'do_lower_case' in model_class_item else False

    tokenizer = model_class_item['tokenizer'].from_pretrained(model_name,
                                                              cache_dir=str(config.PRO_ROOT / "trans_cache"),
                                                              do_lower_case=do_lower_case)

    encoder_model = model_class_item['sequence_classification'].from_pretrained(model_name,
                                                                                cache_dir=str(
                                                                                    config.PRO_ROOT / "trans_cache"),
                                                                                num_labels=num_labels)

    model = MultichoiceModel(encoder_model)

    padding_token_value = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    padding_segement_value = model_class_item["padding_segement_value"]
    padding_att_value = model_class_item["padding_att_value"]
    left_pad = model_class_item['left_pad'] if 'left_pad' in model_class_item else False

    # args.weight_decay = 0.0
    # args.learning_rate = 5e-5
    # args.adam_epsilon = 1e-8

    batch_size_per_gpu_train = args.per_gpu_train_batch_size
    batch_size_per_gpu_eval = args.per_gpu_eval_batch_size

    if not args.cpu and not args.single_gpu:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.global_rank
        )

    o_abdnli_train = common.load_jsonl(config.PRO_ROOT / "data/abductive_commonsense/p_anli/train.jsonl")
    o_abdnli_dev = common.load_jsonl(config.PRO_ROOT / "data/abductive_commonsense/p_anli/dev.jsonl")
    o_abdnli_test = common.load_jsonl(config.PRO_ROOT / "data/abductive_commonsense/p_anli/test.jsonl")

    # change this later to any model tokenizer.
    # tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=str(config.PRO_ROOT / "trans_cache"))
    random.shuffle(o_abdnli_train)

    def equally_sample_examples(training_list, num_of_examples, seed):
        assert num_of_examples / 2 == num_of_examples // 2     # make sure number of examples are balances
        one_list = []
        two_list = []
        for item in training_list:
            if item['label'] == 1:
                one_list.append(item)
            elif item['label'] == 2:
                two_list.append(item)

        assert len(one_list) + len(two_list) == len(training_list)
        balanced_count = num_of_examples // 2
        random.seed(seed)
        random.shuffle(one_list)
        random.shuffle(two_list)
        sampled_list = one_list[:balanced_count] + two_list[:balanced_count]
        random.shuffle(sampled_list)

        return sampled_list

    def increase_list_size_to_match_batch(train_list, the_batch_size, match_multiplication=3):
        base_list = copy.deepcopy(train_list)
        target_size = the_batch_size * match_multiplication
        out_list = train_list
        while len(out_list) < target_size:
            out_list.extend(base_list)

        return out_list

    # sampled_o_train = o_abdnli_train[:int(len(o_abdnli_train) * args.downsample_ratio)]  # default = 1.0
    sampled_o_train = equally_sample_examples(o_abdnli_train, args.number_example, args.seed)  # default = 1.0
    sampled_o_train = increase_list_size_to_match_batch(sampled_o_train, actual_train_batch_size)  # default = 1.0

    if args.total_step > 0:
        if (len(sampled_o_train) / actual_train_batch_size) * args.epochs > args.total_step:
            # if total training step greater than total step then we want to finished the training.
            num_epoch = args.epochs
            args.total_step = -1    # we don't set total step.
            args.eval_frequency = (len(sampled_o_train) // actual_train_batch_size) // 8
            # reset evaluation frequence to at least 5 times per epoch.
        else:
            pass

    nli_transformer = AdbNLITransform(model_name, tokenizer, max_length)

    abd_train_dataset = ListDataSet(sampled_o_train, nli_transformer)
    abd_dev_dataset = ListDataSet(o_abdnli_dev, nli_transformer)
    abd_test_dataset = ListDataSet(o_abdnli_test, nli_transformer)

    # train_sampler = RandomSampler(snli_dev_dataset)
    train_sampler = SequentialSampler(abd_train_dataset)
    abd_dev_sampler = SequentialSampler(abd_dev_dataset)
    abd_test_sampler = SequentialSampler(abd_test_dataset)

    if not args.cpu and not args.single_gpu:
        print("Use distributed sampler.")
        train_sampler = DistributedSampler(abd_train_dataset, args.world_size, args.global_rank,
                                           shuffle=True)

    batching_schema = {
        'uid': RawFlintField(),
        'y': LabelFlintField(),
        'input_ids_1': ArrayIndexFlintField(pad_idx=padding_token_value, left_pad=left_pad),
        'token_type_ids_1': ArrayIndexFlintField(pad_idx=padding_segement_value, left_pad=left_pad),
        'attention_mask_1': ArrayIndexFlintField(pad_idx=padding_att_value, left_pad=left_pad),

        'input_ids_2': ArrayIndexFlintField(pad_idx=padding_token_value, left_pad=left_pad),
        'token_type_ids_2': ArrayIndexFlintField(pad_idx=padding_segement_value, left_pad=left_pad),
        'attention_mask_2': ArrayIndexFlintField(pad_idx=padding_att_value, left_pad=left_pad),
    }

    train_dataloader = DataLoader(dataset=abd_train_dataset,
                                  batch_size=batch_size_per_gpu_train,
                                  shuffle=False,  #
                                  num_workers=0,
                                  pin_memory=True,
                                  sampler=train_sampler,
                                  collate_fn=BaseBatchBuilder(batching_schema))  #

    abd_dev_dataloader = DataLoader(dataset=abd_dev_dataset,
                                    batch_size=batch_size_per_gpu_eval,
                                    shuffle=False,  #
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=abd_dev_sampler,
                                    collate_fn=BaseBatchBuilder(batching_schema))  #

    abd_test_dataloader = DataLoader(dataset=abd_test_dataset,
                                     batch_size=batch_size_per_gpu_eval,
                                     shuffle=False,  #
                                     num_workers=0,
                                     pin_memory=True,
                                     sampler=abd_test_sampler,
                                     collate_fn=BaseBatchBuilder(batching_schema))  #
    if args.total_step <= 0:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * num_epoch
    else:
        t_total = args.total_step
    if args.warmup_steps <= 0:  # set the warmup steps to 0.1 * total step if the given warmup step is -1.
        args.warmup_steps = int(t_total * 0.1)

    # model = RobertaForSequenceClassification.from_pretrained(model_name,
    #                                                          cache_dir=str(config.PRO_ROOT / "trans_cache"),
    #                                                          num_labels=num_labels)

    if not args.cpu:
        torch.cuda.set_device(args.local_rank)
        model.cuda(args.local_rank)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if not args.cpu and not args.single_gpu:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                    output_device=local_rank, find_unused_parameters=True)

    args_dict = dict(vars(args))
    file_path_prefix = '.'
    if args.global_rank in [-1, 0]:
        print("Total Steps:", t_total)
        args.total_step = t_total
        print("Warmup Steps:", args.warmup_steps)
        print("Actual Training Batch Size:", actual_train_batch_size)
        print("Arguments", pp.pprint(args))

    is_finished = False

    # Let build the logger and log everything before the start of the first training epoch.
    if args.global_rank in [-1, 0]:  # only do logging if we use cpu or global_rank=0
        if not args.debug_mode:
            file_path_prefix, date = save_tool.gen_file_prefix(f"{args.experiment_name}")
            # # # Create Log File
            # Save the source code.
            script_name = os.path.basename(__file__)
            with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
                out_f.write(it.read())
                out_f.flush()

            # Save option file
            common.save_json(args_dict, os.path.join(file_path_prefix, "args.json"))
            checkpoints_path = Path(file_path_prefix) / "checkpoints"
            if not checkpoints_path.exists():
                checkpoints_path.mkdir()
            prediction_path = Path(file_path_prefix) / "predictions"
            if not prediction_path.exists():
                prediction_path.mkdir()

    global_step = 0

    # print(f"Global Rank:{args.global_rank} ### ", 'Init!')

    for epoch in tqdm(range(num_epoch), desc="Epoch", disable=args.global_rank not in [-1, 0]):
        print(debug_node_info(args), "epoch: ", epoch)

        if not args.cpu and not args.single_gpu:
            train_sampler.set_epoch(epoch)  # setup the epoch to ensure random sampling at each epoch

        for forward_step, batch in enumerate(tqdm(train_dataloader, desc="Iteration",
                                                  disable=args.global_rank not in [-1, 0]), 0):
            model.train()

            batch = move_to_device(batch, local_rank)
            # print(batch['input_ids'], batch['y'])
            if args.model_class_name in ["distilbert", "bart-large", "roberta-large", "roberta-base"]:
                outputs = model(input_ids_1=batch['input_ids_1'],
                                attention_mask_1=batch['attention_mask_1'],
                                input_ids_2=batch['input_ids_2'],
                                attention_mask_2=batch['attention_mask_2'],
                                labels=batch['y'], with_token_type=False)
            else:
                outputs = model(input_ids_1=batch['input_ids_1'],
                                attention_mask_1=batch['attention_mask_1'],
                                token_type_ids_1=batch['token_type_ids_1'],
                                input_ids_2=batch['input_ids_2'],
                                attention_mask_2=batch['attention_mask_2'],
                                token_type_ids_2=batch['token_type_ids_2'],
                                labels=batch['y'])

            loss, logits = outputs[:2]
            # print(debug_node_info(args), loss, logits, batch['uid'])
            # print(debug_node_info(args), loss, batch['uid'])

            # Accumulated loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # if this forward step need model updates
            # handle fp16
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

                # Gradient clip: if max_grad_norm < 0
            if (forward_step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                global_step += 1

                if args.global_rank in [-1, 0] and args.eval_frequency > 0 and global_step % args.eval_frequency == 0:
                    r_dict = dict()
                    pred_output_list = eval_model(model, abd_dev_dataloader, args.global_rank, args)
                    hit, total = count_acc(o_abdnli_dev, pred_output_list)
                    print(debug_node_info(args), "AbdNLI dev Acc:", hit, total, hit / total)
                    r_dict['abdnli_dev'] = {
                        'acc': hit / total,
                        'correct_count': hit,
                        'total_count': total,
                        'predictions': pred_output_list,
                    }

                    pred_output_list = eval_model(model, abd_test_dataloader, args.global_rank, args)
                    hit, total = count_acc(o_abdnli_test, pred_output_list)
                    print(debug_node_info(args), "AbdNLI test Acc:", hit, total, hit / total)
                    r_dict['abdnli_test'] = {
                        'acc': hit / total,
                        'correct_count': hit,
                        'total_count': total,
                        'predictions': pred_output_list,
                    }

                    # saving checkpoints
                    abdnli_dev_acc = r_dict['abdnli_dev']['acc']
                    abdnli_test_acc = r_dict['abdnli_test']['acc']

                    current_checkpoint_filename = \
                        f'e({epoch})|i({global_step})' \
                        f'|abdnli_dev({round(abdnli_dev_acc, 4)})' \
                        f'|abdnli_test({round(abdnli_test_acc, 4)})'

                    if not args.debug_mode and args.save_checkpoint:
                        # save model:
                        model_output_dir = checkpoints_path / current_checkpoint_filename
                        if not model_output_dir.exists():
                            model_output_dir.mkdir()
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training

                        torch.save(model_to_save.state_dict(), str(model_output_dir / "model.pt"))
                        torch.save(optimizer.state_dict(), str(model_output_dir / "optimizer.pt"))
                        torch.save(scheduler.state_dict(), str(model_output_dir / "scheduler.pt"))

                    # save prediction:
                    if not args.debug_mode and args.save_prediction:
                        cur_results_path = prediction_path / current_checkpoint_filename
                        if not cur_results_path.exists():
                            cur_results_path.mkdir(parents=True)
                        for key, item in r_dict.items():
                            common.save_jsonl(item['predictions'], cur_results_path / f"{key}.jsonl")

                if args.total_step > 0 and global_step == t_total:
                    # if we set total step and global step s t_total.
                    is_finished = True
                    break

        if args.global_rank in [-1, 0] and args.total_step <= 0:
            r_dict = dict()
            pred_output_list = eval_model(model, abd_dev_dataloader, args.global_rank, args)
            hit, total = count_acc(o_abdnli_dev, pred_output_list)
            print(debug_node_info(args), "AbdNLI dev Acc:", hit, total, hit / total)
            r_dict['abdnli_dev'] = {
                'acc': hit / total,
                'correct_count': hit,
                'total_count': total,
                'predictions': pred_output_list,
            }

            pred_output_list = eval_model(model, abd_test_dataloader, args.global_rank, args)
            hit, total = count_acc(o_abdnli_test, pred_output_list)
            print(debug_node_info(args), "AbdNLI test Acc:", hit, total, hit / total)
            r_dict['abdnli_test'] = {
                'acc': hit / total,
                'correct_count': hit,
                'total_count': total,
                'predictions': pred_output_list,
            }

            # saving checkpoints
            abdnli_dev_acc = r_dict['abdnli_dev']['acc']
            abdnli_test_acc = r_dict['abdnli_test']['acc']

            current_checkpoint_filename = \
                f'e({epoch})|i({global_step})' \
                f'|abdnli_dev({round(abdnli_dev_acc, 4)})' \
                f'|abdnli_test({round(abdnli_test_acc, 4)})'

            if not args.debug_mode and args.save_checkpoint:
                # save model:
                model_output_dir = checkpoints_path / current_checkpoint_filename
                if not model_output_dir.exists():
                    model_output_dir.mkdir()
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training

                torch.save(model_to_save.state_dict(), str(model_output_dir / "model.pt"))
                torch.save(optimizer.state_dict(), str(model_output_dir / "optimizer.pt"))
                torch.save(scheduler.state_dict(), str(model_output_dir / "scheduler.pt"))

            # save prediction:
            if not args.debug_mode and args.save_prediction:
                cur_results_path = prediction_path / current_checkpoint_filename
                if not cur_results_path.exists():
                    cur_results_path.mkdir(parents=True)
                for key, item in r_dict.items():
                    common.save_jsonl(item['predictions'], cur_results_path / f"{key}.jsonl")

        if is_finished:
            break

    # print(global_step)


id2label = {
    0: 1,
    1: 2,
    -1: '-',
}


def count_acc(gt_list, pred_list):
    assert len(gt_list) == len(pred_list)
    gt_dict = list_dict_data_tool.list_to_dict(gt_list, 'story_id')
    pred_list = list_dict_data_tool.list_to_dict(pred_list, 'uid')
    total_count = 0
    hit = 0
    for key, value in pred_list.items():
        if gt_dict[key]['label'] == value['predicted_label']:
            hit += 1
        total_count += 1
    return hit, total_count


def eval_model(model, dev_dataloader, device_num, args):
    model.eval()

    uid_list = []
    y_list = []
    pred_list = []
    logits_list = []

    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader, 0):
            batch = move_to_device(batch, device_num)

            if args.model_class_name in ["distilbert", "bart-large", "roberta-large", "roberta-base"]:
                outputs = model(input_ids_1=batch['input_ids_1'],
                                attention_mask_1=batch['attention_mask_1'],
                                input_ids_2=batch['input_ids_2'],
                                attention_mask_2=batch['attention_mask_2'],
                                labels=batch['y'], with_token_type=False)
            else:
                outputs = model(input_ids_1=batch['input_ids_1'],
                                attention_mask_1=batch['attention_mask_1'],
                                token_type_ids_1=batch['token_type_ids_1'],
                                input_ids_2=batch['input_ids_2'],
                                attention_mask_2=batch['attention_mask_2'],
                                token_type_ids_2=batch['token_type_ids_2'],
                                labels=batch['y'])

            loss, logits = outputs[:2]

            uid_list.extend(list(batch['uid']))
            y_list.extend(batch['y'].tolist())
            pred_list.extend(torch.max(logits, 1)[1].view(logits.size(0)).tolist())
            logits_list.extend(logits.tolist())

    assert len(pred_list) == len(logits_list)
    assert len(pred_list) == len(logits_list)

    result_items_list = []
    for i in range(len(uid_list)):
        r_item = dict()
        r_item['uid'] = uid_list[i]
        r_item['logits'] = logits_list[i]
        r_item['predicted_label'] = id2label[pred_list[i]]

        result_items_list.append(r_item)

    return result_items_list


def debug_node_info(args):
    names = ['global_rank', 'local_rank', 'node_rank']
    values = []

    for name in names:
        if name in args:
            values.append(getattr(args, name))
        else:
            return "Pro:No node info "

    return "Pro:" + '|'.join([f"{name}:{value}" for name, value in zip(names, values)]) + "||Print:"


if __name__ == '__main__':
    main()
